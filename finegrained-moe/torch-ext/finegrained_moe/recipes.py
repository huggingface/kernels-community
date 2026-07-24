# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import contextvars
import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal


import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ._ops import add_op_namespace_prefix
from .bayesian_autotuner import bayesian_autotune

from .compat import *  # noqa: F401,F403



def ue8m0_as_uint8(scale: torch.Tensor | None) -> torch.Tensor | None:
    """View UE8M0 (``float8_e8m0fnu``) weight scales as ``uint8`` for the Triton
    binder, which doesn't recognize the dtype; kernels decode ``2^(exp-127)``
    inline. fp32 (non-UE8M0) scales pass through unchanged; ``None`` (an absent
    optional scale) passes through as ``None`` — kernels take it as a dummy pointer."""
    if scale is None:
        return None
    return scale.view(torch.uint8) if scale.dtype == torch.float8_e8m0fnu else scale



def e2m1_as_uint8(weight: torch.Tensor) -> torch.Tensor:
    """View an ``int8``-stored MXFP4 (packed E2M1) weight as ``uint8`` — a zero-cost
    reinterpret. ``tl.dot_scaled`` requires the packed rhs as ``uint8``, so do it once here
    instead of casting in-kernel at every load. E4M3 (MXFP8) weights pass through unchanged."""
    return weight.view(torch.uint8) if weight.dtype == torch.int8 else weight



# UE8M0 group-32 scales arrive either as ``float8_e8m0fnu`` or as raw ``uint8`` — the same 8
# exponent bits, and a common on-disk encoding (e.g. group-32 "mxfp8" checkpoints store the
# scale tensor as uint8). Both are valid MX scales: ``ue8m0_as_uint8`` reinterprets to uint8
# and the kernels decode ``2^(exp-127)`` inline, so the detectors accept either dtype.
UE8M0_SCALE_DTYPES = (torch.float8_e8m0fnu, torch.uint8)



def _shapes_match(weight: torch.Tensor, scale: torch.Tensor, group: int) -> bool:
    """Shape leg of the family predicates: matching leading dims and a last dim of
    one scale per ``group`` unpacked values. Early-return ``if``s, not an ``and``
    chain: callers compare predicate results (``is_x(gate) != is_x(down)``), and an
    ``and`` chain hands them a lazy SymBool under dynamo — the resulting nested
    symbolic Eq crashes ``evaluate_expr``. Control flow forces each comparison to a
    real bool (weight shapes are static parameters, so the guards are correct)."""
    packed = NIBBLES_PER_BYTE if weight.dtype == torch.int8 else 1
    if scale.shape[:-1] != weight.shape[:-1]:
        return False
    if scale.shape[-1] != (weight.shape[-1] * packed) // group:
        return False
    return True



def is_mxfp8(weight: torch.Tensor, scale: torch.Tensor) -> bool:
    """MXFP8 weight/scale pair: E4M3 weights with UE8M0 group-32 scales — last dim
    ``scale.shape[-1] == weight.shape[-1] // MX_SCALE_GROUP_K``, matching leading dims.
    Works for 2D ``(N, K)`` and 3D ``(E, N, K)`` weights. The group-32 layout is what
    separates MXFP8 from 128-block FP8 (which may also carry UE8M0 scales)."""
    return (
        weight.dtype == torch.float8_e4m3fn
        and scale.dtype in UE8M0_SCALE_DTYPES
        and _shapes_match(weight, scale, MX_SCALE_GROUP_K)
    )



def is_mxfp4(weight: torch.Tensor, scale: torch.Tensor) -> bool:
    """MXFP4 weight/scale pair: packed E2M1 weights (``int8``, two codes/byte) with
    UE8M0 group-32 scales — ``scale.shape[-1] == weight.shape[-1] * NIBBLES_PER_BYTE //
    MX_SCALE_GROUP_K`` (unpacked K = ``2 * K_half``), matching leading dims. 2D or 3D."""
    return (
        weight.dtype == torch.int8
        and scale.dtype in UE8M0_SCALE_DTYPES
        and _shapes_match(weight, scale, MX_SCALE_GROUP_K)
    )



def is_nvfp4(weight: torch.Tensor, scale: torch.Tensor) -> bool:
    """NVFP4 weight/scale pair: packed E2M1 weights (``int8``, two codes/byte) with E4M3
    group-16 block scales — the scale DTYPE is the recipe carrier (UE8M0 = MX, E4M3 = NV), and
    the group falls out of the shape. This predicate reads the block scale; the per-tensor
    second-level global is a separate ``b_global_scale`` argument (``nvfp4_quantize_two_level``)."""
    return (
        weight.dtype == torch.int8
        and scale.dtype == torch.float8_e4m3fn
        and _shapes_match(weight, scale, NVFP4_SCALE_GROUP_K)
    )



def combine_global_scales(
    a_global_scale: torch.Tensor | None, b_global_scale: torch.Tensor | None, num_experts: int
) -> torch.Tensor | None:
    """The g_a · g_b product the MX kernels fold onto the accumulator (``AsBsGlobal`` at the kernel,
    ``input_global_scale`` at the wrapper), broadcast to ``(num_experts,)`` (grouped/batched index it
    per expert; the 2D op passes ``num_experts=1`` and reads it unindexed). Only the product matters
    for the acc — ``a_global_scale`` alone is passed separately for the inline-quant arm. Both
    operands' globals are calibrated/provided, never computed here; this just multiplies them.
    ``None`` if neither operand has a global."""
    if a_global_scale is None and b_global_scale is None:
        return None
    glob = (
        b_global_scale if a_global_scale is None
        else a_global_scale if b_global_scale is None
        else a_global_scale * b_global_scale
    )
    if glob.numel() == 1 and num_experts > 1:
        glob = glob.expand(num_experts)
    assert glob.numel() == num_experts, (
        f"global scale has {glob.numel()} elements, expected {num_experts} (per expert)"
    )
    return glob.contiguous()



def is_preswizzled_mx(weight: torch.Tensor, scale: torch.Tensor) -> bool:
    """A weight scale already in the SWIZZLE_32_4_4 layout, swizzled once at model load (the
    deployment contract — the same checkpoint feeds grouped prefill and batched decode with no
    per-call rearrange). The 5D shape ``(1, groups, cols//4, 2, 256)`` is the marker; the scale is
    a 1-byte block scale (UE8M0 for MXFP8/MXFP4, E4M3 for NVFP4) against an MX weight (E4M3 or
    packed E2M1). Recipe-agnostic — NVFP4 pre-swizzles the same way (the layout cuBLAS wants)."""
    return weight.dtype in (torch.float8_e4m3fn, torch.int8) and scale.ndim == 5



def is_mx(weight: torch.Tensor, scale: torch.Tensor) -> bool:
    """Any microscaled weight/scale pair — MXFP8 (``float8_e4m3fn`` values), MXFP4
    (``int8``, two E2M1 codes/byte), both UE8M0 group-32, or NVFP4 (packed E2M1 + E4M3
    group-16); also an already-swizzled MX scale (``is_preswizzled_mx``). The dispatchers route
    on this into the ``mx_*`` kernels; the op picks the format from the dtypes."""
    return (
        is_mxfp8(weight, scale)
        or is_mxfp4(weight, scale)
        or is_nvfp4(weight, scale)
        or is_preswizzled_mx(weight, scale)
    )



def is_tensor_wide(block_size, weight: torch.Tensor) -> bool:
    """True when ``block_size`` selects per-tensor (tensor-dynamic) scaling: ``None`` or
    equal to the weight's full ``(N, K)`` — one scale block spanning the whole matrix.
    Handles 2D ``(N, K)`` and 3D ``(E, N, K)`` weights via the last two dims. (2D path
    only — the grouped/batched dispatchers derive the recipe from the SCALE shape via
    ``weight_block_size``.)"""
    return block_size is None or (
        block_size[0] == weight.shape[-2] and block_size[1] == weight.shape[-1]
    )



def weight_block_size(B: torch.Tensor, Bs: torch.Tensor) -> list[int] | None:
    """The fp8 weight-quantization block ``[block_n, block_k]``, derived from the scale
    tensor's shape — the data already says how it was quantized, so no ``block_size``
    parameter exists to disagree with it. ``None`` = tensor-wide (one scalar per expert:
    ``Bs`` ``(E,)`` or ``(E, 1, 1)`` spanning the full ``(N, K)``). Expects a 3D
    ``(E, N, K)`` fp8 ``B``; MX weights never reach here (recipe keyed by scale dtype)."""
    num_experts, n_rows, K = B.shape
    if Bs.numel() == num_experts:
        return None
    assert Bs.ndim == 3 and K % Bs.shape[2] == 0, (
        f"Bs shape {tuple(Bs.shape)} does not tile B {tuple(B.shape)} along K"
    )
    # K tiles evenly, so block_k is exact; N (n_rows) may be non-aligned (a partial last block,
    # n_rows % Bs.shape[1] != 0) — recover block_n from the even K dim (square FP8 blocks). The
    # routed kernels then reject the non-aligned N via ``require_moe_dims_aligned``; matmul_2d masks it.
    block_k = K // Bs.shape[2]
    block_n = n_rows // Bs.shape[1] if n_rows % Bs.shape[1] == 0 else block_k
    return [block_n, block_k]



def validate_dense_operands(A: torch.Tensor, B: torch.Tensor) -> None:
    """Shared (rows, K) x (num_experts, N, K) operand checks for the unpacked recipes —
    the packed-E2M1 ops do their own (K spans two values per stored byte)."""
    assert A.ndim == 2, f"A must be 2D (rows, K), got ndim={A.ndim}"
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 3, f"B must be 3D (num_experts, N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"
    assert A.shape[1] == B.shape[2], (
        f"K mismatch: A has K={A.shape[1]}, B has K={B.shape[2]}"
    )



def validate_dense_2d_operands(A: torch.Tensor, B: torch.Tensor) -> None:
    """Shared (rows, K) x (N, K) operand checks for the 2D dense unpacked-fp8 wrappers — matching
    K, contiguous A, 2D contiguous B. The packed-E2M1 (MX) 2D op does its own (K is two values per
    stored byte, so B is (N, K // 2))."""
    assert A.shape[-1] == B.shape[-1], (
        f"K mismatch: A has K={A.shape[-1]}, B has K={B.shape[-1]}"
    )
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 2, f"B must be 2D (N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"



def expert_weight_shape(B: torch.Tensor, gate: bool) -> tuple[int, int, int]:
    """(num_experts, n_rows, N) of an expert weight stack — under a gate epilogue B is
    the (E, 2N, K) gate|up stack, so the per-projection width N is half the stored rows."""
    num_experts, n_rows, _ = B.shape
    return num_experts, n_rows, (n_rows // 2 if gate else n_rows)



def routed_rows(A, gather_idx, scatter_idx, expert_start, num_experts) -> int:
    """S (routed rows) for a grouped launch, carried by the (S,) maps — A's rows are
    gather SOURCES and under-count S whenever top_k > 1 (gate_up reading raw hidden);
    only with no maps at all is A itself the expert-sorted (S, K) matrix. Validates the
    maps and the ``(next_power_of_2(E) + 1,)`` ``expert_start`` schedule."""
    if gather_idx is not None:
        S = gather_idx.numel()
    elif scatter_idx is not None:
        S = scatter_idx.numel()
    else:
        S = A.shape[0]
    for perm_map in (gather_idx, scatter_idx):
        assert perm_map is None or (perm_map.numel() == S and perm_map.is_contiguous())
    assert (
        expert_start.is_contiguous()
        and expert_start.numel() == triton.next_power_of_2(num_experts) + 1
    ), "expert_start must be contiguous (next_power_of_2(num_experts) + 1,)"
    return S



def tokens_per_expert_bucket(S: int, num_experts: int) -> int:
    """log2 bucket of the average routed rows per expert — the grouped kernels' autotune
    key (raw S would retune per unique token count)."""
    return int((S + num_experts - 1) // num_experts).bit_length()



def normalize_per_expert_scale(Bs: torch.Tensor, num_experts: int) -> torch.Tensor:
    """One per-tensor scale per expert, normalized to ``(num_experts, 1, 1)`` from
    either that or a bare ``(num_experts,)``."""
    if Bs.ndim == 1:
        assert Bs.shape[0] == num_experts, (
            f"Bs shape {tuple(Bs.shape)} != expected ({num_experts},)"
        )
        return Bs.reshape(num_experts, 1, 1)
    assert Bs.shape == (num_experts, 1, 1), (
        f"Bs shape {tuple(Bs.shape)} != expected ({num_experts}, 1, 1)"
    )
    return Bs



def mx_scale_family(Bs: torch.Tensor, K: int) -> int:
    """The group size of an MX/NV weight-scale tensor, in either layout — the wrapper hands ``Bs``
    as-is and this reads the group off its shape. Row-major (2D/3D): ``K // Bs.shape[-1]``.
    SWIZZLE_32_4_4 (5D ``(1, blocks, cols // 4, 2, 256)``): ``K // (Bs.shape[2] * 4)``. The scale
    dtype IS the recipe carrier (E4M3 = NVFP4 group-16, UE8M0 = MX group-32) and the pairing is
    validated; callers that need the recipe read it off ``Bs.dtype`` (``== torch.float8_e4m3fn``
    is NVFP4)."""
    nvfp4 = Bs.dtype == torch.float8_e4m3fn
    assert nvfp4 or Bs.dtype in UE8M0_SCALE_DTYPES, (
        f"Bs must be UE8M0 (float8_e8m0fnu/uint8) or E4M3 (NVFP4), got {Bs.dtype}"
    )
    scale_group = K // (Bs.shape[2] * 4) if Bs.ndim == 5 else K // Bs.shape[-1]
    assert scale_group == (NVFP4_SCALE_GROUP_K if nvfp4 else MX_SCALE_GROUP_K), (
        f"scale group {scale_group} does not match the scale dtype {Bs.dtype}"
    )
    assert K % scale_group == 0, f"K (={K}) must be a multiple of {scale_group}"
    return scale_group



@dataclass(frozen=True)
class Epilogue:
    """Fused output TRANSFORM of a grouped/batched GEMM (default = plain GEMM) — pure math,
    no quantization (that is ``Quantization``'s side). ``gate`` loads the weight as the
    stacked gate|up projection and applies the ``act_fn``/SwiGLU gated linear unit.
    ``simulate_unfused`` (test-only) rounds each fused intermediate through the output
    dtype (the dispatchers' ``output_dtype`` argument, or its auto rule) to bit-match the
    separate-kernel path. Row order (gather/scatter) is NOT carried here — it is passed
    to the op as standalone ``gather_idx``/``scatter_idx`` maps."""

    gate: bool = False
    act_fn: str = "silu"
    swiglu_alpha: float | None = None
    swiglu_limit: float | None = None
    simulate_unfused: bool = False

    def as_args(self) -> tuple:
        """Flatten to the transform primitives the registered matmul ops take (torch custom
        ops can't accept the dataclass itself); the ops' bundles are ordered
        ``(*Epilogue.as_args(), *Quantization.as_args(), output_dtype)``."""
        return (
            self.gate,
            self.act_fn,
            self.swiglu_alpha,
            self.swiglu_limit,
            self.simulate_unfused,
        )



@dataclass(frozen=True)
class Quantization:
    """How tensors are quantized at the op boundaries — a recipe name per side, validated
    against the weight recipe (a mismatched name fails loudly at the op). ``None`` =
    follow the weights: the recipe's default quant on the way in, a plain high-precision
    store on the way out. A name means one format, identical on either side — an output
    feeds a matching input as-is, and requantized outputs are bit-identical to quantizing
    the same values offline. Pre-quantized activations are the ops' ``As`` parameter (its
    dtype carries the format); the plain-store element type is the dispatchers'
    ``output_dtype`` argument.

    Support matrix (weight recipe → accepted names; default first):

    ==================  =========================  =========================
    weights             input_recipe               output_recipe
    ==================  =========================  =========================
    block-dynamic FP8   "fp8" (E4M3 +              "fp8" (E4M3 +
                        per-block scales)          per-block scales)
    tensor-wide FP8     "fp8" (E4M3 +              —
                        per-token scales)
    MXFP8 / MXFP4       "mxfp8" (E4M3 + UE8M0),    "mxfp8" (E4M3 + UE8M0),
                        "mxfp4" (packed E2M1       "mxfp4" (packed E2M1
                        + UE8M0)                   + UE8M0)
    NVFP4               "nvfp4" (packed E2M1       "nvfp4" (packed E2M1
                        + E4M3 group-16)           + E4M3 group-16)
    full-precision      —                          —
    ==================  =========================  =========================

    (—: tensor-wide's whole-row activation scale can't be formed by a tile-local
    epilogue; the unfused path quantizes on the host between GEMMs.)"""

    input_recipe: Literal["fp8", "mxfp8", "mxfp4", "nvfp4"] | None = None
    output_recipe: Literal["fp8", "mxfp8", "mxfp4", "nvfp4"] | None = None

    def __post_init__(self):
        # catch typos at construction — the closest point to the user; the ops separately
        # assert which of these THEIR recipe implements
        assert self.input_recipe in (None, "fp8", "mxfp8", "mxfp4", "nvfp4"), (
            f"unknown input_recipe {self.input_recipe!r}; "
            "expected None, 'fp8', 'mxfp8', 'mxfp4', or 'nvfp4'"
        )
        assert self.output_recipe in (None, "fp8", "mxfp8", "mxfp4", "nvfp4"), (
            f"unknown output_recipe {self.output_recipe!r}; "
            "expected None, 'fp8', 'mxfp8', 'mxfp4', or 'nvfp4'"
        )

    def as_args(self) -> tuple:
        """Flatten to the fields as-is — ``(input_recipe, output_recipe)``; the registered
        ops interpret and validate them (each op knows which recipes it implements). The
        ops' bundles are ordered ``(*Epilogue.as_args(), *Quantization.as_args(),
        output_dtype)``."""
        return (self.input_recipe, self.output_recipe)



def resolve_input_recipe(
    input_recipe: str | None, output_recipe: str | None, Bs: torch.Tensor
) -> str:
    """GEMM-level activation recipe, keyed off the weight scales' dtype: NVFP4 weights
    (E4M3 scales) pin the whole family to ``"nvfp4"`` (the MMA kind needs matching
    scale formats on both operands); MX weights take E4M3 activations (``"mxfp8"``,
    the default) or packed E2M1 (``"mxfp4"``, W4A4). Validates both recipe names
    against the weight scale family. The MoE-level weight-following default (mxfp4
    weights -> mxfp4 acts) lives in ``moe._block_recipe``; the GEMM wrappers stay
    conservative."""
    if Bs.dtype == torch.float8_e4m3fn:
        assert input_recipe in (None, "nvfp4"), (
            f"NVFP4 activations are packed E2M1 + E4M3 scales, got {input_recipe!r}"
        )
        assert output_recipe in (None, "nvfp4"), (
            f"NVFP4 requantizes to 'nvfp4' (matching scale families), got {output_recipe!r}"
        )
        return "nvfp4"
    assert input_recipe in (None, "mxfp8", "mxfp4"), (
        f"MX activations are E4M3 ('mxfp8', the default) or packed E2M1 ('mxfp4'), "
        f"got {input_recipe!r}"
    )
    assert output_recipe in (None, "mxfp8", "mxfp4"), (
        f"MX recipes requantize to 'mxfp8' or packed 'mxfp4', got {output_recipe!r}"
    )
    return input_recipe or "mxfp8"



def resolve_output_dtype(
    output_dtype: torch.dtype | None,
    activation: torch.Tensor,
    act_scale: torch.Tensor | None,
) -> torch.dtype:
    """Output element type for a quantized matmul: the explicit ``output_dtype`` if given, else
    the raw activation dtype (``act_scale`` is None -> ``activation`` is high precision), else
    ``bfloat16`` (``activation`` is pre-quantized FP8, whose dtype is not a valid output)."""
    if output_dtype is not None:
        return output_dtype
    return activation.dtype if act_scale is None else torch.bfloat16
