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

import math

import torch
import triton

from .compat import MX_SCALE_GROUP_K, NIBBLES_PER_BYTE, NVFP4_SCALE_GROUP_K, UE8M0_DTYPE

# Amortize weight conversion across this many activation rows per GEMM.
FP8_WEIGHT_ONLY_DEQUANT_MIN_M = 1024

def ue8m0_as_uint8(scale: torch.Tensor | None) -> torch.Tensor | None:
    """View UE8M0 (``float8_e8m0fnu``) weight scales as ``uint8`` for the Triton
    binder, which doesn't recognize the dtype; kernels decode ``2^(exp-127)``
    inline. fp32 (non-UE8M0) scales pass through unchanged; ``None`` (an absent
    optional scale) passes through as ``None`` — kernels take the arg as None."""
    if scale is None:
        return None
    return scale.view(torch.uint8) if scale.dtype == UE8M0_DTYPE else scale

def e2m1_as_uint8(weight: torch.Tensor) -> torch.Tensor:
    """View an ``int8``-stored MXFP4 (packed E2M1) weight as ``uint8`` — a zero-cost
    reinterpret. ``tl.dot_scaled`` requires the packed rhs as ``uint8``, so do it once here
    instead of casting in-kernel at every load. E4M3 (MXFP8) weights pass through unchanged."""
    return weight.view(torch.uint8) if weight.dtype == torch.int8 else weight

# UE8M0 group-32 scales arrive either as ``float8_e8m0fnu`` or as raw ``uint8`` — the same 8
# exponent bits, and a common on-disk encoding (e.g. group-32 "mxfp8" checkpoints store the
# scale tensor as uint8). Both are valid MX scales: ``ue8m0_as_uint8`` reinterprets to uint8
# and the kernels decode ``2^(exp-127)`` inline, so the detectors accept either dtype.
UE8M0_SCALE_DTYPES = (torch.uint8,) if UE8M0_DTYPE is None else (UE8M0_DTYPE, torch.uint8)

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
    group-16 block scales, or ModelOpt's group-32 FP4/FP8 recipe. The dtype identifies the scale
    encoding, and the group falls out of the shape. This predicate reads the block scale; the per-tensor
    second-level global is a separate ``b_global_scale`` argument (``nvfp4_quantize_two_level``).
    Layout-agnostic questions ("is this NVFP4, swizzled or not?") go through ``weight_format``,
    which carries the same dtype rule without the shape check."""
    return (
        weight.dtype == torch.int8
        and scale.dtype == torch.float8_e4m3fn
        and (_shapes_match(weight, scale, NVFP4_SCALE_GROUP_K) or _shapes_match(weight, scale, MX_SCALE_GROUP_K))
    )

def is_preswizzled_mx(weight: torch.Tensor, scale: torch.Tensor) -> bool:
    """A weight scale already in the SWIZZLE_32_4_4 layout, swizzled once at model load (the
    deployment contract — the same checkpoint feeds grouped prefill and batched decode with no
    per-call rearrange). The 5D shape ``(G, row_blocks, cols//4, 2, 256)`` is the marker; the scale is
    a 1-byte block scale (UE8M0 for MXFP8/MXFP4, E4M3 for NVFP4) against an MX weight (E4M3 or
    packed E2M1). Format-agnostic — NVFP4 pre-swizzles the same way (the layout cuBLAS wants).
    6D is the gate|up artifact: the same blocks with the gate-interleave carried in the shape."""
    return (
        weight.dtype in (torch.float8_e4m3fn, torch.int8)
        and scale.ndim in (5, 6)
        and scale.element_size() == 1
    )

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
    only — the grouped/batched dispatchers derive the format from the SCALE shape via
    ``weight_block_size``.)"""
    return block_size is None or (
        block_size[0] == weight.shape[-2] and block_size[1] == weight.shape[-1]
    )

def is_rowwise(weight: torch.Tensor, scale: torch.Tensor) -> bool:
    """Row-wise FP8 weight/scale pair: E4M3 weights with one fp32 scale per output row —
    ``scale`` ``(N, 1)`` for a 2D ``(N, K)`` weight, ``(E, N, 1)`` for a 3D stack (e.g. Llama 4
    FP8). The block is ``(1, K)``: the scale is constant along K, so it folds once after the
    K-loop, like a per-tensor one."""
    if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32:
        return False
    if scale.shape != weight.shape[:-1] + (1,):
        return False
    return True

def per_token_activations(B: torch.Tensor, Bs: torch.Tensor | None) -> bool:
    """Whether the weights take one activation scale per token: FP8 with a K-constant scale
    (tensor-wide or row-wise). The scale spans the whole row, so a tile-local epilogue cannot
    form it — such a GEMM never fuses the requant of its output."""
    if Bs is None or is_mx(B, Bs):
        return False
    return Bs.numel() == (B.shape[0] if B.ndim == 3 else 1) or is_rowwise(B, Bs)

def weight_block_size(B: torch.Tensor, Bs: torch.Tensor) -> list[int] | None:
    """The fp8 weight-quantization block ``[block_n, block_k]``, derived from the scale
    tensor's shape — the data already says how it was quantized, so no ``block_size``
    parameter exists to disagree with it. ``None`` = a scale constant along K, folded after
    the K-loop by the tensor-wide kernels: one scalar per expert (``Bs`` ``(E,)`` or
    ``(E, 1, 1)``) or one per output row (``(E, N, 1)``, ``is_rowwise``). Expects a 3D
    ``(E, N, K)`` fp8 ``B``; MX weights never reach here (format keyed by scale dtype)."""
    num_experts, n_rows, K = B.shape
    if Bs.numel() == num_experts or is_rowwise(B, Bs):
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
    """Shared (rows, K) x (num_experts, N, K) operand checks for the unpacked formats —
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
    K, contiguous A, 2D B. The packed-E2M1 (MX) 2D op does its own (K is two values per
    stored byte, so B is (N, K // 2)).

    B may be row-major or a transposed view of a row-major tensor. The kernels read both of B's
    strides, and ``descriptor_box_pruner`` keeps the TMA arms — which require a unit-stride
    innermost dim — off the transposed form, so a caller can contract over the other axis by
    passing ``B.t()`` (with its scale grid transposed to match) instead of materializing a copy."""
    assert A.shape[-1] == B.shape[-1], (
        f"K mismatch: A has K={A.shape[-1]}, B has K={B.shape[-1]}"
    )
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 2, f"B must be 2D (N, K), got ndim={B.ndim}"
    assert B.is_contiguous() or B.t().is_contiguous(), (
        "B must be contiguous or a transposed view of a contiguous tensor, got strides "
        f"{tuple(B.stride())} for shape {tuple(B.shape)}"
    )

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

def normalize_global_scale(
    g: torch.Tensor | None, num_experts: int
) -> torch.Tensor | None:
    """A calibrated/global scale in the kernels' fp32 load layouts: ``(1,)`` per tensor
    or ``(num_experts,)`` per expert (``(E, 1, 1)`` accepted for the latter). The kernels index it
    through ``global_scale_stride``, so a shared value broadcasts by stride and nothing is
    materialized per call. ``None`` passes through.

    A gate|up stack calibrated per half (modelopt's ``(E, 2)`` ``weight_scale_2``) is merged to one
    global per expert at load — SwiGLU is linear in the up half, so its global folds into the down
    projection's — never here: the ops take one global per matrix."""
    if g is None:
        return None
    g = g.float()
    if g.numel() == 1:
        return g.reshape(1)
    assert g.numel() == num_experts, (
        f"global scale of shape {tuple(g.shape)}: expected a scalar or ({num_experts},)"
    )
    return g.reshape(num_experts).contiguous()

def is_per_expert_global(g: torch.Tensor | None) -> bool:
    """Whether an NVFP4 second-level global carries one value per expert rather than one for the
    whole tensor. The activation quant branches on it: a per-expert global has to be applied by
    each quantized row's own expert, which is what decides the grid that quant runs on."""
    return g is not None and g.numel() > 1

def global_scale_stride(g: torch.Tensor | None) -> int:
    """The expert stride of a ``normalize_global_scale`` layout: 0 for a scalar (or ``None``),
    which broadcasts it to every expert, else 1."""
    return 0 if g is None or g.numel() == 1 else 1

def normalize_per_expert_scale(Bs: torch.Tensor, num_experts: int) -> torch.Tensor:
    """A K-constant weight scale in the tensor-wide kernels' ``(num_experts, rows, 1)`` layout:
    one per-tensor scale per expert (``rows`` 1, from that or a bare ``(num_experts,)``), or a
    row-wise ``(num_experts, N, 1)`` passed through. The kernels read row ``n`` at
    ``n * stride(1)``, so ``row_scale_stride`` broadcasts the per-tensor one."""
    if Bs.ndim == 3 and Bs.shape[0] == num_experts and Bs.shape[1] > 1 and Bs.shape[2] == 1:
        return Bs
    if Bs.ndim == 1:
        assert Bs.shape[0] == num_experts, (
            f"Bs shape {tuple(Bs.shape)} != expected ({num_experts},)"
        )
        return Bs.reshape(num_experts, 1, 1)
    assert Bs.shape == (num_experts, 1, 1), (
        f"Bs shape {tuple(Bs.shape)} != expected ({num_experts}, 1, 1)"
    )
    return Bs

def row_scale_stride(Bs: torch.Tensor) -> int:
    """The output-row stride of a K-constant weight scale (``normalize_per_expert_scale``, or a
    2D ``(N, 1)`` / scalar one): 0 broadcasts a per-tensor scale to every row."""
    return 0 if Bs.ndim < 2 or Bs.shape[-2] == 1 else Bs.stride(-2)

def mx_scale_family(Bs: torch.Tensor, K: int) -> int:
    """The group size of an MX/NV weight-scale tensor, in either layout — the wrapper hands ``Bs``
    as-is and this reads the group off its shape. Row-major (2D/3D): ``K // Bs.shape[-1]``.
    SWIZZLE_32_4_4 (5D ``(G, row_blocks, cols // 4, 2, 256)``): ``K // (Bs.shape[2] * 4)``. The scale
    dtype identifies the scale encoding (E4M3 for NVFP4 group-16 or FP4/FP8 group-32; UE8M0 for
    MX group-32), while shape identifies its group width. Both are validated."""
    nvfp4 = Bs.dtype == torch.float8_e4m3fn
    assert nvfp4 or Bs.dtype in UE8M0_SCALE_DTYPES, (
        f"Bs must be UE8M0 (float8_e8m0fnu/uint8) or E4M3 (NVFP4), got {Bs.dtype}"
    )
    scale_group = K // (Bs.shape[2] * 4) if Bs.ndim == 5 else K // Bs.shape[-1]
    assert scale_group in ((NVFP4_SCALE_GROUP_K, MX_SCALE_GROUP_K) if nvfp4 else (MX_SCALE_GROUP_K,)), (
        f"scale group {scale_group} does not match the scale dtype {Bs.dtype}"
    )
    assert K % scale_group == 0, f"K (={K}) must be a multiple of {scale_group}"
    return scale_group

def get_supported_act_fns() -> tuple[str, ...]:
    """The activation names the gate|up epilogue fuses in-kernel; any other activation is applied
    on the host by the MoE forwards (``act_fn`` as a callable)."""
    return ("silu", "gelu", "relu")


ACTIVATION_FORMATS = (None, "bf16", "int8", "fp8", "mxfp8", "mxfp4", "nvfp4")


def check_activation_format(activation_format: str | None, quantize_output: bool) -> None:
    """Validate the dispatchers' activation kwargs. ``activation_format`` is the activations'
    quantization, in the same vocabulary transformers' ``FineGrainedConfig`` uses; the weights are
    never named — their format is read off the weight/scale dtypes (``weight_format``).

    - ``None`` (default): the weights' own format — W8A8 on FP8/MXFP8 weights, the all-fp4 W4A4
      chain on MXFP4/NVFP4 weights.
    - ``"bf16"``: weight-only (W4A16 / W8A16) — the activation stays high precision.
    - ``"fp8"`` / ``"mxfp8"`` / ``"mxfp4"`` / ``"nvfp4"``: an explicit activation format, e.g.
      ``"mxfp8"`` on MXFP4 weights is the W4A8 chain. A raw ``A`` is quantized to it by the op; an
      already-quantized ``A`` arrives with ``As`` (its dtype carries the format).

    ``quantize_output`` stores the (GLU) output requantized into that same format — bit-identical
    to quantizing the values offline — so it feeds the next op's activation side directly; the op
    then returns ``(C, Cs)``. A weight-only activation never requantizes.

    Per weight family:

    ==================  ===============================================================
    weight family       activation_format
    ==================  ===============================================================
    block-dynamic FP8   None / "fp8" (E4M3 + per-block scales); quantize_output ok
    tensor-wide FP8     None / "fp8" (E4M3 + per-token scales); no quantize_output
    row-wise FP8        None / "fp8" (E4M3 + per-token scales); no quantize_output
    MXFP8 / MXFP4       None / "mxfp8" / "mxfp4" / "bf16"; MXFP4 also calibrated "fp8"
    NVFP4               None / "nvfp4" (group-16) / "bf16"; group-32 requires calibrated "fp8"
    integer words       None / "bf16" (weight-only), calibrated "fp8" (INT4) or "int8" (INT8)
    full-precision      None (nothing to quantize)
    ==================  ===============================================================

    (A per-token activation scale spans the whole row, which a tile-local epilogue can't form; the
    next GEMM quantizes the intermediate instead.) A format the weight family doesn't support fails
    loudly at the op."""
    assert activation_format in ACTIVATION_FORMATS, (
        f"unknown activation_format {activation_format!r}; expected one of {ACTIVATION_FORMATS}"
    )
    assert not (quantize_output and activation_format == "bf16"), (
        "quantize_output needs quantized activations (activation_format='bf16' is weight-only)"
    )


def _validate_quantization_args(
    A, As, B, Bs, a_global_scale, b_global_scale, output_global_scale,
    b_zero, b_perm, a_pre_quant_scale, activation_format, quantize_output, activation_scale_ub,
):
    """Shared recipe and metadata checks for the dense and routed dispatchers."""
    assert a_pre_quant_scale is None or B.dtype in INT_WEIGHT_DTYPES or (
        Bs is not None and is_mx(B, Bs) and activation_format != "bf16"
    ), "a_pre_quant_scale requires integer weights or an MX/FP4 activation-quantized path"
    check_activation_format(activation_format, quantize_output)
    check_activation_scale_ub(A, As, B, Bs, activation_format, activation_scale_ub)
    assert b_zero is None or B.dtype in INT_WEIGHT_DTYPES, "zero points (b_zero) are the integer weights'"
    assert b_perm is None or B.dtype in INT_WEIGHT_DTYPES, "an act-order K order (b_perm) is the integer weights'"
    assert (a_global_scale is None and b_global_scale is None) or (
        Bs is not None and weight_format(B, Bs) == "nvfp4"
    ) or (
        B.dtype in INT_WEIGHT_DTYPES and activation_format == "fp8" and a_global_scale is None
    ), "global scales require NVFP4, or integer W4A8 weight_scale_2"
    assert output_global_scale is None or (
        Bs is not None and quantize_output and resolve_activation_format(activation_format, B, Bs) == "nvfp4"
    ), (
        "output_global_scale is the NVFP4 requant second level — it requires quantize_output=True on NVFP4 "
        "(the epilogue would otherwise normalize by it with nothing downstream to compensate)"
    )
    if B.dtype in INT_WEIGHT_DTYPES:
        assert not quantize_output and (
            (activation_format in (None, "bf16") and As is None and b_global_scale is None)
            or (activation_format == "fp8" and As is not None and b_global_scale is not None)
            or (activation_format == "int8" and As is not None and b_global_scale is None)
        ), "integer GEMMs support weight-only, calibrated INT4/FP8 AWQ, or calibrated INT8 SmoothQuant, without requant"


def check_activation_scale_ub(A, As, B, Bs, activation_format, activation_scale_ub):
    """An amax cap controls dynamic per-token FP8 quantization on row-wise weights."""
    if activation_scale_ub is None:
        return
    if not math.isfinite(activation_scale_ub) or activation_scale_ub <= 0:
        raise ValueError("activation_scale_ub must be finite and positive")
    if Bs is None or not is_rowwise(B, Bs) or activation_format not in (None, "fp8"):
        raise ValueError("activation_scale_ub requires row-wise FP8 weights and FP8 activations")
    if As is not None or A.dtype == torch.float8_e4m3fn:
        raise ValueError("activation_scale_ub requires raw activations without a calibrated activation scale")


INT_WEIGHT_BITS = (2, 4, 8)
# uint32: the kernels' interleaved layout; int32 routes to the integer ops only to be refused there (int_weight_bits)
INT_WEIGHT_DTYPES = (torch.uint32, torch.int32)


def int_weight_bits(K: int, B: torch.Tensor) -> int:
    """The value width of an integer weight ``B``: its ``K`` values per row ride in ``B.shape[-1]`` uint32 words,
    ``32 // bits`` each. The width is a fact of the shapes, so the data cannot disagree with a parameter. The dtype
    is the layout: uint32 words are the kernels' interleaved order (``interleave_int_words``, the GPTQ / AWQ
    repacks), and int32 words, a checkpoint's natural packing, are refused rather than read in the wrong order."""
    assert B.dtype == torch.uint32, (
        f"integer weights are uint32 words in the kernels' interleaved layout, got {B.dtype}: convert natural-order "
        "int32 packing with interleave_int_words (GPTQ / AWQ: repack_gptq_int4 / repack_awq_int4)"
    )
    words = B.shape[-1]
    bits = 32 * words // K
    assert bits in INT_WEIGHT_BITS and bits * K == 32 * words, (
        f"{words} uint32 words per row cannot hold K={K} values of 2, 4 or 8 bits"
    )
    return bits


def validate_int_operands(A, B, Bs, b_zero, b_perm, a_scale, b_scale2) -> int:
    """Shared dense/routed integer layout and calibrated activation recipe checks."""
    assert A.dtype in (torch.bfloat16, torch.float16), f"integer GEMMs take raw bf16/fp16 activations, got {A.dtype}"
    K = A.shape[-1]
    bits = int_weight_bits(K, B)
    assert Bs.ndim == B.ndim and Bs.shape[:-1] == B.shape[:-1] and Bs.shape[-1] > 0 and K % Bs.shape[-1] == 0, (
        "weight scales must match the weight rows and divide K into whole groups"
    )
    assert b_zero is None or b_zero.shape == Bs.shape, "zero points are shaped like the scales"
    assert b_perm is None or (b_perm.shape == (*B.shape[:-2], K) and b_perm.dtype == torch.int32), (
        "act-order K orders must be int32 with one K-vector per weight matrix"
    )
    assert b_scale2 is None or a_scale is not None, "weight_scale_2 needs a calibrated input_scale"
    if a_scale is not None:
        smoothquant = b_scale2 is None
        assert bits == (8 if smoothquant else 4) and b_zero is None and b_perm is None, (
            "ModelOpt activation-quantized integer recipes are symmetric and have no act-order"
        )
        if smoothquant:
            assert Bs.shape[-1] == 1 and K * 128 * 128 < 2**31, (
                "INT8 SmoothQuant needs per-channel scales and a K that cannot overflow INT32"
            )
    return bits


def validate_svdquant_operands(A, B, As, Bs, activation_format, output_dtype, lora_a, lora_b, pre_quant_scale):
    if lora_a is None and lora_b is None:
        return
    assert lora_a is not None and lora_b is not None, "SVDQuant requires both low-rank factors"
    assert (
        B.ndim == 2 and B.dtype == torch.int8 and Bs is not None
        and weight_format(B, Bs) == "nvfp4"
        and mx_scale_family(Bs, A.shape[-1]) == NVFP4_SCALE_GROUP_K
        and activation_format in (None, "nvfp4")
    ), "SVDQuant requires NVFP4 W4A4 weights"
    assert As is None and A.dtype == torch.bfloat16 and output_dtype in (None, torch.bfloat16), (
        "SVDQuant requires raw BF16 activations and BF16 output"
    )
    assert A.shape[-1] % 64 == 0, "SVDQuant's fused input preparation requires K divisible by 64"
    assert (
        lora_a.ndim == lora_b.ndim == 2 and lora_a.shape[1] == A.shape[-1]
        and lora_b.shape == (B.shape[0], lora_a.shape[0])
    ), "SVDQuant factors must have shapes (rank, K) and (N, rank)"
    if pre_quant_scale is not None:
        assert pre_quant_scale.shape == (A.shape[-1],) and pre_quant_scale.is_contiguous(), (
            "pre_quant_scale must be contiguous (K,)"
        )
        assert pre_quant_scale.dtype == A.dtype and pre_quant_scale.device == A.device, (
            "pre_quant_scale must match the BF16 activation device and dtype"
        )
    assert not torch.is_grad_enabled() or not any(t.requires_grad for t in (A, lora_a, lora_b)), (
        "SVDQuant currently supports inference only"
    )
    assert lora_a.shape[0] > 0 and lora_a.shape[0] % 32 == 0, "SVDQuant rank must be a positive multiple of 32"
    assert lora_a.dtype == lora_b.dtype == A.dtype and lora_a.device == lora_b.device == A.device, (
        "SVDQuant factors must match the BF16 activation device and dtype"
    )


def weight_format(B: torch.Tensor, Bs: torch.Tensor) -> str:
    """The format the weights are ALREADY in, read off their dtypes — what ``activation_format=None``
    resolves to. E4M3 scales -> ``"nvfp4"``; else packed E2M1 -> ``"mxfp4"``, MX E4M3 -> ``"mxfp8"``,
    block/tensor-scaled E4M3 -> ``"fp8"``; int32 / uint32 words of integer values -> ``"int"`` (2/4/8-bit, weight-only)."""
    if B.dtype in INT_WEIGHT_DTYPES:
        return "int"
    if Bs.dtype == torch.float8_e4m3fn:
        return "nvfp4"
    if not is_mx(B, Bs):
        return "fp8"
    # packed E2M1 is a WEIGHT dtype fact; is_mxfp4's shape check can't see it through a
    # pre-swizzled (5D) scale, which would silently downgrade a W4A4 default to W4A8.
    return "mxfp4" if B.dtype in (torch.int8, torch.uint8) else "mxfp8"




def resolve_activation_format(activation_format: str | None, B: torch.Tensor, Bs: torch.Tensor) -> str:
    """The format an MX/NVFP4 GEMM quantizes its activations to. ``None`` resolves to the weights'
    own format (``weight_format``): group-16 NVFP4 weights pin ``"nvfp4"`` (the MMA kind needs matching
    scale formats on both operands), packed E2M1 -> ``"mxfp4"`` (the all-fp4 W4A4 chain), else
    ``"mxfp8"``. An explicit name overrides (e.g. ``"mxfp8"`` on MXFP4 weights = the W4A8 chain)
    and is validated against the family. Packed group-32 weights also support calibrated ordinary
    ``"fp8"`` activations. ``"bf16"`` never reaches here — a weight-only chain is
    routed to its own kernel by the dispatcher."""
    if activation_format == "fp8":
        assert B.dtype == torch.int8, "ordinary FP8 activations on MX weights require packed FP4 weights"
        assert mx_scale_family(Bs, B.shape[-1] * 2) == MX_SCALE_GROUP_K, (
            "ModelOpt FP4/FP8 recipes use group-32 weight scales"
        )
        return "fp8"
    if Bs.dtype == torch.float8_e4m3fn:
        assert mx_scale_family(Bs, B.shape[-1] * 2) == NVFP4_SCALE_GROUP_K, (
            "E4M3 group-32 FP4 weights require activation_format='fp8'"
        )
        assert activation_format in (None, "nvfp4"), (
            f"NVFP4 activations are packed E2M1 + E4M3 scales, got {activation_format!r}"
        )
        return "nvfp4"
    assert activation_format in (None, "mxfp8", "mxfp4"), (
        f"MX activations are E4M3 ('mxfp8') or packed E2M1 ('mxfp4'), got {activation_format!r}"
    )
    return activation_format if activation_format is not None else weight_format(B, Bs)


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


def block_fp8_weight_scale_grid(B: torch.Tensor, Bs: torch.Tensor, interleaved_scales: bool = False):
    """Normalize scalar/block FP8 scales and infer their block size from the grid.

    A partial square block on N retains the size implied by K. Separately
    calibrated gate/up scale rows alternate per block under interleaved_scales.
    """
    assert B.dtype == torch.float8_e4m3fn and B.ndim in (2, 3)
    experts = B.shape[0] if B.ndim == 3 else 1
    rows, K = B.shape[-2:]
    if Bs.numel() == experts:
        return Bs.reshape(*B.shape[:-2], 1, 1), rows, K
    assert Bs.ndim == B.ndim and Bs.shape[:-2] == B.shape[:-2]
    assert K % Bs.shape[-1] == 0, "the weight scale grid must tile K evenly"
    block_k = K // Bs.shape[-1]
    if interleaved_scales and Bs.shape[-2] > 1:
        assert rows % 2 == 0 and Bs.shape[-2] % 2 == 0, "gate/up scale rows must occur in pairs"
        scale_rows, weight_rows = Bs.shape[-2] // 2, rows // 2
    else:
        scale_rows, weight_rows = Bs.shape[-2], rows
    block_n = block_k if ((weight_rows + block_k - 1) // block_k) == scale_rows else weight_rows // scale_rows
    assert block_n > 0 and ((weight_rows + block_n - 1) // block_n) == scale_rows
    return Bs, block_n, block_k
