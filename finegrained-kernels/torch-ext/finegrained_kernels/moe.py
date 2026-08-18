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

"""MoE forwards — thin orchestrations over the base ``matmul_grouped`` / ``matmul_batched`` ops.

The base ops carry the gate|up ``Epilogue`` (SwiGLU + FP8/MX requant) and the gather/scatter row
maps, so both the fused and unfused MoE forwards are pure sequencing here — no MoE-specific
kernels live in this module:

  fused:   gate_up (``Epilogue(gate=True)`` + ``Quantization(output_recipe=...)``) -> down -> ``weighted_reduce``. The
           SwiGLU + intermediate requant happen inside the gate_up kernel epilogue.
  unfused: gate_up (plain GEMM) -> host ``apply_glu`` -> down (plain GEMM) -> ``weighted_reduce``.
           The activation + requant happen between two plain GEMMs; the GEMMs self-quantize their
           raw inputs (``As=None``). Same math as the fused path, split across kernels.

grouped (prefill) shares one on-device routing pass (``compute_grouped_scheduling``): gate_up
gathers hidden by routed row and leaves its output expert-ordered; down reads it in place and
scatters to routed rows. batched (decode) dispatches per token: ``gather_idx`` reads each routed
row from the unexpanded hidden in-kernel (no copy), and EP-sentinel rows (``id >= num_experts``)
are left uninit by the GEMM and skipped in ``weighted_reduce``. ``moe_fused_*`` / ``moe_unfused_*`` are recipe-neutral:
the base ops dispatch on the weight dtypes / scale layout (block-dynamic FP8, MXFP4/MXFP8,
NVFP4), and the fused forwards take an optional ``recipe`` naming the block's activation
quantization."""

import functools

import torch

from .grouped import matmul_grouped
from .batched import GATE_UNSTACK_MAX_S, matmul_batched
from .compat import MX_SCALE_GROUP_K, NVFP4_SCALE_GROUP_K, weighted_reduce
from .recipes import Epilogue, Quantization, is_mx, is_mxfp4, weight_recipe
from .quant import _launch_act_quant
from .scheduling import compute_grouped_scheduling
from .epilogue import fused_glu


def _validate_moe(gate_up_proj, gate_up_proj_scale, down_proj, down_proj_scale):
    """gate_up and down must share the recipe (both MX or both block-dynamic FP8 — the
    intermediate handed between them carries one quant format). Returns whether the recipe
    is MX (the fused dispatchers branch on it); the fp8 quantization block is derived from
    the scale shapes (``weight_block_size``), never passed. Scales are the pure block scales
    (per the decoupled API — per-tensor globals ride as separate ``*_global_scale`` args);
    the recipe predicates read the block scale's dtype/grouping."""
    gate_up_is_mx = is_mx(gate_up_proj, gate_up_proj_scale)
    if gate_up_is_mx != is_mx(down_proj, down_proj_scale):
        raise ValueError(
            "gate_up_proj and down_proj must use the same recipe (both MX or both block-dynamic FP8)."
        )
    return gate_up_is_mx


def _gather_idx(top_k_index: torch.Tensor) -> torch.Tensor:
    """The batched routed-row gather: routed row ``s`` (``= t*K + k``) reads token ``s // num_top_k``
    of the unexpanded hidden. ``matmul_batched`` applies it in-kernel, so no ``(S, H)`` copy.
    The map depends only on the SHAPE, so it is cached per (tokens, top_k, device) — its two
    elementwise launches (~3µs) were the largest non-GEMM cost of the fp8 decode chain."""
    return _gather_idx_cached(
        top_k_index.shape[0], top_k_index.shape[1], top_k_index.device
    )


@functools.lru_cache(maxsize=64)
def _gather_idx_cached(
    num_tokens: int, num_top_k: int, device: torch.device
) -> torch.Tensor:
    return (
        torch.arange(num_tokens * num_top_k, device=device, dtype=torch.int32)
        // num_top_k
    )


def _torch_weighted_reduce(down_out, top_k_index, top_k_weights, num_experts):
    """Naive (unfused) routing-weighted top-k reduce in plain torch — NOT the fused
    ``weighted_reduce`` kernel. Materializes the (bf16) weighted contribs, masks EP-sentinel rows
    (``id >= num_experts``, left uninit in ``down_out``) to 0, and torch-sums to ``(num_tokens, H)``
    (fp32 accumulate, activation-dtype out). This is the independent reference the fused
    ``weighted_reduce`` is checked against; the fused path's ``simulate_unfused`` reproduces its
    bf16-contrib rounding."""
    num_tokens, num_top_k = top_k_index.shape
    dropped = (top_k_index.reshape(-1) >= num_experts).reshape(-1, 1)
    # masked in place on the PRODUCT: a sentinel row's down_out is uninitialized, so zeroing the
    # weight instead would leave 0 * NaN == NaN.
    contrib = down_out * top_k_weights.reshape(-1, 1)
    contrib.masked_fill_(dropped, 0)
    return contrib.view(num_tokens, num_top_k, down_out.size(1)).sum(dim=1)


# ── Fused (gate_up epilogue owns SwiGLU + intermediate requant) ──────────────


def _block_recipe(gate_up_proj, gate_up_proj_scale, down_proj, down_proj_scale, recipe):
    """The MoE block's activation recipe: validates the weight pairing; an explicit
    ``recipe`` is respected as-is, ``"weights"`` follows the weight recipe (fp8 / mxfp8 /
    mxfp4 / nvfp4 — mxfp4 weights default to mxfp4 activations, the all-fp4 W4A4
    chain; unquantized BF16/FP16 weights carry no scales and stay ``None``, the
    full-precision path)."""
    _validate_moe(gate_up_proj, gate_up_proj_scale, down_proj, down_proj_scale)
    if is_mxfp4(gate_up_proj, gate_up_proj_scale) != is_mxfp4(
        down_proj, down_proj_scale
    ):
        raise ValueError(
            "gate_up_proj and down_proj must use the same MX format (both MXFP4 or both MXFP8)."
        )
    if recipe != "weights":  # None (weight-only) or an explicit format — respected as-is
        return recipe
    if gate_up_proj_scale is None:
        return None
    return weight_recipe(gate_up_proj, gate_up_proj_scale)


def moe_fused_grouped(
    hidden_states: torch.Tensor,  # (T, H)
    top_k_index: torch.Tensor,  # (T, K) int
    top_k_weights: torch.Tensor,  # (T, K)
    gate_up_proj: torch.Tensor,  # (E, 2I, H)
    down_proj: torch.Tensor,  # (E, H, I)
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    gate_up_proj_global_scale: torch.Tensor | None = None,
    down_proj_global_scale: torch.Tensor | None = None,
    gate_up_input_global_scale: torch.Tensor | None = None,
    down_input_global_scale: torch.Tensor | None = None,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    recipe: str | None = "weights",
) -> torch.Tensor:
    """Fused grouped MoE (prefill): gather gate_up + SiLU + requant epilogue → quantized
    expert-ordered intermediate → grouped down → routing-weighted top-k reduce. Returns
    ``(num_tokens, hidden_dim)``. The base ops dispatch on the weight dtypes / scale
    layout (block-dynamic FP8, MXFP8/MXFP4, NVFP4); ``recipe`` names the activation
    quantization for the whole block — activations and the fused intermediate requant
    carry it ("mxfp4"/"nvfp4" run all-fp4 W4A4 chains), ``"weights"`` picks the weight
    family's recipe, and the ops validate the pairing. ``gate_up_input_global_scale`` / ``down_input_global_scale`` are the
    NVFP4 activation second level (a checkpoint's calibrated per-projection ``input_scale``):
    the gate_up quantizes hidden against its input global, requants the intermediate against
    the down's, and the down consumes it as its activation global — ``None`` (dynamic quant)
    everywhere else. ``simulate_unfused`` (testing) rounds each step through
    the activation dtype so the output matches the unfused reference to reduce order."""
    recipe = _block_recipe(
        gate_up_proj, gate_up_proj_scale_inv, down_proj, down_proj_scale_inv, recipe
    )
    num_top_k = top_k_index.size(-1)
    NUM_EXPERTS = gate_up_proj.size(0)
    expert_start, gather_idx, scatter_idx = compute_grouped_scheduling(
        top_k_index, NUM_EXPERTS, num_top_k
    )

    # Phase 1: gate_up + SiLU + requant in the block recipe -> expert-ordered quantized
    # intermediate (the op quantizes the raw hidden itself and owns the expand-vs-gather
    # regime policy — this forward is pure sequencing). scatter_idx=None: the down reads
    # the intermediate in place. (C, Cs) under a requant recipe; a bare Tensor otherwise.
    gate_up_out = matmul_grouped(
        hidden_states,
        gate_up_proj,
        Bs=gate_up_proj_scale_inv,
        a_global_scale=gate_up_input_global_scale,
        b_global_scale=gate_up_proj_global_scale,
        # the intermediate requant normalizes against the DOWN's calibrated input global,
        # which the down then consumes as its activation global — the two-level handoff
        output_global_scale=down_input_global_scale,
        expert_start=expert_start,
        epilogue=Epilogue(
            gate=True,
            act_fn=act_fn,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            simulate_unfused=simulate_unfused,
        ),
        # recipe is the resolved format; None (weight-only) leaves the GLU intermediate
        # bf16, no requant.
        quantization=Quantization(input_recipe=recipe, output_recipe=recipe),
        output_dtype=hidden_states.dtype,
        gather_idx=gather_idx,
    )
    inter, inter_scale = (
        gate_up_out if isinstance(gate_up_out, tuple) else (gate_up_out, None)
    )
    # Phase 2: grouped down over the expert-ordered pre-quantized intermediate (its dtypes
    # carry the recipe; gather_idx=None), scattering to routed rows (scatter_idx).
    down_out = matmul_grouped(
        inter,
        down_proj,
        As=inter_scale,
        Bs=down_proj_scale_inv,
        a_global_scale=down_input_global_scale,
        b_global_scale=down_proj_global_scale,
        expert_start=expert_start,
        # weight-only: the intermediate is bf16 (As None) — the down goes weight-only too.
        quantization=Quantization(input_recipe=recipe) if recipe is None else None,
        output_dtype=hidden_states.dtype,
        scatter_idx=scatter_idx,
    )

    # Phase 3: routing-weighted top-k reduce -> (num_tokens, hidden_dim). simulate_unfused
    # rounds each weighted contrib to the activation dtype before summing, matching the
    # unfused path's torch reduce (which materializes bf16 contribs); production
    # accumulates in fp32.
    return weighted_reduce(
        down_out, top_k_index, top_k_weights, NUM_EXPERTS, simulate_unfused
    )


def moe_fused_batched(
    hidden_states: torch.Tensor,  # (T, H)
    top_k_index: torch.Tensor,  # (T, K) int
    top_k_weights: torch.Tensor,  # (T, K)
    gate_up_proj: torch.Tensor,  # (E, 2I, H)
    down_proj: torch.Tensor,  # (E, H, I)
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    gate_up_proj_global_scale: torch.Tensor | None = None,
    down_proj_global_scale: torch.Tensor | None = None,
    gate_up_input_global_scale: torch.Tensor | None = None,
    down_input_global_scale: torch.Tensor | None = None,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    recipe: str | None = "weights",
) -> torch.Tensor:
    """Fused batched MoE (decode): gate_up + SiLU + requant epilogue → per-row quantized
    intermediate → batched down → routing-weighted top-k reduce. Returns
    ``(num_tokens, hidden_dim)``. The base ops dispatch on the weight dtypes / scale
    layout (block-dynamic FP8, MXFP8/MXFP4, NVFP4 — decode runs the software/swap arms
    below the native mxf4nvf4 M=128 staging); ``recipe`` names the activation
    quantization for the whole block — activations and the fused intermediate requant
    carry it ("mxfp4" runs the all-fp4 W4A4 chain), ``"weights"`` picks the weight family's
    recipe, and the ops validate the pairing. ``gate_up_input_global_scale`` / ``down_input_global_scale`` are the
    NVFP4 activation second level (a checkpoint's calibrated per-projection ``input_scale``):
    the gate_up quantizes hidden against its input global, requants the intermediate against
    the down's, and the down consumes it as its activation global — ``None`` (dynamic quant)
    everywhere else. ``simulate_unfused`` (testing) rounds each
    step through the activation dtype so the output matches the unfused reference to
    reduce order."""
    recipe = _block_recipe(
        gate_up_proj, gate_up_proj_scale_inv, down_proj, down_proj_scale_inv, recipe
    )
    NUM_EXPERTS = gate_up_proj.size(0)
    expert_ids = top_k_index.reshape(-1)
    gather_idx = _gather_idx(top_k_index)

    # Phase 1: gate_up + SiLU + requant in the block recipe -> per-row quantized
    # intermediate (the op quantizes the raw activations). gather_idx reads each routed
    # row from the unexpanded hidden in-kernel (no copy).
    # (C, Cs) under a requant recipe; a bare Tensor on the full-precision path
    gate_up_out = matmul_batched(
        hidden_states,
        gate_up_proj,
        Bs=gate_up_proj_scale_inv,
        a_global_scale=gate_up_input_global_scale,
        b_global_scale=gate_up_proj_global_scale,
        # the two-level handoff, as in the grouped sibling
        output_global_scale=down_input_global_scale,
        expert_ids=expert_ids,
        epilogue=Epilogue(
            gate=True,
            act_fn=act_fn,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            simulate_unfused=simulate_unfused,
        ),
        # Decode (batched): recipe None (weight-only) leaves the intermediate bf16, no requant.
        # Block-FP8: INSIDE the unstacked decode band the requant fuses into the GLU kernel
        # (``fused_glu(quant_group=...)`` — one launch, hands the down a ready fp8+scales intermediate and
        # kills its offline act quant); ABOVE the band the stacked epilogue's requant pins
        # the gate|up tile to the whole block scale and halves the grid, so the bf16 handoff
        # (down inline-quants) stays the win there.
        quantization=Quantization(
            input_recipe=recipe,
            output_recipe=(
                recipe
                if recipe != "fp8" or expert_ids.numel() <= GATE_UNSTACK_MAX_S
                else None
            ),
        ),
        output_dtype=hidden_states.dtype,
        gather_idx=gather_idx,
    )
    inter, inter_scale = (
        gate_up_out if isinstance(gate_up_out, tuple) else (gate_up_out, None)
    )
    # Phase 2: batched down over the intermediate (its dtypes carry the recipe; already
    # routed-order, no gather).
    down_out = matmul_batched(
        inter,
        down_proj,
        As=inter_scale,
        Bs=down_proj_scale_inv,
        a_global_scale=down_input_global_scale,
        b_global_scale=down_proj_global_scale,
        expert_ids=expert_ids,
        # weight-only / block-FP8: the intermediate is bf16 (As is None), so the down carries the recipe
        # and quantizes it, mirroring the unfused sibling.
        quantization=(
            Quantization(input_recipe=recipe) if recipe in (None, "fp8") else None
        ),
        output_dtype=hidden_states.dtype,
    )
    # Phase 3: routing-weighted top-k reduce -> (num_tokens, hidden_dim). simulate_unfused
    # rounds each weighted contrib to the activation dtype before summing, matching the
    # unfused path's torch reduce (which materializes bf16 contribs); production
    # accumulates in fp32.
    return weighted_reduce(
        down_out, top_k_index, top_k_weights, NUM_EXPERTS, simulate_unfused
    )


# ── Unfused (plain GEMMs + host GLU) ──────────────────────────────────────────


def moe_unfused_grouped(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    gate_up_proj_global_scale: torch.Tensor | None = None,
    down_proj_global_scale: torch.Tensor | None = None,
    gate_up_input_global_scale: torch.Tensor | None = None,
    down_input_global_scale: torch.Tensor | None = None,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    recipe: str | None = "weights",
) -> torch.Tensor:
    """Unfused grouped MoE: gate_up (plain grouped GEMM, gather hidden) → host ``apply_glu`` →
    down (plain grouped GEMM, scatter to routed rows) → routing-weighted reduce. Same math as
    ``moe_fused_grouped`` but the SwiGLU + intermediate quant happen between two plain GEMMs
    rather than inside the gate_up epilogue; each GEMM quantizes its raw input in ``recipe``
    (``"weights"`` follows the weight recipe, mirroring the fused forward — mxfp4 weights run the
    all-fp4 W4A4 chain). All recipes route through the shared ``matmul_grouped``. The NVFP4 activation globals thread the same way as the fused sibling:
    each GEMM quantizes its raw input against its ``*_input_global_scale``."""
    recipe = _block_recipe(
        gate_up_proj, gate_up_proj_scale_inv, down_proj, down_proj_scale_inv, recipe
    )

    num_top_k = top_k_index.size(-1)
    NUM_EXPERTS = gate_up_proj.size(0)
    expert_start, gather_idx, scatter_idx = compute_grouped_scheduling(
        top_k_index, NUM_EXPERTS, num_top_k
    )

    # gate_up as a plain GEMM (no gate epilogue) over gathered hidden -> expert-ordered (S, 2I).
    gate_up_out = matmul_grouped(
        hidden_states,
        gate_up_proj,
        Bs=gate_up_proj_scale_inv,
        a_global_scale=gate_up_input_global_scale,
        b_global_scale=gate_up_proj_global_scale,
        expert_start=expert_start,
        quantization=Quantization(input_recipe=recipe),
        output_dtype=hidden_states.dtype,
        gather_idx=gather_idx,
    )
    inter = fused_glu(gate_up_out, act_fn, swiglu_alpha, swiglu_limit)
    # down over the expert-ordered intermediate (quantized in the same recipe), scattering
    # to routed rows.
    down_out = matmul_grouped(
        inter,
        down_proj,
        Bs=down_proj_scale_inv,
        a_global_scale=down_input_global_scale,
        b_global_scale=down_proj_global_scale,
        expert_start=expert_start,
        quantization=Quantization(input_recipe=recipe),
        output_dtype=hidden_states.dtype,
        scatter_idx=scatter_idx,
    )
    return _torch_weighted_reduce(down_out, top_k_index, top_k_weights, NUM_EXPERTS)


def moe_torch_grouped(
    hidden_states: torch.Tensor,  # (T, H)
    top_k_index: torch.Tensor,  # (T, K) int
    top_k_weights: torch.Tensor,  # (T, K)
    gate_up_proj: torch.Tensor,  # (E, 2I, H) E4M3
    down_proj: torch.Tensor,  # (E, H, I) E4M3
    gate_up_proj_scale_inv: torch.Tensor,  # gate_up scale through torchao's triton_mx_block_rearrange_per_group_3d (NOT swizzle_mx_scales)
    down_proj_scale_inv: torch.Tensor,  # down scale through the same torchao rearrange
    gate_up_proj_global_scale: torch.Tensor | None = None,
    down_proj_global_scale: torch.Tensor | None = None,
    gate_up_input_global_scale: torch.Tensor | None = None,
    down_input_global_scale: torch.Tensor | None = None,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    recipe: str | None = "weights",
) -> torch.Tensor:
    """Torch-only MX grouped MoE — the fair cuBLAS baseline for ``moe_fused_grouped`` /
    ``moe_unfused_grouped`` on the PUBLIC ``torch.nn.functional.scaled_grouped_mm``. Same weights,
    scales, and routing as our forwards; the only difference is the machinery torch forces:

    - Routing by **sort**, not our on-device gather/scatter: stable-argsort the ``T*K`` routed slots
      by expert into contiguous groups (cumulative ``offs``).
    - Two ``scaled_grouped_mm`` calls (per-recipe ``ScalingType``: group-32 ``BlockWise1x32`` for
      mxfp8/mxfp4, group-16 ``BlockWise1x16`` for nvfp4; fp4 operands viewed as ``e2m1_x2``).
    - Our Triton MX act-quant (so torch is timed on the same fast quant), the shared host ``apply_glu``,
      and the shared ``_torch_weighted_reduce``. All three MX recipes.

    WEIGHT scales arrive already SWIZZLE_32_4_4-blocked by **torchao's**
    ``triton_mx_block_rearrange_per_group_3d`` (done once offline — a real deployment doesn't
    reblock a fixed weight every forward); this is scaled_grouped_mm's own layout, NOT the
    ``swizzle_mx_scales`` artifact the other four forwards consume. The timed loop only blocks the
    ACTIVATION scale (which changes each call). The recipe is read off the dtypes (the block
    preserves them: E4M3 scale = NVFP4, uint8 = MX; packed-E2M1 weight = int8) since the blocked
    shape no longer matches the group-shape detectors."""
    assert gate_up_proj.dtype in (torch.int8, torch.float8_e4m3fn), (
        "torch grouped baseline is MX-only (packed E2M1 or E4M3 weights)"
    )
    assert gate_up_proj_scale_inv.ndim not in (5, 6) and down_proj_scale_inv.ndim not in (5, 6), (
        "the torch baseline consumes torchao's triton_mx_block_rearrange_per_group_3d layout, "
        "not the swizzle_mx_scales artifact — a byte-compatible wrong layout would misread silently"
    )
    assert recipe is not None, (
        "the torch baseline always quantizes activations (scaled_grouped_mm has no bf16-act x "
        "MX-weight form) — recipe=None (W4A16/W8A16) is not representable here"
    )

    import torch.nn.functional as F
    from torch.nn.functional import ScalingType, SwizzleType

    # torchao >= 0.18 required with cutlass-dsl >= 4.6 (0.17 imports a helper path 4.6
    # removed; fixed upstream in pytorch/ao).
    # torchao's per-group blocked-scale builder for the per-forward ACTIVATION scale (graph-capturable
    # @triton_op, same S+128·E static padding + SWIZZLE_32_4_4 layout scaled_grouped_mm consumes). The
    # weight scale is already blocked (offline); only the act scale is blocked here.
    from torchao.prototype.moe_training.kernels.mxfp8 import (
        triton_mx_block_rearrange_2d_M_groups,
    )

    nvfp4 = gate_up_proj_scale_inv.dtype == torch.float8_e4m3fn
    packed = gate_up_proj.dtype == torch.int8  # fp4 recipes pack e2m1
    family = "nvfp4" if nvfp4 else "mxfp4" if packed else "mxfp8"
    act_recipe = family if recipe == "weights" else recipe
    scale_group = NVFP4_SCALE_GROUP_K if nvfp4 else MX_SCALE_GROUP_K
    scale_dtype = (
        torch.float8_e4m3fn if nvfp4 else torch.uint8
    )  # our act-quant/swizzle carry uint8
    # scaled_grouped_mm dispatches on the scale dtype — view the uint8 MX scales as e8m0 for it
    f_dtype = torch.float8_e4m3fn if nvfp4 else torch.float8_e8m0fnu
    SWZ = SwizzleType.SWIZZLE_32_4_4
    BW = ScalingType.BlockWise1x16 if nvfp4 else ScalingType.BlockWise1x32
    FP4 = getattr(torch, "float4_e2m1fn_x2", None)
    E = gate_up_proj.shape[0]
    # NVFP4's tcgen05 MMA kind requires TWO-level scaling: the e4m3 per-16 block scale AND a
    # per-tensor global fp32 scale — unlike our kernels there is no "no global" form, so a
    # missing calibrated global rides as identity 1.0 (dynamic quant). Weight globals arrive
    # as the separate *_global_scale args; a calibrated *_input_global_scale normalizes that
    # GEMM's activation quant and rides as its TensorWise scale. MX recipes are single-level.
    tensorwise = ScalingType.TensorWise

    def _tensorwise_global(g, n):  # (n,) fp32 TensorWise operand, identity when uncalibrated
        if g is None:
            return torch.ones(n, device=hidden_states.device, dtype=torch.float32)
        return g.reshape(-1).expand(n).contiguous() if g.numel() == 1 else g.reshape(n)

    top_k = top_k_index.shape[1]
    out_dtype = hidden_states.dtype

    # route: stable-sort routed slots by expert into contiguous groups (torch has no gather/scatter fuse)
    flat_e = top_k_index.reshape(-1)
    order = torch.argsort(flat_e, stable=True)
    counts = torch.histc(flat_e.float(), bins=E, min=0, max=E - 1).to(torch.int32)
    offs = counts.cumsum(0).to(torch.int32)
    tok = (order // top_k).to(torch.long)  # source token of each sorted slot

    def pk(t):  # view a packed-e2m1 operand as torch's fp4 dtype for scaled_grouped_mm
        return t.view(FP4) if packed else t

    def aswz(a_s):  # (S, K//G) -> per-group blocked layout, one launch
        return triton_mx_block_rearrange_2d_M_groups(a_s.view(torch.uint8), offs).view(
            f_dtype
        )

    def wswz(
        w_s,
    ):  # weight scale is pre-blocked offline (SWIZZLE_32_4_4) — pass through, no per-call kernel
        return w_s.view(f_dtype)

    def grouped_mm(a, w_q, w_s, w_g=None, a_g=None):
        assert a_g is None or nvfp4, "an activation global is NVFP4-only"  # match the ops
        # our Triton MX act-quant (recipe-taking launcher) — torch is timed on the same fast quant
        aq, a_s = _launch_act_quant(
            a, act_recipe, scale_group, scale_dtype, global_scale=a_g
        )
        sa, ra = aswz(a_s), BW
        sb, rb = wswz(w_s), BW
        if nvfp4:  # two-level: block e4m3 + the per-tensor/per-expert fp32 globals
            sa, ra = [sa, _tensorwise_global(a_g, 1)], [BW, tensorwise]
            sb, rb = [sb, _tensorwise_global(w_g, E)], [BW, tensorwise]
        return F.scaled_grouped_mm(
            pk(aq),
            pk(w_q).transpose(-2, -1),
            sa,
            ra,
            sb,
            rb,
            swizzle_a=SWZ,
            swizzle_b=SWZ,
            offs=offs,
            output_dtype=out_dtype,
        )

    gate_up = grouped_mm(
        hidden_states[tok],
        gate_up_proj,
        gate_up_proj_scale_inv,
        gate_up_proj_global_scale,
        gate_up_input_global_scale,
    )
    inter = fused_glu(gate_up, act_fn, swiglu_alpha, swiglu_limit)
    down_out = grouped_mm(
        inter, down_proj, down_proj_scale_inv, down_proj_global_scale, down_input_global_scale
    )

    # One weighted scatter-reduce: down_out is expert-sorted, so index_add_ over the source-token
    # map fuses unroute + routing-weight + top-k sum into (T, H) directly — no separate unsort pass.
    out = torch.zeros_like(hidden_states)
    w = top_k_weights.reshape(-1)[order].unsqueeze(-1).to(out.dtype)
    return out.index_add_(0, tok, down_out * w)


def moe_unfused_batched(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    gate_up_proj_global_scale: torch.Tensor | None = None,
    down_proj_global_scale: torch.Tensor | None = None,
    gate_up_input_global_scale: torch.Tensor | None = None,
    down_input_global_scale: torch.Tensor | None = None,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    recipe: str | None = "weights",
) -> torch.Tensor:
    """Unfused batched MoE: gate_up (plain batched GEMM, gather hidden) → host ``apply_glu`` →
    down (plain batched GEMM) → routing-weighted reduce. Same math as ``moe_fused_batched`` but
    the SwiGLU + intermediate quant happen between two plain GEMMs; each GEMM quantizes its raw
    input in ``recipe`` (``"weights"`` follows the weight recipe, ``None`` is weight-only). All
    recipes route through the shared ``matmul_batched``. The NVFP4 activation globals thread the same way as the fused sibling:
    each GEMM quantizes its raw input against its ``*_input_global_scale``."""
    recipe = _block_recipe(
        gate_up_proj, gate_up_proj_scale_inv, down_proj, down_proj_scale_inv, recipe
    )
    NUM_EXPERTS = gate_up_proj.size(0)
    expert_ids = top_k_index.reshape(-1)
    gather_idx = _gather_idx(top_k_index)

    # gate_up as a plain GEMM (no gate epilogue) over gathered hidden -> (S, 2I).
    gate_up_out = matmul_batched(
        hidden_states,
        gate_up_proj,
        Bs=gate_up_proj_scale_inv,
        a_global_scale=gate_up_input_global_scale,
        b_global_scale=gate_up_proj_global_scale,
        expert_ids=expert_ids,
        quantization=Quantization(input_recipe=recipe),
        output_dtype=hidden_states.dtype,
        gather_idx=gather_idx,
    )
    inter = fused_glu(gate_up_out, act_fn, swiglu_alpha, swiglu_limit)
    # down over the intermediate (quantized in the same recipe), routed-order output.
    down_out = matmul_batched(
        inter,
        down_proj,
        Bs=down_proj_scale_inv,
        a_global_scale=down_input_global_scale,
        b_global_scale=down_proj_global_scale,
        expert_ids=expert_ids,
        quantization=Quantization(input_recipe=recipe),
        output_dtype=hidden_states.dtype,
    )
    return _torch_weighted_reduce(down_out, top_k_index, top_k_weights, NUM_EXPERTS)
