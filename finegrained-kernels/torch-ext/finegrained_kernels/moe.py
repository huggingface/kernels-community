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

The base ops carry the gate|up fusion (SwiGLU + FP8/MX requant) and the gather/scatter row
maps, so both the fused and unfused MoE forwards are pure sequencing here — no MoE-specific
kernels live in this module:

  fused:   gate_up (``gate=True`` + ``quantize_output=True``) -> down -> ``weighted_reduce``. The
           SwiGLU + intermediate requant happen inside the gate_up kernel epilogue. A model's
           ``post_expert_norm`` runs on the down's routed rows, before the reduce.
  unfused: gate_up (plain GEMM) -> host ``apply_glu`` -> down (plain GEMM) -> ``weighted_reduce``.
           The activation + requant happen between two plain GEMMs; the GEMMs self-quantize their
           raw inputs (``As=None``). Same math as the fused path, split across kernels.

grouped (prefill) shares one on-device routing pass (``compute_grouped_scheduling``): gate_up
gathers hidden by routed row and leaves its output expert-ordered; down reads it in place and
scatters to routed rows. batched (decode) dispatches per token: ``gather_idx`` reads each routed
row from the unexpanded hidden in-kernel (no copy), and EP-sentinel rows (``id >= num_experts``)
are left uninit by the GEMM and skipped in ``weighted_reduce``. ``moe_fused_*`` / ``moe_unfused_*`` are format-neutral:
the base ops dispatch on the weight dtypes / scale layout (block-dynamic FP8, MXFP4/MXFP8,
NVFP4), and every forward takes ``activation_format`` naming the block's activation quantization
(``None`` = the weights' own format, ``"bf16"`` = weight-only, or an explicit format such as
``"mxfp8"`` on MXFP4 weights for W4A8)."""

import functools
from collections.abc import Callable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from .grouped import matmul_grouped
from .batched import GATE_UNSTACK_MAX_S, matmul_batched
from .bayesian_autotuner import bayesian_autotune
from .compat import (
    MX_SCALE_GROUP_K,
    NVFP4_SCALE_GROUP_K,
    ScalingType,
    SwizzleType,
    compile_time_only_triton_wrap,
    decode_pdl,
    device_context,
)
from .formats import get_supported_act_fns, is_mx, is_mxfp4, weight_format
from .norm import norm_column_factor, rms_inv_rows, rms_norm_rows
from .quant import _launch_act_quant
from .scheduling import compute_grouped_scheduling
from .epilogue import fused_glu


@bayesian_autotune(
    [
        triton.Config({"BLOCK_H": block_h}, num_warps=warps)
        for block_h in (256, 512, 1024, 2048)
        for warps in (4, 8)
    ],
    # the H tile width trades off against grid occupancy: at few groups (decode) narrow
    # tiles spread more H-blocks across SMs, at many groups (prefill) wide tiles amortize
    # the per-row weight load — so key on H and the group-count bucket.
    ["H", "num_groups_bit_length", "NORM"],
    n_trials=8,
)
@triton.jit
def weighted_reduce_kernel(
    Rows,  # (num_groups * NUM_TOP_K, H) — rows to reduce, group-major
    Out,  # (num_groups, H) — one reduced row per group
    Ids,  # (num_groups, NUM_TOP_K) — per-row id; a row is skipped when its id >= NUM_EXPERTS
    Weights,  # (num_groups * NUM_TOP_K,) — per-row scale
    NormWeight,  # (H,) fused post-expert norm weight; None (with NormInv) folds the arm out
    NormInv,  # (num_groups * NUM_TOP_K,) fp32 per-row rsqrt from rms_inv_rows
    H,
    stride_rows_m,
    stride_rows_h,
    stride_o_m,
    stride_o_h,
    stride_ids_m,
    stride_ids_k,
    num_groups_bit_length,  # autotune key only (log2 group-count bucket); unused in body
    NUM_TOP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NORM: tl.constexpr = None,  # a get_supported_norms() name folded into the reduce
    SIMULATE_UNFUSED: tl.constexpr = False,
    PDL: tl.constexpr = False,
):
    """Per group ``g``, the weighted sum of its ``NUM_TOP_K`` rows into ``Out[g]``:
    ``sum_k Weights[g*NUM_TOP_K + k] * Rows[g*NUM_TOP_K + k]``, skipping rows whose id is
    ``>= NUM_EXPERTS`` (out-of-range rows are never written upstream and contribute 0).
    fp32 accumulate; ~2.8x a generic ``view(g, k, H).sum(1)``. ``SIMULATE_UNFUSED`` rounds
    each weighted row to ``Out``'s dtype before summing, matching a reference that weights
    in that dtype; production leaves the accumulation in fp32.

    ``NORM`` folds a per-expert output norm into the pass: each row is scaled by its
    ``NormInv`` (``rms_inv_rows``, one pass ahead of this one) and the norm's column factor
    before the routing weight, so the normalized rows are never written or re-read."""
    if PDL:
        gdc_wait()
    g = tl.program_id(0)
    offs_h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs_h < H
    acc = tl.zeros((BLOCK_H,), tl.float32)
    if NORM is not None:  # the tile's column factor is row-independent — load it once
        factor = norm_column_factor(NormWeight, offs_h, mask, NORM)
    for k in tl.static_range(NUM_TOP_K):
        flat = g * NUM_TOP_K + k
        valid = tl.load(Ids + g * stride_ids_m + k * stride_ids_k) < NUM_EXPERTS
        weight = tl.load(Weights + flat)
        row = tl.load(
            Rows + flat * stride_rows_m + offs_h * stride_rows_h,
            mask=mask & valid,
            other=0.0,
        ).to(tl.float32)
        if NORM is not None:
            # masked like the value load: a sentinel row's rsqrt comes off uninitialized memory,
            # and 0 * NaN would poison the sum
            row = row * factor * tl.load(NormInv + flat, mask=valid, other=0.0)
            if SIMULATE_UNFUSED:  # the reference materializes the normalized row in Out's dtype
                row = row.to(Out.dtype.element_ty).to(tl.float32)
        contrib = weight * row
        if SIMULATE_UNFUSED:
            contrib = contrib.to(Out.dtype.element_ty).to(tl.float32)
        acc += contrib
    if PDL:
        gdc_launch_dependents()
    tl.store(
        Out + g * stride_o_m + offs_h * stride_o_h,
        acc.to(Out.dtype.element_ty),
        mask=mask,
    )



def weighted_reduce(
    rows: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_experts: int,
    simulate_unfused: bool = False,
    norm: str | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_inv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Routing-weighted top-k reduce — the bookend of the fused-MoE chain. Folds each token's
    ``num_top_k`` expert-output rows (``rows``, group-major, scaled by ``top_k_weights``, with
    EP-sentinel rows ``id >= num_experts`` skipped) from the routed-row layout back to
    ``(num_tokens, H)``. See ``weighted_reduce_kernel``. ``norm`` (a ``get_supported_norms()``
    name, with ``norm_weight`` and the ``rms_inv_rows`` ``norm_inv``) folds a per-expert output
    norm into this pass rather than normalizing the rows in one of their own."""
    assert (norm is None) == (norm_inv is None), "a fused norm needs its per-row rsqrt"
    num_tokens, num_top_k = top_k_index.shape
    H = rows.size(1)
    reduced = torch.empty(num_tokens, H, device=rows.device, dtype=rows.dtype)
    with device_context(rows.device):
        compile_time_only_triton_wrap(weighted_reduce_kernel)[
            lambda meta: (num_tokens, triton.cdiv(H, meta["BLOCK_H"]))
        ](
            rows,
            reduced,
            top_k_index,
            top_k_weights,
            norm_weight,
            norm_inv,
            H,
            rows.stride(0),
            rows.stride(1),
            reduced.stride(0),
            reduced.stride(1),
            top_k_index.stride(0),
            top_k_index.stride(1),
            num_groups_bit_length=int(num_tokens).bit_length(),
            NUM_TOP_K=num_top_k,
            NUM_EXPERTS=num_experts,
            NORM=norm,
            SIMULATE_UNFUSED=simulate_unfused,
            PDL=decode_pdl(),
            launch_pdl=decode_pdl(),
        )
    return reduced


def _validate_moe(gate_up_proj, gate_up_proj_scale, down_proj, down_proj_scale):
    """gate_up and down must share the format (both MX or both block-dynamic FP8 — the
    intermediate handed between them carries one quant format). Returns whether the format
    is MX (the fused dispatchers branch on it); the fp8 quantization block is derived from
    the scale shapes (``weight_block_size``), never passed. Scales are the pure block scales
    (per the decoupled API — per-tensor globals ride as separate ``*_global_scale`` args);
    the format predicates read the block scale's dtype/grouping."""
    gate_up_is_mx = is_mx(gate_up_proj, gate_up_proj_scale)
    if gate_up_is_mx != is_mx(down_proj, down_proj_scale):
        raise ValueError(
            "gate_up_proj and down_proj must use the same format (both MX or both block-dynamic FP8)."
        )
    if is_mxfp4(gate_up_proj, gate_up_proj_scale) != is_mxfp4(down_proj, down_proj_scale):
        raise ValueError("gate_up_proj and down_proj must use the same MX format (both MXFP4 or both MXFP8).")
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


def _host_glu(gate_up_out, act_fn, swiglu_alpha, swiglu_limit, gate):
    """The activation on the host, for the arms the epilogue does not fuse: a callable takes the
    raw GEMM output (interleaved gate|up columns when ``gate``) and returns the intermediate —
    the caller owns its semantics, so any activation runs without a kernel change; a name from
    ``get_supported_act_fns()`` runs the torch GLU (``gate``) or the bare activation."""
    if callable(act_fn):
        return act_fn(gate_up_out)
    if act_fn not in get_supported_act_fns():
        raise ValueError(f"act_fn must be one of {get_supported_act_fns()} or a callable, got {act_fn!r}")
    if gate:
        return fused_glu(gate_up_out, act_fn, swiglu_alpha, swiglu_limit)
    return {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}[act_fn](gate_up_out)


def _fused_glu(act_fn, swiglu_alpha, swiglu_limit, simulate_unfused, gate) -> dict:
    """The gate_up GEMM's fused-GLU kwargs when the activation is a fusable name; ``{}`` (a plain
    GEMM) sends a callable — or an unfusable name, rejected in ``_host_glu`` — to the host."""
    if not isinstance(act_fn, str) or act_fn not in get_supported_act_fns():
        return {}
    return dict(
        gate=gate,
        act_fn=act_fn,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        simulate_unfused=simulate_unfused,
    )


# ── Fused (gate_up epilogue owns SwiGLU + intermediate requant) ──────────────


def _block_format(gate_up_proj, gate_up_proj_scale, down_proj, down_proj_scale, activation_format):
    """The MoE block's activation format, resolved to a name: validates the weight pairing; an
    explicit ``activation_format`` (``"bf16"`` weight-only, or a format) is respected as-is,
    ``None`` follows the weights (fp8 / mxfp8 / mxfp4 / nvfp4 — mxfp4 weights default to mxfp4
    activations, the all-fp4 W4A4 chain; unquantized BF16/FP16 weights carry no scales, so their
    activations stay ``"bf16"``, the full-precision path)."""
    _validate_moe(gate_up_proj, gate_up_proj_scale, down_proj, down_proj_scale)
    if activation_format is not None:
        return activation_format
    if gate_up_proj_scale is None:
        return "bf16"
    return weight_format(gate_up_proj, gate_up_proj_scale)


def _post_expert_norm(down_out, post_expert_norm, weight, eps):
    """A model's per-expert output norm applied to the down projection's routed rows — one row
    per (token, expert) application, which is what such a norm is defined over. ``None`` passes
    through; a ``get_supported_norms()`` name runs ``rms_norm_rows`` against ``weight``; anything
    else is a host callable, like ``act_fn``'s."""
    if post_expert_norm is None:
        return down_out
    if isinstance(post_expert_norm, str):
        return rms_norm_rows(down_out, weight, eps, post_expert_norm)
    return post_expert_norm(down_out)


def _fused_post_expert_norm(down_out, post_expert_norm, weight, eps):
    """``(rows, weighted_reduce kwargs)`` for a fused forward: a named norm rides INTO the reduce
    — one pass for the row's ``rsqrt``, then the reduce it already runs applies that and the
    column factor — so the normalized rows are never materialized. A callable has to run first."""
    if isinstance(post_expert_norm, str):
        return down_out, {
            "norm": post_expert_norm,
            "norm_weight": weight,
            "norm_inv": rms_inv_rows(down_out, weight, eps, post_expert_norm),
        }
    return _post_expert_norm(down_out, post_expert_norm, weight, eps), {}


def moe_fused_grouped(
    hidden_states: torch.Tensor,  # (T, H)
    top_k_index: torch.Tensor,  # (T, K) int
    top_k_weights: torch.Tensor,  # (T, K)
    gate_up_proj: torch.Tensor,  # (E, 2I, H)
    down_proj: torch.Tensor,  # (E, H, I)
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    gate_up_proj_weight_global_scale: torch.Tensor | None = None,
    down_proj_weight_global_scale: torch.Tensor | None = None,
    gate_up_proj_input_global_scale: torch.Tensor | None = None,
    down_proj_input_global_scale: torch.Tensor | None = None,
    gate_up_proj_activation_scale: torch.Tensor | None = None,
    down_proj_activation_scale: torch.Tensor | None = None,
    post_expert_norm=None,  # the model's per-expert output norm on the routed rows: a
    # get_supported_norms() name (fused into the reduce) or a host callable
    post_expert_norm_weight: torch.Tensor | None = None,  # (H,) weight of a named norm
    post_expert_norm_eps: float = 1e-6,
    gate_up_proj_bias: torch.Tensor | None = None,  # (E, 2I) pre-activation bias
    down_proj_bias: torch.Tensor | None = None,  # (E, H)
    act_fn: str | Callable = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    activation_format: str | None = None,
    gate: bool = True,
) -> torch.Tensor:
    """Fused grouped MoE (prefill): gather gate_up + SiLU + requant epilogue → quantized
    expert-ordered intermediate → grouped down → routing-weighted top-k reduce. Returns
    ``(num_tokens, hidden_dim)``. The base ops dispatch on the weight dtypes / scale
    layout (block-dynamic FP8, MXFP8/MXFP4, NVFP4); ``activation_format`` names the activation
    quantization for the whole block — activations and the fused intermediate requant
    carry it (``"mxfp4"``/``"nvfp4"`` run all-fp4 W4A4 chains, ``"bf16"`` is weight-only);
    ``None`` follows the weight format, and the ops validate the pairing. The
    ``*_proj_input_global_scale`` pair is the NVFP4 activation second level (a checkpoint's
    calibrated per-projection ``input_scale``): the gate_up quantizes hidden against its own,
    requants the intermediate against the down's, and the down consumes that as its activation
    global. ``None`` is dynamic quant everywhere else. ``simulate_unfused`` (testing) rounds each step through
    the activation dtype so the output matches the unfused reference to reduce order. ``act_fn`` is a
    ``get_supported_act_fns()`` name (fused into the gate_up epilogue where the forward fuses) or any
    callable applied on the host to the raw gate_up output; ``gate=False`` runs an ungated
    projection. Scales are affine or pre-swizzled (``SWIZZLE_32_4_4``, self-describing 5-D)
    — the swizzled layout is for the dot_scaled arm; weight-only chains take affine."""
    fmt = _block_format(
        gate_up_proj, gate_up_proj_scale_inv, down_proj, down_proj_scale_inv, activation_format
    )
    num_top_k = top_k_index.size(-1)
    glu = _fused_glu(act_fn, swiglu_alpha, swiglu_limit, simulate_unfused, gate)
    NUM_EXPERTS = gate_up_proj.size(0)
    expert_start, gather_idx, scatter_idx = compute_grouped_scheduling(
        top_k_index, NUM_EXPERTS, num_top_k
    )

    # Phase 1: gate_up + SiLU + requant in the block format -> expert-ordered quantized
    # intermediate (the op quantizes the raw hidden itself and owns the expand-vs-gather
    # regime policy — this forward is pure sequencing). scatter_idx=None: the down reads
    # the intermediate in place. (C, Cs) under a requant format; a bare Tensor otherwise.
    static_act = gate_up_proj_activation_scale is not None or down_proj_activation_scale is not None
    gate_up_out = matmul_grouped(
        hidden_states,
        gate_up_proj,
        As=gate_up_proj_activation_scale,
        Bs=gate_up_proj_scale_inv,
        a_global_scale=gate_up_proj_input_global_scale,
        b_global_scale=gate_up_proj_weight_global_scale,
        # the intermediate requant normalizes against the DOWN's calibrated input global,
        # which the down then consumes as its activation global — the two-level handoff
        output_global_scale=down_proj_input_global_scale,
        expert_start=expert_start,
        **glu,
        bias=gate_up_proj_bias,
        # fmt is the resolved format; "bf16" (weight-only) leaves the GLU intermediate bf16,
        # no requant. So does static, whose epilogue has no calibrated scale to requant against.
        activation_format=fmt,
        quantize_output=bool(glu) and fmt != "bf16" and not static_act,
        output_dtype=hidden_states.dtype,
        gather_idx=gather_idx,
    )
    inter, inter_scale = (
        gate_up_out if isinstance(gate_up_out, tuple) else (gate_up_out, None)
    )
    if not glu:
        inter = _host_glu(inter, act_fn, swiglu_alpha, swiglu_limit, gate)
    # Phase 2: grouped down over the expert-ordered pre-quantized intermediate (its dtypes
    # carry the format; gather_idx=None), scattering to routed rows (scatter_idx).
    down_out = matmul_grouped(
        inter,
        down_proj,
        As=down_proj_activation_scale if static_act else inter_scale,
        Bs=down_proj_scale_inv,
        a_global_scale=down_proj_input_global_scale,
        b_global_scale=down_proj_weight_global_scale,
        expert_start=expert_start,
        bias=down_proj_bias,
        # weight-only: the intermediate is bf16 (As None) — the down goes weight-only too; a
        # quantized intermediate carries its format in its own dtypes.
        activation_format=fmt if inter_scale is None else None,
        output_dtype=hidden_states.dtype,
        scatter_idx=scatter_idx,
    )

    # Phase 3: routing-weighted top-k reduce -> (num_tokens, hidden_dim). simulate_unfused
    # rounds each weighted contrib to the activation dtype before summing, matching the
    # unfused path's torch reduce (which materializes bf16 contribs); production
    # accumulates in fp32.
    rows, norm_kwargs = _fused_post_expert_norm(
        down_out, post_expert_norm, post_expert_norm_weight, post_expert_norm_eps
    )
    return weighted_reduce(
        rows, top_k_index, top_k_weights, NUM_EXPERTS, simulate_unfused, **norm_kwargs
    )


def moe_fused_batched(
    hidden_states: torch.Tensor,  # (T, H)
    top_k_index: torch.Tensor,  # (T, K) int
    top_k_weights: torch.Tensor,  # (T, K)
    gate_up_proj: torch.Tensor,  # (E, 2I, H)
    down_proj: torch.Tensor,  # (E, H, I)
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    gate_up_proj_weight_global_scale: torch.Tensor | None = None,
    down_proj_weight_global_scale: torch.Tensor | None = None,
    gate_up_proj_input_global_scale: torch.Tensor | None = None,
    down_proj_input_global_scale: torch.Tensor | None = None,
    gate_up_proj_activation_scale: torch.Tensor | None = None,
    down_proj_activation_scale: torch.Tensor | None = None,
    post_expert_norm=None,  # the model's per-expert output norm on the routed rows: a
    # get_supported_norms() name (fused into the reduce) or a host callable
    post_expert_norm_weight: torch.Tensor | None = None,  # (H,) weight of a named norm
    post_expert_norm_eps: float = 1e-6,
    gate_up_proj_bias: torch.Tensor | None = None,  # (E, 2I) pre-activation bias
    down_proj_bias: torch.Tensor | None = None,  # (E, H)
    act_fn: str | Callable = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    activation_format: str | None = None,
    gate: bool = True,
) -> torch.Tensor:
    """Fused batched MoE (decode): gate_up + SiLU + requant epilogue → per-row quantized
    intermediate → batched down → routing-weighted top-k reduce. Returns
    ``(num_tokens, hidden_dim)``. The base ops dispatch on the weight dtypes / scale
    layout (block-dynamic FP8, MXFP8/MXFP4, NVFP4 — decode runs the software/swap arms
    below the native mxf4nvf4 M=128 staging); ``activation_format`` names the activation
    quantization for the whole block — activations and the fused intermediate requant
    carry it (``"mxfp4"`` runs the all-fp4 W4A4 chain, ``"bf16"`` is weight-only); ``None``
    follows the weight format, and the ops validate the pairing. The
    ``*_proj_input_global_scale`` pair is the NVFP4 activation second level (a checkpoint's
    calibrated per-projection ``input_scale``): the gate_up quantizes hidden against its own,
    requants the intermediate against the down's, and the down consumes that as its activation
    global. ``None`` is dynamic quant everywhere else. ``simulate_unfused`` (testing) rounds each
    step through the activation dtype so the output matches the unfused reference to
    reduce order. ``act_fn`` is a
    ``get_supported_act_fns()`` name (fused into the gate_up epilogue where the forward fuses) or any
    callable applied on the host to the raw gate_up output; ``gate=False`` runs an ungated
    projection. Scales are affine or pre-swizzled (``SWIZZLE_32_4_4``, self-describing 5-D)
    — the swizzled layout is for the dot_scaled arm; weight-only chains take affine."""
    fmt = _block_format(
        gate_up_proj, gate_up_proj_scale_inv, down_proj, down_proj_scale_inv, activation_format
    )
    glu = _fused_glu(act_fn, swiglu_alpha, swiglu_limit, simulate_unfused, gate)
    NUM_EXPERTS = gate_up_proj.size(0)
    expert_ids = top_k_index.reshape(-1)
    gather_idx = _gather_idx(top_k_index)

    # Phase 1: gate_up + SiLU + requant in the block format -> per-row quantized
    # intermediate (the op quantizes the raw activations). gather_idx reads each routed
    # row from the unexpanded hidden in-kernel (no copy).
    # (C, Cs) under a requant format; a bare Tensor on the full-precision path
    static_act = gate_up_proj_activation_scale is not None or down_proj_activation_scale is not None
    gate_up_out = matmul_batched(
        hidden_states,
        gate_up_proj,
        As=gate_up_proj_activation_scale,
        Bs=gate_up_proj_scale_inv,
        a_global_scale=gate_up_proj_input_global_scale,
        b_global_scale=gate_up_proj_weight_global_scale,
        # the two-level handoff, as in the grouped sibling
        output_global_scale=down_proj_input_global_scale,
        expert_ids=expert_ids,
        **glu,
        bias=gate_up_proj_bias,
        # Decode (batched): "bf16" (weight-only) leaves the intermediate bf16, no requant.
        # Block-FP8: INSIDE the unstacked decode band the requant fuses into the GLU kernel
        # (``fused_glu(quant_group=...)`` — one launch, hands the down a ready fp8+scales intermediate and
        # kills its offline act quant); ABOVE the band the stacked epilogue's requant pins
        # the gate|up tile to the whole block scale and halves the grid, so the bf16 handoff
        # (down inline-quants) stays the win there. Static keeps bf16 either way, its epilogue
        # having no calibrated scale to requant against.
        activation_format=fmt,
        quantize_output=(
            bool(glu)
            and fmt != "bf16"
            and (fmt != "fp8" or expert_ids.numel() <= GATE_UNSTACK_MAX_S)
            and not static_act
        ),
        output_dtype=hidden_states.dtype,
        gather_idx=gather_idx,
    )
    inter, inter_scale = (
        gate_up_out if isinstance(gate_up_out, tuple) else (gate_up_out, None)
    )
    if not glu:
        inter = _host_glu(inter, act_fn, swiglu_alpha, swiglu_limit, gate)
    # Phase 2: batched down over the intermediate (its dtypes carry the format; already
    # routed-order, no gather).
    down_out = matmul_batched(
        inter,
        down_proj,
        As=down_proj_activation_scale if static_act else inter_scale,
        Bs=down_proj_scale_inv,
        a_global_scale=down_proj_input_global_scale,
        b_global_scale=down_proj_weight_global_scale,
        expert_ids=expert_ids,
        bias=down_proj_bias,
        # weight-only / block-FP8: the intermediate is bf16 (As is None), so the down carries the
        # format and quantizes it, mirroring the unfused sibling; a quantized MX intermediate
        # carries its format in its own dtypes.
        activation_format=fmt if inter_scale is None or fmt == "fp8" else None,
        output_dtype=hidden_states.dtype,
    )
    # Phase 3: routing-weighted top-k reduce -> (num_tokens, hidden_dim). simulate_unfused
    # rounds each weighted contrib to the activation dtype before summing, matching the
    # unfused path's torch reduce (which materializes bf16 contribs); production
    # accumulates in fp32.
    rows, norm_kwargs = _fused_post_expert_norm(
        down_out, post_expert_norm, post_expert_norm_weight, post_expert_norm_eps
    )
    return weighted_reduce(
        rows, top_k_index, top_k_weights, NUM_EXPERTS, simulate_unfused, **norm_kwargs
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
    gate_up_proj_weight_global_scale: torch.Tensor | None = None,
    down_proj_weight_global_scale: torch.Tensor | None = None,
    gate_up_proj_input_global_scale: torch.Tensor | None = None,
    down_proj_input_global_scale: torch.Tensor | None = None,
    gate_up_proj_activation_scale: torch.Tensor | None = None,
    down_proj_activation_scale: torch.Tensor | None = None,
    post_expert_norm=None,  # the model's per-expert output norm on the routed rows: a
    # get_supported_norms() name (fused into the reduce) or a host callable
    post_expert_norm_weight: torch.Tensor | None = None,  # (H,) weight of a named norm
    post_expert_norm_eps: float = 1e-6,
    gate_up_proj_bias: torch.Tensor | None = None,  # (E, 2I) pre-activation bias
    down_proj_bias: torch.Tensor | None = None,  # (E, H)
    act_fn: str | Callable = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    activation_format: str | None = None,
    gate: bool = True,
) -> torch.Tensor:
    """Unfused grouped MoE: gate_up (plain grouped GEMM, gather hidden) → host ``apply_glu`` →
    down (plain grouped GEMM, scatter to routed rows) → routing-weighted reduce. Same math as
    ``moe_fused_grouped`` but the SwiGLU + intermediate quant happen between two plain GEMMs
    rather than inside the gate_up epilogue; each GEMM quantizes its raw input in
    ``activation_format`` (``None`` follows the weight format, mirroring the fused forward — mxfp4
    weights run the all-fp4 W4A4 chain). All formats route through the shared ``matmul_grouped``.
    The NVFP4 activation globals thread the same way as the fused sibling: each GEMM quantizes its
    raw input against its ``*_input_global_scale``, and a ``*_activation_scale`` quantizes it
    against that calibrated scale instead of a runtime one. ``act_fn`` is a
    ``get_supported_act_fns()`` name (fused into the gate_up epilogue where the forward fuses) or any
    callable applied on the host to the raw gate_up output; ``gate=False`` runs an ungated
    projection. Scales are affine or pre-swizzled (``SWIZZLE_32_4_4``, self-describing 5-D)
    — the swizzled layout is for the dot_scaled arm; weight-only chains take affine."""
    fmt = _block_format(
        gate_up_proj, gate_up_proj_scale_inv, down_proj, down_proj_scale_inv, activation_format
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
        As=gate_up_proj_activation_scale,
        Bs=gate_up_proj_scale_inv,
        a_global_scale=gate_up_proj_input_global_scale,
        b_global_scale=gate_up_proj_weight_global_scale,
        expert_start=expert_start,
        bias=gate_up_proj_bias,
        activation_format=fmt,
        output_dtype=hidden_states.dtype,
        gather_idx=gather_idx,
    )
    inter = _host_glu(gate_up_out, act_fn, swiglu_alpha, swiglu_limit, gate)
    # down over the expert-ordered intermediate (quantized in the same format), scattering
    # to routed rows.
    down_out = matmul_grouped(
        inter,
        down_proj,
        As=down_proj_activation_scale,
        Bs=down_proj_scale_inv,
        a_global_scale=down_proj_input_global_scale,
        b_global_scale=down_proj_weight_global_scale,
        expert_start=expert_start,
        bias=down_proj_bias,
        activation_format=fmt,
        output_dtype=hidden_states.dtype,
        scatter_idx=scatter_idx,
    )
    return _torch_weighted_reduce(
        _post_expert_norm(down_out, post_expert_norm, post_expert_norm_weight, post_expert_norm_eps),
        top_k_index, top_k_weights, NUM_EXPERTS,
    )


def moe_torch_grouped(
    hidden_states: torch.Tensor,  # (T, H)
    top_k_index: torch.Tensor,  # (T, K) int
    top_k_weights: torch.Tensor,  # (T, K)
    gate_up_proj: torch.Tensor,  # (E, 2I, H) E4M3
    down_proj: torch.Tensor,  # (E, H, I) E4M3
    gate_up_proj_scale_inv: torch.Tensor,  # gate_up scale through torchao's triton_mx_block_rearrange_per_group_3d (NOT swizzle_mx_scales)
    down_proj_scale_inv: torch.Tensor,  # down scale through the same torchao rearrange
    gate_up_proj_weight_global_scale: torch.Tensor | None = None,
    down_proj_weight_global_scale: torch.Tensor | None = None,
    gate_up_proj_input_global_scale: torch.Tensor | None = None,
    down_proj_input_global_scale: torch.Tensor | None = None,
    gate_up_proj_activation_scale: torch.Tensor | None = None,
    down_proj_activation_scale: torch.Tensor | None = None,
    post_expert_norm=None,  # the model's per-expert output norm on the routed rows: a
    # get_supported_norms() name (fused into the reduce) or a host callable
    post_expert_norm_weight: torch.Tensor | None = None,  # (H,) weight of a named norm
    post_expert_norm_eps: float = 1e-6,
    gate_up_proj_bias: torch.Tensor | None = None,  # (E, 2I) pre-activation bias
    down_proj_bias: torch.Tensor | None = None,  # (E, H)
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    activation_format: str | None = None,
) -> torch.Tensor:
    """Torch-only MX grouped MoE — the fair cuBLAS baseline for ``moe_fused_grouped`` /
    ``moe_unfused_grouped`` on the PUBLIC ``torch.nn.functional.scaled_grouped_mm``. Same weights,
    scales, and routing as our forwards; the only difference is the machinery torch forces:

    - Routing by **sort**, not our on-device gather/scatter: stable-argsort the ``T*K`` routed slots
      by expert into contiguous groups (cumulative ``offs``).
    - Two ``scaled_grouped_mm`` calls (per-format ``ScalingType``: group-32 ``BlockWise1x32`` for
      mxfp8/mxfp4, group-16 ``BlockWise1x16`` for nvfp4; fp4 operands viewed as ``e2m1_x2``).
    - Our Triton MX act-quant (so torch is timed on the same fast quant), the shared host ``apply_glu``,
      and the shared ``_torch_weighted_reduce``. All three MX formats.

    WEIGHT scales arrive already SWIZZLE_32_4_4-blocked by **torchao's**
    ``triton_mx_block_rearrange_per_group_3d`` (done once offline — a real deployment doesn't
    reblock a fixed weight every forward); this is scaled_grouped_mm's own layout, NOT the
    ``swizzle_mx_scales`` artifact the other four forwards consume. The timed loop only blocks the
    ACTIVATION scale (which changes each call). The format is read off the dtypes (the block
    preserves them: E4M3 scale = NVFP4, uint8 = MX; packed-E2M1 weight = int8) since the blocked
    shape no longer matches the group-shape detectors."""
    assert ScalingType is not None, (
        "this torch has no torch.nn.functional.ScalingType — the baseline's scaled_grouped_mm "
        "scaling enums arrived with the op itself, so an older torch cannot run it"
    )
    assert gate_up_proj.dtype in (torch.int8, torch.float8_e4m3fn), (
        "torch grouped baseline is MX-only (packed E2M1 or E4M3 weights)"
    )
    assert gate_up_proj_scale_inv.ndim not in (5, 6) and down_proj_scale_inv.ndim not in (5, 6), (
        "the torch baseline consumes torchao's triton_mx_block_rearrange_per_group_3d layout, "
        "not the swizzle_mx_scales artifact — a byte-compatible wrong layout would misread silently"
    )
    assert activation_format != "bf16", (
        "the torch baseline always quantizes activations (scaled_grouped_mm has no bf16-act x "
        "MX-weight form) — activation_format='bf16' (W4A16/W8A16) is not representable here"
    )
    assert gate_up_proj_activation_scale is None and down_proj_activation_scale is None, (
        "the torch baseline quantizes each activation against a scale it derives per call — a "
        "calibrated (static) scale is not representable here"
    )

    # torchao >= 0.18 required with cutlass-dsl >= 4.6 (0.17 imports a helper path 4.6
    # removed; fixed upstream in pytorch/ao).
    # torchao's per-group blocked-scale builder for the per-forward ACTIVATION scale (graph-capturable
    # @triton_op, same S+128·E static padding + SWIZZLE_32_4_4 layout scaled_grouped_mm consumes). The
    # weight scale is already blocked (offline); only the act scale is blocked here.
    from torchao.prototype.moe_training.kernels.mxfp8 import (
        triton_mx_block_rearrange_2d_M_groups,
    )

    nvfp4 = gate_up_proj_scale_inv.dtype == torch.float8_e4m3fn
    packed = gate_up_proj.dtype == torch.int8  # fp4 formats pack e2m1
    family = "nvfp4" if nvfp4 else "mxfp4" if packed else "mxfp8"
    act_format = family if activation_format is None else activation_format
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
    # GEMM's activation quant and rides as its TensorWise scale. MX formats are single-level.
    tensorwise = ScalingType.TensorWise

    def _tensorwise_global(g, n):  # (n,) fp32 TensorWise operand, identity when uncalibrated
        if g is None:
            return torch.ones(n, device=hidden_states.device, dtype=torch.float32)
        if g.numel() == 1:
            return g.reshape(-1).float().expand(n).contiguous()
        assert g.shape == (n,), (
            f"torch's scaled_grouped_mm takes ONE TensorWise global per expert, got "
            f"{tuple(g.shape)} — a per-expert activation global has no slot in this baseline "
            "(the Triton ops take one)"
        )
        return g.float()

    top_k = top_k_index.shape[1]
    out_dtype = hidden_states.dtype

    # route: stable-sort routed slots by expert into contiguous groups (torch has no gather/scatter fuse)
    flat_e = top_k_index.reshape(-1)
    order = torch.argsort(flat_e, stable=True)
    counts = torch.histc(flat_e.float(), bins=E, min=0, max=E - 1).to(torch.int32)
    offs = counts.cumsum(0).to(torch.int32)
    tok = (order // top_k).to(torch.long)  # source token of each sorted slot
    slot_e = flat_e[order].to(torch.long)  # expert of each sorted slot, for the per-expert biases

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
        assert a_g is None or a_g.numel() == 1, (
            "this baseline quantizes the routed rows in one pass, so the activation global is "
            "per tensor; per-expert activation globals are a Triton-op path"
        )
        # our Triton MX act-quant (format-taking launcher) — torch is timed on the same fast quant
        aq, a_s = _launch_act_quant(
            a, act_format, scale_group, scale_dtype, global_scale=a_g
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
        gate_up_proj_weight_global_scale,
        gate_up_proj_input_global_scale,
    )
    # torch has no fused bias, so this baseline adds both host-side; rows are expert-sorted here,
    # so the per-expert bias indexes by the sorted expert ids
    if gate_up_proj_bias is not None:
        gate_up = gate_up + gate_up_proj_bias[slot_e]
    inter = fused_glu(gate_up, act_fn, swiglu_alpha, swiglu_limit)
    down_out = grouped_mm(
        inter, down_proj, down_proj_scale_inv,
        down_proj_weight_global_scale, down_proj_input_global_scale,
    )

    # One weighted scatter-reduce: down_out is expert-sorted, so index_add_ over the source-token
    # map fuses unroute + routing-weight + top-k sum into (T, H) directly — no separate unsort pass.
    out = torch.zeros_like(hidden_states)
    w = top_k_weights.reshape(-1)[order].unsqueeze(-1).to(out.dtype)
    if down_proj_bias is not None:
        down_out = down_out + down_proj_bias[slot_e]
    down_out = _post_expert_norm(
        down_out, post_expert_norm, post_expert_norm_weight, post_expert_norm_eps
    )
    return out.index_add_(0, tok, down_out * w)


def moe_unfused_batched(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    gate_up_proj_weight_global_scale: torch.Tensor | None = None,
    down_proj_weight_global_scale: torch.Tensor | None = None,
    gate_up_proj_input_global_scale: torch.Tensor | None = None,
    down_proj_input_global_scale: torch.Tensor | None = None,
    gate_up_proj_activation_scale: torch.Tensor | None = None,
    down_proj_activation_scale: torch.Tensor | None = None,
    post_expert_norm=None,  # the model's per-expert output norm on the routed rows: a
    # get_supported_norms() name (fused into the reduce) or a host callable
    post_expert_norm_weight: torch.Tensor | None = None,  # (H,) weight of a named norm
    post_expert_norm_eps: float = 1e-6,
    gate_up_proj_bias: torch.Tensor | None = None,  # (E, 2I) pre-activation bias
    down_proj_bias: torch.Tensor | None = None,  # (E, H)
    act_fn: str | Callable = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    activation_format: str | None = None,
    gate: bool = True,
) -> torch.Tensor:
    """Unfused batched MoE: gate_up (plain batched GEMM, gather hidden) → host ``apply_glu`` →
    down (plain batched GEMM) → routing-weighted reduce. Same math as ``moe_fused_batched`` but
    the SwiGLU + intermediate quant happen between two plain GEMMs; each GEMM quantizes its raw
    input in ``activation_format`` (``None`` follows the weight format, ``"bf16"`` is weight-only). All
    formats route through the shared ``matmul_batched``. The NVFP4 activation globals thread the
    same way as the fused sibling: each GEMM quantizes its raw input against its
    ``*_input_global_scale``, and a ``*_activation_scale`` quantizes it against that calibrated
    scale instead of a runtime one. ``act_fn`` is a
    ``get_supported_act_fns()`` name (fused into the gate_up epilogue where the forward fuses) or any
    callable applied on the host to the raw gate_up output; ``gate=False`` runs an ungated
    projection. Scales are affine or pre-swizzled (``SWIZZLE_32_4_4``, self-describing 5-D)
    — the swizzled layout is for the dot_scaled arm; weight-only chains take affine."""
    fmt = _block_format(
        gate_up_proj, gate_up_proj_scale_inv, down_proj, down_proj_scale_inv, activation_format
    )
    NUM_EXPERTS = gate_up_proj.size(0)
    expert_ids = top_k_index.reshape(-1)
    gather_idx = _gather_idx(top_k_index)

    # gate_up as a plain GEMM (no gate epilogue) over gathered hidden -> (S, 2I).
    gate_up_out = matmul_batched(
        hidden_states,
        gate_up_proj,
        As=gate_up_proj_activation_scale,
        Bs=gate_up_proj_scale_inv,
        a_global_scale=gate_up_proj_input_global_scale,
        b_global_scale=gate_up_proj_weight_global_scale,
        expert_ids=expert_ids,
        bias=gate_up_proj_bias,
        activation_format=fmt,
        output_dtype=hidden_states.dtype,
        gather_idx=gather_idx,
    )
    inter = _host_glu(gate_up_out, act_fn, swiglu_alpha, swiglu_limit, gate)
    # down over the intermediate (quantized in the same format), routed-order output.
    down_out = matmul_batched(
        inter,
        down_proj,
        As=down_proj_activation_scale,
        Bs=down_proj_scale_inv,
        a_global_scale=down_proj_input_global_scale,
        b_global_scale=down_proj_weight_global_scale,
        expert_ids=expert_ids,
        bias=down_proj_bias,
        activation_format=fmt,
        output_dtype=hidden_states.dtype,
    )
    return _torch_weighted_reduce(
        _post_expert_norm(down_out, post_expert_norm, post_expert_norm_weight, post_expert_norm_eps),
        top_k_index, top_k_weights, NUM_EXPERTS,
    )
