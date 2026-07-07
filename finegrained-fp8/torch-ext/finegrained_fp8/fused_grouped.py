# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fused grouped MoE (prefill counterpart to the fused batched/decode path).

Two **persistent** kernels over expert-grouped M-tiles, no ``A_sorted`` materialization:
  1. Routing (on-device): an atomic counting-sort scatter (``_scatter_kernel``, O(S)) →
     ``perm`` / ``perm_token`` + an ``expert_start`` exclusive offset vector — the sort is
     just an index; activations stay unsorted.
  2. Phase 1 — gate_up, **gathering** each tile's hidden rows via ``perm_token``, + SiLU,
     then **FP8 requant** of the intermediate — same trick as ``fused_batched``: the down
     kernel reads FP8 directly instead of a bf16 round-trip.
  3. Phase 2 — grouped down over the FP8 intermediate, then **fuses the routing-weight ×
     reorder** (no atomics): each row is scaled by its weight and **scattered** to its flat
     ``(token, slot)`` row via ``perm``, leaving the host only a top-k sum back to (T, H).

Scheduling is persistent: a fixed ``NUM_SMS`` programs grid-stride over all tiles. The
expert-tile layout is built once into registers from ``expert_start`` (``_build_tile_layout``)
and each tile's owner is resolved inline (``_resolve_tile_inline``) — no padded grid, no
per-tile binary search, no host-side tile-offset precompute.

Recipe-named to mirror ``fused_batched``: ``w8a8_block_dynamic_fp8_moe_grouped`` (128x128
block scales, FP8 intermediate) and ``mxfp_dynamic_moe_grouped`` (MXFP4/MXFP8, UE8M0
group-32 scales, MXFP8 intermediate); ``moe_fused_grouped`` is the neutral dispatcher.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from .bayesian_autotuner import bayesian_autotune
from .grouped import store_tile
from .utils import (
    FP8_DTYPE,
    MX_SCALE_GROUP_K,
    NIBBLES_PER_BYTE,
    decode_ue8m0_scale,
    device_context,
    sm_count,
    fp8_act_quant_inline,
    topk_reduce_kernel,
    TOPK_REDUCE_BLOCK_H,
    get_accelerator_autotuning_configs,
    get_mxfp_autotuning_configs,
    is_mxfp,
    is_mxfp4,
    mxfp_act_quant_inline,
    mx_compute,
    smem_config_pruner,
    glu,
    e2m1_as_uint8,
    ue8m0_as_uint8,
)


# ── Persistent tile scheduling (register-resident expert-tile layout) ─────────

# Flat-slot tile per program for the O(S) routing kernels (count + scatter). These are small
# latency-bound atomic kernels that want many programs: a sweep over {256..4096} x prefill shapes
# put 256 best (or within ~1%) for both, with 1024 up to ~1.5x slower. The grid derives from it
# so the two can't drift. Power of 2.
_ROUTING_BLOCK_SIZE = 256


@triton.jit
def _build_tile_layout(
    ExpertStart, NUM_EXPERTS: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
):
    """Load ``expert_start`` once and derive the per-BM tile layout vectors (kept in
    registers for the whole persistent loop): per-expert first sorted row, token count,
    exclusive tile-start cumsum, and the total M-tile count. ``ExpertStart`` is
    ``(NUM_EXPERTS + 1,)`` with a trailing ``S`` sentinel (``expert_start[E] == S``)."""
    e_offs = tl.arange(0, NUM_EXPERTS)
    exp_start = tl.load(ExpertStart + e_offs)
    exp_end = tl.load(ExpertStart + e_offs + 1)
    freqs = exp_end - exp_start
    tiles_per_e = (freqs + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    tile_start_excl = (
        tl.cumsum(tiles_per_e, 0) - tiles_per_e
    )  # first tile index of expert e
    total_m_tiles = tl.sum(tiles_per_e, 0)
    return exp_start, freqs, tile_start_excl, total_m_tiles, e_offs


@triton.jit
def _resolve_tile_inline(
    pid_m, exp_start, freqs, tile_start_excl, e_offs, BLOCK_SIZE_M: tl.constexpr
):
    """Map an M-tile id to its owning expert + the tile's sorted row range, from the
    register-resident layout (no global loads). Returns ``(expert_id, sorted_indices,
    row_mask)``."""
    # Bucketize via the exclusive tile cumsum: #experts whose tile-start <= pid_m, minus 1.
    expert_id = tl.sum((tile_start_excl <= pid_m).to(tl.int32), 0) - 1
    sel = (
        e_offs == expert_id
    )  # scalar-index the E-vectors via mask-sum (no dynamic index)
    e_start = tl.sum(tl.where(sel, exp_start, 0), 0)
    e_tile_start = tl.sum(tl.where(sel, tile_start_excl, 0), 0)
    freq = tl.sum(tl.where(sel, freqs, 0), 0)
    within = pid_m - e_tile_start
    m_start = e_start + within * BLOCK_SIZE_M
    offs = tl.arange(0, BLOCK_SIZE_M)
    row_mask = offs < freq - within * BLOCK_SIZE_M
    sorted_indices = tl.max_contiguous(m_start + offs, BLOCK_SIZE_M)
    return expert_id, sorted_indices, row_mask


@triton.jit
def _exclusive_offsets_kernel(
    ExpertFreq, ExpertStart, Counters, NUM_EXPERTS: tl.constexpr
):
    """Exclusive cumsum of per-expert token counts → ``expert_start`` (leading 0, trailing
    S), and zero the scatter counters — one launch."""
    offs = tl.arange(0, NUM_EXPERTS)
    incl = tl.cumsum(tl.load(ExpertFreq + offs), 0)
    tl.store(ExpertStart, 0)
    tl.store(ExpertStart + 1 + offs, incl)
    tl.store(Counters + offs, tl.zeros([NUM_EXPERTS], tl.int32))


@triton.jit
def _scatter_kernel(
    ExpertIds,
    Perm,
    PermToken,
    ExpertStart,
    Counters,
    S,
    NUM_TOP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Counting-sort scatter: each flat slot atomically claims the next slot of its expert
    (``expert_start[e] + counter[e]++``). O(S), replaces an O(S·logS) argsort. Within-expert
    order is arbitrary (atomic race) — fine, the per-token reduce is order-invariant. Slots whose
    expert is non-local (EP sentinel id ``>= NUM_EXPERTS``) are skipped — matches ``_count_kernel``,
    and avoids the atomic/store landing at an out-of-range (invalid) global address."""
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    expert_id = tl.load(ExpertIds + offs, mask=offs < S, other=NUM_EXPERTS)
    valid = expert_id < NUM_EXPERTS
    dest = tl.load(ExpertStart + expert_id, mask=valid, other=0) + tl.atomic_add(
        Counters + expert_id, 1, mask=valid
    )
    tl.store(Perm + dest, offs, mask=valid)
    tl.store(PermToken + dest, offs // NUM_TOP_K, mask=valid)


@triton.jit
def _count_kernel(
    ExpertIds, ExpertFreq, S, NUM_EXPERTS: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """Per-expert token count via atomics — replaces ``torch.histc`` (no float cast), fixed
    ``(NUM_EXPERTS,)`` output stays CUDA-graph friendly. ``ExpertFreq`` is pre-zeroed."""
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < S
    expert_id = tl.load(ExpertIds + offs, mask=mask, other=NUM_EXPERTS)
    tl.atomic_add(ExpertFreq + expert_id, 1, mask=mask & (expert_id < NUM_EXPERTS))


def _grouped_routing(expert_ids: torch.Tensor, num_experts: int, num_top_k: int):
    """On-device routing: expert-sorted index (no copy of the activations) via two Triton
    launches — exclusive offsets + an atomic counting-sort scatter (replaces host
    ``argsort``). Returns ``(perm_token, perm, expert_start, num_experts, num_routed_tokens)``.
    ``perm`` is the sorted-position → flat
    ``(t*K + j)`` map (gate_up gathers via ``perm_token = perm // K``, down scatters via
    ``perm``); ``expert_start`` is ``(E+1,)`` padded with S so the kernels build the tile
    layout in-register (E is a power of 2)."""
    device = expert_ids.device
    expert_ids = expert_ids.int()
    num_routed_tokens = expert_ids.numel()  # S = num_tokens * num_top_k
    expert_freq = torch.zeros(num_experts, dtype=torch.int32, device=device)
    expert_start = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    counters = torch.empty(num_experts, dtype=torch.int32, device=device)
    perm = torch.empty(num_routed_tokens, dtype=torch.int32, device=device)
    perm_token = torch.empty(num_routed_tokens, dtype=torch.int32, device=device)
    with device_context(device):
        _count_kernel[(triton.cdiv(num_routed_tokens, _ROUTING_BLOCK_SIZE),)](
            expert_ids,
            expert_freq,
            num_routed_tokens,
            NUM_EXPERTS=num_experts,
            BLOCK_SIZE=_ROUTING_BLOCK_SIZE,
        )
        _exclusive_offsets_kernel[(1,)](
            expert_freq,
            expert_start,
            counters,
            NUM_EXPERTS=num_experts,
        )
        _scatter_kernel[(triton.cdiv(num_routed_tokens, _ROUTING_BLOCK_SIZE),)](
            expert_ids,
            perm,
            perm_token,
            expert_start,
            counters,
            num_routed_tokens,
            NUM_TOP_K=num_top_k,
            NUM_EXPERTS=num_experts,
            BLOCK_SIZE=_ROUTING_BLOCK_SIZE,
        )
    return perm_token, perm, expert_start, num_experts, num_routed_tokens


# MX MoE autotune axes: the MMA flavor and the weight-load mechanism. The kernels implement all
# three memory modes; get_mxfp_autotuning_configs drops the device-inapplicable descriptor
# flavor (host on XPU, device on CUDA).
_MX_COMPUTE_MODES = ("dot_scaled", "dot")
_MX_MEMORY_MODES = ("descriptor", "pointer")


# ── Block-dynamic FP8 ────────────────────────────────────────────────────────


@bayesian_autotune(
    get_accelerator_autotuning_configs(tune_block_m=True),
    ["INTERMEDIATE_DIM", "HIDDEN_DIM", "tokens_per_sm_bit_length"],
    n_trials=60,
    # bf16 activation tile + fused gate|up weight tiles
    prune_configs_by={
        "early_config_prune": smem_config_pruner(act_bytes=2, n_weight_tiles=2)
    },
)
@triton.jit
def w8a8_block_dynamic_fp8_moe_grouped_gate_up_kernel(
    Hidden,  # (T, H) raw activations, UNSORTED
    PermToken,  # (S,) int32 — sorted position -> source token id
    GateUp,  # (E, 2I, H) FP8
    GateUpScale,  # (E, 2I//bn, H//bk) UE8M0 block scales
    ExpertStart,  # (NUM_EXPERTS+1,) int32 — exclusive sorted-row start per expert (pad S)
    Inter,  # (S, I) FP8 — output (expert-sorted)
    InterScale,  # (S, NUM_I_TILES) fp32 — per-row, per-I-tile activation scale
    stride_h_t,
    stride_h_k,
    stride_gu_e,
    stride_gu_n,
    stride_gu_k,
    stride_gus_e,
    stride_gus_n,
    stride_gus_k,
    stride_int_m,
    stride_int_n,
    stride_is_m,
    stride_is_n,
    stride_pt,
    tokens_per_sm_bit_length,  # autotune key only (log2 tokens-per-SM bucket); unused in body
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
):
    """Phase 1: persistent grid-stride over (M-tile, I-tile). Gather hidden rows per expert
    M-tile, gate + up block-FP8 matmuls, SiLU-combine, FP8-requant the intermediate."""
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = _build_tile_layout(
        ExpertStart, NUM_EXPERTS, BLOCK_SIZE_M
    )
    num_n_tiles = tl.cdiv(INTERMEDIATE_DIM, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles
        expert_id, offs_global_m, row_mask = _resolve_tile_inline(
            pid_m, exp_start, freqs, tile_start_excl, e_offs, BLOCK_SIZE_M
        )
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        token = tl.load(PermToken + offs_global_m * stride_pt, mask=row_mask, other=0)
        a_ptrs = Hidden + token[:, None] * stride_h_t + offs_k[None, :] * stride_h_k
        gate_ptr = (
            GateUp
            + expert_id * stride_gu_e
            + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_gu_k
            + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_gu_n
        )
        up_ptr = (
            GateUp
            + expert_id * stride_gu_e
            + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_gu_k
            + (INTERMEDIATE_DIM + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[
                None, :
            ]
            * stride_gu_n
        )
        gate_s_ptr = GateUpScale + expert_id * stride_gus_e + pid_n * stride_gus_n
        up_s_ptr = (
            GateUpScale
            + expert_id * stride_gus_e
            + (num_n_tiles + pid_n) * stride_gus_n
        )

        acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for _ in range(0, tl.cdiv(HIDDEN_DIM, BLOCK_SIZE_K)):
            a_raw = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
            a, a_s = fp8_act_quant_inline(a_raw)
            w_gate = tl.load(gate_ptr)
            w_up = tl.load(up_ptr)
            w_s_gate = decode_ue8m0_scale(tl.load(gate_s_ptr))
            w_s_up = decode_ue8m0_scale(tl.load(up_s_ptr))
            acc_gate += tl.dot(a, w_gate) * a_s[:, None] * w_s_gate
            acc_up += tl.dot(a, w_up) * a_s[:, None] * w_s_up
            a_ptrs += BLOCK_SIZE_K * stride_h_k
            gate_ptr += BLOCK_SIZE_K * stride_gu_k
            up_ptr += BLOCK_SIZE_K * stride_gu_k
            gate_s_ptr += stride_gus_k
            up_s_ptr += stride_gus_k

        intermediate = glu(
            acc_gate,
            acc_up,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            Hidden.dtype.element_ty,
            SIMULATE_UNFUSED,
        )
        inter, inter_s = fp8_act_quant_inline(
            intermediate
        )  # FP8 requant, per-row scale
        int_ptrs = (
            Inter
            + offs_global_m[:, None] * stride_int_m
            + offs_bn[None, :] * stride_int_n
        )
        tl.store(int_ptrs, inter, mask=row_mask[:, None])
        tl.store(
            InterScale + offs_global_m * stride_is_m + pid_n * stride_is_n,
            inter_s,
            mask=row_mask,
        )


@bayesian_autotune(
    get_accelerator_autotuning_configs(tune_block_m=True),
    ["INTERMEDIATE_DIM", "HIDDEN_DIM", "tokens_per_sm_bit_length"],
    n_trials=60,
    # fp8 intermediate activation tile + single down weight tile
    prune_configs_by={
        "early_config_prune": smem_config_pruner(act_bytes=1, n_weight_tiles=1)
    },
)
@triton.jit
def w8a8_block_dynamic_fp8_moe_grouped_down_kernel(
    Inter,  # (S, I) FP8 — expert-sorted intermediate
    InterScale,  # (S, NUM_I_TILES) fp32 — per-row, per-I-tile scale
    Down,  # (E, H, I) FP8
    DownScale,  # (E, H//bn, I//bk) UE8M0 block scales
    ExpertStart,  # (NUM_EXPERTS+1,) int32 — exclusive sorted-row start per expert (pad S)
    Perm,  # (S,) int32 — sorted position -> flat (token, slot) row
    SampleWeights,  # (S,) routing weight per flat row
    ProjOut,  # (S, H) bf16, flat (token, slot) order
    stride_int_m,
    stride_int_n,
    stride_is_m,
    stride_is_n,
    stride_down_e,
    stride_down_h,
    stride_down_i,
    stride_downs_e,
    stride_downs_h,
    stride_downs_i,
    stride_po_m,
    stride_po_n,
    stride_perm,
    tokens_per_sm_bit_length,  # autotune key only (log2 tokens-per-SM bucket); unused in body
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    NUM_I_TILES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    SIMULATE_UNFUSED: tl.constexpr,
):
    """Phase 2: persistent grid-stride over (M-tile, H-tile). Grouped down over the FP8
    intermediate, then routing-weight × scatter to the flat (token, slot) row."""
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = _build_tile_layout(
        ExpertStart, NUM_EXPERTS, BLOCK_SIZE_M
    )
    num_h_tiles = tl.cdiv(HIDDEN_DIM, BLOCK_SIZE_N)
    offs_i = tl.arange(0, BLOCK_SIZE_K)
    for tile_id in tl.range(start_pid, total_m_tiles * num_h_tiles, NUM_SMS):
        pid_m = tile_id // num_h_tiles
        pid_h = tile_id % num_h_tiles
        expert_id, offs_global_m, row_mask = _resolve_tile_inline(
            pid_m, exp_start, freqs, tile_start_excl, e_offs, BLOCK_SIZE_M
        )
        offs_h = pid_h * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        w_down_ptr = (
            Down
            + expert_id * stride_down_e
            + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_down_i
            + (pid_h * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :]
            * stride_down_h
        )
        ws_down_ptr = DownScale + expert_id * stride_downs_e + pid_h * stride_downs_h

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for i_tile in range(0, NUM_I_TILES):
            i_off = i_tile * BLOCK_SIZE_K
            inter = tl.load(
                Inter
                + offs_global_m[:, None] * stride_int_m
                + (i_off + offs_i)[None, :] * stride_int_n,
                mask=row_mask[:, None],
                other=0.0,
            )
            inter_s = tl.load(
                InterScale + offs_global_m * stride_is_m + i_tile * stride_is_n,
                mask=row_mask,
                other=0.0,
            )
            w_s_down = decode_ue8m0_scale(tl.load(ws_down_ptr))
            w_down = tl.load(w_down_ptr)
            acc += tl.dot(inter, w_down) * inter_s[:, None] * w_s_down
            w_down_ptr += BLOCK_SIZE_K * stride_down_i
            ws_down_ptr += stride_downs_i

        if SIMULATE_UNFUSED:
            acc = acc.to(ProjOut.dtype.element_ty).to(tl.float32)
        # Fused routing-weight × top-k reorder (no atomics): scale each row by its weight
        # and scatter to its flat (token, slot) row; the host then just sums over slots.
        flat = tl.load(Perm + offs_global_m * stride_perm, mask=row_mask, other=0)
        weight = tl.load(SampleWeights + flat, mask=row_mask, other=0.0)
        acc = acc * weight[:, None]
        store_tile(ProjOut, acc, flat, offs_h, row_mask, stride_po_m, stride_po_n)


def w8a8_block_dynamic_fp8_moe_grouped(
    hidden_states: torch.Tensor,  # (T, H)
    top_k_index: torch.Tensor,  # (T, K) int
    top_k_weights: torch.Tensor,  # (T, K)
    gate_up_proj: torch.Tensor,  # (E, 2I, H) FP8
    down_proj: torch.Tensor,  # (E, H, I) FP8
    gate_up_proj_scale: torch.Tensor,
    down_proj_scale: torch.Tensor,
    block_size: list[int],
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """Block-dynamic FP8 fused grouped MoE: gather gate_up+SiLU → FP8 intermediate →
    grouped down → routing-weighted top-k reduce. Returns ``(num_tokens, hidden_dim)``."""
    device = hidden_states.device
    num_tokens = hidden_states.size(0)
    num_top_k = top_k_index.size(-1)
    HIDDEN_DIM = hidden_states.size(1)
    INTERMEDIATE_DIM = down_proj.size(2)
    BLOCK_SIZE_N, BLOCK_SIZE_K = block_size
    NUM_I_TILES = INTERMEDIATE_DIM // BLOCK_SIZE_N

    perm_token, perm, expert_start, NUM_EXPERTS, num_routed_tokens = _grouped_routing(
        top_k_index, gate_up_proj.size(0), num_top_k
    )
    num_sms = sm_count(device.index)
    tokens_per_sm_bit_length = (num_routed_tokens // num_sms).bit_length()

    gate_up_scale_u8 = ue8m0_as_uint8(gate_up_proj_scale)
    down_scale_u8 = ue8m0_as_uint8(down_proj_scale)

    inter = torch.empty(
        num_routed_tokens, INTERMEDIATE_DIM, device=device, dtype=FP8_DTYPE
    )
    inter_scale = torch.empty(
        num_routed_tokens, NUM_I_TILES, device=device, dtype=torch.float32
    )
    out = torch.empty(
        num_routed_tokens, HIDDEN_DIM, device=device, dtype=hidden_states.dtype
    )
    reduced = torch.empty(
        num_tokens, HIDDEN_DIM, device=device, dtype=hidden_states.dtype
    )
    with device_context(device):
        w8a8_block_dynamic_fp8_moe_grouped_gate_up_kernel[(num_sms,)](
            hidden_states,
            perm_token,
            gate_up_proj,
            gate_up_scale_u8,
            expert_start,
            inter,
            inter_scale,
            hidden_states.stride(0),
            hidden_states.stride(1),
            gate_up_proj.stride(0),
            gate_up_proj.stride(1),
            gate_up_proj.stride(2),
            gate_up_scale_u8.stride(0),
            gate_up_scale_u8.stride(1),
            gate_up_scale_u8.stride(2),
            inter.stride(0),
            inter.stride(1),
            inter_scale.stride(0),
            inter_scale.stride(1),
            perm_token.stride(0),
            tokens_per_sm_bit_length=tokens_per_sm_bit_length,
            HIDDEN_DIM=HIDDEN_DIM,
            INTERMEDIATE_DIM=INTERMEDIATE_DIM,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            NUM_EXPERTS=NUM_EXPERTS,
            NUM_SMS=num_sms,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            SIMULATE_UNFUSED=simulate_unfused,
        )
        w8a8_block_dynamic_fp8_moe_grouped_down_kernel[(num_sms,)](
            inter,
            inter_scale,
            down_proj,
            down_scale_u8,
            expert_start,
            perm,
            top_k_weights,
            out,
            inter.stride(0),
            inter.stride(1),
            inter_scale.stride(0),
            inter_scale.stride(1),
            down_proj.stride(0),
            down_proj.stride(1),
            down_proj.stride(2),
            down_scale_u8.stride(0),
            down_scale_u8.stride(1),
            down_scale_u8.stride(2),
            out.stride(0),
            out.stride(1),
            perm.stride(0),
            tokens_per_sm_bit_length=tokens_per_sm_bit_length,
            HIDDEN_DIM=HIDDEN_DIM,
            INTERMEDIATE_DIM=INTERMEDIATE_DIM,
            NUM_I_TILES=NUM_I_TILES,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            NUM_EXPERTS=NUM_EXPERTS,
            NUM_SMS=num_sms,
            SIMULATE_UNFUSED=simulate_unfused,
        )
        topk_reduce_kernel[(num_tokens, triton.cdiv(HIDDEN_DIM, TOPK_REDUCE_BLOCK_H))](
            out,
            reduced,
            top_k_index,
            HIDDEN_DIM,
            out.stride(0),
            out.stride(1),
            reduced.stride(0),
            reduced.stride(1),
            top_k_index.stride(0),
            top_k_index.stride(1),
            NUM_TOP_K=num_top_k,
            NUM_EXPERTS=NUM_EXPERTS,
            BLOCK_H=TOPK_REDUCE_BLOCK_H,
        )

    return reduced


# ── MXFP4/MXFP8 (UE8M0 group-32, tunable tile, MXFP8 intermediate) ────────────


def _set_gate_up_descriptor(nargs):
    """Per-config pre_hook: set the gate_up TMA descriptor box to the autotuned
    [2 (gate|up), BLOCK_SIZE_N, BK//VALUES_PER_BYTE] over the (2E, I, H/vpb) weight view.

    MUST mutate ``block_shape`` in place, not rebind ``nargs[...]`` to a fresh descriptor:
    the autotuner launches with the original ``*args`` object, so a rebind never reaches the
    kernel (the placeholder box survives → all but BN=32,BK=32 configs fail the reshape). An
    in-place edit of the passed object propagates to both compile-time specialization and the
    launch-time tensor-map fill. ``shape``/``strides`` are already correct from the wrapper's
    ``from_tensor`` placeholder; only the per-config box changes. No-op unless MEMORY_MODE is host_descriptor (pointer /
    device_descriptor don't use the host-built descriptor)."""
    if nargs["MEMORY_MODE"] != "host_descriptor":
        return
    nargs["GateUpDescriptor"].block_shape = [
        2,
        nargs["BLOCK_SIZE_N"],
        nargs["BLOCK_SIZE_K"] // nargs["VALUES_PER_BYTE"],
    ]


@bayesian_autotune(
    get_mxfp_autotuning_configs(
        pre_hook=_set_gate_up_descriptor,
        compute_modes=_MX_COMPUTE_MODES,
        memory_modes=_MX_MEMORY_MODES,
        tune_block_m=True,
    ),  # prefill: no scalar; combined gate∪up dot ([2*BN,K] reshape) — TMA vs block-ptr load
    ["INTERMEDIATE_DIM", "HIDDEN_DIM", "tokens_per_sm_bit_length"],
    n_trials=60,
    # bf16 activation tile + fused gate|up weight tiles
    prune_configs_by={
        "early_config_prune": smem_config_pruner(
            act_bytes=2, n_weight_tiles=2, reduction_dim="HIDDEN_DIM"
        )
    },
)
@triton.jit
def mxfp_dynamic_moe_grouped_gate_up_kernel(
    Hidden,  # (T, H) raw activations, UNSORTED
    PermToken,  # (S,) int32 — sorted position -> source token id
    GateUp,  # (E, 2I, H//VALUES_PER_BYTE) MXFP4/MXFP8
    GateUpScale,  # (E, 2I, H//32) UE8M0 group-32 scales
    GateUpDescriptor,  # TMA descriptor over the (2E, I, H//VALUES_PER_BYTE) weight view
    ExpertStart,  # (NUM_EXPERTS+1,) int32 — exclusive sorted-row start per expert (pad S)
    Inter,  # (S, I) E4M3 — MXFP8 intermediate (expert-sorted)
    InterScale,  # (S, I//32) UE8M0 group-32 scales
    stride_h_t,
    stride_h_k,
    stride_gu_e,
    stride_gu_n,
    stride_gu_k,
    stride_gus_e,
    stride_gus_n,
    stride_gus_k,
    stride_int_m,
    stride_int_n,
    stride_is_m,
    stride_is_n,
    stride_pt,
    tokens_per_sm_bit_length,  # autotune key only (log2 tokens-per-SM bucket); unused in body
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
    MEMORY_MODE: tl.constexpr,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
):
    """MXFP4/MXFP8 phase 1 (persistent): gather hidden rows per expert M-tile, gate + up MX
    matmuls (``tl.dot_scaled`` or fp8 ``tl.dot`` + per-group rescale), SiLU, MXFP8-requant."""
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = _build_tile_layout(
        ExpertStart, NUM_EXPERTS, BLOCK_SIZE_M
    )
    num_n_tiles = tl.cdiv(INTERMEDIATE_DIM, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles
        expert_id, offs_global_m, row_mask = _resolve_tile_inline(
            pid_m, exp_start, freqs, tile_start_excl, e_offs, BLOCK_SIZE_M
        )
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        token = tl.load(PermToken + offs_global_m * stride_pt, mask=row_mask, other=0)
        a_ptrs = Hidden + token[:, None] * stride_h_t + offs_k[None, :] * stride_h_k
        n_off = pid_n * BLOCK_SIZE_N

        # Load gate (row 2e) + up (row 2e+1) of the (2E, I, H) view as ONE combined tile, then
        # split — this is the shared loop. MEMORY_MODE picks the LOAD only (decoupled from COMPUTE_MODE):
        # a host/device descriptor (TMA on NVIDIA) vs explicit rank-3 pointers. COMPUTE_MODE picks the compute on the
        # loaded tile: scaled-MMA (dot_scaled) or fp8 dot + per-group software rescale (dot).
        gu_row = expert_id * 2
        # (E, 2I, H//vpb) reinterpreted as (2E, I, H//vpb); host_descriptor uses the passed
        # GateUpDescriptor, device_descriptor builds one in-kernel, pointer indexes it directly.
        if MEMORY_MODE == "pointer":
            gu_ptr = (
                GateUp
                + (gu_row + tl.arange(0, 2))[:, None, None]
                * (INTERMEDIATE_DIM * stride_gu_n)
                + (n_off + tl.arange(0, BLOCK_SIZE_N))[None, :, None] * stride_gu_n
                + tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)[None, None, :]
                * stride_gu_k
            )
        elif MEMORY_MODE == "device_descriptor":
            gu_desc = tl.make_tensor_descriptor(
                GateUp,
                shape=(
                    2 * NUM_EXPERTS,
                    INTERMEDIATE_DIM,
                    HIDDEN_DIM // VALUES_PER_BYTE,
                ),
                strides=(INTERMEDIATE_DIM * stride_gu_n, stride_gu_n, stride_gu_k),
                block_shape=(2, BLOCK_SIZE_N, BLOCK_SIZE_K // VALUES_PER_BYTE),
            )
        gu_scale_ptr = (
            GateUpScale
            + expert_id * stride_gus_e
            + tl.arange(0, 2)[:, None, None] * (INTERMEDIATE_DIM * stride_gus_n)
            + (n_off + tl.arange(0, BLOCK_SIZE_N))[None, :, None] * stride_gus_n
            + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)[None, None, :] * stride_gus_k
        )
        acc = tl.zeros((BLOCK_SIZE_M, 2 * BLOCK_SIZE_N), dtype=tl.float32)
        for k_off in tl.range(0, HIDDEN_DIM, BLOCK_SIZE_K):
            a_raw = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
            a, a_scale = mxfp_act_quant_inline(
                a_raw, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K
            )
            if MEMORY_MODE == "host_descriptor":
                gu = tl.reshape(
                    GateUpDescriptor.load([gu_row, n_off, k_off // VALUES_PER_BYTE]),
                    [2 * BLOCK_SIZE_N, BLOCK_SIZE_K // VALUES_PER_BYTE],
                )
            elif MEMORY_MODE == "device_descriptor":
                gu = tl.reshape(
                    gu_desc.load([gu_row, n_off, k_off // VALUES_PER_BYTE]),
                    [2 * BLOCK_SIZE_N, BLOCK_SIZE_K // VALUES_PER_BYTE],
                )
            else:
                gu = tl.reshape(
                    tl.load(gu_ptr),
                    [2 * BLOCK_SIZE_N, BLOCK_SIZE_K // VALUES_PER_BYTE],
                )
                gu_ptr += (BLOCK_SIZE_K // VALUES_PER_BYTE) * stride_gu_k
            gu_scale = tl.reshape(
                tl.load(gu_scale_ptr + (k_off // SCALE_GROUP_K) * stride_gus_k),
                [2 * BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_K],
            )
            gu_t = tl.trans(
                gu
            )  # [BK, 2*BN] gate∪up weight; mx_compute decodes it for the fp8 path
            acc = mx_compute(
                acc,
                a,
                a_scale,
                gu_t,
                gu_scale,
                COMPUTE_MODE,
                VALUES_PER_BYTE,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                SCALE_GROUP_K,
            )
            a_ptrs += BLOCK_SIZE_K * stride_h_k
        acc_3d = tl.permute(tl.reshape(acc, [BLOCK_SIZE_M, 2, BLOCK_SIZE_N]), (0, 2, 1))
        acc_gate, acc_up = tl.split(acc_3d)

        intermediate = glu(
            acc_gate,
            acc_up,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            Hidden.dtype.element_ty,
            SIMULATE_UNFUSED,
        )

        # MXFP8 requant of the intermediate (E4M3 + UE8M0 group-32 along this N-tile).
        inter, inter_scale = mxfp_act_quant_inline(
            intermediate, BLOCK_SIZE_M, BLOCK_SIZE_N, SCALE_GROUP_K
        )
        int_ptrs = (
            Inter
            + offs_global_m[:, None] * stride_int_m
            + offs_bn[None, :] * stride_int_n
        )
        tl.store(int_ptrs, inter, mask=row_mask[:, None])
        offs_sc = pid_n * (BLOCK_SIZE_N // SCALE_GROUP_K) + tl.arange(
            0, BLOCK_SIZE_N // SCALE_GROUP_K
        )
        sc_ptrs = (
            InterScale
            + offs_global_m[:, None] * stride_is_m
            + offs_sc[None, :] * stride_is_n
        )
        tl.store(sc_ptrs, inter_scale, mask=row_mask[:, None])


def _set_down_descriptor(nargs):
    """Per-config pre_hook: in-place set the down TMA descriptor box to
    [BLOCK_SIZE_N, BK//VALUES_PER_BYTE] over the (E*H, I/vpb) weight view. In-place, not
    rebind — see _set_gate_up_descriptor. No-op unless MEMORY_MODE is host_descriptor."""
    if nargs["MEMORY_MODE"] != "host_descriptor":
        return
    nargs["DownDescriptor"].block_shape = [
        nargs["BLOCK_SIZE_N"],
        nargs["BLOCK_SIZE_K"] // nargs["VALUES_PER_BYTE"],
    ]


@bayesian_autotune(
    get_mxfp_autotuning_configs(
        pre_hook=_set_down_descriptor,
        compute_modes=_MX_COMPUTE_MODES,
        memory_modes=_MX_MEMORY_MODES,
        tune_block_m=True,
    ),  # prefill: no scalar; TMA vs block-ptr load
    ["INTERMEDIATE_DIM", "HIDDEN_DIM", "tokens_per_sm_bit_length"],
    n_trials=60,
    # fp8 intermediate activation tile + single down weight tile
    prune_configs_by={
        "early_config_prune": smem_config_pruner(
            act_bytes=1, n_weight_tiles=1, reduction_dim="INTERMEDIATE_DIM"
        )
    },
)
@triton.jit
def mxfp_dynamic_moe_grouped_down_kernel(
    Inter,  # (S, I) E4M3 — MXFP8 intermediate (expert-sorted)
    InterScale,  # (S, I//32) UE8M0 group-32 scales
    Down,  # (E, H, I//VALUES_PER_BYTE) MXFP4/MXFP8
    DownScale,  # (E, H, I//32) UE8M0 group-32 scales
    DownDescriptor,  # TMA descriptor over the (E*H, I//VALUES_PER_BYTE) weight view
    ExpertStart,  # (NUM_EXPERTS+1,) int32 — exclusive sorted-row start per expert (pad S)
    Perm,  # (S,) int32 — sorted position -> flat (token, slot) row
    SampleWeights,  # (S,) routing weight per flat row
    ProjOut,  # (S, H) bf16, flat (token, slot) order
    stride_int_m,
    stride_int_n,
    stride_is_m,
    stride_is_n,
    stride_down_e,
    stride_down_n,
    stride_down_k,
    stride_downs_e,
    stride_downs_n,
    stride_downs_k,
    stride_po_m,
    stride_po_n,
    stride_perm,
    tokens_per_sm_bit_length,  # autotune key only (log2 tokens-per-SM bucket); unused in body
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    SIMULATE_UNFUSED: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
    MEMORY_MODE: tl.constexpr,
):
    """MXFP4/MXFP8 phase 2 (persistent): MXFP8 intermediate → down proj, then routing-weight
    × scatter to the flat (token, slot) row. N = hidden tile, K = intermediate tile."""
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = _build_tile_layout(
        ExpertStart, NUM_EXPERTS, BLOCK_SIZE_M
    )
    num_n_tiles = tl.cdiv(HIDDEN_DIM, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)
    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles
        expert_id, offs_global_m, row_mask = _resolve_tile_inline(
            pid_m, exp_start, freqs, tile_start_excl, e_offs, BLOCK_SIZE_M
        )
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        a_ptrs = (
            Inter
            + offs_global_m[:, None] * stride_int_m
            + offs_k[None, :] * stride_int_n
        )
        as_ptrs = (
            InterScale
            + offs_global_m[:, None] * stride_is_m
            + offs_sf[None, :] * stride_is_n
        )
        n_off = pid_n * BLOCK_SIZE_N
        ws_down_ptr = (
            DownScale
            + expert_id * stride_downs_e
            + (n_off + tl.arange(0, BLOCK_SIZE_N))[:, None] * stride_downs_n
            + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)[None, :] * stride_downs_k
        )

        # Down weight tile [BK//vpb, BN] loaded once per K-chunk: MEMORY_MODE picks the LOAD (a
        # host/device descriptor over the (E*H, I//vpb) view vs explicit pointers), COMPUTE_MODE
        # picks the compute. Scales stay a pointer load (a descriptor needs >=16B inner; the
        # BK//32 row is too narrow).
        if MEMORY_MODE == "pointer":
            w_down_ptr = (
                Down
                + expert_id * stride_down_e
                + tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)[:, None] * stride_down_k
                + (n_off + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_down_n
            )
        elif MEMORY_MODE == "device_descriptor":
            down_desc = tl.make_tensor_descriptor(
                Down,  # (E, H, I//vpb) flattened to the (E*H, I//vpb) view
                shape=(NUM_EXPERTS * HIDDEN_DIM, INTERMEDIATE_DIM // VALUES_PER_BYTE),
                strides=(stride_down_n, stride_down_k),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K // VALUES_PER_BYTE),
            )
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k_off in tl.range(0, INTERMEDIATE_DIM, BLOCK_SIZE_K):
            a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
            a_scale = tl.load(as_ptrs, mask=row_mask[:, None], other=0)
            if MEMORY_MODE == "host_descriptor":
                w = tl.trans(
                    tl.reshape(
                        DownDescriptor.load(
                            [expert_id * HIDDEN_DIM + n_off, k_off // VALUES_PER_BYTE]
                        ),
                        [BLOCK_SIZE_N, BLOCK_SIZE_K // VALUES_PER_BYTE],
                    )
                )
            elif MEMORY_MODE == "device_descriptor":
                w = tl.trans(
                    tl.reshape(
                        down_desc.load(
                            [expert_id * HIDDEN_DIM + n_off, k_off // VALUES_PER_BYTE]
                        ),
                        [BLOCK_SIZE_N, BLOCK_SIZE_K // VALUES_PER_BYTE],
                    )
                )
            else:
                w = tl.load(w_down_ptr)
                w_down_ptr += (BLOCK_SIZE_K // VALUES_PER_BYTE) * stride_down_k
            w_scale = tl.load(ws_down_ptr + (k_off // SCALE_GROUP_K) * stride_downs_k)
            acc = mx_compute(
                acc,
                a,
                a_scale,
                w,
                w_scale,
                COMPUTE_MODE,
                VALUES_PER_BYTE,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                SCALE_GROUP_K,
            )
            a_ptrs += BLOCK_SIZE_K * stride_int_n
            as_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_is_n

        if SIMULATE_UNFUSED:
            acc = acc.to(ProjOut.dtype.element_ty).to(tl.float32)
        # Fused routing-weight × top-k reorder (no atomics): scale each row by its weight
        # and scatter to its flat (token, slot) row; the host then just sums over slots.
        flat = tl.load(Perm + offs_global_m * stride_perm, mask=row_mask, other=0)
        weight = tl.load(SampleWeights + flat, mask=row_mask, other=0.0)
        acc = acc * weight[:, None]
        store_tile(ProjOut, acc, flat, offs_bn, row_mask, stride_po_m, stride_po_n)


def mxfp_dynamic_moe_grouped(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale: torch.Tensor,
    down_proj_scale: torch.Tensor,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """MXFP4/MXFP8 fused grouped MoE (UE8M0 group-32); gate_up and down must share the same MX
    format. Same structure as the block-dynamic path but with a tunable tile and MXFP8 intermediate."""
    gate_up_is_fp4 = is_mxfp4(gate_up_proj, gate_up_proj_scale)
    down_is_fp4 = is_mxfp4(down_proj, down_proj_scale)
    if gate_up_is_fp4 != down_is_fp4:
        raise ValueError(
            "gate_up_proj and down_proj must use the same MX format (both MXFP4 or both MXFP8)."
        )

    device = hidden_states.device
    num_sms = sm_count(device.index)
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    HIDDEN_DIM = hidden_states.size(1)
    INTERMEDIATE_DIM = gate_up_proj.size(1) // 2
    perm_token, perm, expert_start, NUM_EXPERTS, num_routed_tokens = _grouped_routing(
        top_k_index, gate_up_proj.size(0), num_top_k
    )
    tokens_per_sm_bit_length = (num_routed_tokens // num_sms).bit_length()
    VALUES_PER_BYTE = NIBBLES_PER_BYTE if gate_up_is_fp4 else 1
    gate_up_proj_u8 = e2m1_as_uint8(gate_up_proj)
    down_proj_u8 = e2m1_as_uint8(down_proj)
    gate_up_scale_u8 = ue8m0_as_uint8(gate_up_proj_scale)
    down_scale_u8 = ue8m0_as_uint8(down_proj_scale)

    inter = torch.empty(
        num_routed_tokens, INTERMEDIATE_DIM, device=device, dtype=FP8_DTYPE
    )
    inter_scale = torch.empty(
        num_routed_tokens,
        INTERMEDIATE_DIM // MX_SCALE_GROUP_K,
        device=device,
        dtype=torch.uint8,
    )
    out = torch.empty(
        num_routed_tokens, HIDDEN_DIM, device=device, dtype=hidden_states.dtype
    )
    reduced = torch.empty(
        num_tokens, HIDDEN_DIM, device=device, dtype=hidden_states.dtype
    )

    # Host-built descriptors for host_descriptor mode: views (not reshapes) of the weight
    # buffers — gate_up (E, 2I, H/vpb) → (2E, I, H/vpb) so one [2, BN, BK/vpb] box loads gate
    # (row 2e) + up (2e+1); down (E, H, I/vpb) → (E*H, I/vpb), one [BN, BK/vpb] box per (expert,
    # hidden-tile). Cheap to build, so always created (device_descriptor/pointer ignore them).
    gate_up_2e = gate_up_proj_u8.view(
        2 * gate_up_proj_u8.size(0), INTERMEDIATE_DIM, gate_up_proj_u8.size(2)
    )
    gate_up_descriptor = TensorDescriptor.from_tensor(
        gate_up_2e, [2, 32, 32 // VALUES_PER_BYTE]
    )
    down_eh = down_proj_u8.view(down_proj_u8.size(0) * HIDDEN_DIM, down_proj_u8.size(2))
    down_descriptor = TensorDescriptor.from_tensor(down_eh, [32, 32 // VALUES_PER_BYTE])
    with device_context(device):
        mxfp_dynamic_moe_grouped_gate_up_kernel[(num_sms,)](
            hidden_states,
            perm_token,
            gate_up_proj_u8,
            gate_up_scale_u8,
            gate_up_descriptor,
            expert_start,
            inter,
            inter_scale,
            hidden_states.stride(0),
            hidden_states.stride(1),
            gate_up_proj_u8.stride(0),
            gate_up_proj_u8.stride(1),
            gate_up_proj_u8.stride(2),
            gate_up_scale_u8.stride(0),
            gate_up_scale_u8.stride(1),
            gate_up_scale_u8.stride(2),
            inter.stride(0),
            inter.stride(1),
            inter_scale.stride(0),
            inter_scale.stride(1),
            perm_token.stride(0),
            tokens_per_sm_bit_length=tokens_per_sm_bit_length,
            HIDDEN_DIM=HIDDEN_DIM,
            INTERMEDIATE_DIM=INTERMEDIATE_DIM,
            VALUES_PER_BYTE=VALUES_PER_BYTE,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
            NUM_EXPERTS=NUM_EXPERTS,
            NUM_SMS=num_sms,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            SIMULATE_UNFUSED=simulate_unfused,
        )
        mxfp_dynamic_moe_grouped_down_kernel[(num_sms,)](
            inter,
            inter_scale,
            down_proj_u8,
            down_scale_u8,
            down_descriptor,
            expert_start,
            perm,
            top_k_weights,
            out,
            inter.stride(0),
            inter.stride(1),
            inter_scale.stride(0),
            inter_scale.stride(1),
            down_proj_u8.stride(0),
            down_proj_u8.stride(1),
            down_proj_u8.stride(2),
            down_scale_u8.stride(0),
            down_scale_u8.stride(1),
            down_scale_u8.stride(2),
            out.stride(0),
            out.stride(1),
            perm.stride(0),
            tokens_per_sm_bit_length=tokens_per_sm_bit_length,
            HIDDEN_DIM=HIDDEN_DIM,
            INTERMEDIATE_DIM=INTERMEDIATE_DIM,
            VALUES_PER_BYTE=VALUES_PER_BYTE,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
            NUM_EXPERTS=NUM_EXPERTS,
            NUM_SMS=num_sms,
            SIMULATE_UNFUSED=simulate_unfused,
        )
        topk_reduce_kernel[(num_tokens, triton.cdiv(HIDDEN_DIM, TOPK_REDUCE_BLOCK_H))](
            out,
            reduced,
            top_k_index,
            HIDDEN_DIM,
            out.stride(0),
            out.stride(1),
            reduced.stride(0),
            reduced.stride(1),
            top_k_index.stride(0),
            top_k_index.stride(1),
            NUM_TOP_K=num_top_k,
            NUM_EXPERTS=NUM_EXPERTS,
            BLOCK_H=TOPK_REDUCE_BLOCK_H,
        )

    return reduced


# ── Dispatcher ────────────────────────────────────────────────────────────────


def moe_fused_grouped(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    block_size: list[int] | None,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """Fused grouped-MoE dispatcher — routes to the recipe matching the weight dtype /
    scale layout, mirroring ``moe_fused_batched``. Implemented: block-dynamic FP8 and MXFP8 /
    MXFP4 (UE8M0 group-32). ``simulate_unfused`` (testing) rounds each step through the
    activation dtype so the output matches the unfused reference to reduce order."""
    gate_up_is_mx = is_mxfp(gate_up_proj, gate_up_proj_scale_inv)
    down_is_mx = is_mxfp(down_proj, down_proj_scale_inv)
    if gate_up_is_mx != down_is_mx:
        raise ValueError(
            "gate_up_proj and down_proj must use the same recipe (both MX or both block-dynamic FP8)."
        )

    if gate_up_is_mx:
        return mxfp_dynamic_moe_grouped(
            hidden_states,
            top_k_index,
            top_k_weights,
            gate_up_proj,
            down_proj,
            gate_up_proj_scale_inv,
            down_proj_scale_inv,
            act_fn,
            swiglu_alpha,
            swiglu_limit,
            simulate_unfused,
        )

    if block_size is None:
        raise ValueError("block_size is required for block-dynamic FP8 weights.")

    return w8a8_block_dynamic_fp8_moe_grouped(
        hidden_states,
        top_k_index,
        top_k_weights,
        gate_up_proj,
        down_proj,
        gate_up_proj_scale_inv,
        down_proj_scale_inv,
        block_size,
        act_fn,
        swiglu_alpha,
        swiglu_limit,
        simulate_unfused,
    )
