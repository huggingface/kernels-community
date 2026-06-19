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
group-32 scales, MXFP8 intermediate); ``moe_grouped`` is the neutral dispatcher.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from .bayesian_autotuner import bayesian_autotune
from .grouped import _store_tile
from .utils import (
    MX_SCALE_GROUP_K,
    NIBBLES_PER_BYTE,
    adaptive_block_size_m,
    decode_ue8m0_scale,
    device_context,
    fp8_act_quant_inline,
    get_accelerator_autotuning_configs,
    get_mxfp_autotuning_configs,
    is_mxfp,
    is_mxfp4,
    mxfp_act_quant_inline,
    mxfp4_e2m1_to_e4m3,
    smem_config_pruner,
    ue8m0_as_uint8,
)

FP8 = torch.float8_e4m3fn


# ── Persistent tile scheduling (register-resident expert-tile layout) ─────────


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
    BLOCK_SIZE: tl.constexpr,
):
    """Counting-sort scatter: each flat slot atomically claims the next slot of its expert
    (``expert_start[e] + counter[e]++``). O(S), replaces an O(S·logS) argsort. Within-expert
    order is arbitrary (atomic race) — fine, the per-token reduce is order-invariant."""
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < S
    expert_id = tl.load(ExpertIds + offs, mask=mask, other=0)
    dest = tl.load(ExpertStart + expert_id, mask=mask, other=0) + tl.atomic_add(
        Counters + expert_id, 1, mask=mask
    )
    tl.store(Perm + dest, offs, mask=mask)
    tl.store(PermToken + dest, offs // NUM_TOP_K, mask=mask)


def _grouped_routing(expert_ids: torch.Tensor, num_experts: int, num_top_k: int):
    """On-device routing: expert-sorted index (no copy of the activations) via two Triton
    launches — exclusive offsets + an atomic counting-sort scatter (replaces host
    ``argsort``). Returns ``(perm_token, perm, expert_start, num_experts,
    block_size_m, num_sms, num_routed_tokens)``. ``perm`` is the sorted-position → flat
    ``(t*K + j)`` map (gate_up gathers via ``perm_token = perm // K``, down scatters via
    ``perm``); ``expert_start`` is ``(E+1,)`` padded with S so the kernels build the tile
    layout in-register (E is a power of 2)."""
    device = expert_ids.device
    expert_ids = expert_ids.int()
    num_routed_tokens = expert_ids.numel()  # S = num_tokens * num_top_k
    # histc not bincount: its fixed-size output is CUDA-graph friendly.
    expert_freq = torch.histc(
        expert_ids.float(), bins=num_experts, min=0, max=num_experts - 1
    ).int()

    expert_start = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    counters = torch.empty(num_experts, dtype=torch.int32, device=device)
    with device_context(device):
        _exclusive_offsets_kernel[(1,)](
            expert_freq, expert_start, counters, NUM_EXPERTS=num_experts
        )
        perm = torch.empty(num_routed_tokens, dtype=torch.int32, device=device)
        perm_token = torch.empty(num_routed_tokens, dtype=torch.int32, device=device)
        _scatter_kernel[(triton.cdiv(num_routed_tokens, 1024),)](
            expert_ids,
            perm,
            perm_token,
            expert_start,
            counters,
            num_routed_tokens,
            NUM_TOP_K=num_top_k,
            BLOCK_SIZE=1024,
        )
    block_size_m = adaptive_block_size_m(
        (num_routed_tokens + num_experts - 1) // num_experts
    )
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    return (
        perm_token,
        perm,
        expert_start,
        num_experts,
        block_size_m,
        num_sms,
        num_routed_tokens,
    )


# ── Block-dynamic FP8 ────────────────────────────────────────────────────────


@bayesian_autotune(
    get_accelerator_autotuning_configs(),
    ["INTERMEDIATE_DIM", "HIDDEN_DIM", "BLOCK_SIZE_M"],
    n_trials=60,
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
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    SIMULATE_UNFUSED: tl.constexpr,
):
    """Phase 1: persistent grid-stride over (M-tile, I-tile). Gather hidden rows per expert
    M-tile, gate + up block-FP8 matmuls, SiLU-combine, FP8-requant the intermediate."""
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = _build_tile_layout(
        ExpertStart, NUM_EXPERTS, BLOCK_SIZE_M
    )
    num_n_tiles = tl.cdiv(INTERMEDIATE_DIM, BLOCK_SIZE_N)
    n_scale_blocks = INTERMEDIATE_DIM // BLOCK_SIZE_N
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
        gate_ptrs = (
            GateUp
            + expert_id * stride_gu_e
            + offs_k[:, None] * stride_gu_k
            + offs_bn[None, :] * stride_gu_n
        )
        up_ptrs = (
            GateUp
            + expert_id * stride_gu_e
            + offs_k[:, None] * stride_gu_k
            + (INTERMEDIATE_DIM + offs_bn)[None, :] * stride_gu_n
        )
        gs_base = GateUpScale + expert_id * stride_gus_e
        gate_s_ptrs = gs_base + pid_n * stride_gus_n
        up_s_ptrs = gs_base + (n_scale_blocks + pid_n) * stride_gus_n

        acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(HIDDEN_DIM, BLOCK_SIZE_K)):
            a_raw = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
            a, a_s = fp8_act_quant_inline(a_raw)
            gw = tl.load(gate_ptrs)
            uw = tl.load(up_ptrs)
            gs = decode_ue8m0_scale(tl.load(gate_s_ptrs + k * stride_gus_k))
            us = decode_ue8m0_scale(tl.load(up_s_ptrs + k * stride_gus_k))
            acc_gate += tl.dot(a, gw) * a_s[:, None] * gs[None, :]
            acc_up += tl.dot(a, uw) * a_s[:, None] * us[None, :]
            a_ptrs += BLOCK_SIZE_K * stride_h_k
            gate_ptrs += BLOCK_SIZE_K * stride_gu_k
            up_ptrs += BLOCK_SIZE_K * stride_gu_k

        if SIMULATE_UNFUSED:
            dtype = Hidden.dtype.element_ty
            acc_gate = acc_gate.to(dtype).to(tl.float32)
            acc_up = acc_up.to(dtype).to(tl.float32)
            silu = (acc_gate * tl.sigmoid(acc_gate)).to(dtype).to(tl.float32)
            intermediate = (silu * acc_up).to(dtype).to(tl.float32)
        else:
            intermediate = acc_gate * tl.sigmoid(acc_gate) * acc_up
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
    get_accelerator_autotuning_configs(),
    ["INTERMEDIATE_DIM", "HIDDEN_DIM", "BLOCK_SIZE_M"],
    n_trials=60,
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
    stride_sw,
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    NUM_I_TILES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
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
    num_h_tiles = tl.cdiv(HIDDEN_DIM, BLOCK_SIZE_H)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    for tile_id in tl.range(start_pid, total_m_tiles * num_h_tiles, NUM_SMS):
        pid_m = tile_id // num_h_tiles
        pid_h = tile_id % num_h_tiles
        expert_id, offs_global_m, row_mask = _resolve_tile_inline(
            pid_m, exp_start, freqs, tile_start_excl, e_offs, BLOCK_SIZE_M
        )
        offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

        w_down_ptr = tl.make_block_ptr(
            base=Down + expert_id * stride_down_e,
            shape=(INTERMEDIATE_DIM, HIDDEN_DIM),
            strides=(stride_down_i, stride_down_h),
            offsets=(0, pid_h * BLOCK_SIZE_H),
            block_shape=(BLOCK_SIZE_I, BLOCK_SIZE_H),
            order=(0, 1),
        )
        ws_base = DownScale + expert_id * stride_downs_e + pid_h * stride_downs_h

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_H), dtype=tl.float32)
        for i_tile in range(0, NUM_I_TILES):
            i_off = i_tile * BLOCK_SIZE_I
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
            ws_down = decode_ue8m0_scale(tl.load(ws_base + i_tile * stride_downs_i))
            w_down = tl.load(w_down_ptr)
            acc += tl.dot(inter, w_down) * inter_s[:, None] * ws_down
            w_down_ptr = tl.advance(w_down_ptr, (BLOCK_SIZE_I, 0))

        if SIMULATE_UNFUSED:
            acc = acc.to(ProjOut.dtype.element_ty).to(tl.float32)
        # Fused routing-weight × top-k reorder (no atomics): scale each row by its weight
        # and scatter to its flat (token, slot) row; the host then just sums over slots.
        flat = tl.load(Perm + offs_global_m * stride_perm, mask=row_mask, other=0)
        weight = tl.load(SampleWeights + flat * stride_sw, mask=row_mask, other=0.0)
        acc = acc * weight[:, None]
        _store_tile(ProjOut, acc, flat, offs_h, row_mask, stride_po_m, stride_po_n)


def w8a8_block_dynamic_fp8_moe_grouped(
    hidden_states: torch.Tensor,  # (T, H)
    top_k_index: torch.Tensor,  # (T, K) int
    top_k_weights: torch.Tensor,  # (T, K)
    gate_up_proj: torch.Tensor,  # (E, 2I, H) FP8
    down_proj: torch.Tensor,  # (E, H, I) FP8
    gate_up_proj_scale: torch.Tensor,
    down_proj_scale: torch.Tensor,
    block_size: list[int],
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """Block-dynamic FP8 fused grouped MoE: gather gate_up+SiLU → FP8 intermediate →
    grouped down → routing-weighted top-k reduce. Returns ``(num_tokens, hidden_dim)``."""
    num_tokens = hidden_states.size(0)
    HIDDEN_DIM = hidden_states.size(1)
    num_top_k = top_k_index.size(-1)
    INTERMEDIATE_DIM = down_proj.size(2)
    BLOCK_SIZE_N, BLOCK_SIZE_K = block_size
    device = hidden_states.device

    expert_ids = top_k_index.reshape(-1)
    sample_weights = top_k_weights.reshape(-1)
    (
        perm_token,
        perm,
        expert_start,
        NUM_EXPERTS,
        BLOCK_SIZE_M,
        num_sms,
        num_routed_tokens,
    ) = _grouped_routing(expert_ids, gate_up_proj.size(0), num_top_k)
    NUM_I_TILES = INTERMEDIATE_DIM // BLOCK_SIZE_N

    # ── Phase 1: gate_up + SiLU + FP8 requant → FP8 intermediate (sorted) ──
    gate_up_scale_u8 = ue8m0_as_uint8(gate_up_proj_scale)
    inter = torch.empty(num_routed_tokens, INTERMEDIATE_DIM, device=device, dtype=FP8)
    inter_scale = torch.empty(
        num_routed_tokens, NUM_I_TILES, device=device, dtype=torch.float32
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
            HIDDEN_DIM=HIDDEN_DIM,
            INTERMEDIATE_DIM=INTERMEDIATE_DIM,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            NUM_EXPERTS=NUM_EXPERTS,
            NUM_SMS=num_sms,
            SIMULATE_UNFUSED=simulate_unfused,
        )

    # ── Phase 2: grouped down over the FP8 intermediate → proj_out (flat order) ──
    down_scale_u8 = ue8m0_as_uint8(down_proj_scale)
    proj_out = torch.empty(
        num_routed_tokens, HIDDEN_DIM, device=device, dtype=hidden_states.dtype
    )
    with device_context(device):
        w8a8_block_dynamic_fp8_moe_grouped_down_kernel[(num_sms,)](
            inter,
            inter_scale,
            down_proj,
            down_scale_u8,
            expert_start,
            perm,
            sample_weights,
            proj_out,
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
            proj_out.stride(0),
            proj_out.stride(1),
            perm.stride(0),
            sample_weights.stride(0),
            HIDDEN_DIM=HIDDEN_DIM,
            INTERMEDIATE_DIM=INTERMEDIATE_DIM,
            NUM_I_TILES=NUM_I_TILES,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_H=BLOCK_SIZE_N,
            BLOCK_SIZE_I=BLOCK_SIZE_K,
            NUM_EXPERTS=NUM_EXPERTS,
            NUM_SMS=num_sms,
            SIMULATE_UNFUSED=simulate_unfused,
        )

    # The down kernel already applied the routing weight and scattered each row to its
    # flat (token, slot) position, so the host only sums over the slot axis (in the
    # activation dtype, like fused_batched, to match the unfused ref).
    return (
        proj_out.view(num_tokens, num_top_k, HIDDEN_DIM)
        .sum(dim=1)
        .to(hidden_states.dtype)
    )


# ── MXFP4/MXFP8 (UE8M0 group-32, tunable tile, MXFP8 intermediate) ────────────


def _set_gate_up_descriptor(nargs):
    """Per-config: build the gate_up TMA descriptor with box [2 (gate|up), BLOCK_SIZE_N,
    BK//VALUES_PER_BYTE] over the (2E, I, H/vpb) weight view. Recreated fresh (not
    block_shape-mutated) so TMA derives the swizzle for this box — mutating one descriptor
    across swizzle classes (e.g. 32B→128B inner) yields a misaligned tensor map."""
    w = nargs["GateUp"]
    w2e = w.view(2 * w.size(0), nargs["INTERMEDIATE_DIM"], w.size(2))
    nargs["GateUpDescriptor"] = TensorDescriptor.from_tensor(
        w2e, [2, nargs["BLOCK_SIZE_N"], nargs["BLOCK_SIZE_K"] // nargs["VALUES_PER_BYTE"]]
    )


@bayesian_autotune(
    get_mxfp_autotuning_configs(pre_hook=_set_gate_up_descriptor),
    ["INTERMEDIATE_DIM", "HIDDEN_DIM", "BLOCK_SIZE_M"],
    n_trials=60,
    # bf16 activation tile + fused gate|up weight tiles
    prune_configs_by={"early_config_prune": smem_config_pruner(act_bytes=2, n_weight_tiles=2)},
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
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_DOT_SCALED: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    SIMULATE_UNFUSED: tl.constexpr,
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

        if USE_DOT_SCALED:
            # Scaled-MMA path: load gate (row 2e) + up (row 2e+1) in ONE TMA load (stride trick
            # over the (2E, I, H) view), single fused dot over [gate|up], then split. Weight
            # scales read from GateUpScale via the r2*I index (gate cols 0:I, up cols I:2I).
            gu_row = expert_id * 2
            sg = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)
            nn = tl.arange(0, BLOCK_SIZE_N)
            r2 = tl.arange(0, 2)
            acc = tl.zeros((BLOCK_SIZE_M, 2 * BLOCK_SIZE_N), dtype=tl.float32)
            for k_off in tl.range(0, HIDDEN_DIM, BLOCK_SIZE_K):
                a_raw = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
                a, a_scale = mxfp_act_quant_inline(
                    a_raw, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K
                )
                gu = tl.reshape(
                    GateUpDescriptor.load([gu_row, n_off, k_off // VALUES_PER_BYTE]),
                    [2 * BLOCK_SIZE_N, BLOCK_SIZE_K // VALUES_PER_BYTE],
                )
                sk = k_off // SCALE_GROUP_K + sg
                ws = tl.load(
                    GateUpScale
                    + expert_id * stride_gus_e
                    + (r2[:, None, None] * INTERMEDIATE_DIM + (n_off + nn)[None, :, None])
                    * stride_gus_n
                    + sk[None, None, :] * stride_gus_k
                )
                ws = tl.reshape(ws, [2 * BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_K]).to(
                    tl.uint8
                )
                if VALUES_PER_BYTE == 2:  # MXFP4: packed E2M1 weight bytes
                    acc = tl.dot_scaled(a, a_scale, "e4m3", tl.trans(gu), ws, "e2m1", acc)
                else:  # MXFP8: unpacked E4M3 weights
                    acc = tl.dot_scaled(a, a_scale, "e4m3", tl.trans(gu), ws, "e4m3", acc)
                a_ptrs += BLOCK_SIZE_K * stride_h_k
            acc_3d = tl.permute(tl.reshape(acc, [BLOCK_SIZE_M, 2, BLOCK_SIZE_N]), (0, 2, 1))
            acc_gate, acc_up = tl.split(acc_3d)
        else:
            # fp8 dot + per-group software rescale (BK=32); strided pointers, gate cols
            # [0:I) and up cols [I:2I) of the (H/vpb, 2I) weight, scales likewise.
            okv = tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)
            nn = tl.arange(0, BLOCK_SIZE_N)
            sg = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)
            w_base = GateUp + expert_id * stride_gu_e + okv[:, None] * stride_gu_k
            gw_ptrs = w_base + (n_off + nn)[None, :] * stride_gu_n
            uw_ptrs = w_base + (INTERMEDIATE_DIM + n_off + nn)[None, :] * stride_gu_n
            s_base = GateUpScale + expert_id * stride_gus_e + sg[None, :] * stride_gus_k
            gs_ptrs = s_base + (n_off + nn)[:, None] * stride_gus_n
            us_ptrs = s_base + (INTERMEDIATE_DIM + n_off + nn)[:, None] * stride_gus_n
            acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for _ in range(0, tl.cdiv(HIDDEN_DIM, BLOCK_SIZE_K)):
                a_raw = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
                a, a_scale = mxfp_act_quant_inline(
                    a_raw, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K
                )
                if VALUES_PER_BYTE == 2:
                    b_gate = tl.load(gw_ptrs).to(tl.uint8)
                    b_up = tl.load(uw_ptrs).to(tl.uint8)
                else:
                    b_gate = tl.load(gw_ptrs)
                    b_up = tl.load(uw_ptrs)
                a_s = decode_ue8m0_scale(a_scale)
                wg = tl.trans(decode_ue8m0_scale(tl.load(gs_ptrs).to(tl.uint8)))
                wu = tl.trans(decode_ue8m0_scale(tl.load(us_ptrs).to(tl.uint8)))
                if VALUES_PER_BYTE == 2:
                    acc_gate += tl.dot(a, mxfp4_e2m1_to_e4m3(b_gate)) * a_s * wg
                    acc_up += tl.dot(a, mxfp4_e2m1_to_e4m3(b_up)) * a_s * wu
                else:
                    acc_gate += tl.dot(a, b_gate) * a_s * wg
                    acc_up += tl.dot(a, b_up) * a_s * wu
                a_ptrs += BLOCK_SIZE_K * stride_h_k
                gw_ptrs += (BLOCK_SIZE_K // VALUES_PER_BYTE) * stride_gu_k
                uw_ptrs += (BLOCK_SIZE_K // VALUES_PER_BYTE) * stride_gu_k
                gs_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_gus_k
                us_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_gus_k

        if SIMULATE_UNFUSED:
            dtype = Hidden.dtype.element_ty
            acc_gate = acc_gate.to(dtype).to(tl.float32)
            acc_up = acc_up.to(dtype).to(tl.float32)
            silu = (acc_gate * tl.sigmoid(acc_gate)).to(dtype).to(tl.float32)
            intermediate = (silu * acc_up).to(dtype).to(tl.float32)
        else:
            intermediate = acc_gate * tl.sigmoid(acc_gate) * acc_up

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
    """Per-config: build the down TMA descriptor with box [BLOCK_SIZE_N, BK//VALUES_PER_BYTE]
    over the (E*H, I/vpb) weight view. Recreated fresh per config (see _set_gate_up_descriptor
    for why mutating block_shape across swizzle classes misaligns the tensor map)."""
    w = nargs["Down"]
    w_eh = w.view(w.size(0) * nargs["HIDDEN_DIM"], w.size(2))
    nargs["DownDescriptor"] = TensorDescriptor.from_tensor(
        w_eh, [nargs["BLOCK_SIZE_N"], nargs["BLOCK_SIZE_K"] // nargs["VALUES_PER_BYTE"]]
    )


@bayesian_autotune(
    get_mxfp_autotuning_configs(pre_hook=_set_down_descriptor),
    ["INTERMEDIATE_DIM", "HIDDEN_DIM", "BLOCK_SIZE_M"],
    n_trials=60,
    # fp8 intermediate activation tile + single down weight tile
    prune_configs_by={"early_config_prune": smem_config_pruner(act_bytes=1, n_weight_tiles=1)},
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
    stride_sw,
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_DOT_SCALED: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    SIMULATE_UNFUSED: tl.constexpr,
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
        ws_down_ptr = tl.make_block_ptr(
            base=DownScale + expert_id * stride_downs_e,
            shape=(HIDDEN_DIM, INTERMEDIATE_DIM // SCALE_GROUP_K),
            strides=(stride_downs_n, stride_downs_k),
            offsets=(n_off, 0),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_K),
            order=(1, 0),
        )

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        if USE_DOT_SCALED:
            # Scaled-MMA path: down weight tile via TMA over the (E*H, I//vpb) view (one load),
            # then trans to [K, N]. Scales stay a pointer load (TMA needs ≥16B inner; the BK//32
            # scale row is too narrow).
            for k_off in tl.range(0, INTERMEDIATE_DIM, BLOCK_SIZE_K):
                a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
                a_scale = tl.load(as_ptrs, mask=row_mask[:, None], other=0).to(tl.uint8)
                w = tl.trans(
                    tl.reshape(
                        DownDescriptor.load([expert_id * HIDDEN_DIM + n_off, k_off // VALUES_PER_BYTE]),
                        [BLOCK_SIZE_N, BLOCK_SIZE_K // VALUES_PER_BYTE],
                    )
                )
                ws = tl.load(ws_down_ptr).to(tl.uint8)
                if VALUES_PER_BYTE == 2:
                    acc = tl.dot_scaled(a, a_scale, "e4m3", w, ws, "e2m1", acc)
                else:
                    acc = tl.dot_scaled(a, a_scale, "e4m3", w, ws, "e4m3", acc)
                a_ptrs += BLOCK_SIZE_K * stride_int_n
                as_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_is_n
                ws_down_ptr = tl.advance(ws_down_ptr, (0, BLOCK_SIZE_K // SCALE_GROUP_K))
        else:
            # fp8 dot + per-group software rescale (BK=32), pointer weights.
            w_down_ptr = tl.make_block_ptr(
                base=Down + expert_id * stride_down_e,
                shape=(INTERMEDIATE_DIM // VALUES_PER_BYTE, HIDDEN_DIM),
                strides=(stride_down_k, stride_down_n),
                offsets=(0, n_off),
                block_shape=(BLOCK_SIZE_K // VALUES_PER_BYTE, BLOCK_SIZE_N),
                order=(0, 1),
            )
            for _ in range(0, tl.cdiv(INTERMEDIATE_DIM, BLOCK_SIZE_K)):
                a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
                a_scale = tl.load(as_ptrs, mask=row_mask[:, None], other=0).to(tl.uint8)
                if VALUES_PER_BYTE == 2:
                    w = tl.load(w_down_ptr).to(tl.uint8)
                else:
                    w = tl.load(w_down_ptr)
                a_s = decode_ue8m0_scale(a_scale)
                ws_d = tl.trans(decode_ue8m0_scale(tl.load(ws_down_ptr).to(tl.uint8)))
                if VALUES_PER_BYTE == 2:
                    acc += tl.dot(a, mxfp4_e2m1_to_e4m3(w)) * a_s * ws_d
                else:
                    acc += tl.dot(a, w) * a_s * ws_d
                a_ptrs += BLOCK_SIZE_K * stride_int_n
                as_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_is_n
                w_down_ptr = tl.advance(w_down_ptr, (BLOCK_SIZE_K // VALUES_PER_BYTE, 0))
                ws_down_ptr = tl.advance(ws_down_ptr, (0, BLOCK_SIZE_K // SCALE_GROUP_K))

        if SIMULATE_UNFUSED:
            acc = acc.to(ProjOut.dtype.element_ty).to(tl.float32)
        # Fused routing-weight × top-k reorder (no atomics): scale each row by its weight
        # and scatter to its flat (token, slot) row; the host then just sums over slots.
        flat = tl.load(Perm + offs_global_m * stride_perm, mask=row_mask, other=0)
        weight = tl.load(SampleWeights + flat * stride_sw, mask=row_mask, other=0.0)
        acc = acc * weight[:, None]
        _store_tile(ProjOut, acc, flat, offs_bn, row_mask, stride_po_m, stride_po_n)


def mxfp_dynamic_moe_grouped(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale: torch.Tensor,
    down_proj_scale: torch.Tensor,
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """MXFP4/MXFP8 fused grouped MoE — format picked per-weight (UE8M0 group-32). Same
    structure as the block-dynamic path but with a tunable tile and an MXFP8 intermediate."""

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    HIDDEN_DIM = hidden_states.size(1)
    INTERMEDIATE_DIM = gate_up_proj.size(1) // 2

    expert_ids = top_k_index.reshape(-1)
    sample_weights = top_k_weights.reshape(-1)
    (
        perm_token,
        perm,
        expert_start,
        NUM_EXPERTS,
        BLOCK_SIZE_M,
        num_sms,
        num_routed_tokens,
    ) = _grouped_routing(expert_ids, gate_up_proj.size(0), num_top_k)
    # MXFP4 (packed E2M1, 2 codes/byte) vs MXFP8 (E4M3, 1/byte) — one recipe per layer,
    # so gate_up and down share it.
    values_per_byte = (
        NIBBLES_PER_BYTE if is_mxfp4(gate_up_proj, gate_up_proj_scale) else 1
    )

    # ── Phase 1: gate_up + SiLU + MXFP8 requant → MXFP8 intermediate (sorted) ──
    gate_up_scale_u8 = ue8m0_as_uint8(gate_up_proj_scale)
    inter = torch.empty(num_routed_tokens, INTERMEDIATE_DIM, device=device, dtype=FP8)
    inter_scale = torch.empty(
        num_routed_tokens,
        INTERMEDIATE_DIM // MX_SCALE_GROUP_K,
        device=device,
        dtype=torch.uint8,
    )
    # (E, 2I, H/vpb) → (2E, I, H/vpb): gate=row 2e, up=row 2e+1, so one TMA box [2, BN, BK/vpb]
    # loads both. A view (not reshape) — the descriptor must point at the real weight buffer,
    # and contiguous weights make this a pure stride reinterpretation.
    gate_up_2e = gate_up_proj.view(
        2 * gate_up_proj.size(0), INTERMEDIATE_DIM, gate_up_proj.size(2)
    )
    gate_up_descriptor = TensorDescriptor.from_tensor(
        gate_up_2e, [2, 32, 32 // values_per_byte]
    )
    with device_context(device):
        mxfp_dynamic_moe_grouped_gate_up_kernel[(num_sms,)](
            hidden_states,
            perm_token,
            gate_up_proj,
            gate_up_scale_u8,
            gate_up_descriptor,
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
            HIDDEN_DIM=HIDDEN_DIM,
            INTERMEDIATE_DIM=INTERMEDIATE_DIM,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            VALUES_PER_BYTE=values_per_byte,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
            NUM_EXPERTS=NUM_EXPERTS,
            NUM_SMS=num_sms,
            SIMULATE_UNFUSED=simulate_unfused,
        )

    # ── Phase 2: grouped down over the MXFP8 intermediate → proj_out (flat order) ──
    down_scale_u8 = ue8m0_as_uint8(down_proj_scale)
    proj_out = torch.empty(
        num_routed_tokens, HIDDEN_DIM, device=device, dtype=hidden_states.dtype
    )
    # (E, H, I/vpb) → (E*H, I/vpb): one TMA box [BN, BK/vpb] per (expert, hidden-tile). View,
    # not reshape — same reason as gate_up: the descriptor must alias the real weight buffer.
    down_eh = down_proj.view(down_proj.size(0) * HIDDEN_DIM, down_proj.size(2))
    down_descriptor = TensorDescriptor.from_tensor(down_eh, [32, 32 // values_per_byte])
    with device_context(device):
        mxfp_dynamic_moe_grouped_down_kernel[(num_sms,)](
            inter,
            inter_scale,
            down_proj,
            down_scale_u8,
            down_descriptor,
            expert_start,
            perm,
            sample_weights,
            proj_out,
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
            proj_out.stride(0),
            proj_out.stride(1),
            perm.stride(0),
            sample_weights.stride(0),
            HIDDEN_DIM=HIDDEN_DIM,
            INTERMEDIATE_DIM=INTERMEDIATE_DIM,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            VALUES_PER_BYTE=values_per_byte,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
            NUM_EXPERTS=NUM_EXPERTS,
            NUM_SMS=num_sms,
            SIMULATE_UNFUSED=simulate_unfused,
        )

    return (
        proj_out.view(num_tokens, num_top_k, HIDDEN_DIM)
        .sum(dim=1)
        .to(hidden_states.dtype)
    )


# ── Dispatcher ────────────────────────────────────────────────────────────────


def moe_grouped(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    block_size: list[int] | None,
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """Fused grouped-MoE dispatcher — routes to the recipe matching the weight dtype /
    scale layout, mirroring ``moe_batched``. Implemented: block-dynamic FP8 and MXFP8 /
    MXFP4 (UE8M0 group-32). ``simulate_unfused`` (testing) rounds each step through the
    activation dtype so the output matches the unfused reference to reduce order."""
    if is_mxfp(gate_up_proj, gate_up_proj_scale_inv):
        return mxfp_dynamic_moe_grouped(
            hidden_states,
            top_k_index,
            top_k_weights,
            gate_up_proj,
            down_proj,
            gate_up_proj_scale_inv,
            down_proj_scale_inv,
            simulate_unfused,
        )

    return w8a8_block_dynamic_fp8_moe_grouped(
        hidden_states,
        top_k_index,
        top_k_weights,
        gate_up_proj,
        down_proj,
        gate_up_proj_scale_inv,
        down_proj_scale_inv,
        block_size,
        simulate_unfused,
    )
