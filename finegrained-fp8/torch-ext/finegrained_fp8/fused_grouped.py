# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fused grouped MoE (prefill counterpart to the fused batched/decode path).

Two **persistent** kernels over expert-grouped M-tiles, no ``A_sorted`` materialization:
  1. Routing (on-device): an atomic counting-sort scatter (``_scatter_kernel``, O(S)) →
     ``perm_routed`` / ``perm_token`` + an ``expert_start`` exclusive offset vector — the sort is
     just an index; activations stay unsorted.
  2. Phase 1 — gate_up, **gathering** each tile's hidden rows via ``perm_token``, + SiLU,
     then **FP8 requant** of the intermediate — same trick as ``fused_batched``: the down
     kernel reads FP8 directly instead of a bf16 round-trip.
  3. Phase 2 — grouped down over the FP8 intermediate, then **fuses the routing-weight ×
     reorder** (no atomics): each row is scaled by its weight and **scattered** to its flat
     ``(token, slot)`` row via ``perm_routed``, leaving the host only a top-k sum back to (T, H).

Scheduling is persistent: a fixed ``NUM_SMS`` programs grid-stride over all tiles. The
expert-tile layout is built once into registers from ``expert_start`` (``build_tile_layout``)
and each tile's owner is resolved inline (``resolve_tile_inline``) — no padded grid, no
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
    compute_grouped_scheduling,
    build_tile_layout,
    resolve_tile_inline,
    batched_mx_pruner,
    FP8_DTYPE,
    MX_SCALE_GROUP_K,
    NIBBLES_PER_BYTE,
    decode_ue8m0_scale,
    device_context,
    sm_count,
    tl_dtype,
    fp8_act_quant_2d,
    fp8_act_quant_inline,
    topk_reduce_kernel,
    TOPK_REDUCE_BLOCK_H,
    get_accelerator_autotuning_configs,
    is_mxfp,
    is_mxfp4,
    mxfp_act_quant,
    load_mx_act_tile,
    mxfp_act_quant_inline,
    mx_compute,
    compose_pruners,
    smem_config_pruner,
    warp_spec_compile_guard_pruner,
    block_dynamic_grouped_gate_up_pruner,
    glu,
    e2m1_as_uint8,
    ue8m0_as_uint8,
)


# ── Persistent tile scheduling (register-resident expert-tile layout) ─────────


@triton.jit
def scatter_weighted_tile(
    ProjOut,
    acc,
    offs_global_m,
    offs_cols,
    row_mask,
    PermRouted,
    SampleWeights,
    stride_perm,
    stride_po_m,
    stride_po_n,
    SIMULATE_UNFUSED: tl.constexpr,
):
    """Down-projection epilogue: optionally round through the output dtype (unfused parity), then
    apply each row's routing weight and scatter to its flat (token, slot) row — the fused top-k
    reorder (no atomics) that lets the host just sum over slots."""
    if SIMULATE_UNFUSED:
        acc = acc.to(ProjOut.dtype.element_ty).to(tl.float32)
    flat = tl.load(PermRouted + offs_global_m * stride_perm, mask=row_mask, other=0)
    weight = tl.load(SampleWeights + flat, mask=row_mask, other=0.0)
    acc = acc * weight[:, None]
    store_tile(ProjOut, acc, flat, offs_cols, row_mask, stride_po_m, stride_po_n)


# MX MoE autotune axes: the MMA flavor and the weight-load mechanism. The kernels implement all
# three memory modes; resolve_memory_modes maps "descriptor" to the device's flavor
# flavor (host on XPU, device on CUDA).
_MX_COMPUTE_MODES = ("dot_scaled", "dot")
_MX_MEMORY_MODES = ("descriptor", "pointer")


# ── Block-dynamic FP8 ────────────────────────────────────────────────────────


@bayesian_autotune(
    get_accelerator_autotuning_configs(tune_block_m=True, warp_spec=True),
    ["INTERMEDIATE_DIM", "HIDDEN_DIM", "tokens_per_sm_bit_length"],
    n_trials=100,
    # fp8 pre-quantized activation tile + fused gate|up weight tiles; WS-race guard (the
    # non-WS dual-dot loop races at w<8 / BM>64 on Triton 3.7.1 — see the pruner).
    prune_configs_by={
        "early_config_prune": block_dynamic_grouped_gate_up_pruner(n_weight_tiles=2)
    },
)
@triton.jit
def w8a8_block_dynamic_fp8_moe_grouped_gate_up_kernel(
    Hidden,  # (T, H) E4M3 activations (pre-quantized once by the wrapper), UNSORTED
    HiddenScale,  # (T, H//BLOCK_SIZE_K) fp32 per-row, per-K-block scales
    PermToken,  # (S,) int32 — sorted position -> source token id
    GateUp,  # (E, 2I, H) FP8
    GateUpScale,  # (E, 2I//bn, H//bk) UE8M0 block scales
    ExpertStart,  # (NUM_EXPERTS+1,) int32 — exclusive sorted-row start per expert (pad S)
    Inter,  # (S, I) FP8 — output (expert-sorted)
    InterScale,  # (S, NUM_I_TILES) fp32 — per-row, per-I-tile activation scale
    stride_h_t,
    stride_h_k,
    stride_hs_t,
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
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
    WARP_SPEC: tl.constexpr = False,
):
    """Phase 1: persistent grid-stride over (M-tile, I-tile). Gather pre-quantized fp8 hidden
    rows + per-K-block scales per expert M-tile, gate + up block-FP8 matmuls, SiLU-combine,
    FP8-requant the intermediate.

    ``WARP_SPEC`` (CUDA): warp-specialize the K-loop — +21% AND load-bearing for correctness:
    Triton 3.7.1's default pipeliner RACES this loop's six load streams + dual dot at
    num_warps < 8 or BM = 128 (nondeterministic output; WS's explicit producer/consumer
    barriers are what make those configs sound). The single-dot combined form is race-free
    but measured 20% slower."""
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
        ExpertStart, NUM_EXPERTS, BLOCK_SIZE_M
    )
    num_n_tiles = tl.cdiv(INTERMEDIATE_DIM, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles
        expert_id, offs_global_m, row_mask = resolve_tile_inline(
            pid_m, exp_start, freqs, tile_start_excl, e_offs, BLOCK_SIZE_M
        )
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        # int64 against offset overflow: expert_id * stride_gu_e reaches E*2I*H > 2^31 at
        # full (non-EP) expert counts on the big-model dims — int32 wraps to a garbage pointer.
        expert_id64 = expert_id.to(tl.int64)
        token = tl.load(PermToken + offs_global_m * stride_pt, mask=row_mask, other=0)
        a_ptrs = Hidden + token[:, None] * stride_h_t + offs_k[None, :] * stride_h_k
        as_ptrs = HiddenScale + token * stride_hs_t
        gate_ptr = (
            GateUp
            + expert_id64 * stride_gu_e
            + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_gu_k
            + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_gu_n
        )
        up_ptr = (
            GateUp
            + expert_id64 * stride_gu_e
            + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_gu_k
            + (INTERMEDIATE_DIM + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[
                None, :
            ]
            * stride_gu_n
        )
        gate_s_ptr = GateUpScale + expert_id64 * stride_gus_e + pid_n * stride_gus_n
        up_s_ptr = (
            GateUpScale
            + expert_id64 * stride_gus_e
            + (num_n_tiles + pid_n) * stride_gus_n
        )

        acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for _ in tl.range(
            0, tl.cdiv(HIDDEN_DIM, BLOCK_SIZE_K), warp_specialize=WARP_SPEC
        ):
            a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
            a_s = tl.load(as_ptrs, mask=row_mask, other=0.0)
            w_gate = tl.load(gate_ptr)
            w_up = tl.load(up_ptr)
            w_s_gate = decode_ue8m0_scale(tl.load(gate_s_ptr))
            w_s_up = decode_ue8m0_scale(tl.load(up_s_ptr))
            acc_gate += tl.dot(a, w_gate) * a_s[:, None] * w_s_gate
            acc_up += tl.dot(a, w_up) * a_s[:, None] * w_s_up
            a_ptrs += BLOCK_SIZE_K * stride_h_k
            as_ptrs += 1
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
            SIMULATE_UNFUSED,
            INTERMEDIATE_DTYPE,
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
    get_accelerator_autotuning_configs(tune_block_m=True, warp_spec=True),
    ["INTERMEDIATE_DIM", "HIDDEN_DIM", "tokens_per_sm_bit_length"],
    n_trials=100,
    # fp8 intermediate activation tile + single down weight tile; WS is a pure perf
    # axis (non-WS is the validated state), compile-guarded.
    prune_configs_by={
        "early_config_prune": compose_pruners(
            smem_config_pruner(n_weight_tiles=1), warp_spec_compile_guard_pruner()
        )
    },
)
@triton.jit
def w8a8_block_dynamic_fp8_moe_grouped_down_kernel(
    Inter,  # (S, I) FP8 — expert-sorted intermediate
    InterScale,  # (S, NUM_I_TILES) fp32 — per-row, per-I-tile scale
    Down,  # (E, H, I) FP8
    DownScale,  # (E, H//bn, I//bk) UE8M0 block scales
    ExpertStart,  # (NUM_EXPERTS+1,) int32 — exclusive sorted-row start per expert (pad S)
    PermRouted,  # (S,) int32 — sorted position -> flat (token, slot) row
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
    WARP_SPEC: tl.constexpr = False,
):
    """Phase 2: persistent grid-stride over (M-tile, H-tile). Grouped down over the FP8
    intermediate, then routing-weight × scatter to the flat (token, slot) row."""
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
        ExpertStart, NUM_EXPERTS, BLOCK_SIZE_M
    )
    num_h_tiles = tl.cdiv(HIDDEN_DIM, BLOCK_SIZE_N)
    offs_i = tl.arange(0, BLOCK_SIZE_K)
    for tile_id in tl.range(start_pid, total_m_tiles * num_h_tiles, NUM_SMS):
        pid_m = tile_id // num_h_tiles
        pid_h = tile_id % num_h_tiles
        expert_id, offs_global_m, row_mask = resolve_tile_inline(
            pid_m, exp_start, freqs, tile_start_excl, e_offs, BLOCK_SIZE_M
        )
        offs_h = pid_h * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        # int64 against offset overflow: E*H*I > 2^31 at full expert counts (see gate_up).
        expert_id64 = expert_id.to(tl.int64)
        w_down_ptr = (
            Down
            + expert_id64 * stride_down_e
            + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_down_i
            + (pid_h * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :]
            * stride_down_h
        )
        ws_down_ptr = DownScale + expert_id64 * stride_downs_e + pid_h * stride_downs_h

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for i_tile in tl.range(0, NUM_I_TILES, warp_specialize=WARP_SPEC):
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

        scatter_weighted_tile(
            ProjOut,
            acc,
            offs_global_m,
            offs_h,
            row_mask,
            PermRouted,
            SampleWeights,
            stride_perm,
            stride_po_m,
            stride_po_n,
            SIMULATE_UNFUSED,
        )


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
    num_sms = sm_count(device.index)
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    num_routed_tokens = top_k_index.numel()
    tokens_per_sm_bit_length = int(num_routed_tokens // num_sms).bit_length()

    NUM_EXPERTS = gate_up_proj.size(0)
    HIDDEN_DIM = hidden_states.size(1)
    INTERMEDIATE_DIM = down_proj.size(2)
    BLOCK_SIZE_N, BLOCK_SIZE_K = block_size
    NUM_I_TILES = INTERMEDIATE_DIM // BLOCK_SIZE_N

    perm_token, perm_routed, expert_start = compute_grouped_scheduling(
        top_k_index, NUM_EXPERTS, num_top_k
    )

    gate_up_scale_u8 = ue8m0_as_uint8(gate_up_proj_scale)
    down_scale_u8 = ue8m0_as_uint8(down_proj_scale)
    # One-pass block-FP8 pre-quant of the activations (see the MX wrapper / mxfp_act_quant:
    # the kernel used to re-run the inline quant per N-tile). Bit-exact — the quant span
    # equals the kernel's BLOCK_SIZE_K.
    hidden_q, hidden_scale = fp8_act_quant_2d(hidden_states, BLOCK_SIZE_K)

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
            hidden_q,
            hidden_scale,
            perm_token,
            gate_up_proj,
            gate_up_scale_u8,
            expert_start,
            inter,
            inter_scale,
            hidden_q.stride(0),
            hidden_q.stride(1),
            hidden_scale.stride(0),
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
            # the dtype the UNFUSED path lands the GLU intermediate in
            INTERMEDIATE_DTYPE=tl_dtype(hidden_states.dtype),
        )
        w8a8_block_dynamic_fp8_moe_grouped_down_kernel[(num_sms,)](
            inter,
            inter_scale,
            down_proj,
            down_scale_u8,
            expert_start,
            perm_routed,
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
            perm_routed.stride(0),
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
    get_accelerator_autotuning_configs(
        pre_hook=_set_gate_up_descriptor,
        mx=True,
        tune_block_nk=True,
        compute_modes=_MX_COMPUTE_MODES,
        memory_modes=_MX_MEMORY_MODES,
        tune_block_m=True,
    ),  # prefill: no scalar; combined gate_up dot ([2*BN,K] reshape) — TMA vs pointer load
    # VALUES_PER_BYTE keys the MXFP4/MXFP8 split — the packing halves the weight tile bytes, so a
    # winner is only valid for its own recipe.
    ["INTERMEDIATE_DIM", "HIDDEN_DIM", "tokens_per_sm_bit_length", "VALUES_PER_BYTE"],
    n_trials=100,
    # fp8 pre-quantized activation tile + fused gate|up weight tiles
    prune_configs_by={
        "early_config_prune": compose_pruners(
            batched_mx_pruner("HIDDEN_DIM", stacked_gate_up=True),
            smem_config_pruner(
                n_weight_tiles=2, reduction_dim="HIDDEN_DIM", double_mma=True
            ),
        )
    },
)
@triton.jit
def mxfp_dynamic_moe_grouped_gate_up_kernel(
    Hidden,  # (T, H) E4M3 activations (pre-quantized once by the wrapper), UNSORTED
    HiddenScale,  # (T, H//32) UE8M0 group-32 activation scales
    PermToken,  # (S,) int32 — sorted position -> source token id
    GateUp,  # (E, 2I, H//VALUES_PER_BYTE) MXFP4/MXFP8
    GateUpScale,  # (E, 2I, H//32) UE8M0 group-32 scales
    GateUpDescriptor,  # TMA descriptor over the (2E, I, H//VALUES_PER_BYTE) weight view
    ExpertStart,  # (NUM_EXPERTS+1,) int32 — exclusive sorted-row start per expert (pad S)
    Inter,  # (S, I) E4M3 — MXFP8 intermediate (expert-sorted)
    InterScale,  # (S, I//32) UE8M0 group-32 scales
    stride_h_t,
    stride_h_k,
    stride_hs_t,
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
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
):
    """MXFP4/MXFP8 phase 1 (persistent): gather hidden rows per expert M-tile, gate + up MX
    matmuls (``tl.dot_scaled`` or fp8 ``tl.dot`` + per-group rescale), SiLU, MXFP8-requant."""
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
        ExpertStart, NUM_EXPERTS, BLOCK_SIZE_M
    )
    num_n_tiles = tl.cdiv(INTERMEDIATE_DIM, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles
        expert_id, offs_global_m, row_mask = resolve_tile_inline(
            pid_m, exp_start, freqs, tile_start_excl, e_offs, BLOCK_SIZE_M
        )
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        # int64 against offset overflow: weight offsets reach 2E*I*(H//vpb) > 2^31 at full
        # (non-EP) expert counts. Descriptor loads keep the int32 ids — TMA takes row indices
        # (bounded by 2E), not byte offsets.
        expert_id64 = expert_id.to(tl.int64)
        token = tl.load(PermToken + offs_global_m * stride_pt, mask=row_mask, other=0)
        a_ptrs = Hidden + token[:, None] * stride_h_t + offs_k[None, :] * stride_h_k
        as_ptrs = (
            HiddenScale
            + token[:, None] * stride_hs_t
            + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)[None, :]
        )
        n_off = pid_n * BLOCK_SIZE_N

        # Load gate (row 2e) + up (row 2e+1) of the (2E, I, H) view as ONE combined tile and
        # run ONE combined [BM, 2*BN] MMA — measured fastest form on B200 for both MX recipes
        # (double-load and split-dot variants lose or fail to compile at the fast tile sizes).
        # On sm_10x a scaled MMA caps at N=256 (Triton miscompiles wider: packed-E2M1 rhs → device
        # "misaligned address" trap), so dot_scaled with 2*BN > 256 must not run there — the
        # smem_config_pruner (n_weight_tiles=2) drops those configs on sm_10x before benching.
        gu_row = expert_id * 2
        # (E, 2I, H//vpb) reinterpreted as (2E, I, H//vpb); host_descriptor uses the passed
        # GateUpDescriptor, device_descriptor builds one in-kernel, pointer indexes it directly.
        if MEMORY_MODE == "pointer":
            gu_ptr = (
                GateUp
                + (expert_id64 * 2 + tl.arange(0, 2))[:, None, None]
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
            + expert_id64 * stride_gus_e
            + tl.arange(0, 2)[:, None, None] * (INTERMEDIATE_DIM * stride_gus_n)
            + (n_off + tl.arange(0, BLOCK_SIZE_N))[None, :, None] * stride_gus_n
            + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)[None, None, :] * stride_gus_k
        )
        acc = tl.zeros((BLOCK_SIZE_M, 2 * BLOCK_SIZE_N), dtype=tl.float32)
        for k_off in tl.range(0, HIDDEN_DIM, BLOCK_SIZE_K):
            a, a_scale = load_mx_act_tile(
                a_ptrs, as_ptrs, row_mask, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K
            )
            as_ptrs += BLOCK_SIZE_K // SCALE_GROUP_K
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
            )  # [BK, 2*BN] gate_up weight; mx_compute decodes it for the fp8 path
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
            SIMULATE_UNFUSED,
            INTERMEDIATE_DTYPE,
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
    get_accelerator_autotuning_configs(
        pre_hook=_set_down_descriptor,
        mx=True,
        tune_block_nk=True,
        compute_modes=_MX_COMPUTE_MODES,
        memory_modes=_MX_MEMORY_MODES,
        tune_block_m=True,
    ),  # prefill: no scalar; TMA vs block-ptr load
    # VALUES_PER_BYTE keys the MXFP4/MXFP8 split — see the gate_up kernel's note above.
    ["INTERMEDIATE_DIM", "HIDDEN_DIM", "tokens_per_sm_bit_length", "VALUES_PER_BYTE"],
    n_trials=100,
    # fp8 intermediate activation tile + single down weight tile
    prune_configs_by={
        "early_config_prune": compose_pruners(
            batched_mx_pruner("INTERMEDIATE_DIM"),
            smem_config_pruner(n_weight_tiles=1, reduction_dim="INTERMEDIATE_DIM"),
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
    PermRouted,  # (S,) int32 — sorted position -> flat (token, slot) row
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
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
        ExpertStart, NUM_EXPERTS, BLOCK_SIZE_M
    )
    num_n_tiles = tl.cdiv(HIDDEN_DIM, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)
    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles
        expert_id, offs_global_m, row_mask = resolve_tile_inline(
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
        # int64 against offset overflow (see gate_up); descriptor loads below keep the int32
        # expert_id — TMA takes row indices (bounded by E*H), not byte offsets.
        expert_id64 = expert_id.to(tl.int64)
        ws_down_ptr = (
            DownScale
            + expert_id64 * stride_downs_e
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
                + expert_id64 * stride_down_e
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

        scatter_weighted_tile(
            ProjOut,
            acc,
            offs_global_m,
            offs_bn,
            row_mask,
            PermRouted,
            SampleWeights,
            stride_perm,
            stride_po_m,
            stride_po_n,
            SIMULATE_UNFUSED,
        )


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
    num_routed_tokens = top_k_index.numel()
    tokens_per_sm_bit_length = int(num_routed_tokens // num_sms).bit_length()

    NUM_EXPERTS = gate_up_proj.size(0)
    HIDDEN_DIM = hidden_states.size(1)
    INTERMEDIATE_DIM = gate_up_proj.size(1) // 2
    VALUES_PER_BYTE = NIBBLES_PER_BYTE if gate_up_is_fp4 else 1

    perm_token, perm_routed, expert_start = compute_grouped_scheduling(
        top_k_index, NUM_EXPERTS, num_top_k
    )
    gate_up_proj_u8 = e2m1_as_uint8(gate_up_proj)
    down_proj_u8 = e2m1_as_uint8(down_proj)
    gate_up_scale_u8 = ue8m0_as_uint8(gate_up_proj_scale)
    down_scale_u8 = ue8m0_as_uint8(down_proj_scale)
    # One-pass MX pre-quant of the activations: the gate_up kernel used to re-run the inline
    # quant per N-tile (16x redundant ALU + 2x act bytes) — it held gate_up at ~380 TFLOPS
    # while the pre-quantized down kernel ran ~1080. Bit-exact (same group-32 boundaries).
    hidden_q, hidden_scale = mxfp_act_quant(hidden_states)

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
            hidden_q,
            hidden_scale,
            perm_token,
            gate_up_proj_u8,
            gate_up_scale_u8,
            gate_up_descriptor,
            expert_start,
            inter,
            inter_scale,
            hidden_q.stride(0),
            hidden_q.stride(1),
            hidden_scale.stride(0),
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
            # the dtype the UNFUSED path lands the GLU intermediate in (the model's
            # activation dtype) — the kernel can't read it off the fp8 activations
            INTERMEDIATE_DTYPE=tl_dtype(hidden_states.dtype),
        )
        mxfp_dynamic_moe_grouped_down_kernel[(num_sms,)](
            inter,
            inter_scale,
            down_proj_u8,
            down_scale_u8,
            down_descriptor,
            expert_start,
            perm_routed,
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
            perm_routed.stride(0),
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
