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

"""Fused MoE: two-kernel deterministic approach.

Kernel 1 (M×N grid): gather + gate_up + SiLU + FP8 quant → fp8 intermediate buffer
Kernel 2 (M×H grid): fp8 intermediate → down projection → output

Both kernels are deterministic (no atomic_add). The fp8 intermediate buffer
is ~half the size of the bf16 intermediate in the unfused path.

The full pipeline:
  1. histc + cumsum (expert offsets)
  2. sort (expert grouping)
  3. Kernel 1: gather → gate_up → SiLU → FP8 quant → (S, N_inter) fp8 + scales
  4. Kernel 2: fp8 intermediate → down_proj → (S, hidden)
  5. routing weights + unsort + top_k reduce
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from .utils import device_context


# ── Kernel 1: gather + gate_up + SiLU + FP8 quant ──────────────────────────


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [4, 8, 16]
        for s in [2, 3, 4]
    ],
    key=["N_inter", "K", "BLOCK_SIZE_M"],
)
@triton.jit
def fused_gate_up_silu_kernel(
    # Unsorted inputs
    A,  # (num_tokens, K) raw BF16/FP16 — NOT sorted
    Perm,  # (S,) int64 — sorted_pos → original flat index
    # Expert weights
    W_gu,  # (E, 2*N_inter, K) FP8 gate_up weights
    Ws_gu,  # gate_up scales
    # Outputs
    Inter,  # (S, N_inter) FP8 intermediate
    Inter_s,  # (S,) fp32 per-row scales
    # Expert scheduling
    Offsets,
    TileOffsets,
    # Shapes
    N_inter,
    K,
    num_top_k,
    # Strides — A, W_gu, Ws_gu, Inter
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_bs_e,
    stride_bs_k,
    stride_bs_n,
    stride_im,
    stride_in,
    # Constexprs
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    NUM_EXPERTS_BIT_LENGTH: tl.constexpr,
    SIMULATE_UNFUSED: tl.constexpr,
):
    """Kernel 1: gather A from unsorted → gate_up GEMM → SiLU → FP8 quant.

    Grid: (M-tiles, N-tiles). Each program writes one (BLOCK_M, BLOCK_N)
    tile of the fp8 intermediate + per-row scales. No atomics.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    total_tiles = tl.load(TileOffsets + NUM_EXPERTS - 1)
    if pid_m >= total_tiles:
        return

    # Binary search for expert
    lo = 0
    hi = NUM_EXPERTS
    for _ in tl.static_range(NUM_EXPERTS_BIT_LENGTH):
        mid = (lo + hi) >> 1
        mid_val = tl.load(TileOffsets + mid)
        is_left = mid_val <= pid_m
        lo = tl.where(is_left, mid + 1, lo)
        hi = tl.where(is_left, hi, mid)
    expert_id = lo.to(tl.int64)

    prev_eid = tl.maximum(expert_id - 1, 0)
    expert_start = tl.where(expert_id == 0, 0, tl.load(Offsets + prev_eid))
    expert_end = tl.load(Offsets + expert_id)
    M_expert = expert_end - expert_start

    expert_tile_start = tl.where(expert_id == 0, 0, tl.load(TileOffsets + prev_eid))
    local_tile = pid_m - expert_tile_start
    m_off = local_tile * BLOCK_SIZE_M

    offs_am = m_off + tl.arange(0, BLOCK_SIZE_M)
    row_mask = offs_am < M_expert
    sorted_indices = expert_start + offs_am

    # Gather from unsorted A via Perm
    perm_vals = tl.load(Perm + sorted_indices, mask=row_mask, other=0)
    original_tokens = perm_vals // num_top_k

    # Gate + Up projection
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + original_tokens[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_base = W_gu + expert_id * stride_be + offs_k[:, None] * stride_bk
    b_gate_ptrs = b_base + offs_bn[None, :] * stride_bn
    b_up_ptrs = b_base + (N_inter + offs_bn)[None, :] * stride_bn

    n_scale_blocks = N_inter // BLOCK_SIZE_N
    bs_base = Ws_gu + expert_id * stride_bs_e
    bs_gate_ptrs = bs_base + pid_n * stride_bs_n
    bs_up_ptrs = bs_base + (n_scale_blocks + pid_n) * stride_bs_n

    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        b_gate = tl.load(b_gate_ptrs)
        b_up = tl.load(b_up_ptrs)
        bs_gate = tl.load(bs_gate_ptrs + k * stride_bs_k)
        bs_up = tl.load(bs_up_ptrs + k * stride_bs_k)

        a_raw = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        a_s = tl.max(tl.abs(a_raw), axis=1) / 448.0
        a = (a_raw / tl.maximum(a_s[:, None], 1e-12)).to(tl.float8e4nv)

        acc_gate += tl.dot(a, b_gate) * a_s[:, None] * bs_gate[None, :]
        acc_up += tl.dot(a, b_up) * a_s[:, None] * bs_up[None, :]

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_gate_ptrs += BLOCK_SIZE_K * stride_bk
        b_up_ptrs += BLOCK_SIZE_K * stride_bk

    # SiLU(gate) * up
    if SIMULATE_UNFUSED:
        acc_gate = acc_gate.to(tl.bfloat16).to(tl.float32)
        acc_up = acc_up.to(tl.bfloat16).to(tl.float32)
        intermediate = (acc_gate * tl.sigmoid(acc_gate)).to(tl.bfloat16).to(
            tl.float32
        ) * acc_up
        intermediate = intermediate.to(tl.bfloat16).to(tl.float32)
    else:
        intermediate = acc_gate * tl.sigmoid(acc_gate) * acc_up

    # FP8 quantize — per-row scale across this N-tile
    inter_s = tl.max(tl.abs(intermediate), axis=1) / 448.0
    inter_fp8 = (intermediate / tl.maximum(inter_s[:, None], 1e-12)).to(tl.float8e4nv)

    # Store fp8 intermediate tile
    inter_ptrs = (
        Inter + sorted_indices[:, None] * stride_im + offs_bn[None, :] * stride_in
    )
    tl.store(inter_ptrs, inter_fp8, mask=row_mask[:, None])

    # Store per-row scale (one per row per N-tile)
    # Layout: Inter_s[sorted_idx, pid_n]
    scale_ptrs = Inter_s + sorted_indices * tl.cdiv(N_inter, BLOCK_SIZE_N) + pid_n
    tl.store(scale_ptrs, inter_s, mask=row_mask)


# ── Kernel 2: down projection from fp8 intermediate ─────────────────────────


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [4, 8, 16]
        for s in [2, 3, 4]
    ],
    key=["N_inter", "hidden", "BLOCK_SIZE_M"],
)
@triton.jit
def fused_down_proj_kernel(
    # Inputs
    Inter,  # (S, N_inter) FP8 intermediate
    Inter_s,  # (S, num_N_tiles) fp32 per-row-per-N-tile scales
    W_down,  # (E, hidden, N_inter) FP8 down weights
    Ws_down,  # down scales
    SampleWeights,  # (S,) routing weights in sorted order
    Perm,  # (S,) int64 — sorted_pos → original flat index
    # Output
    Out,  # (num_tokens * top_k, hidden) output in original flat order
    # Expert scheduling
    Offsets,
    TileOffsets,
    # Shapes
    N_inter,
    hidden,
    # Strides — Inter, W_down, Ws_down, Out
    stride_im,
    stride_in,
    stride_be,
    stride_bk,
    stride_bn,
    stride_bs_e,
    stride_bs_k,
    stride_bs_n,
    stride_om,
    stride_oh,
    # Constexprs
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    NUM_EXPERTS_BIT_LENGTH: tl.constexpr,
    SIMULATE_UNFUSED: tl.constexpr,
):
    """Kernel 2: fp8 intermediate → down_proj → output.

    Grid: (M-tiles, H-tiles). Each program reads (BLOCK_M, N_inter) fp8
    intermediate, does the down projection tiled over N, stores (BLOCK_M, BLOCK_H).
    No atomics — deterministic.
    """
    pid_m = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    total_tiles = tl.load(TileOffsets + NUM_EXPERTS - 1)
    if pid_m >= total_tiles:
        return

    # Binary search for expert
    lo = 0
    hi = NUM_EXPERTS
    for _ in tl.static_range(NUM_EXPERTS_BIT_LENGTH):
        mid = (lo + hi) >> 1
        mid_val = tl.load(TileOffsets + mid)
        is_left = mid_val <= pid_m
        lo = tl.where(is_left, mid + 1, lo)
        hi = tl.where(is_left, hi, mid)
    expert_id = lo.to(tl.int64)

    prev_eid = tl.maximum(expert_id - 1, 0)
    expert_start = tl.where(expert_id == 0, 0, tl.load(Offsets + prev_eid))
    expert_end = tl.load(Offsets + expert_id)
    M_expert = expert_end - expert_start

    expert_tile_start = tl.where(expert_id == 0, 0, tl.load(TileOffsets + prev_eid))
    local_tile = pid_m - expert_tile_start
    m_off = local_tile * BLOCK_SIZE_M

    offs_am = m_off + tl.arange(0, BLOCK_SIZE_M)
    row_mask = offs_am < M_expert
    sorted_indices = expert_start + offs_am

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_H), dtype=tl.float32)

    for n_tile in range(0, NUM_N_TILES):
        n_offs = n_tile * BLOCK_SIZE_N + offs_n

        # Load fp8 intermediate tile
        inter_ptrs = (
            Inter + sorted_indices[:, None] * stride_im + n_offs[None, :] * stride_in
        )
        inter_fp8 = tl.load(inter_ptrs, mask=row_mask[:, None], other=0.0)

        # Load per-row scale for this N-tile
        scale_ptrs = Inter_s + sorted_indices * NUM_N_TILES + n_tile
        inter_s = tl.load(scale_ptrs, mask=row_mask, other=0.0)

        # Load down weights
        w_down_ptrs = (
            W_down
            + expert_id * stride_be
            + n_offs[:, None] * stride_bk
            + offs_h[None, :] * stride_bn
        )
        w_down = tl.load(w_down_ptrs)
        ws_down = tl.load(
            Ws_down
            + expert_id * stride_bs_e
            + pid_h * stride_bs_n
            + n_tile * stride_bs_k
        )

        acc += tl.dot(inter_fp8, w_down) * inter_s[:, None] * ws_down

    # Apply routing weights and scatter to original flat order via Perm
    if SIMULATE_UNFUSED:
        acc = acc.to(tl.bfloat16).to(tl.float32)
    routing_w = tl.load(SampleWeights + sorted_indices, mask=row_mask, other=0.0)
    acc = acc * routing_w[:, None]
    original_flat_idx = tl.load(Perm + sorted_indices, mask=row_mask, other=0)
    out_ptrs = (
        Out + original_flat_idx[:, None] * stride_om + offs_h[None, :] * stride_oh
    )
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=row_mask[:, None])


# ── Wrapper ──────────────────────────────────────────────────────────────────


@triton_op("finegrained_fp8::moe_grouped_fused", mutates_args=())
def _moe_grouped_fused(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    block_size: list[int],
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """Two-kernel fused MoE expert layer: deterministic, no atomics.

    Input:  unsorted hidden_states + router outputs (top_k_index, top_k_weights)
    Output: (num_tokens, hidden) — accumulated across top_k experts

    Pipeline: sort → fused_gate_up_silu_kernel → fused_down_proj_kernel → routing + unsort + reduce
    Deterministic: no atomic_add, intermediate goes through HBM as fp8.
    """
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_experts = gate_up_proj.size(0)
    num_tokens = hidden_states.size(0)
    hidden_dim = down_proj.size(1)
    intermediate_dim = down_proj.size(2)
    block_n, block_k = block_size

    # S is the number of selected token-expert pairs (S = num_tokens * num_top_k)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)
    S = expert_ids.size(0)

    # Sort by expert for grouped processing
    _, perm = expert_ids.sort(stable=True)
    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]

    # Compute offsets for grouped processing
    histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
    tokens_per_expert = torch.histc(
        histc_input, bins=num_experts, min=0, max=num_experts - 1
    )
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    # Tile setup
    BLOCK_SIZE_M = min(
        max(triton.next_power_of_2((S + num_experts - 1) // num_experts), 16), 128
    )
    tiles_per_expert = (tokens_per_expert + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    tile_offsets = torch.cumsum(tiles_per_expert, dim=0).to(torch.int32)
    max_M_tiles = triton.cdiv(S, BLOCK_SIZE_M) + num_experts
    num_N_tiles = triton.cdiv(intermediate_dim, block_n)
    num_H_tiles = triton.cdiv(hidden_dim, block_n)

    # Temp buffer: fp8 intermediate + per-row-per-N-tile scales
    inter_fp8 = torch.empty(
        S, intermediate_dim, device=device, dtype=torch.float8_e4m3fn
    )
    inter_scales = torch.empty(S, num_N_tiles, device=device, dtype=torch.float32)

    # --- Kernel 1: gate_up + SiLU + FP8 quant ---
    grid1 = (max_M_tiles, num_N_tiles)
    with device_context(device):
        wrap_triton(fused_gate_up_silu_kernel)[grid1](
            hidden_states,
            perm,
            gate_up_proj,
            gate_up_proj_scale_inv,
            inter_fp8,
            inter_scales,
            offsets,
            tile_offsets,
            intermediate_dim,
            hidden_states.shape[1],
            num_top_k,
            hidden_states.stride(0),
            hidden_states.stride(1),
            gate_up_proj.stride(0),
            gate_up_proj.stride(2),
            gate_up_proj.stride(1),
            gate_up_proj_scale_inv.stride(0),
            gate_up_proj_scale_inv.stride(2),
            gate_up_proj_scale_inv.stride(1),
            inter_fp8.stride(0),
            inter_fp8.stride(1),
            NUM_EXPERTS=num_experts,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            NUM_EXPERTS_BIT_LENGTH=num_experts.bit_length(),
            SIMULATE_UNFUSED=simulate_unfused,
        )

    # --- Kernel 2: down projection + routing weights + scatter to original order ---
    proj_out = torch.empty(S, hidden_dim, device=device, dtype=hidden_states.dtype)
    grid2 = (max_M_tiles, num_H_tiles)
    with device_context(device):
        wrap_triton(fused_down_proj_kernel)[grid2](
            inter_fp8,
            inter_scales,
            down_proj,
            down_proj_scale_inv,
            sample_weights_g,
            perm,
            proj_out,
            offsets,
            tile_offsets,
            intermediate_dim,
            hidden_dim,
            inter_fp8.stride(0),
            inter_fp8.stride(1),
            down_proj.stride(0),
            down_proj.stride(2),
            down_proj.stride(1),
            down_proj_scale_inv.stride(0),
            down_proj_scale_inv.stride(2),
            down_proj_scale_inv.stride(1),
            proj_out.stride(0),
            proj_out.stride(1),
            NUM_EXPERTS=num_experts,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_H=block_n,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            NUM_N_TILES=num_N_tiles,
            NUM_EXPERTS_BIT_LENGTH=num_experts.bit_length(),
            SIMULATE_UNFUSED=simulate_unfused,
        )

    # Output already in original flat order — just reduce across top_k
    final_hidden_states = proj_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)


def moe_grouped_fused(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    block_size: list[int],
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """Two-kernel fused MoE expert layer: deterministic, no atomics.

    Input:  unsorted hidden_states + router outputs (top_k_index, top_k_weights)
    Output: (num_tokens, hidden) — accumulated across top_k experts

    Pipeline: sort → fused_gate_up_silu_kernel → fused_down_proj_kernel → routing + unsort + reduce
    Deterministic: no atomic_add, intermediate goes through HBM as fp8.
    """
    return torch.ops.finegrained_fp8.moe_grouped_fused(
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


# ── Batched fused: gate_up + SiLU + down (no sorting, no atomics) ───────────
#
# Each program handles one (token, H-tile) and loops over N-tiles sequentially.
# The intermediate stays entirely in registers. No sorting needed — expert
# lookup is per-token via ExpertIds.


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [4, 8, 16]
        for s in [2, 3, 4]
    ],
    key=["N_inter", "K", "BLOCK_SIZE_M"],
)
@triton.jit
def moe_batched_fused_kernel(
    A,  # (S, K) raw BF16/FP16 activations
    W_gu,  # (E, 2*N_inter, K) FP8 gate_up weights
    W_down,  # (E, hidden, N_inter) FP8 down weights
    Out,  # (S, hidden) output
    Ws_gu,  # gate_up scales
    Ws_down,  # down scales
    ExpertIds,  # (S,) expert index per token
    SampleWeights,  # (S,) routing weights
    # Shapes
    N_inter,
    K,
    hidden,
    # Strides — A
    stride_am,
    stride_ak,
    # Strides — W_gu, Ws_gu
    stride_be_gu,
    stride_bk_gu,
    stride_bn_gu,
    stride_bs_e_gu,
    stride_bs_k_gu,
    stride_bs_n_gu,
    # Strides — W_down, Ws_down
    stride_be_down,
    stride_bk_down,
    stride_bn_down,
    stride_bs_e_down,
    stride_bs_k_down,
    stride_bs_n_down,
    # Strides — Out
    stride_om,
    stride_oh,
    # Constexprs
    NUM_N_TILES: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    SIMULATE_UNFUSED: tl.constexpr,
):
    """Batched fused MoE kernel: gate_up + SiLU + down in one kernel, no atomics.

    Grid: (S, H-tiles). Each program handles one (token, H-tile) and loops
    over N-tiles. The intermediate stays entirely in registers.
    """
    batch_id = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    expert_id = tl.load(ExpertIds + batch_id).to(tl.int64)

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    acc_down = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_H), dtype=tl.float32)

    for n_inter in range(0, NUM_N_TILES):
        offs_n = n_inter * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        # ── Gate + Up projection for this N-tile ──
        a_ptrs = A + batch_id * stride_am + offs_k[None, :] * stride_ak
        b_base = W_gu + expert_id * stride_be_gu + offs_k[:, None] * stride_bk_gu
        b_gate_ptrs = b_base + offs_n[None, :] * stride_bn_gu
        b_up_ptrs = b_base + (N_inter + offs_n)[None, :] * stride_bn_gu

        n_scale_blocks = N_inter // BLOCK_SIZE_N
        bs_base = Ws_gu + expert_id * stride_bs_e_gu
        bs_gate_ptr = bs_base + n_inter * stride_bs_n_gu
        bs_up_ptr = bs_base + (n_scale_blocks + n_inter) * stride_bs_n_gu

        acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, NUM_K_TILES):
            a_raw = tl.load(a_ptrs + offs_m[:, None] * 0).to(tl.float32)
            a_s = tl.max(tl.abs(a_raw)) / 448.0
            a = (a_raw / tl.maximum(a_s, 1e-12)).to(tl.float8e4nv)

            b_gate = tl.load(b_gate_ptrs)
            b_up = tl.load(b_up_ptrs)
            bs_gate = tl.load(bs_gate_ptr + k * stride_bs_k_gu)
            bs_up = tl.load(bs_up_ptr + k * stride_bs_k_gu)

            acc_gate += tl.dot(a, b_gate) * a_s * bs_gate[None, :]
            acc_up += tl.dot(a, b_up) * a_s * bs_up[None, :]

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_gate_ptrs += BLOCK_SIZE_K * stride_bk_gu
            b_up_ptrs += BLOCK_SIZE_K * stride_bk_gu

        # ── SiLU(gate) * up ──
        if SIMULATE_UNFUSED:
            acc_gate = acc_gate.to(tl.bfloat16).to(tl.float32)
            acc_up = acc_up.to(tl.bfloat16).to(tl.float32)
            intermediate = (acc_gate * tl.sigmoid(acc_gate)).to(tl.bfloat16).to(
                tl.float32
            ) * acc_up
            intermediate = intermediate.to(tl.bfloat16).to(tl.float32)
        else:
            intermediate = acc_gate * tl.sigmoid(acc_gate) * acc_up

        # ── Quantize intermediate to FP8 ──
        inter_s = tl.max(tl.abs(intermediate)) / 448.0
        inter_fp8 = (intermediate / tl.maximum(inter_s, 1e-12)).to(tl.float8e4nv)

        # ── Partial down projection ──
        w_down_ptrs = (
            W_down
            + expert_id * stride_be_down
            + offs_n[:, None] * stride_bk_down
            + offs_h[None, :] * stride_bn_down
        )
        w_down = tl.load(w_down_ptrs)
        ws_down = tl.load(
            Ws_down
            + expert_id * stride_bs_e_down
            + pid_h * stride_bs_n_down
            + n_inter * stride_bs_k_down
        )

        acc_down += tl.dot(inter_fp8, w_down) * inter_s * ws_down

    # ── Apply routing weight and store ──
    if SIMULATE_UNFUSED:
        acc_down = acc_down.to(tl.bfloat16).to(tl.float32)
    routing_w = tl.load(SampleWeights + batch_id)
    acc_down = acc_down * routing_w

    if Out.dtype.element_ty == tl.bfloat16:
        c = acc_down.to(tl.bfloat16)
    elif Out.dtype.element_ty == tl.float16:
        c = acc_down.to(tl.float16)
    else:
        c = acc_down.to(tl.float32)

    c_ptrs = (
        Out + batch_id * stride_om + offs_h[None, :] * stride_oh + offs_m[:, None] * 0
    )
    tl.store(c_ptrs, c)


@triton_op("finegrained_fp8::moe_batched_fused", mutates_args=())
def _moe_batched_fused(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    block_size: list[int],
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """Batched fused MoE expert layer: deterministic, no sorting, no atomics.

    Input:  unsorted hidden_states + router outputs (top_k_index, top_k_weights)
    Output: (num_tokens, hidden) — accumulated across top_k experts

    Pipeline: expand → ONE kernel (gate_up + SiLU + down per token) → routing + reduce
    Deterministic: each token processed independently. Intermediate stays in registers.
    """
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = down_proj.shape[1]
    intermediate_dim = down_proj.shape[2]
    block_n, block_k = block_size

    # S is the number of selected token-expert pairs (S = num_tokens * num_top_k)
    token_idx = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1)
        .expand(-1, num_top_k)
        .reshape(-1)
    )
    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)

    selected_hidden_states = hidden_states[token_idx]
    S = expert_ids.size(0)

    BLOCK_SIZE_M = min(
        max(
            triton.next_power_of_2(
                (S + gate_up_proj.shape[0] - 1) // gate_up_proj.shape[0]
            ),
            16,
        ),
        128,
    )
    Out = selected_hidden_states.new_empty(S, hidden_dim)
    num_H_tiles = triton.cdiv(hidden_dim, block_n)
    grid = (S, num_H_tiles)

    with device_context(device):
        wrap_triton(moe_batched_fused_kernel)[grid](
            selected_hidden_states,
            gate_up_proj,
            down_proj,
            Out,
            gate_up_proj_scale_inv,
            down_proj_scale_inv,
            expert_ids,
            sample_weights,
            intermediate_dim,
            hidden_states.shape[1],
            hidden_dim,
            selected_hidden_states.stride(0),
            selected_hidden_states.stride(1),
            gate_up_proj.stride(0),
            gate_up_proj.stride(2),
            gate_up_proj.stride(1),
            gate_up_proj_scale_inv.stride(0),
            gate_up_proj_scale_inv.stride(2),
            gate_up_proj_scale_inv.stride(1),
            down_proj.stride(0),
            down_proj.stride(2),
            down_proj.stride(1),
            down_proj_scale_inv.stride(0),
            down_proj_scale_inv.stride(2),
            down_proj_scale_inv.stride(1),
            Out.stride(0),
            Out.stride(1),
            NUM_N_TILES=triton.cdiv(intermediate_dim, block_n),
            NUM_K_TILES=triton.cdiv(hidden_states.shape[1], block_k),
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            BLOCK_SIZE_H=block_n,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            SIMULATE_UNFUSED=simulate_unfused,
        )

    # Routing weights already applied in kernel — just reduce
    final_hidden_states = Out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)


def moe_batched_fused(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    block_size: list[int],
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """Batched fused MoE expert layer: deterministic, no sorting, no atomics.

    Input:  unsorted hidden_states + router outputs (top_k_index, top_k_weights)
    Output: (num_tokens, hidden) — accumulated across top_k experts

    Pipeline: expand → ONE kernel (gate_up + SiLU + down per token) → routing + reduce
    Deterministic: each token processed independently. Intermediate stays in registers.
    """
    return torch.ops.finegrained_fp8.moe_batched_fused(
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
