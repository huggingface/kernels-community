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
#
# Optimizations: block pointers for weight loads, L2 swizzle (GROUP_SIZE_M),
# tile-to-expert via bucketize (replaces binary search), BLOCK_SIZE_M capped at 64.


@triton.autotune(
    configs=[
        triton.Config({"GROUP_SIZE_M": g}, num_warps=w, num_stages=s)
        for w in [2, 4, 8, 16]
        for s in [2, 3, 4, 5]
        for g in [1, 8]
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
    TileToExpert,  # (max_M_tiles,) int32 — tile → expert lookup
    # Shapes
    N_inter,
    K,
    num_top_k,
    num_M_tiles,
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
    SIMULATE_UNFUSED: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel 1: gather A from unsorted → gate_up GEMM → SiLU → FP8 quant.

    Grid: (M-tiles, N-tiles). Each program writes one (BLOCK_M, BLOCK_N)
    tile of the fp8 intermediate + per-row scales. No atomics.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_N_tiles = tl.cdiv(N_inter, BLOCK_SIZE_N)
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_M_tiles, num_N_tiles, GROUP_SIZE_M)

    total_tiles = tl.load(TileOffsets + NUM_EXPERTS - 1)
    if pid_m >= total_tiles:
        return

    # O(1) tile → expert lookup
    expert_id = tl.load(TileToExpert + pid_m).to(tl.int64)

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

    b_gate_ptr = tl.make_block_ptr(
        base=W_gu + expert_id * stride_be,
        shape=(K, N_inter * 2),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(0, 1),
    )
    b_up_ptr = tl.make_block_ptr(
        base=W_gu + expert_id * stride_be,
        shape=(K, N_inter * 2),
        strides=(stride_bk, stride_bn),
        offsets=(0, N_inter + pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(0, 1),
    )

    n_scale_blocks = N_inter // BLOCK_SIZE_N
    bs_base = Ws_gu + expert_id * stride_bs_e
    bs_gate_ptrs = bs_base + pid_n * stride_bs_n
    bs_up_ptrs = bs_base + (n_scale_blocks + pid_n) * stride_bs_n

    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        b_gate = tl.load(b_gate_ptr)
        b_up = tl.load(b_up_ptr)
        bs_gate = tl.load(bs_gate_ptrs + k * stride_bs_k)
        bs_up = tl.load(bs_up_ptrs + k * stride_bs_k)

        a_raw = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        a_s = tl.max(tl.abs(a_raw), axis=1) / 448.0
        a = (a_raw / tl.maximum(a_s[:, None], 1e-12)).to(tl.float8e4nv)

        acc_gate += tl.dot(a, b_gate) * a_s[:, None] * bs_gate[None, :]
        acc_up += tl.dot(a, b_up) * a_s[:, None] * bs_up[None, :]

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_gate_ptr = tl.advance(b_gate_ptr, (BLOCK_SIZE_K, 0))
        b_up_ptr = tl.advance(b_up_ptr, (BLOCK_SIZE_K, 0))

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
    scale_ptrs = Inter_s + sorted_indices * tl.cdiv(N_inter, BLOCK_SIZE_N) + pid_n
    tl.store(scale_ptrs, inter_s, mask=row_mask)


# ── Kernel 2: down projection from fp8 intermediate ─────────────────────────


@triton.autotune(
    configs=[
        triton.Config({"GROUP_SIZE_M": g}, num_warps=w, num_stages=s)
        for w in [2, 4, 8, 16]
        for s in [2, 3, 4, 5]
        for g in [1, 8]
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
    TileToExpert,  # (max_M_tiles,) int32 — tile → expert lookup
    # Shapes
    N_inter,
    hidden,
    num_M_tiles,
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
    SIMULATE_UNFUSED: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel 2: fp8 intermediate → down_proj → output.

    Grid: (M-tiles, H-tiles). Each program reads (BLOCK_M, N_inter) fp8
    intermediate, does the down projection tiled over N, stores (BLOCK_M, BLOCK_H).
    No atomics — deterministic.
    """
    pid_m = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    num_H_tiles = tl.cdiv(hidden, BLOCK_SIZE_H)
    pid_m, pid_h = tl.swizzle2d(pid_m, pid_h, num_M_tiles, num_H_tiles, GROUP_SIZE_M)

    total_tiles = tl.load(TileOffsets + NUM_EXPERTS - 1)
    if pid_m >= total_tiles:
        return

    # O(1) tile → expert lookup
    expert_id = tl.load(TileToExpert + pid_m).to(tl.int64)

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

    # Block pointer for down weights
    w_down_ptr = tl.make_block_ptr(
        base=W_down + expert_id * stride_be,
        shape=(N_inter, hidden),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_h * BLOCK_SIZE_H),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_H),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_H), dtype=tl.float32)

    for n_tile in range(0, NUM_N_TILES):
        n_offs = n_tile * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        # Load fp8 intermediate tile
        inter_ptrs = (
            Inter + sorted_indices[:, None] * stride_im + n_offs[None, :] * stride_in
        )
        inter_fp8 = tl.load(inter_ptrs, mask=row_mask[:, None], other=0.0)

        # Load per-row scale for this N-tile
        scale_ptrs = Inter_s + sorted_indices * NUM_N_TILES + n_tile
        inter_s = tl.load(scale_ptrs, mask=row_mask, other=0.0)

        # Load down weights via block pointer
        w_down = tl.load(w_down_ptr)
        ws_down = tl.load(
            Ws_down
            + expert_id * stride_bs_e
            + pid_h * stride_bs_n
            + n_tile * stride_bs_k
        )

        acc += tl.dot(inter_fp8, w_down) * inter_s[:, None] * ws_down
        w_down_ptr = tl.advance(w_down_ptr, (BLOCK_SIZE_N, 0))

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

    # Tile setup — BLOCK_SIZE_M capped at 64 for better SM utilization with many experts
    BLOCK_SIZE_M = min(
        max(triton.next_power_of_2((S + num_experts - 1) // num_experts), 16), 64
    )
    tiles_per_expert = (tokens_per_expert + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    tile_offsets = torch.cumsum(tiles_per_expert, dim=0).to(torch.int32)
    max_M_tiles = triton.cdiv(S, BLOCK_SIZE_M) + num_experts
    tile_to_expert = torch.bucketize(
        torch.arange(max_M_tiles, device=device, dtype=torch.int32),
        tile_offsets,
        right=True,
    )

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
            tile_to_expert,
            intermediate_dim,
            hidden_states.shape[1],
            num_top_k,
            max_M_tiles,
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
            tile_to_expert,
            intermediate_dim,
            hidden_dim,
            max_M_tiles,
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


# ── Batched fused: two-kernel approach (no sorting, no atomics) ──────────────
#
# Same two-kernel architecture as grouped fused but with per-token dispatch:
# Kernel 1: (S, N-tiles) — gate_up + SiLU + FP8 quant → intermediate buffer
# Kernel 2: (S, H-tiles) — fp8 intermediate → down proj → output
# No sorting needed — expert lookup is per-token via ExpertIds.


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [4, 8, 16]
        for s in [2, 3, 4]
    ],
    key=["N_inter", "K", "BLOCK_SIZE_M"],
)
@triton.jit
def batched_gate_up_silu_kernel(
    A,
    W_gu,
    Ws_gu,
    Inter,
    Inter_s,
    ExpertIds,
    N_inter,
    K,
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
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    SIMULATE_UNFUSED: tl.constexpr,
):
    """Batched kernel 1: per-token gate_up + SiLU + FP8 quant. Grid: (S, N-tiles)."""
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    expert_id = tl.load(ExpertIds + batch_id).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    a_ptrs = A + batch_id * stride_am + offs_k[None, :] * stride_ak

    b_gate_ptr = tl.make_block_ptr(
        base=W_gu + expert_id * stride_be,
        shape=(K, N_inter * 2),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(0, 1),
    )
    b_up_ptr = tl.make_block_ptr(
        base=W_gu + expert_id * stride_be,
        shape=(K, N_inter * 2),
        strides=(stride_bk, stride_bn),
        offsets=(0, N_inter + pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(0, 1),
    )

    n_scale_blocks = N_inter // BLOCK_SIZE_N
    bs_base = Ws_gu + expert_id * stride_bs_e
    bs_gate_ptrs = bs_base + pid_n * stride_bs_n
    bs_up_ptrs = bs_base + (n_scale_blocks + pid_n) * stride_bs_n

    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_raw = tl.load(a_ptrs + offs_m[:, None] * 0).to(tl.float32)
        a_s = tl.max(tl.abs(a_raw)) / 448.0
        a = (a_raw / tl.maximum(a_s, 1e-12)).to(tl.float8e4nv)

        b_gate = tl.load(b_gate_ptr)
        b_up = tl.load(b_up_ptr)
        bs_gate = tl.load(bs_gate_ptrs + k * stride_bs_k)
        bs_up = tl.load(bs_up_ptrs + k * stride_bs_k)

        acc_gate += tl.dot(a, b_gate) * a_s * bs_gate[None, :]
        acc_up += tl.dot(a, b_up) * a_s * bs_up[None, :]

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_gate_ptr = tl.advance(b_gate_ptr, (BLOCK_SIZE_K, 0))
        b_up_ptr = tl.advance(b_up_ptr, (BLOCK_SIZE_K, 0))

    if SIMULATE_UNFUSED:
        acc_gate = acc_gate.to(tl.bfloat16).to(tl.float32)
        acc_up = acc_up.to(tl.bfloat16).to(tl.float32)
        intermediate = (acc_gate * tl.sigmoid(acc_gate)).to(tl.bfloat16).to(
            tl.float32
        ) * acc_up
        intermediate = intermediate.to(tl.bfloat16).to(tl.float32)
    else:
        intermediate = acc_gate * tl.sigmoid(acc_gate) * acc_up

    inter_s = tl.max(tl.abs(intermediate)) / 448.0
    inter_fp8 = (intermediate / tl.maximum(inter_s, 1e-12)).to(tl.float8e4nv)

    inter_ptrs = (
        Inter
        + batch_id * stride_im
        + offs_bn[None, :] * stride_in
        + offs_m[:, None] * 0
    )
    tl.store(inter_ptrs, inter_fp8)

    num_N_tiles = tl.cdiv(N_inter, BLOCK_SIZE_N)
    tl.store(Inter_s + batch_id * num_N_tiles + pid_n, inter_s)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [4, 8, 16]
        for s in [2, 3, 4]
    ],
    key=["N_inter", "hidden", "BLOCK_SIZE_M"],
)
@triton.jit
def batched_down_proj_kernel(
    Inter,
    Inter_s,
    W_down,
    Ws_down,
    ExpertIds,
    SampleWeights,
    Out,
    N_inter,
    hidden,
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
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    SIMULATE_UNFUSED: tl.constexpr,
):
    """Batched kernel 2: fp8 intermediate → down proj → output. Grid: (S, H-tiles)."""
    batch_id = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    expert_id = tl.load(ExpertIds + batch_id).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    w_down_ptr = tl.make_block_ptr(
        base=W_down + expert_id * stride_be,
        shape=(N_inter, hidden),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_h * BLOCK_SIZE_H),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_H),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_H), dtype=tl.float32)

    for n_tile in range(0, NUM_N_TILES):
        n_offs = n_tile * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        inter_ptrs = (
            Inter
            + batch_id * stride_im
            + n_offs[None, :] * stride_in
            + offs_m[:, None] * 0
        )
        inter_fp8 = tl.load(inter_ptrs)

        inter_s = tl.load(Inter_s + batch_id * NUM_N_TILES + n_tile)

        w_down = tl.load(w_down_ptr)
        ws_down = tl.load(
            Ws_down
            + expert_id * stride_bs_e
            + pid_h * stride_bs_n
            + n_tile * stride_bs_k
        )

        acc += tl.dot(inter_fp8, w_down) * inter_s * ws_down
        w_down_ptr = tl.advance(w_down_ptr, (BLOCK_SIZE_N, 0))

    if SIMULATE_UNFUSED:
        acc = acc.to(tl.bfloat16).to(tl.float32)
    routing_w = tl.load(SampleWeights + batch_id)
    acc = acc * routing_w

    if Out.dtype.element_ty == tl.bfloat16:
        c = acc.to(tl.bfloat16)
    elif Out.dtype.element_ty == tl.float16:
        c = acc.to(tl.float16)
    else:
        c = acc.to(tl.float32)

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
    """Two-kernel batched fused MoE: deterministic, no sorting, no atomics.

    Pipeline: expand → kernel 1 (gate_up + SiLU + FP8 quant) → kernel 2 (down proj + routing) → reduce
    """
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = down_proj.shape[1]
    intermediate_dim = down_proj.shape[2]
    block_n, block_k = block_size

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

    num_N_tiles = triton.cdiv(intermediate_dim, block_n)
    num_H_tiles = triton.cdiv(hidden_dim, block_n)

    inter_fp8 = torch.empty(
        S, intermediate_dim, device=device, dtype=torch.float8_e4m3fn
    )
    inter_scales = torch.empty(S, num_N_tiles, device=device, dtype=torch.float32)

    # Kernel 1: gate_up + SiLU + FP8 quant — grid (S, N-tiles)
    BLOCK_SIZE_M = min(
        max(
            triton.next_power_of_2(
                (S + gate_up_proj.shape[0] - 1) // gate_up_proj.shape[0]
            ),
            16,
        ),
        64,
    )
    grid1 = (S, num_N_tiles)
    with device_context(device):
        wrap_triton(batched_gate_up_silu_kernel)[grid1](
            selected_hidden_states,
            gate_up_proj,
            gate_up_proj_scale_inv,
            inter_fp8,
            inter_scales,
            expert_ids,
            intermediate_dim,
            hidden_states.shape[1],
            selected_hidden_states.stride(0),
            selected_hidden_states.stride(1),
            gate_up_proj.stride(0),
            gate_up_proj.stride(2),
            gate_up_proj.stride(1),
            gate_up_proj_scale_inv.stride(0),
            gate_up_proj_scale_inv.stride(2),
            gate_up_proj_scale_inv.stride(1),
            inter_fp8.stride(0),
            inter_fp8.stride(1),
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            SIMULATE_UNFUSED=simulate_unfused,
        )

    # Kernel 2: down proj + routing — grid (S, H-tiles)
    Out = selected_hidden_states.new_empty(S, hidden_dim)
    grid2 = (S, num_H_tiles)
    with device_context(device):
        wrap_triton(batched_down_proj_kernel)[grid2](
            inter_fp8,
            inter_scales,
            down_proj,
            down_proj_scale_inv,
            expert_ids,
            sample_weights,
            Out,
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
            Out.stride(0),
            Out.stride(1),
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_H=block_n,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            NUM_N_TILES=num_N_tiles,
            SIMULATE_UNFUSED=simulate_unfused,
        )

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
    """Two-kernel batched fused MoE: deterministic, no sorting, no atomics."""
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
