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

"""Atomic fused MoE: gate_up + SiLU + down in one pass with atomic split-K.

The intermediate tensor NEVER hits HBM — gate_up, SiLU, FP8 quantization,
and down projection all happen in registers. Output is accumulated via
atomic_add (split-K across intermediate N-tiles).

Non-deterministic due to atomic_add ordering. Faster than the two-kernel
deterministic path at small token counts (decode), but degrades at high
token counts due to atomic contention.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from .utils import device_context


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [4, 8, 16]
        for s in [2, 3, 4]
    ],
    key=["N_inter", "K", "BLOCK_SIZE_M"],
    reset_to_zero=["Out"],
)
@triton.jit
def moe_atomic_kernel(
    # Unsorted inputs
    A,  # (num_tokens, K) raw BF16/FP16 — NOT sorted
    Perm,  # (S,) int64 — sorted_pos → original flat index
    SampleWeights,  # (S,) routing weights in sorted order
    # Expert weights
    W_gu,  # (E, 2*N_inter, K) FP8 gate_up weights
    W_down,  # (E, hidden, N_inter) FP8 down weights
    Ws_gu,  # gate_up scales
    Ws_down,  # down scales
    # Output
    Out,  # (S, hidden) — accumulated via atomic_add
    # Expert scheduling
    Offsets,
    TileOffsets,
    # Shapes
    N_inter,
    K,
    hidden,
    num_top_k,
    # Strides — A, W_gu, Ws_gu, W_down, Ws_down, Out
    stride_am,
    stride_ak,
    stride_be_gu,
    stride_bk_gu,
    stride_bn_gu,
    stride_bs_e_gu,
    stride_bs_k_gu,
    stride_bs_n_gu,
    stride_be_down,
    stride_bk_down,
    stride_bn_down,
    stride_bs_e_down,
    stride_bs_k_down,
    stride_bs_n_down,
    stride_om,
    stride_oh,
    # Constexprs
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    NUM_H_TILES: tl.constexpr,
    NUM_EXPERTS_BIT_LENGTH: tl.constexpr,
    SIMULATE_UNFUSED: tl.constexpr,
):
    """Single fused MoE kernel: gather + gate_up + SiLU + down in one pass.

    Grid: (M-tiles, N-tiles). Each program:
    1. Gathers A from unsorted hidden_states via Perm
    2. Gate+up GEMM → SiLU → FP8 quant (intermediate in registers)
    3. Loops over H-tiles: down projection → atomic_add to output
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

    # ── Gate + Up projection ──
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + original_tokens[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_base = W_gu + expert_id * stride_be_gu + offs_k[:, None] * stride_bk_gu
    b_gate_ptrs = b_base + offs_bn[None, :] * stride_bn_gu
    b_up_ptrs = b_base + (N_inter + offs_bn)[None, :] * stride_bn_gu

    n_scale_blocks = N_inter // BLOCK_SIZE_N
    bs_base = Ws_gu + expert_id * stride_bs_e_gu
    bs_gate_ptrs = bs_base + pid_n * stride_bs_n_gu
    bs_up_ptrs = bs_base + (n_scale_blocks + pid_n) * stride_bs_n_gu

    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        b_gate = tl.load(b_gate_ptrs)
        b_up = tl.load(b_up_ptrs)
        bs_gate = tl.load(bs_gate_ptrs + k * stride_bs_k_gu)
        bs_up = tl.load(bs_up_ptrs + k * stride_bs_k_gu)

        a_raw = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        a_s = tl.max(tl.abs(a_raw), axis=1) / 448.0
        a = (a_raw / tl.maximum(a_s[:, None], 1e-12)).to(tl.float8e4nv)

        acc_gate += tl.dot(a, b_gate) * a_s[:, None] * bs_gate[None, :]
        acc_up += tl.dot(a, b_up) * a_s[:, None] * bs_up[None, :]

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
    inter_s = tl.max(tl.abs(intermediate), axis=1) / 448.0
    inter_fp8 = (intermediate / tl.maximum(inter_s[:, None], 1e-12)).to(tl.float8e4nv)

    # ── Down projection + atomic accumulate ──
    offs_h = tl.arange(0, BLOCK_SIZE_H)
    for h in range(0, NUM_H_TILES):
        h_offs = h * BLOCK_SIZE_H + offs_h
        w_down_ptrs = (
            W_down
            + expert_id * stride_be_down
            + offs_bn[:, None] * stride_bk_down
            + h_offs[None, :] * stride_bn_down
        )
        w_down = tl.load(w_down_ptrs)
        ws_down = tl.load(
            Ws_down
            + expert_id * stride_bs_e_down
            + h * stride_bs_n_down
            + pid_n * stride_bs_k_down
        )

        partial = tl.dot(inter_fp8, w_down) * inter_s[:, None] * ws_down

        out_ptrs = (
            Out + sorted_indices[:, None] * stride_om + h_offs[None, :] * stride_oh
        )
        tl.atomic_add(
            out_ptrs,
            partial.to(Out.dtype.element_ty),
            mask=row_mask[:, None],
            sem="relaxed",
        )


# ── Wrapper ──────────────────────────────────────────────────────────────────


@triton_op("finegrained_fp8::moe_grouped_atomic", mutates_args=())
def _moe_grouped_atomic(
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
    """Single-kernel fused MoE expert layer: non-deterministic (atomic split-K).

    Input:  unsorted hidden_states + router outputs (top_k_index, top_k_weights)
    Output: (num_tokens, hidden) — accumulated across top_k experts

    Pipeline: sort → ONE kernel (gather + gate_up + SiLU + down, atomic split-K) → routing + unsort + reduce
    Non-deterministic: atomic_add across intermediate N-tiles.
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

    # fp32 output for atomic accumulation
    proj_out = torch.zeros(S, hidden_dim, device=device, dtype=torch.float32)

    grid = (max_M_tiles, num_N_tiles)
    with device_context(device):
        wrap_triton(moe_atomic_kernel)[grid](
            hidden_states,
            perm,
            sample_weights_g,
            gate_up_proj,
            down_proj,
            gate_up_proj_scale_inv,
            down_proj_scale_inv,
            proj_out,
            offsets,
            tile_offsets,
            intermediate_dim,
            hidden_states.shape[1],
            hidden_dim,
            num_top_k,
            hidden_states.stride(0),
            hidden_states.stride(1),
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
            proj_out.stride(0),
            proj_out.stride(1),
            NUM_EXPERTS=num_experts,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            BLOCK_SIZE_H=block_n,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            NUM_H_TILES=triton.cdiv(hidden_dim, block_n),
            NUM_EXPERTS_BIT_LENGTH=num_experts.bit_length(),
            SIMULATE_UNFUSED=simulate_unfused,
        )

    # Apply routing weights + unsort + reduce
    proj_out = proj_out.to(hidden_states.dtype)
    weighted_out = proj_out * sample_weights_g.to(proj_out.dtype).unsqueeze(-1)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(S, device=device)
    final_hidden_states = (
        weighted_out[inv_perm].view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
    )

    return final_hidden_states.to(hidden_states.dtype)


def moe_grouped_atomic(
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
    """Single-kernel fused MoE expert layer: non-deterministic (atomic split-K).

    Input:  unsorted hidden_states + router outputs (top_k_index, top_k_weights)
    Output: (num_tokens, hidden) — accumulated across top_k experts

    Pipeline: sort → ONE kernel (gather + gate_up + SiLU + down, atomic split-K) → routing + unsort + reduce
    Non-deterministic: atomic_add across intermediate N-tiles. Intermediate stays in registers.
    """
    return torch.ops.finegrained_fp8.moe_grouped_atomic(
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


# ── Batched atomic: gate_up + SiLU + down per token (atomic split-K) ────────


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [4, 8, 16]
        for s in [2, 3, 4]
    ],
    key=["N_inter", "K", "BLOCK_SIZE_M"],
    reset_to_zero=["Out"],
)
@triton.jit
def moe_batched_atomic_kernel(
    A,  # (S, K) raw BF16/FP16 activations
    W_gu,  # (E, 2*N_inter, K) FP8 gate_up weights
    W_down,  # (E, hidden, N_inter) FP8 down weights
    Out,  # (S, hidden) output — accumulated via atomic_add
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
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    NUM_H_TILES: tl.constexpr,
    NUM_EXPERTS_BIT_LENGTH: tl.constexpr,
    SIMULATE_UNFUSED: tl.constexpr,
):
    """Batched atomic fused MoE kernel: gate_up + SiLU + down per token.

    Grid: (S, N-tiles). Each program handles one (token, N-tile):
    1. Gate+up GEMM → SiLU → FP8 quant (intermediate in registers)
    2. Loops over H-tiles: down projection → atomic_add to output
    """
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    expert_id = tl.load(ExpertIds + batch_id).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    # ── Gate + Up projection ──
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + batch_id * stride_am + offs_k[None, :] * stride_ak
    b_base = W_gu + expert_id * stride_be_gu + offs_k[:, None] * stride_bk_gu
    b_gate_ptrs = b_base + offs_bn[None, :] * stride_bn_gu
    b_up_ptrs = b_base + (N_inter + offs_bn)[None, :] * stride_bn_gu

    n_scale_blocks = N_inter // BLOCK_SIZE_N
    bs_base = Ws_gu + expert_id * stride_bs_e_gu
    bs_gate_ptrs = bs_base + pid_n * stride_bs_n_gu
    bs_up_ptrs = bs_base + (n_scale_blocks + pid_n) * stride_bs_n_gu

    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_raw = tl.load(a_ptrs + offs_m[:, None] * 0).to(tl.float32)
        a_s = tl.max(tl.abs(a_raw)) / 448.0
        a = (a_raw / tl.maximum(a_s, 1e-12)).to(tl.float8e4nv)

        b_gate = tl.load(b_gate_ptrs)
        b_up = tl.load(b_up_ptrs)
        bs_gate = tl.load(bs_gate_ptrs + k * stride_bs_k_gu)
        bs_up = tl.load(bs_up_ptrs + k * stride_bs_k_gu)

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

    # ── Down projection + atomic accumulate ──
    offs_h = tl.arange(0, BLOCK_SIZE_H)
    for h in range(0, NUM_H_TILES):
        h_offs = h * BLOCK_SIZE_H + offs_h
        w_down_ptrs = (
            W_down
            + expert_id * stride_be_down
            + offs_bn[:, None] * stride_bk_down
            + h_offs[None, :] * stride_bn_down
        )
        w_down = tl.load(w_down_ptrs)
        ws_down = tl.load(
            Ws_down
            + expert_id * stride_bs_e_down
            + h * stride_bs_n_down
            + pid_n * stride_bs_k_down
        )

        partial = tl.dot(inter_fp8, w_down) * inter_s * ws_down

        out_ptrs = (
            Out
            + batch_id * stride_om
            + h_offs[None, :] * stride_oh
            + offs_m[:, None] * 0
        )
        # Only write row 0 — all rows are identical (batched: 1 token per program)
        row_mask = offs_m[:, None] == 0
        tl.atomic_add(
            out_ptrs, partial.to(Out.dtype.element_ty), mask=row_mask, sem="relaxed"
        )


@triton_op("finegrained_fp8::moe_batched_atomic", mutates_args=())
def _moe_batched_atomic(
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
    """Batched atomic fused MoE expert layer: non-deterministic (atomic split-K).

    Input:  unsorted hidden_states + router outputs (top_k_index, top_k_weights)
    Output: (num_tokens, hidden) — accumulated across top_k experts

    Pipeline: expand → ONE kernel (gate_up + SiLU + down per token, atomic split-K) → routing + reduce
    Non-deterministic: atomic_add across intermediate N-tiles. Intermediate stays in registers.
    """
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = down_proj.shape[1]
    intermediate_dim = down_proj.shape[2]
    block_n, block_k = block_size
    num_experts = gate_up_proj.shape[0]

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
        max(triton.next_power_of_2((S + num_experts - 1) // num_experts), 16), 128
    )
    num_N_tiles = triton.cdiv(intermediate_dim, block_n)

    Out = torch.zeros(S, hidden_dim, device=device, dtype=torch.float32)
    grid = (S, num_N_tiles)

    with device_context(device):
        wrap_triton(moe_batched_atomic_kernel)[grid](
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
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            BLOCK_SIZE_H=block_n,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            NUM_H_TILES=triton.cdiv(hidden_dim, block_n),
            NUM_EXPERTS_BIT_LENGTH=num_experts.bit_length(),
            SIMULATE_UNFUSED=simulate_unfused,
        )

    # Apply routing weights + reduce
    Out = Out.to(hidden_states.dtype)
    weighted_out = Out * sample_weights.to(Out.dtype).unsqueeze(-1)
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(
        dim=1
    )

    return final_hidden_states.to(hidden_states.dtype)


def moe_batched_atomic(
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
    """Batched atomic fused MoE expert layer: non-deterministic (atomic split-K).

    Input:  unsorted hidden_states + router outputs (top_k_index, top_k_weights)
    Output: (num_tokens, hidden) — accumulated across top_k experts

    Pipeline: expand → ONE kernel (gate_up + SiLU + down per token, atomic split-K) → routing + reduce
    Non-deterministic: atomic_add across intermediate N-tiles. Intermediate stays in registers.
    """
    return torch.ops.finegrained_fp8.moe_batched_atomic(
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
