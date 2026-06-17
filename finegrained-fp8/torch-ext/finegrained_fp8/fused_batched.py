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

"""Fused batched MoE: two deterministic kernels (no atomics, no sort), per routed token.

  Kernel 1 (S x N-tiles): gather -> gate_up GEMM -> SiLU -> FP8 requant -> fp8 intermediate
  Kernel 2 (S x H-tiles): fp8 intermediate -> down GEMM -> routing weight -> output

Recipe-named to mirror ``batched.py``; ``moe_batched`` is the neutral dispatcher.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ._ops import add_op_namespace_prefix, ops
from .utils import device_context, fp8_act_quant_inline
from .batched import _store_row


# ── Batched fused: two-kernel approach (no sorting, no atomics) ──────────────
#
# Same two-kernel architecture as grouped fused but with per-token dispatch:
# Kernel 1: (S, N-tiles) — gate_up + SiLU + FP8 quant → intermediate buffer
# Kernel 2: (S, H-tiles) — fp8 intermediate → down proj → output
# No sorting needed — expert lookup is per-token via ExpertIds.


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [2, 4, 8, 16]
        for s in [2, 3, 4, 5, 6]
    ],
    # num_routed_tokens (= grid axis-0, runtime) is in the key so decode re-tunes per
    # batch size — a config tuned at S=8 is wrong at S=256 (GPU unsaturated at S=8). The
    # constexpr dims auto-partition the cache, so they don't need to be in the key.
    key=["num_routed_tokens"],
)
@triton.jit
def w8a8_block_dynamic_fp8_moe_batched_gate_up_kernel(
    HiddenStates,
    GateUp,
    GateUpScale,
    Intermediate,
    IntermediateScale,
    ExpertIds,
    stride_a_m,
    stride_a_k,
    stride_gu_e,
    stride_gu_n,
    stride_gu_k,
    stride_gus_e,
    stride_gus_n,
    stride_gus_k,
    num_routed_tokens,  # tuning knob: in the autotune key so decode re-tunes per batch
    NUM_TOP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    SIMULATE_UNFUSED: tl.constexpr,
):
    """Batched kernel 1: per-token gate_up + SiLU + FP8 quant. Grid: (S, N-tiles)."""
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Each token is replicated NUM_TOP_K times in the routed (S,) layout, so the
    # source row is batch_id // NUM_TOP_K — gathered from the unexpanded activations
    # (no token-index tensor, no top_k-replicated copy). int64 against offset overflow.
    token = (batch_id // NUM_TOP_K).to(tl.int64)
    expert_id = tl.load(ExpertIds + batch_id).to(tl.int64)
    # EP sentinel: row routed to a non-local expert — skip. The intermediate row is left
    # uninit; the down kernel writes zeros for this row, so the output stays well-defined.
    if expert_id >= NUM_EXPERTS:
        return
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = HiddenStates + token * stride_a_m + offs_k[None, :] * stride_a_k

    b_gate_ptr = tl.make_block_ptr(
        base=GateUp + expert_id * stride_gu_e,
        shape=(HIDDEN_DIM, INTERMEDIATE_DIM * 2),
        strides=(stride_gu_k, stride_gu_n),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(0, 1),
    )
    b_up_ptr = tl.make_block_ptr(
        base=GateUp + expert_id * stride_gu_e,
        shape=(HIDDEN_DIM, INTERMEDIATE_DIM * 2),
        strides=(stride_gu_k, stride_gu_n),
        offsets=(0, INTERMEDIATE_DIM + pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(0, 1),
    )

    n_scale_blocks = INTERMEDIATE_DIM // BLOCK_SIZE_N
    bs_base = GateUpScale + expert_id * stride_gus_e
    bs_gate_ptrs = bs_base + pid_n * stride_gus_n
    bs_up_ptrs = bs_base + (n_scale_blocks + pid_n) * stride_gus_n

    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(HIDDEN_DIM, BLOCK_SIZE_K)):
        a_raw = tl.load(a_ptrs).to(tl.float32)
        a, a_s = fp8_act_quant_inline(a_raw)

        b_gate = tl.load(b_gate_ptr)
        b_up = tl.load(b_up_ptr)
        bs_gate = tl.load(bs_gate_ptrs + k * stride_gus_k)
        bs_up = tl.load(bs_up_ptrs + k * stride_gus_k)

        acc_gate += tl.dot(a, b_gate) * a_s[:, None] * bs_gate[None, :]
        acc_up += tl.dot(a, b_up) * a_s[:, None] * bs_up[None, :]

        a_ptrs += BLOCK_SIZE_K * stride_a_k
        b_gate_ptr = tl.advance(b_gate_ptr, (BLOCK_SIZE_K, 0))
        b_up_ptr = tl.advance(b_up_ptr, (BLOCK_SIZE_K, 0))

    if SIMULATE_UNFUSED:
        # Round each step through the activation dtype to match the numerics of the
        # unfused path, where every op's output is materialized in that dtype.
        dtype = HiddenStates.dtype.element_ty
        acc_gate = acc_gate.to(dtype).to(tl.float32)
        acc_up = acc_up.to(dtype).to(tl.float32)
        silu = (acc_gate * tl.sigmoid(acc_gate)).to(dtype).to(tl.float32)
        intermediate = (silu * acc_up).to(dtype).to(tl.float32)
    else:
        intermediate = acc_gate * tl.sigmoid(acc_gate) * acc_up

    # Requant the intermediate to FP8 — the same inline per-row act quant as the inputs;
    # with BLOCK_SIZE_M=1 the per-row scale is the single per-tile scalar we store.
    inter, inter_s = fp8_act_quant_inline(intermediate)

    _store_row(
        Intermediate + batch_id * INTERMEDIATE_DIM,
        inter,
        pid_n,
        1,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    tl.store(IntermediateScale + batch_id * NUM_N_TILES + pid_n, tl.max(inter_s))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [2, 4, 8, 16]
        for s in [2, 3, 4, 5, 6]
    ],
    key=["num_routed_tokens"],
)
@triton.jit
def w8a8_block_dynamic_fp8_moe_batched_down_kernel(
    Intermediate,
    IntermediateScale,
    Down,
    DownScale,
    ExpertIds,
    SampleWeights,
    Out,
    stride_down_e,
    stride_down_n,
    stride_down_k,
    stride_downs_e,
    stride_downs_n,
    stride_downs_k,
    num_routed_tokens,  # tuning knob: in the autotune key so decode re-tunes per batch
    NUM_EXPERTS: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
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
    # EP sentinel: row routed to a non-local expert. The program is already launched, so
    # write its zero tile here (skipping the weight load) — cheaper than a host-side mask
    # pass, and it leaves the output fully defined for a plain top-k sum.
    if expert_id >= NUM_EXPERTS:
        z = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_H), dtype=tl.float32)
        _store_row(Out + batch_id * HIDDEN_DIM, z, pid_h, 1, BLOCK_SIZE_M, BLOCK_SIZE_H)
        return

    w_down_ptr = tl.make_block_ptr(
        base=Down + expert_id * stride_down_e,
        shape=(INTERMEDIATE_DIM, HIDDEN_DIM),
        strides=(stride_down_k, stride_down_n),
        offsets=(0, pid_h * BLOCK_SIZE_H),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_H),
        order=(0, 1),
    )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_H), dtype=tl.float32)

    for n_tile in range(0, NUM_N_TILES):
        n_offs = n_tile * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        inter_ptrs = Intermediate + batch_id * INTERMEDIATE_DIM + n_offs[None, :]
        inter_s_ptrs = IntermediateScale + batch_id * NUM_N_TILES + n_tile
        ws_down_ptr = (
            DownScale
            + expert_id * stride_downs_e
            + pid_h * stride_downs_n
            + n_tile * stride_downs_k
        )
        inter = tl.load(inter_ptrs)
        w_down = tl.load(w_down_ptr)
        ws_down = tl.load(ws_down_ptr)
        inter_s = tl.load(inter_s_ptrs)
        acc += tl.dot(inter, w_down) * inter_s * ws_down
        w_down_ptr = tl.advance(w_down_ptr, (BLOCK_SIZE_N, 0))

    if SIMULATE_UNFUSED:
        acc = acc.to(Out.dtype.element_ty).to(tl.float32)

    acc = acc * tl.load(SampleWeights + batch_id)
    _store_row(Out + batch_id * HIDDEN_DIM, acc, pid_h, 1, BLOCK_SIZE_M, BLOCK_SIZE_H)


@triton_op(add_op_namespace_prefix("moe_batched_gate_up"), mutates_args=())
def _moe_batched_gate_up(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_proj_scale_inv: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int],
    simulate_unfused: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched fused MoE kernel 1: gate_up + SiLU + FP8 quant. Gathers each routed
    row directly from the unexpanded ``hidden_states`` (source row ``s // NUM_TOP_K``,
    so no replicated copy is materialized), and returns the ``(fp8 intermediate,
    per-tile scales)`` pair consumed by the down op."""

    # One routed token per program; the M lanes are aliased duplicates (only lane 0
    # is stored), so M is the fake-batch row count — 1, as in the non-fused batched
    # kernels (Triton pads the dot's M to the MMA minimum internally).
    BLOCK_SIZE_M = 1
    device = hidden_states.device
    HIDDEN_DIM = hidden_states.size(1)
    NUM_EXPERTS = gate_up_proj.size(0)
    num_routed_tokens = expert_ids.size(0)
    BLOCK_SIZE_N, BLOCK_SIZE_K = block_size
    INTERMEDIATE_DIM = gate_up_proj.size(1) // 2
    NUM_TOP_K = num_routed_tokens // hidden_states.size(0)
    NUM_N_TILES = triton.cdiv(INTERMEDIATE_DIM, BLOCK_SIZE_N)
    inter = torch.empty(
        num_routed_tokens, INTERMEDIATE_DIM, device=device, dtype=torch.float8_e4m3fn
    )
    inter_scales = torch.empty(
        num_routed_tokens, NUM_N_TILES, device=device, dtype=torch.float32
    )
    grid = (num_routed_tokens, NUM_N_TILES)
    with device_context(device):
        wrap_triton(w8a8_block_dynamic_fp8_moe_batched_gate_up_kernel)[grid](
            hidden_states,
            gate_up_proj,
            gate_up_proj_scale_inv,
            inter,
            inter_scales,
            expert_ids,
            hidden_states.stride(0),
            hidden_states.stride(1),
            gate_up_proj.stride(0),
            gate_up_proj.stride(1),
            gate_up_proj.stride(2),
            gate_up_proj_scale_inv.stride(0),
            gate_up_proj_scale_inv.stride(1),
            gate_up_proj_scale_inv.stride(2),
            num_routed_tokens,
            NUM_TOP_K=NUM_TOP_K,
            NUM_EXPERTS=NUM_EXPERTS,
            HIDDEN_DIM=HIDDEN_DIM,
            INTERMEDIATE_DIM=INTERMEDIATE_DIM,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            NUM_N_TILES=NUM_N_TILES,
            SIMULATE_UNFUSED=simulate_unfused,
        )
    return inter, inter_scales


@triton_op(add_op_namespace_prefix("moe_batched_down"), mutates_args=())
def _moe_batched_down(
    inter: torch.Tensor,
    inter_scales: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    expert_ids: torch.Tensor,
    sample_weights: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """Batched fused MoE kernel 2: fp8 intermediate → down proj → routing-weighted
    per-(token, expert) output (the caller reduces over the top-k axis)."""

    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = block_size[0]
    HIDDEN_DIM = down_proj.size(1)
    NUM_EXPERTS = down_proj.size(0)
    INTERMEDIATE_DIM = inter.size(1)
    num_routed_tokens = inter.size(0)
    NUM_N_TILES = inter_scales.size(1)
    NUM_H_TILES = triton.cdiv(HIDDEN_DIM, BLOCK_SIZE_N)
    Out = inter.new_empty(num_routed_tokens, HIDDEN_DIM, dtype=output_dtype)
    grid = (num_routed_tokens, NUM_H_TILES)

    with device_context(inter.device):
        wrap_triton(w8a8_block_dynamic_fp8_moe_batched_down_kernel)[grid](
            inter,
            inter_scales,
            down_proj,
            down_proj_scale_inv,
            expert_ids,
            sample_weights,
            Out,
            down_proj.stride(0),
            down_proj.stride(1),
            down_proj.stride(2),
            down_proj_scale_inv.stride(0),
            down_proj_scale_inv.stride(1),
            down_proj_scale_inv.stride(2),
            num_routed_tokens,
            NUM_EXPERTS=NUM_EXPERTS,
            HIDDEN_DIM=HIDDEN_DIM,
            INTERMEDIATE_DIM=INTERMEDIATE_DIM,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_H=BLOCK_SIZE_N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            NUM_N_TILES=NUM_N_TILES,
            SIMULATE_UNFUSED=simulate_unfused,
        )
    return Out


def w8a8_block_dynamic_fp8_moe_batched(
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

    The gate_up and down kernels are separate ``triton_op``s. The gate_up op gathers
    each routed row directly from the unexpanded ``hidden_states`` (source row
    ``s // num_top_k``), so no top_k-replicated copy is materialized; the top-k reduce
    stays plain torch so ``torch.compile`` can fuse it with the surrounding model graph.
    """

    hidden_dim = down_proj.size(1)
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)

    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)

    inter, inter_scales = ops.moe_batched_gate_up(
        hidden_states,
        gate_up_proj,
        gate_up_proj_scale_inv,
        expert_ids,
        block_size,
        simulate_unfused,
    )
    out = ops.moe_batched_down(
        inter,
        inter_scales,
        down_proj,
        down_proj_scale_inv,
        expert_ids,
        sample_weights,
        block_size,
        hidden_states.dtype,
        simulate_unfused,
    )
    # The down kernel already zeros EP-sentinel rows, so the top-k reduce is a plain sum.
    return (
        out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)
    )


# ── Dispatcher ────────────────────────────────────────────────────────────────


def moe_batched(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """Fused batched-MoE dispatcher — routes to the recipe matching the weight dtype /
    scale layout, mirroring ``matmul_batched``. Implemented: block-dynamic FP8
    (tensor-dynamic, MXFP8, MXFP4 to follow)."""

    return w8a8_block_dynamic_fp8_moe_batched(
        hidden_states,
        top_k_index,
        top_k_weights,
        gate_up_proj,
        down_proj,
        gate_up_proj_scale_inv,
        down_proj_scale_inv,
        block_size,
    )
