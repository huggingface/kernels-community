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

Recipe-named to mirror ``batched.py``; ``moe_fused_batched`` is the neutral dispatcher.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ._ops import add_op_namespace_prefix, ops
from .utils import (
    FP8_DTYPE,
    MX_SCALE_GROUP_K,
    NIBBLES_PER_BYTE,
    DECODE_BLOCK_SIZE_M,
    device_context,
    fp8_act_quant_inline,
    get_mxfp_autotuning_configs,
    is_mxfp,
    is_mxfp4,
    mxfp_act_quant_inline,
    mx_compute,
    mx_compute_gate_up,
    glu,
    topk_reduce_kernel,
    TOPK_REDUCE_BLOCK_H,
    e2m1_as_uint8,
    ue8m0_as_uint8,
)
from .batched import store_row
from .bayesian_autotuner import bayesian_autotune


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
    NUM_EXPERTS: tl.constexpr,
    NUM_TOP_K: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
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
    a_ptr = (
        HiddenStates
        + (token + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_a_m
        + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_a_k
    )
    b_gate_ptr = (
        GateUp
        + expert_id * stride_gu_e
        + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_gu_k
        + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_gu_n
    )
    b_up_ptr = (
        GateUp
        + expert_id * stride_gu_e
        + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_gu_k
        + (INTERMEDIATE_DIM + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[
            None, :
        ]
        * stride_gu_n
    )
    bs_gate_ptr = GateUpScale + expert_id * stride_gus_e + pid_n * stride_gus_n
    bs_up_ptr = (
        GateUpScale + expert_id * stride_gus_e + (NUM_N_TILES + pid_n) * stride_gus_n
    )

    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(HIDDEN_DIM, BLOCK_SIZE_K)):
        a_raw = tl.load(a_ptr).to(tl.float32)
        a, a_s = fp8_act_quant_inline(a_raw)

        w_gate = tl.load(b_gate_ptr)
        w_up = tl.load(b_up_ptr)
        w_s_gate = tl.load(bs_gate_ptr)
        w_s_up = tl.load(bs_up_ptr)

        acc_gate += tl.dot(a, w_gate) * a_s[:, None] * w_s_gate
        acc_up += tl.dot(a, w_up) * a_s[:, None] * w_s_up

        a_ptr += BLOCK_SIZE_K * stride_a_k
        b_gate_ptr += BLOCK_SIZE_K * stride_gu_k
        b_up_ptr += BLOCK_SIZE_K * stride_gu_k
        bs_gate_ptr += stride_gus_k
        bs_up_ptr += stride_gus_k

    intermediate = glu(
        acc_gate,
        acc_up,
        ACT_FN,
        SWIGLU_ALPHA,
        SWIGLU_LIMIT,
        HiddenStates.dtype.element_ty,
        SIMULATE_UNFUSED,
    )

    # Requant the intermediate to FP8 — the same inline per-row act quant as the inputs;
    # with BLOCK_SIZE_M=1 the per-row scale is the single per-tile scalar we store.
    inter, inter_s = fp8_act_quant_inline(intermediate)
    store_row(
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
        store_row(Out + batch_id * HIDDEN_DIM, z, pid_h, 1, BLOCK_SIZE_M, BLOCK_SIZE_H)
        return

    w_down_ptr = (
        Down
        + expert_id * stride_down_e
        + tl.arange(0, BLOCK_SIZE_N)[:, None] * stride_down_k
        + (pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H))[None, :] * stride_down_n
    )
    inter_ptr = (
        Intermediate
        + (batch_id + tl.arange(0, BLOCK_SIZE_M))[:, None] * INTERMEDIATE_DIM
        + tl.arange(0, BLOCK_SIZE_N)[None, :]
    )
    inter_s_ptr = (
        IntermediateScale
        + (batch_id + tl.arange(0, BLOCK_SIZE_M))[:, None] * NUM_N_TILES
    )
    ws_down_ptr = DownScale + expert_id * stride_downs_e + pid_h * stride_downs_n

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_H), dtype=tl.float32)

    for _ in range(0, NUM_N_TILES):
        inter = tl.load(inter_ptr)
        inter_s = tl.load(inter_s_ptr)
        w_s_down = tl.load(ws_down_ptr)
        w_down = tl.load(w_down_ptr)
        acc += tl.dot(inter, w_down) * inter_s * w_s_down
        inter_ptr += BLOCK_SIZE_N
        inter_s_ptr += 1
        ws_down_ptr += stride_downs_k
        w_down_ptr += BLOCK_SIZE_N * stride_down_k

    if SIMULATE_UNFUSED:
        acc = acc.to(Out.dtype.element_ty).to(tl.float32)

    acc = acc * tl.load(SampleWeights + batch_id)
    store_row(Out + batch_id * HIDDEN_DIM, acc, pid_h, 1, BLOCK_SIZE_M, BLOCK_SIZE_H)


@triton_op(
    add_op_namespace_prefix("w8a8_block_dynamic_fp8_moe_batched"), mutates_args=()
)
def _w8a8_block_dynamic_fp8_moe_batched(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_proj_scale: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_scale: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    block_size: list[int],
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """Block-dynamic FP8 batched fused MoE in ONE op: gate_up + SiLU + FP8 requant →
    grouped down → routing-weighted per-(token, expert) output. gate_up gathers each routed
    row directly from the unexpanded ``hidden_states`` (source row ``s // NUM_TOP_K``, no
    replicated copy). ``inter``/``inter_scales`` are internal; the caller reduces over top-k."""
    device = hidden_states.device
    HIDDEN_DIM = hidden_states.size(1)
    NUM_EXPERTS = gate_up_proj.size(0)
    num_tokens = hidden_states.size(0)
    num_routed_tokens = top_k_index.numel()
    INTERMEDIATE_DIM = gate_up_proj.size(1) // 2
    NUM_TOP_K = num_routed_tokens // hidden_states.size(0)
    BLOCK_SIZE_N, BLOCK_SIZE_K = block_size
    NUM_N_TILES = triton.cdiv(INTERMEDIATE_DIM, BLOCK_SIZE_N)
    NUM_H_TILES = triton.cdiv(HIDDEN_DIM, BLOCK_SIZE_N)

    inter = torch.empty(
        num_routed_tokens, INTERMEDIATE_DIM, device=device, dtype=FP8_DTYPE
    )
    inter_scales = torch.empty(
        num_routed_tokens, NUM_N_TILES, device=device, dtype=torch.float32
    )

    out = torch.empty(num_routed_tokens, HIDDEN_DIM, device=device, dtype=hidden_states.dtype)
    reduced = torch.empty(num_tokens, HIDDEN_DIM, device=device, dtype=hidden_states.dtype)
    with device_context(device):
        wrap_triton(w8a8_block_dynamic_fp8_moe_batched_gate_up_kernel)[
            (num_routed_tokens, NUM_N_TILES)
        ](
            hidden_states,
            gate_up_proj,
            gate_up_proj_scale,
            inter,
            inter_scales,
            top_k_index,
            hidden_states.stride(0),
            hidden_states.stride(1),
            gate_up_proj.stride(0),
            gate_up_proj.stride(1),
            gate_up_proj.stride(2),
            gate_up_proj_scale.stride(0),
            gate_up_proj_scale.stride(1),
            gate_up_proj_scale.stride(2),
            num_routed_tokens=num_routed_tokens,
            NUM_TOP_K=NUM_TOP_K,
            NUM_EXPERTS=NUM_EXPERTS,
            HIDDEN_DIM=HIDDEN_DIM,
            INTERMEDIATE_DIM=INTERMEDIATE_DIM,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_M=DECODE_BLOCK_SIZE_M,
            NUM_N_TILES=NUM_N_TILES,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            SIMULATE_UNFUSED=simulate_unfused,
        )
        wrap_triton(w8a8_block_dynamic_fp8_moe_batched_down_kernel)[
            (num_routed_tokens, NUM_H_TILES)
        ](
            inter,
            inter_scales,
            down_proj,
            down_proj_scale,
            top_k_index,
            top_k_weights,
            out,
            down_proj.stride(0),
            down_proj.stride(1),
            down_proj.stride(2),
            down_proj_scale.stride(0),
            down_proj_scale.stride(1),
            down_proj_scale.stride(2),
            num_routed_tokens=num_routed_tokens,
            NUM_EXPERTS=NUM_EXPERTS,
            HIDDEN_DIM=HIDDEN_DIM,
            INTERMEDIATE_DIM=INTERMEDIATE_DIM,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_H=BLOCK_SIZE_N,
            BLOCK_SIZE_M=DECODE_BLOCK_SIZE_M,
            NUM_N_TILES=NUM_N_TILES,
            SIMULATE_UNFUSED=simulate_unfused,
        )
        wrap_triton(topk_reduce_kernel)[
            (num_tokens, triton.cdiv(HIDDEN_DIM, TOPK_REDUCE_BLOCK_H))
        ](
            out,
            reduced,
            HIDDEN_DIM,
            out.stride(0),
            out.stride(1),
            reduced.stride(0),
            reduced.stride(1),
            NUM_TOP_K=NUM_TOP_K,
            BLOCK_H=TOPK_REDUCE_BLOCK_H,
        )

    return reduced


def w8a8_block_dynamic_fp8_moe_batched(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale: torch.Tensor,
    down_proj_scale: torch.Tensor,
    block_size: list[int],
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """Batched fused MoE (deterministic, no sorting, no atomics): a single ``triton_op``
    runs gate_up + down (gathering each routed row from the unexpanded ``hidden_states``,
    source row ``s // num_top_k``, no replicated copy). The top-k reduce stays plain torch
    so ``torch.compile`` can fuse it with the surrounding model graph."""

    out = ops.w8a8_block_dynamic_fp8_moe_batched(
        hidden_states,
        gate_up_proj,
        gate_up_proj_scale,
        down_proj,
        down_proj_scale,
        top_k_index,
        top_k_weights,
        block_size,
        act_fn,
        swiglu_alpha,
        swiglu_limit,
        simulate_unfused,
    )
    return out


# ── MXFP8 fused (tl.dot_scaled, UE8M0 group-32 scales, tunable tiles) ─────────
#
# Mirrors the block-dynamic kernels but: activations/intermediate quantize via the MX
# group-32 inline quant, weights/scales feed tl.dot_scaled, and the tile (BLOCK_SIZE_N/K)
# is free to autotune — the group-32 scale runs along K independently of the compute
# tile (the whole reason MXFP8 can tune the tile that the block recipe must lock to 128).


@bayesian_autotune(
    get_mxfp_autotuning_configs(),
    ["num_routed_tokens"],
    n_trials=60,
)
@triton.jit
def mxfp_dynamic_moe_batched_gate_up_kernel(
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
    NUM_EXPERTS: tl.constexpr,
    NUM_TOP_K: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    INTERMEDIATE_DIM: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
):
    """MXFP4/MXFP8 kernel 1: gate_up + SiLU + MXFP8 requant. N = intermediate
    (output) tile, K = hidden (contraction) tile — both tunable. Grid: (S, N-tiles)."""
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    token = (batch_id // NUM_TOP_K).to(tl.int64)
    expert_id = tl.load(ExpertIds + batch_id).to(tl.int64)
    if expert_id >= NUM_EXPERTS:  # EP sentinel; down kernel zeros the output row
        return

    a_ptr = (
        HiddenStates
        + (token + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_a_m
        + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_a_k
    )
    b_gate_ptr = (
        GateUp
        + expert_id * stride_gu_e
        + tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)[:, None] * stride_gu_k
        + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_gu_n
    )
    b_up_ptr = (
        GateUp
        + expert_id * stride_gu_e
        + tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)[:, None] * stride_gu_k
        + (INTERMEDIATE_DIM + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[
            None, :
        ]
        * stride_gu_n
    )
    bs_gate_ptr = (
        GateUpScale
        + expert_id * stride_gus_e
        + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[:, None] * stride_gus_n
        + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)[None, :] * stride_gus_k
    )
    bs_up_ptr = (
        GateUpScale
        + expert_id * stride_gus_e
        + (INTERMEDIATE_DIM + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[
            :, None
        ]
        * stride_gus_n
        + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)[None, :] * stride_gus_k
    )

    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(HIDDEN_DIM, BLOCK_SIZE_K)):
        a_raw = tl.load(a_ptr).to(tl.float32)
        a, a_scale = mxfp_act_quant_inline(
            a_raw, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K
        )
        b_gate = tl.load(b_gate_ptr)
        b_up = tl.load(b_up_ptr)
        bs_gate = tl.load(bs_gate_ptr)
        bs_up = tl.load(bs_up_ptr)
        acc_gate, acc_up = mx_compute_gate_up(
            acc_gate,
            acc_up,
            a,
            a_scale,
            b_gate,
            b_up,
            bs_gate,
            bs_up,
            COMPUTE_MODE,
            VALUES_PER_BYTE,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            SCALE_GROUP_K,
        )
        a_ptr += BLOCK_SIZE_K * stride_a_k
        b_gate_ptr += (BLOCK_SIZE_K // VALUES_PER_BYTE) * stride_gu_k
        b_up_ptr += (BLOCK_SIZE_K // VALUES_PER_BYTE) * stride_gu_k
        bs_gate_ptr += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_gus_k
        bs_up_ptr += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_gus_k

    intermediate = glu(
        acc_gate,
        acc_up,
        ACT_FN,
        SWIGLU_ALPHA,
        SWIGLU_LIMIT,
        HiddenStates.dtype.element_ty,
        SIMULATE_UNFUSED,
    )

    # MXFP8 requant of the intermediate (E4M3 + UE8M0 group-32 along this N-tile).
    inter, inter_scale = mxfp_act_quant_inline(
        intermediate, BLOCK_SIZE_M, BLOCK_SIZE_N, SCALE_GROUP_K
    )
    store_row(
        Intermediate + batch_id * INTERMEDIATE_DIM,
        inter,
        pid_n,
        1,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    offs_sc = pid_n * (BLOCK_SIZE_N // SCALE_GROUP_K) + tl.arange(
        0, BLOCK_SIZE_N // SCALE_GROUP_K
    )
    sc_ptrs = (
        IntermediateScale
        + batch_id * (INTERMEDIATE_DIM // SCALE_GROUP_K)
        + offs_sc[None, :]
    )
    tl.store(sc_ptrs, inter_scale)


@bayesian_autotune(
    get_mxfp_autotuning_configs(),
    ["num_routed_tokens"],
    n_trials=60,
)
@triton.jit
def mxfp_dynamic_moe_batched_down_kernel(
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
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    SIMULATE_UNFUSED: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
):
    """MXFP4/MXFP8 kernel 2: MXFP8 intermediate → down proj. N = hidden
    (output) tile, K = intermediate (contraction) tile. Grid: (S, H-tiles)."""
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    expert_id = tl.load(ExpertIds + batch_id).to(tl.int64)
    if expert_id >= NUM_EXPERTS:  # EP sentinel: zero the output row for the top-k sum
        z = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        store_row(Out + batch_id * HIDDEN_DIM, z, pid_n, 1, BLOCK_SIZE_M, BLOCK_SIZE_N)
        return

    a_ptr = (
        Intermediate
        + (batch_id + tl.arange(0, BLOCK_SIZE_M))[:, None] * INTERMEDIATE_DIM
        + tl.arange(0, BLOCK_SIZE_K)[None, :]
    )
    as_ptr = (
        IntermediateScale
        + (batch_id + tl.arange(0, BLOCK_SIZE_M))[:, None]
        * (INTERMEDIATE_DIM // SCALE_GROUP_K)
        + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)[None, :]
    )
    w_down_ptr = (
        Down
        + expert_id * stride_down_e
        + tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)[:, None] * stride_down_k
        + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_down_n
    )
    ws_down_ptr = (
        DownScale
        + expert_id * stride_downs_e
        + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[:, None] * stride_downs_n
        + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)[None, :] * stride_downs_k
    )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(INTERMEDIATE_DIM, BLOCK_SIZE_K)):
        a = tl.load(a_ptr)
        a_scale = tl.load(as_ptr)
        w = tl.load(w_down_ptr)
        w_scale = tl.load(ws_down_ptr)
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
        a_ptr += BLOCK_SIZE_K
        as_ptr += BLOCK_SIZE_K // SCALE_GROUP_K
        w_down_ptr += (BLOCK_SIZE_K // VALUES_PER_BYTE) * stride_down_k
        ws_down_ptr += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_downs_k

    if SIMULATE_UNFUSED:
        acc = acc.to(Out.dtype.element_ty).to(tl.float32)
    acc = acc * tl.load(SampleWeights + batch_id)
    store_row(Out + batch_id * HIDDEN_DIM, acc, pid_n, 1, BLOCK_SIZE_M, BLOCK_SIZE_N)


@triton_op(add_op_namespace_prefix("mxfp_dynamic_moe_batched"), mutates_args=())
def _mxfp_dynamic_moe_batched(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_proj_scale: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_scale: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """MXFP4/MXFP8 batched fused MoE in ONE op: gate_up + SiLU + MXFP8 requant → grouped
    down → routing-weighted per-(token, expert) output. gate_up and down must share the same MX
    format. ``inter``/``inter_scales`` are internal; caller reduces over top-k."""
    gate_up_is_fp4 = is_mxfp4(gate_up_proj, gate_up_proj_scale)
    down_is_fp4 = is_mxfp4(down_proj, down_proj_scale)
    if gate_up_is_fp4 != down_is_fp4:
        raise ValueError(
            "gate_up_proj and down_proj must use the same MX format (both MXFP4 or both MXFP8)."
        )

    device = hidden_states.device
    HIDDEN_DIM = hidden_states.size(1)
    NUM_EXPERTS = gate_up_proj.size(0)
    num_tokens = hidden_states.size(0)
    num_routed_tokens = top_k_index.numel()
    INTERMEDIATE_DIM = gate_up_proj.size(1) // 2
    NUM_TOP_K = num_routed_tokens // hidden_states.size(0)
    VALUES_PER_BYTE = NIBBLES_PER_BYTE if gate_up_is_fp4 else 1
    gate_up_proj_u8 = e2m1_as_uint8(gate_up_proj)
    gate_up_proj_scale_u8 = ue8m0_as_uint8(gate_up_proj_scale)
    down_proj_u8 = e2m1_as_uint8(down_proj)
    down_proj_scale_u8 = ue8m0_as_uint8(down_proj_scale)

    inter = torch.empty(
        num_routed_tokens, INTERMEDIATE_DIM, device=device, dtype=FP8_DTYPE
    )
    inter_scales = torch.empty(
        num_routed_tokens,
        INTERMEDIATE_DIM // MX_SCALE_GROUP_K,
        device=device,
        dtype=torch.uint8,
    )
    out = torch.empty(num_routed_tokens, HIDDEN_DIM, device=device, dtype=hidden_states.dtype)
    reduced = torch.empty(num_tokens, HIDDEN_DIM, device=device, dtype=hidden_states.dtype)

    def gate_up_grid(META):
        return (num_routed_tokens, triton.cdiv(INTERMEDIATE_DIM, META["BLOCK_SIZE_N"]))

    def down_grid(META):
        return (num_routed_tokens, triton.cdiv(HIDDEN_DIM, META["BLOCK_SIZE_N"]))

    with device_context(device):
        wrap_triton(mxfp_dynamic_moe_batched_gate_up_kernel)[gate_up_grid](
            hidden_states,
            gate_up_proj_u8,
            gate_up_proj_scale_u8,
            inter,
            inter_scales,
            top_k_index,
            hidden_states.stride(0),
            hidden_states.stride(1),
            gate_up_proj_u8.stride(0),
            gate_up_proj_u8.stride(1),
            gate_up_proj_u8.stride(2),
            gate_up_proj_scale_u8.stride(0),
            gate_up_proj_scale_u8.stride(1),
            gate_up_proj_scale_u8.stride(2),
            num_routed_tokens=num_routed_tokens,
            NUM_TOP_K=NUM_TOP_K,
            NUM_EXPERTS=NUM_EXPERTS,
            HIDDEN_DIM=HIDDEN_DIM,
            INTERMEDIATE_DIM=INTERMEDIATE_DIM,
            BLOCK_SIZE_M=DECODE_BLOCK_SIZE_M,
            VALUES_PER_BYTE=VALUES_PER_BYTE,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            SIMULATE_UNFUSED=simulate_unfused,
        )
        wrap_triton(mxfp_dynamic_moe_batched_down_kernel)[down_grid](
            inter,
            inter_scales,
            down_proj_u8,
            down_proj_scale_u8,
            top_k_index,
            top_k_weights,
            out,
            down_proj_u8.stride(0),
            down_proj_u8.stride(1),
            down_proj_u8.stride(2),
            down_proj_scale_u8.stride(0),
            down_proj_scale_u8.stride(1),
            down_proj_scale_u8.stride(2),
            num_routed_tokens=num_routed_tokens,
            NUM_EXPERTS=NUM_EXPERTS,
            HIDDEN_DIM=HIDDEN_DIM,
            INTERMEDIATE_DIM=INTERMEDIATE_DIM,
            BLOCK_SIZE_M=DECODE_BLOCK_SIZE_M,
            VALUES_PER_BYTE=VALUES_PER_BYTE,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
            SIMULATE_UNFUSED=simulate_unfused,
        )
        wrap_triton(topk_reduce_kernel)[
            (num_tokens, triton.cdiv(HIDDEN_DIM, TOPK_REDUCE_BLOCK_H))
        ](
            out,
            reduced,
            HIDDEN_DIM,
            out.stride(0),
            out.stride(1),
            reduced.stride(0),
            reduced.stride(1),
            NUM_TOP_K=NUM_TOP_K,
            BLOCK_H=TOPK_REDUCE_BLOCK_H,
        )

    return reduced


def mxfp_dynamic_moe_batched(
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
    """Two-kernel batched fused MX MoE — MXFP4 or MXFP8 weights (UE8M0 group-32), the
    format picked per-weight by the ops. Same structure as the block-dynamic path but
    with a tunable tile and an MXFP8 group-32 intermediate; ``block_size`` is unused."""
    out = ops.mxfp_dynamic_moe_batched(
        hidden_states,
        gate_up_proj,
        gate_up_proj_scale,
        down_proj,
        down_proj_scale,
        top_k_index,
        top_k_weights,
        act_fn,
        swiglu_alpha,
        swiglu_limit,
        simulate_unfused,
    )
    return out


# ── Dispatcher ────────────────────────────────────────────────────────────────


def moe_fused_batched(
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
    """Fused batched-MoE dispatcher — routes to the recipe matching the weight dtype /
    scale layout, mirroring ``matmul_batched``. Implemented: block-dynamic FP8 and MXFP8
    (tensor-dynamic) and MXFP4. ``simulate_unfused`` (testing) rounds each step through the
    activation dtype so the output matches the unfused reference to reduce order."""

    gate_up_is_mx = is_mxfp(gate_up_proj, gate_up_proj_scale_inv)
    down_is_mx = is_mxfp(down_proj, down_proj_scale_inv)
    if gate_up_is_mx != down_is_mx:
        raise ValueError(
            "gate_up_proj and down_proj must use the same recipe (both MX or both block-dynamic FP8)."
        )

    if gate_up_is_mx:
        return mxfp_dynamic_moe_batched(
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

    return w8a8_block_dynamic_fp8_moe_batched(
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
