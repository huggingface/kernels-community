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
    decode_bm_swap_pairs,
    batched_mx_pruner,
    device_context,
    fp8_act_quant_inline,
    get_mxfp_autotuning_configs,
    is_mxfp,
    is_mxfp4,
    mxfp_act_quant_inline,
    mx_compute,
    stacked_gate_up_ptrs,
    stacked_gate_up_flatten,
    split_gate_up,
    oriented_weight_ptrs,
    acc_init,
    fp8_dot,
    acc_finalize,
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

# Decode (BLOCK_SIZE_M, SWAP_AB) pairs for the fp8 fused kernels: at M=1 the token is replicated to
# fill the MMA 16-atom (non-swap BM=16, ~40% over the degenerate BM=1 on plain tl.dot) or the weight
# output rows go in M (swap, BM=1). Swap needs BM=1, so (16, swap) is excluded. Same coupling as the
# MXFP / matmul kernels' for_decode generators.
_DECODE_BM_SWAP = decode_bm_swap_pairs()


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": bm, "SWAP_AB": sw}, num_warps=w, num_stages=s)
        for w in [2, 4, 8, 16]
        for s in [2, 3, 4, 5, 6]
        for bm, sw in _DECODE_BM_SWAP
    ],
    # Autotune key: num_routed_tokens (grid axis-0 — the batch varies at decode; a config tuned at
    # S=8 is wrong at S=256, GPU unsaturated) + the problem dims. A constexpr change forces a JIT
    # recompile but NOT a retune (the autotune config cache is keyed separately, on these names +
    # tensor dtypes), so INTERMEDIATE_DIM/HIDDEN_DIM must be listed to partition tuning per shape.
    # (BLOCK_SIZE_M, SWAP_AB) is the coupled fill-the-atom axis (replicate token in M, or weight
    # rows in M) — a config axis, not a key.
    key=["num_routed_tokens", "INTERMEDIATE_DIM", "HIDDEN_DIM"],
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
    SWAP_AB: tl.constexpr = False,
):
    """Batched kernel 1: per-token gate_up + SiLU + FP8 quant. Grid: (S, N-tiles).

    ``SWAP_AB`` (tuner axis, M=1 decode): load the weights output-rows-major ``[BN, BK]`` and put
    those rows in the MMA M dim, padding the single token to the N=16 atom; column 0 of the
    ``[BN, 16]`` accumulator is the result. No-swap keeps the token in M (padded to 16)."""
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
    # Replicate the single token across all BLOCK_SIZE_M rows (arange*0): at BM=16 (non-swap) this
    # fills the MMA 16-atom with copies of the same token — store_row keeps lane 0 — avoiding the
    # degenerate M=1 lowering. BM=1 is unchanged; the swap path (coupled to BM=1) uses only row 0.
    a_ptr = (
        HiddenStates
        + (token + tl.arange(0, BLOCK_SIZE_M) * 0)[:, None] * stride_a_m
        + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_a_k
    )
    gate_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    up_n = INTERMEDIATE_DIM + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    kk = tl.arange(0, BLOCK_SIZE_K)
    gu_base = GateUp + expert_id * stride_gu_e
    b_gate_ptr = oriented_weight_ptrs(gu_base, gate_n, kk, stride_gu_n, stride_gu_k, SWAP_AB)
    b_up_ptr = oriented_weight_ptrs(gu_base, up_n, kk, stride_gu_n, stride_gu_k, SWAP_AB)
    bs_gate_ptr = GateUpScale + expert_id * stride_gus_e + pid_n * stride_gus_n
    bs_up_ptr = (
        GateUpScale + expert_id * stride_gus_e + (NUM_N_TILES + pid_n) * stride_gus_n
    )

    acc_gate = acc_init("dot", BLOCK_SIZE_M, BLOCK_SIZE_N, SWAP_AB)
    acc_up = acc_init("dot", BLOCK_SIZE_M, BLOCK_SIZE_N, SWAP_AB)

    for _ in range(0, tl.cdiv(HIDDEN_DIM, BLOCK_SIZE_K)):
        a_raw = tl.load(a_ptr).to(tl.float32)
        a, a_s = fp8_act_quant_inline(a_raw)

        w_gate = tl.load(b_gate_ptr)
        w_up = tl.load(b_up_ptr)
        w_s_gate = tl.load(bs_gate_ptr)
        w_s_up = tl.load(bs_up_ptr)

        # a_s is [BM], w_s_* per-block scalars; a_s[:, None] broadcasts onto the acc either way (swap
        # BM=1 → the single token's scale), so no swap branch — the dot orientation is in fp8_dot.
        acc_gate += fp8_dot(a, w_gate, SWAP_AB, BLOCK_SIZE_K) * a_s[:, None] * w_s_gate
        acc_up += fp8_dot(a, w_up, SWAP_AB, BLOCK_SIZE_K) * a_s[:, None] * w_s_up

        a_ptr += BLOCK_SIZE_K * stride_a_k
        b_gate_ptr += BLOCK_SIZE_K * stride_gu_k
        b_up_ptr += BLOCK_SIZE_K * stride_gu_k
        bs_gate_ptr += stride_gus_k
        bs_up_ptr += stride_gus_k

    acc_gate = acc_finalize(acc_gate, "dot", BLOCK_SIZE_N, SWAP_AB)
    acc_up = acc_finalize(acc_up, "dot", BLOCK_SIZE_N, SWAP_AB)

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
        triton.Config({"BLOCK_SIZE_M": bm, "SWAP_AB": sw}, num_warps=w, num_stages=s)
        for w in [2, 4, 8, 16]
        for s in [2, 3, 4, 5, 6]
        for bm, sw in _DECODE_BM_SWAP
    ],
    # See the gate_up kernel: problem dims must be keyed (constexprs recompile but don't retune).
    key=["num_routed_tokens", "INTERMEDIATE_DIM", "HIDDEN_DIM"],
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
    SWAP_AB: tl.constexpr = False,
):
    """Batched kernel 2: fp8 intermediate → down proj → output. Grid: (S, H-tiles).

    ``SWAP_AB`` (tuner axis, M=1 decode): load the down weight output-rows-major ``[BH, BN]`` and
    put the hidden output rows in the MMA M dim (intermediate token padded to the N=16 atom);
    column 0 of the ``[BH, 16]`` accumulator is the result. No-swap keeps the token in M."""
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

    down_n = tl.arange(0, BLOCK_SIZE_N)
    down_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    down_base = Down + expert_id * stride_down_e
    # Down output rows are the hidden tile (BH); the contraction tile is BN (intermediate).
    w_down_ptr = oriented_weight_ptrs(down_base, down_h, down_n, stride_down_n, stride_down_k, SWAP_AB)
    # Replicate the single intermediate row across BLOCK_SIZE_M (arange*0) so BM=16 (non-swap) fills
    # the MMA 16-atom with copies (store_row keeps lane 0); BM=1 unchanged, swap (BM=1) uses row 0.
    inter_ptr = (
        Intermediate
        + (batch_id + tl.arange(0, BLOCK_SIZE_M) * 0)[:, None] * INTERMEDIATE_DIM
        + tl.arange(0, BLOCK_SIZE_N)[None, :]
    )
    inter_s_ptr = (
        IntermediateScale
        + (batch_id + tl.arange(0, BLOCK_SIZE_M) * 0)[:, None] * NUM_N_TILES
    )
    ws_down_ptr = DownScale + expert_id * stride_downs_e + pid_h * stride_downs_n

    acc = acc_init("dot", BLOCK_SIZE_M, BLOCK_SIZE_H, SWAP_AB)

    for _ in range(0, NUM_N_TILES):
        inter = tl.load(inter_ptr)
        inter_s = tl.load(inter_s_ptr)
        w_s_down = tl.load(ws_down_ptr)
        w_down = tl.load(w_down_ptr)
        # inter_s is [BM, 1] and w_s_down a per-tile scalar, so both broadcast onto the acc in either
        # orientation (under swap BM=1 → inter_s is the single token's scale) — no swap branch needed.
        acc += fp8_dot(inter, w_down, SWAP_AB, BLOCK_SIZE_N) * inter_s * w_s_down
        inter_ptr += BLOCK_SIZE_N
        inter_s_ptr += 1
        ws_down_ptr += stride_downs_k
        # contraction (intermediate) dim uses stride_down_k in both layouts (rows when [BN,BH],
        # cols when swapped [BH,BN]), so the K-advance is identical.
        w_down_ptr += BLOCK_SIZE_N * stride_down_k

    acc = acc_finalize(acc, "dot", BLOCK_SIZE_H, SWAP_AB)

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
            NUM_N_TILES=NUM_N_TILES,
            SIMULATE_UNFUSED=simulate_unfused,
        )
        wrap_triton(topk_reduce_kernel)[
            (num_tokens, triton.cdiv(HIDDEN_DIM, TOPK_REDUCE_BLOCK_H))
        ](
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
            NUM_TOP_K=NUM_TOP_K,
            NUM_EXPERTS=NUM_EXPERTS,
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
    # batched = M=1 decode: dot is never competitive here (its per-group tl.dot still pays the
    # M->16 MMA pad without dot_scaled's wide-K amortization), so only scalar + dot_scaled are
    # emitted. SWAP_AB is orthogonal — the tuner picks {dot_scaled, scalar} x {swap, no-swap} per
    # (expert shape, recipe): swap puts the weight's output rows in the MMA M dim (down ~1.78x,
    # gate_up ~1.07x on the measured decode), no-swap keeps the token in M (padded to 16).
    get_mxfp_autotuning_configs(
        compute_modes=("dot_scaled", "scalar"), swap_ab=True, for_decode=True
    ),
    ["num_routed_tokens", "INTERMEDIATE_DIM", "HIDDEN_DIM", "VALUES_PER_BYTE"],
    n_trials=100,
    # K-loop loads are unmasked; drop configs whose BK exceeds the contraction dim (hidden).
    # BK-within-K + the sm_10x MMA-shape guards over the stacked 2*BN gate|up extent.
    prune_configs_by={"early_config_prune": batched_mx_pruner("HIDDEN_DIM", stacked_gate_up=True)},
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
    SWAP_AB: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
):
    """MXFP4/MXFP8 kernel 1: gate_up + SiLU + MXFP8 requant. N = intermediate
    (output) tile, K = hidden (contraction) tile — both tunable. Grid: (S, N-tiles).

    Gate and up are ONE stacked load and ONE compute over the stacked 2*BN extent in both
    orientations, split back after the K-loop. ``SWAP_AB`` (tuner axis) only orients it: swap
    puts the ``[2*BN, BK]`` weight rows in the MMA M dim (token padded to the N=16 atom —
    native mxfp M=128 at BN=64 with 2x the CTAs); no-swap keeps the token in M against the
    ``[BK, 2*BN]`` combined tile (MMA width 2*BN, capped at 256 on sm_10x by the pruner)."""
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    token = (batch_id // NUM_TOP_K).to(tl.int64)
    expert_id = tl.load(ExpertIds + batch_id).to(tl.int64)
    if expert_id >= NUM_EXPERTS:  # EP sentinel; down kernel zeros the output row
        return

    # Replicate the single token across BLOCK_SIZE_M rows (arange*0): BM=16 (non-swap) fills the
    # MMA 16-atom with copies (store keeps lane 0); BM=1 unchanged, swap (BM=1) uses row 0.
    a_ptr = (
        HiddenStates
        + (token + tl.arange(0, BLOCK_SIZE_M) * 0)[:, None] * stride_a_m
        + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_a_k
    )
    # Gate (rows n) and up (rows I + n) are ONE stacked [2, ...] load in BOTH orientations
    # (stacked_gate_up_ptrs) and the K-loop runs a single mx_compute over the stacked 2*BN
    # extent — swap: [2*BN, BK] rows in the MMA M dim (native mxfp M=128 at BN=64, 2x the
    # CTAs); no-swap: [BK, 2*BN] gate|up along N, one combined dot (the grouped kernel's
    # form; configs with MMA width 2*BN > 256 are pruned on sm_10x). split_gate_up undoes
    # the stacking after the loop. The scale tile is [2*BN, NG] in both orientations.
    gate_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    kb = tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)
    sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)
    b_gu_ptr = stacked_gate_up_ptrs(
        GateUp + expert_id * stride_gu_e, gate_n, kb,
        INTERMEDIATE_DIM * stride_gu_n, stride_gu_n, stride_gu_k, SWAP_AB,
    )
    rows2 = tl.arange(0, 2)[:, None] * INTERMEDIATE_DIM + gate_n[None, :]
    bs_gu_ptr = (
        GateUpScale + expert_id * stride_gus_e
        + rows2[:, :, None] * stride_gus_n
        + sf[None, None, :] * stride_gus_k
    )
    acc = acc_init(COMPUTE_MODE, BLOCK_SIZE_M, 2 * BLOCK_SIZE_N, SWAP_AB)

    for _ in range(0, tl.cdiv(HIDDEN_DIM, BLOCK_SIZE_K)):
        a_raw = tl.load(a_ptr).to(tl.float32)
        a, a_scale = mxfp_act_quant_inline(a_raw, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K)
        w2 = stacked_gate_up_flatten(
            tl.load(b_gu_ptr),
            2 * BLOCK_SIZE_N, BLOCK_SIZE_K // VALUES_PER_BYTE, SWAP_AB,
        )
        ws2 = tl.reshape(
            tl.load(bs_gu_ptr),
            (2 * BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_K),
        )
        acc = mx_compute(
            acc, a, a_scale, w2, ws2, COMPUTE_MODE, VALUES_PER_BYTE,
            BLOCK_SIZE_M, 2 * BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, SWAP_AB,
        )
        b_gu_ptr += (BLOCK_SIZE_K // VALUES_PER_BYTE) * stride_gu_k
        bs_gu_ptr += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_gus_k
        a_ptr += BLOCK_SIZE_K * stride_a_k

    acc_gate, acc_up = split_gate_up(
        acc, COMPUTE_MODE, BLOCK_SIZE_M, BLOCK_SIZE_N, SWAP_AB
    )

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
    # inter_scale is [BLOCK_SIZE_M, BN//G]; at BM>1 the token is replicated so all rows are
    # identical — collapse to the single row 0 ([1, BN//G]) to match sc_ptrs (bit-identical at BM=1).
    tl.store(sc_ptrs, tl.reshape(tl.max(inter_scale, axis=0), (1, BLOCK_SIZE_N // SCALE_GROUP_K)))


@bayesian_autotune(
    # batched = M=1 decode: dot is never competitive here (see the gate_up decorator). SWAP_AB is
    # orthogonal — the tuner picks {dot_scaled, scalar} x {swap, no-swap}. Swap wins big on the
    # down GEMV (~1.78x on the measured decode); the tuner confirms it per (expert shape, recipe).
    get_mxfp_autotuning_configs(
        compute_modes=("dot_scaled", "scalar"), swap_ab=True, for_decode=True
    ),
    ["num_routed_tokens", "INTERMEDIATE_DIM", "HIDDEN_DIM", "VALUES_PER_BYTE"],
    n_trials=100,
    # BK-within-K + the sm_10x MMA-shape guards (swapped dot_scaled needs BN >= 128 for the
    # native scaled-MMA; smaller-BN swap configs never win and mislead the TPE).
    prune_configs_by={"early_config_prune": batched_mx_pruner("INTERMEDIATE_DIM")},
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
    SWAP_AB: tl.constexpr = False,
):
    """MXFP4/MXFP8 kernel 2: MXFP8 intermediate → down proj. N = hidden
    (output) tile, K = intermediate (contraction) tile. Grid: (S, H-tiles).

    ``SWAP_AB`` (tuner axis): swap loads the weight ``[BN, BK]`` and puts its output rows in the
    MMA M dim (token padded to N=16); no-swap loads ``[BK, BN]`` and keeps the token in M."""
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    expert_id = tl.load(ExpertIds + batch_id).to(tl.int64)
    if expert_id >= NUM_EXPERTS:  # EP sentinel: zero the output row for the top-k sum
        z = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        store_row(Out + batch_id * HIDDEN_DIM, z, pid_n, 1, BLOCK_SIZE_M, BLOCK_SIZE_N)
        return

    # Replicate the single intermediate row across BLOCK_SIZE_M (arange*0): BM=16 (non-swap) fills
    # the MMA 16-atom with copies (store keeps lane 0); BM=1 unchanged, swap (BM=1) uses row 0.
    a_ptr = (
        Intermediate
        + (batch_id + tl.arange(0, BLOCK_SIZE_M) * 0)[:, None] * INTERMEDIATE_DIM
        + tl.arange(0, BLOCK_SIZE_K)[None, :]
    )
    as_ptr = (
        IntermediateScale
        + (batch_id + tl.arange(0, BLOCK_SIZE_M) * 0)[:, None]
        * (INTERMEDIATE_DIM // SCALE_GROUP_K)
        + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)[None, :]
    )
    # Weight pointer orientation follows SWAP_AB: [BN, BK] output-rows-major when swapped, else
    # [BK, BN] K-major. The scale is [BN, NG] either way; the K-advance is a shared scalar step.
    down_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    kb = tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)
    down_base = Down + expert_id * stride_down_e
    w_down_ptr = oriented_weight_ptrs(down_base, down_n, kb, stride_down_n, stride_down_k, SWAP_AB)
    ws_down_ptr = (
        DownScale
        + expert_id * stride_downs_e
        + down_n[:, None] * stride_downs_n
        + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)[None, :] * stride_downs_k
    )
    acc = acc_init(COMPUTE_MODE, BLOCK_SIZE_M, BLOCK_SIZE_N, SWAP_AB)

    for _ in range(0, tl.cdiv(INTERMEDIATE_DIM, BLOCK_SIZE_K)):
        inter = tl.load(a_ptr)
        inter_s = tl.load(as_ptr)
        w_down = tl.load(w_down_ptr)
        ws_down = tl.load(ws_down_ptr)
        acc = mx_compute(
            acc, inter, inter_s, w_down, ws_down, COMPUTE_MODE, VALUES_PER_BYTE,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, SWAP_AB,
        )
        a_ptr += BLOCK_SIZE_K
        as_ptr += BLOCK_SIZE_K // SCALE_GROUP_K
        w_down_ptr += (BLOCK_SIZE_K // VALUES_PER_BYTE) * stride_down_k
        ws_down_ptr += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_downs_k

    acc = acc_finalize(acc, COMPUTE_MODE, BLOCK_SIZE_N, SWAP_AB)

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
            VALUES_PER_BYTE=VALUES_PER_BYTE,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
            SIMULATE_UNFUSED=simulate_unfused,
        )
        wrap_triton(topk_reduce_kernel)[
            (num_tokens, triton.cdiv(HIDDEN_DIM, TOPK_REDUCE_BLOCK_H))
        ](
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
            NUM_TOP_K=NUM_TOP_K,
            NUM_EXPERTS=NUM_EXPERTS,
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
