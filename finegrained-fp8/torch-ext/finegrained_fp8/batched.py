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

import torch
import triton
import triton.language as tl

from ._ops import add_op_namespace_prefix

from .bayesian_autotuner import bayesian_autotune
from .utils import (
    compile_time_only_triton_op,
    compile_time_only_triton_wrap,
    Epilogue,
    FP8_DTYPE,
    MX_SCALE_GROUP_K,
    NIBBLES_PER_BYTE,
    UE8M0_SCALE_DTYPES,
    decode_ue8m0_scale,
    device_context,
    tl_dtype,
    resolve_output_dtype,
    mx_compute,
    oriented_tile_ptrs,
    acc_init,
    fp8_dot,
    mx_config_pruner,
    block_k_within_k_pruner,
    acc_finalize,
    glu,
    split_gate_up_glu,
    stacked_gate_up_ptrs,
    stacked_gate_up_flatten,
    fp8_act_quant_inline,
    mxfp_act_quant_inline,
    fp8_act_quant_tensor_wide,
    fp8_act_quant_block_dynamic,
    load_block_fp8_act_tile,
    get_accelerator_autotuning_configs,
    is_mxfp,
    is_tensor_wide,
    e2m1_as_uint8,
    ue8m0_as_uint8,
)


@triton.jit
def expert_setup(
    A,
    B,
    C,
    Bs,
    ExpertIds,
    GatherIdx,
    ScatterIdx,
    stride_a_m,
    stride_b_e,
    stride_c_m,
    stride_bs_e,
    stride_eid,
    HAS_GATHER: tl.constexpr,
    HAS_SCATTER: tl.constexpr,
):
    """Per-(row, expert) prologue shared by the batched kernels: read the program
    ids, look up the routed expert, and advance the A/B/C/Bs base pointers to this
    row's slice. Returns ``(batch_id, pid_n, expert_id, A, B, C, Bs)``.

    ``A``'s source row is ``GatherIdx[batch_id]`` when ``HAS_GATHER`` (the gate_up reading
    unexpanded activations, many-to-one for top_k > 1) else ``batch_id``; ``C``'s destination
    row is ``ScatterIdx[batch_id]`` when ``HAS_SCATTER`` else ``batch_id`` — the same virtual
    gather/scatter ``matmul_grouped`` does, so the routed rows need no materialized copy.

    The caller must early-return on the EP sentinel (``expert_id >= num_experts``)
    before any load — the pointer arithmetic itself is harmless, only the loads on a
    non-local expert would be out of bounds."""
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    # Cast to int64 to prevent overflow on expert_id * stride_b_e.
    expert_id = tl.load(ExpertIds + batch_id * stride_eid).to(tl.int64)
    in_row = tl.load(GatherIdx + batch_id).to(tl.int64) if HAS_GATHER else batch_id
    out_row = tl.load(ScatterIdx + batch_id).to(tl.int64) if HAS_SCATTER else batch_id
    A = A + in_row * stride_a_m
    B = B + expert_id * stride_b_e
    C = C + out_row * stride_c_m
    Bs = Bs + expert_id * stride_bs_e
    return batch_id, pid_n, expert_id, A, B, C, Bs, in_row, out_row


@triton.jit
def store_row(
    C,
    accumulator,
    pid_n,
    stride_c_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Output epilogue shared by the batched kernels (``C`` already advanced to the
    row). The fake-batch trick aliases all ``BLOCK_SIZE_M`` lanes to the same C row,
    so a plain store would issue ``BLOCK_SIZE_M`` duplicate-address writes — benign on
    NVIDIA WGMMA (last-write-wins of identical bytes) but hardware-undefined on Intel
    XPU, where it corrupts the output. Mask so only lane 0 stores; the accumulator
    rows are mathematically identical (same A row × same B), so lane 0 is correct."""
    c = accumulator.to(C.dtype.element_ty)
    offs_cm = tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offs_cm[:, None] * 0 + stride_c_n * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm == 0)[:, None])


@bayesian_autotune(
    get_accelerator_autotuning_configs(swap_ab=True),
    ["N", "K", "S"],
    n_trials=100,
)
@triton.jit
def w8a8_block_dynamic_fp8_matmul_batched_kernel(
    A,  # (S, K) E4M3 activations (pre-quantized once by the wrapper)
    As,  # (S, K // BLOCK_SIZE_K) fp32 per-row, per-K-block activation scales
    B,  # (num_experts, N, K) FP8 weights; under GATE the (num_experts, 2N, K) gate|up stack
    Bs,  # (num_experts, N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales (2N under GATE)
    C,  # (S, N) output; under REQUANT the FP8-requantized intermediate
    Cs,  # (S, N // BLOCK_SIZE_N) per-(row, N-tile) output scale; written iff REQUANT
    ExpertIds,  # (S,) — which expert each batch element routes to
    GatherIdx,  # (S,) int — batch_id -> source row of A; read iff HAS_GATHER
    ScatterIdx,  # (S,) int — batch_id -> destination row of C; read iff HAS_SCATTER
    # Shape
    S,
    N,
    K,
    # Strides
    stride_a_m,
    stride_a_k,
    stride_as_m,
    stride_b_e,
    stride_b_k,
    stride_b_n,
    stride_bs_e,
    stride_bs_k,
    stride_bs_n,
    stride_c_m,
    stride_c_n,
    stride_cs_m,
    stride_cs_n,
    stride_eid,
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
    HAS_GATHER: tl.constexpr = False,
    HAS_SCATTER: tl.constexpr = False,
    # Gate|up fusion epilogue (GATE=False -> plain batched GEMM, every arm below folds out)
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    REQUANT: tl.constexpr = False,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
):
    """Block-scale batched FP8 expert matmul kernel.

    Each program handles one routed token row and one N-tile, looking up the
    owning expert from ``ExpertIds``. Activations arrive pre-quantized (one wrapper
    pass — an inline quant would repeat per N-tile and pay a per-tile amax reduction).

    ``SWAP_AB`` (tuner axis, M=1 decode): load the weight output-rows-major ``[BN, BK]`` and put
    those rows in the MMA M dim, padding the single token to the N=16 atom; column 0 of the
    ``[BN, 16]`` accumulator is the result. No-swap keeps the token in M (padded to 16).

    ``GATE`` fuses the gate|up projection: ``B`` is the ``(E, 2N, K)`` stack (gate rows [0,N),
    up rows [N,2N)), run as two dots (the decode-validated form), SwiGLU-combined, and — under
    ``REQUANT`` — FP8-requantized into ``C`` + a per-(row, N-tile) scalar ``Cs``. Every gate arm
    folds out at compile time; ``GATE=False`` is the plain GEMM, bit-identical."""
    batch_id, pid_n, expert_id, A, B, C, Bs, in_row, out_row = expert_setup(
        A,
        B,
        C,
        Bs,
        ExpertIds,
        GatherIdx,
        ScatterIdx,
        stride_a_m,
        stride_b_e,
        stride_c_m,
        stride_bs_e,
        stride_eid,
        HAS_GATHER,
        HAS_SCATTER,
    )
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= num_experts:
        return

    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 + offs_k[None, :] * stride_a_k
    as_ptrs = As + in_row * stride_as_m + tl.zeros((BLOCK_SIZE_M,), tl.int32)
    # GATE loads gate (rows [0,N)) and up (rows [N,2N)) as two oriented tiles + two dots — the
    # decode-validated form (the swapped stacked-scale orientation is why we keep two dots here).
    if GATE:
        up_n = N + offs_bn
        b_gate_ptr = oriented_tile_ptrs(B, offs_bn, offs_k, stride_b_n, stride_b_k, SWAP_AB)
        b_up_ptr = oriented_tile_ptrs(B, up_n, offs_k, stride_b_n, stride_b_k, SWAP_AB)
        bs_gate_ptr = Bs + pid_n * stride_bs_n
        bs_up_ptr = Bs + (num_n_tiles + pid_n) * stride_bs_n
        acc_gate = acc_init("dot", BLOCK_SIZE_M, BLOCK_SIZE_N, SWAP_AB)
        acc_up = acc_init("dot", BLOCK_SIZE_M, BLOCK_SIZE_N, SWAP_AB)
    else:
        b_ptrs = oriented_tile_ptrs(B, offs_bn, offs_k, stride_b_n, stride_b_k, SWAP_AB)
        bs_ptrs = Bs + pid_n * stride_bs_n
        accumulator = acc_init("dot", BLOCK_SIZE_M, BLOCK_SIZE_N, SWAP_AB)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a, a_s = load_block_fp8_act_tile(a_ptrs, as_ptrs)
        # a_s is [BM], the weight scales per-block scalars; a_s[:, None] broadcasts onto the acc
        # either way (under swap BM=1, so it is the single token's scale) — no swap branch.
        if GATE:
            acc_gate += (
                fp8_dot(a, tl.load(b_gate_ptr), SWAP_AB, BLOCK_SIZE_K)
                * a_s[:, None]
                * decode_ue8m0_scale(tl.load(bs_gate_ptr))
            )
            acc_up += (
                fp8_dot(a, tl.load(b_up_ptr), SWAP_AB, BLOCK_SIZE_K)
                * a_s[:, None]
                * decode_ue8m0_scale(tl.load(bs_up_ptr))
            )
            b_gate_ptr += BLOCK_SIZE_K * stride_b_k
            b_up_ptr += BLOCK_SIZE_K * stride_b_k
            bs_gate_ptr += stride_bs_k
            bs_up_ptr += stride_bs_k
        else:
            accumulator += (
                fp8_dot(a, tl.load(b_ptrs), SWAP_AB, BLOCK_SIZE_K)
                * a_s[:, None]
                * decode_ue8m0_scale(tl.load(bs_ptrs))
            )
            b_ptrs += BLOCK_SIZE_K * stride_b_k
            bs_ptrs += stride_bs_k
        a_ptrs += BLOCK_SIZE_K * stride_a_k
        as_ptrs += 1

    if GATE:
        acc_gate = acc_finalize(acc_gate, "dot", BLOCK_SIZE_N, SWAP_AB)
        acc_up = acc_finalize(acc_up, "dot", BLOCK_SIZE_N, SWAP_AB)
        intermediate = glu(
            acc_gate,
            acc_up,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SIMULATE_UNFUSED,
            INTERMEDIATE_DTYPE,
        )
        if REQUANT:
            inter, inter_s = fp8_act_quant_inline(intermediate)
            store_row(C, inter, pid_n, stride_c_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
            tl.store(Cs + out_row * stride_cs_m + pid_n * stride_cs_n, tl.max(inter_s))
        else:
            store_row(C, intermediate, pid_n, stride_c_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
    else:
        accumulator = acc_finalize(accumulator, "dot", BLOCK_SIZE_N, SWAP_AB)
        store_row(C, accumulator, pid_n, stride_c_n, BLOCK_SIZE_M, BLOCK_SIZE_N)


@bayesian_autotune(
    # S (routed rows) keyed like the block-dynamic/mxfp batched siblings — decode re-tunes per batch.
    get_accelerator_autotuning_configs(tune_block_nk=True, swap_ab=True),
    ["N", "K", "S"],
    n_trials=100,
    # BLOCK_SIZE_K is a tuned axis and the K-loop is maskless — veto non-dividing BKs.
    prune_configs_by={"early_config_prune": block_k_within_k_pruner("K")},
)
@triton.jit
def w8a8_tensor_dynamic_fp8_matmul_batched_kernel(
    A,  # (S, K) pre-quantized FP8 activations
    As,  # (S,) per-token activation scales
    B,  # (num_experts, N, K) FP8 weight matrices
    Bs,  # (num_experts, 1, 1) per-tensor weight scales
    C,  # (S, N) output
    ExpertIds,  # (S,) — which expert each batch element routes to
    GatherIdx,  # (S,) int — batch_id -> source row of A; read iff HAS_GATHER
    ScatterIdx,  # (S,) int — batch_id -> destination row of C; read iff HAS_SCATTER
    # Shape
    S,
    N,
    K,
    # Strides
    stride_a_m,
    stride_a_k,
    stride_as_m,
    stride_b_e,
    stride_b_k,
    stride_b_n,
    stride_bs_e,
    stride_c_m,
    stride_c_n,
    stride_eid,
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
    HAS_GATHER: tl.constexpr = False,
    HAS_SCATTER: tl.constexpr = False,
):
    """Tensor-scale batched FP8 expert matmul kernel.

    Activations are already quantized; the kernel applies per-token activation
    scales and per-expert tensor weight scales.

    ``SWAP_AB`` (tuner axis, M=1 decode): weight output rows in the MMA M dim (``B`` as ``[BN, BK]``,
    single token padded to N=16); column 0 of the ``[BN, 16]`` accumulator is the result. Both
    scales are per-token/per-tensor scalars, applied once after the loop, orientation-agnostic."""
    batch_id, pid_n, expert_id, A, B, C, Bs, in_row, out_row = expert_setup(
        A,
        B,
        C,
        Bs,
        ExpertIds,
        GatherIdx,
        ScatterIdx,
        stride_a_m,
        stride_b_e,
        stride_c_m,
        stride_bs_e,
        stride_eid,
        HAS_GATHER,
        HAS_SCATTER,
    )
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= num_experts:
        return

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 + offs_k[None, :] * stride_a_k
    b_ptrs = oriented_tile_ptrs(B, offs_bn, offs_k, stride_b_n, stride_b_k, SWAP_AB)
    b_s = tl.load(Bs)
    a_s = tl.load(As + in_row * stride_as_m)

    accumulator = acc_init("dot", BLOCK_SIZE_M, BLOCK_SIZE_N, SWAP_AB)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += fp8_dot(a, b, SWAP_AB, BLOCK_SIZE_K)
        a_ptrs += BLOCK_SIZE_K * stride_a_k
        b_ptrs += BLOCK_SIZE_K * stride_b_k

    accumulator = acc_finalize(accumulator, "dot", BLOCK_SIZE_N, SWAP_AB) * a_s * b_s
    store_row(C, accumulator, pid_n, stride_c_n, BLOCK_SIZE_M, BLOCK_SIZE_N)


# VALUES_PER_BYTE keys the MXFP4/MXFP8 split so a cached winner is only reused for its packing.
# BLOCK_SIZE_M is always 1 here (per-token decode), so — like the fused MXFP batched kernels —
# dot is excluded — MEASURED TWICE (2026-07-10): no-swap BM16 within noise of the
# scalar/dot_scaled-swap champions, and fielding it WITH the swapped form
# (mx_dot_rescale_swapped) poisoned the TPE (dsv4 +27%, M3 +12% tuner misses) — the
# can't-win dot-swap configs skew the per-dimension densities. The swapped helper stays
# implemented for future shapes; don't re-emit without new evidence.
@bayesian_autotune(
    get_accelerator_autotuning_configs(
        mx=True,
        tune_block_nk=True,
        compute_modes=("dot_scaled", "scalar"),
        swap_ab=True,
    ),
    ["N", "K", "S", "VALUES_PER_BYTE"],
    n_trials=100,
    # BK-within-K + the sm_10x MMA-shape guards (swapped dot_scaled needs BN >= 128 for the
    # native scaled-MMA; smaller-BN swap configs never win and mislead the TPE).
    prune_configs_by={"early_config_prune": mx_config_pruner("K")},
)
@triton.jit
def mxfp_dynamic_matmul_batched_kernel(
    A,  # (S, K) activations: raw BF16/FP16 (inline-quant) or E4M3 (PRE_QUANTIZED)
    As,  # (S, K // SCALE_GROUP_K) UE8M0 act scales; read iff PRE_QUANTIZED
    B,  # (num_experts, N, K[/2]); under GATE the (num_experts, 2N, K[/2]) gate|up stack
    Bs,  # (num_experts, N, K // SCALE_GROUP_K) UE8M0 weight scales (2N under GATE)
    C,  # (S, N) output; under REQUANT the MX-requantized intermediate
    Cs,  # (S, N // SCALE_GROUP_K) UE8M0 output scale; written iff REQUANT
    ExpertIds,  # (S,) — which expert each routed row uses
    GatherIdx,  # (S,) int — batch_id -> source row of A; read iff HAS_GATHER
    ScatterIdx,  # (S,) int — batch_id -> destination row of C; read iff HAS_SCATTER
    # Shape
    S,
    N,
    K,
    # Strides
    stride_a_m,
    stride_a_k,
    stride_as_m,
    stride_b_e,
    stride_b_k,
    stride_b_n,
    stride_bs_e,
    stride_bs_k,
    stride_bs_n,
    stride_c_m,
    stride_c_n,
    stride_cs_m,
    stride_cs_n,
    stride_eid,
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
    HAS_GATHER: tl.constexpr = False,
    HAS_SCATTER: tl.constexpr = False,
    # Gate|up fusion epilogue (GATE=False -> plain batched GEMM, every arm below folds out)
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    REQUANT: tl.constexpr = False,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
    # PRE_QUANTIZED: A is E4M3 + As UE8M0 (the down reading a requantized intermediate); else
    # A is raw and quantized inline (gate_up / plain, decode-free UE8M0).
    PRE_QUANTIZED: tl.constexpr = False,
):
    """Unified batched MXFP4/MXFP8 (W4A8/W8A8) expert matmul with fused act quant.

    One routed row + one N-tile per program; expert looked up from ``ExpertIds``. ``A`` is
    quantized to E4M3 per K-group inline (UE8M0 scale). ``VALUES_PER_BYTE`` picks the
    weight format (2 = packed E2M1 / MXFP4, 1 = unpacked E4M3 / MXFP8); ``COMPUTE_MODE``
    picks ``tl.dot_scaled`` (native M=128) vs the scalar CUDA-core reduce (wins at decode).

    ``SWAP_AB`` (tuner axis, M=1 decode): weight output rows in the MMA M dim (``B`` as ``[BN, BK]``,
    single token padded to N=16); column 0 of the ``[BN, 16]`` accumulator is the result. dot_scaled
    uses the swapped scaled-MMA; scalar reduces over K with the weight output-rows-major.
    """
    batch_id, pid_n, expert_id, A, B, C, Bs, in_row, out_row = expert_setup(
        A,
        B,
        C,
        Bs,
        ExpertIds,
        GatherIdx,
        ScatterIdx,
        stride_a_m,
        stride_b_e,
        stride_c_m,
        stride_bs_e,
        stride_eid,
        HAS_GATHER,
        HAS_SCATTER,
    )
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= num_experts:
        return

    n_width: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_kb = tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)
    a_ptrs = A + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 + offs_k[None, :] * stride_a_k
    as_ptrs = (
        As
        + in_row * stride_as_m
        + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0
        + offs_sf[None, :]
    )  # read iff PRE_QUANTIZED
    # GATE stacks gate|up into one weight tile + its [2*BN, NG] scale (the up block sits N rows
    # away). Weight [.., 2*BN or BN] oriented by SWAP_AB; scales are always N-major.
    b_ptrs = stacked_gate_up_ptrs(
        B, offs_bn, offs_kb, N * stride_b_n, stride_b_n, stride_b_k, GATE, SWAP_AB
    )
    if GATE:
        rows2 = tl.arange(0, 2)[:, None] * N + offs_bn[None, :]
        bs_ptrs = (
            Bs + rows2[:, :, None] * stride_bs_n + offs_sf[None, None, :] * stride_bs_k
        )
    else:
        bs_ptrs = Bs + offs_bn[:, None] * stride_bs_n + offs_sf[None, :] * stride_bs_k

    accumulator = acc_init(COMPUTE_MODE, BLOCK_SIZE_M, n_width, SWAP_AB)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if PRE_QUANTIZED:
            a = tl.load(a_ptrs)
            a_scale = tl.load(as_ptrs).to(tl.uint8)
            as_ptrs += BLOCK_SIZE_K // SCALE_GROUP_K
        else:
            a_raw = tl.load(a_ptrs).to(tl.float32)
            a, a_scale = mxfp_act_quant_inline(
                a_raw, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K
            )
        b = stacked_gate_up_flatten(
            tl.load(b_ptrs), 2 * BLOCK_SIZE_N, BLOCK_SIZE_K // VALUES_PER_BYTE, GATE, SWAP_AB
        )
        b_s = tl.reshape(
            tl.load(bs_ptrs), (n_width, BLOCK_SIZE_K // SCALE_GROUP_K)
        ).to(tl.uint8)
        accumulator = mx_compute(
            accumulator,
            a,
            a_scale,
            b,
            b_s,
            COMPUTE_MODE,
            VALUES_PER_BYTE,
            BLOCK_SIZE_M,
            n_width,
            BLOCK_SIZE_K,
            SCALE_GROUP_K,
            SWAP_AB,
        )
        a_ptrs += BLOCK_SIZE_K * stride_a_k
        b_ptrs += (BLOCK_SIZE_K // VALUES_PER_BYTE) * stride_b_k
        bs_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_bs_k

    if GATE:
        intermediate = split_gate_up_glu(
            accumulator,
            COMPUTE_MODE,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            SWAP_AB,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SIMULATE_UNFUSED,
            INTERMEDIATE_DTYPE,
        )
        if REQUANT:
            inter, inter_scale = mxfp_act_quant_inline(
                intermediate, BLOCK_SIZE_M, BLOCK_SIZE_N, SCALE_GROUP_K
            )
            store_row(C, inter, pid_n, stride_c_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
            offs_sc = pid_n * (BLOCK_SIZE_N // SCALE_GROUP_K) + tl.arange(
                0, BLOCK_SIZE_N // SCALE_GROUP_K
            )
            tl.store(
                Cs + out_row * stride_cs_m + offs_sc[None, :] * stride_cs_n,
                tl.reshape(
                    tl.max(inter_scale, axis=0), (1, BLOCK_SIZE_N // SCALE_GROUP_K)
                ),
            )
        else:
            store_row(C, intermediate, pid_n, stride_c_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
    else:
        accumulator = acc_finalize(accumulator, COMPUTE_MODE, BLOCK_SIZE_N, SWAP_AB)
        store_row(C, accumulator, pid_n, stride_c_n, BLOCK_SIZE_M, BLOCK_SIZE_N)


@bayesian_autotune(
    get_accelerator_autotuning_configs(tune_block_nk=True, swap_ab=True),
    # S (routed rows) keyed like the fp8/mx batched siblings — decode re-tunes per batch;
    # GATE keys the gate|up arm separately (its stacked dot is 2*BN wide).
    ["N", "K", "S", "GATE"],
    n_trials=100,
    # BLOCK_SIZE_K is a tuned axis and the K-loop is maskless — veto non-dividing BKs.
    prune_configs_by={"early_config_prune": block_k_within_k_pruner("K")},
)
@triton.jit
def full_precision_matmul_batched_kernel(
    A,  # (rows, K) BF16/FP16 activations
    B,  # (num_experts, N, K) weights in A's dtype; under GATE the (num_experts, 2N, K) gate|up stack
    C,  # (S, N) output; under GATE the GLU intermediate
    ExpertIds,  # (S,) — which expert each batch element routes to
    GatherIdx,  # (S,) int — batch_id -> source row of A; read iff HAS_GATHER
    ScatterIdx,  # (S,) int — batch_id -> destination row of C; read iff HAS_SCATTER
    # Shape
    S,
    N,
    K,
    # Strides
    stride_a_m,
    stride_a_k,
    stride_b_e,
    stride_b_k,
    stride_b_n,
    stride_c_m,
    stride_c_n,
    stride_eid,
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
    HAS_GATHER: tl.constexpr = False,
    HAS_SCATTER: tl.constexpr = False,
    # Gate|up fusion epilogue (GATE=False -> plain batched GEMM). No requant arm: the
    # full-precision chain has no quantized intermediate — down consumes the GLU output as is.
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
):
    """Full-precision batched expert matmul kernel: plain ``tl.dot`` over unquantized
    BF16/FP16 activations and weights, fp32 accumulation, no scales anywhere. ``GATE``
    computes gate|up as ONE stacked tile + dot (straight-line, both orientations) and
    applies the ``ACT_FN``/SwiGLU ``glu``. ``SWAP_AB`` (tuner axis, M=1 decode): weight
    output rows in the MMA M dim, the single token padded to the N=16 atom."""
    batch_id, pid_n, expert_id, A, B, C, _, in_row, out_row = expert_setup(
        A,
        B,
        C,
        B,  # dummy Bs (no scales); stride 0 keeps the advance a no-op
        ExpertIds,
        GatherIdx,
        ScatterIdx,
        stride_a_m,
        stride_b_e,
        stride_c_m,
        0,
        stride_eid,
        HAS_GATHER,
        HAS_SCATTER,
    )
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= num_experts:
        return

    n_width: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 + offs_k[None, :] * stride_a_k
    # GATE stacks gate|up into one weight tile (the up block sits N rows away);
    # GATE=False -> the plain oriented tile.
    b_ptrs = stacked_gate_up_ptrs(
        B, offs_bn, offs_k, N * stride_b_n, stride_b_n, stride_b_k, GATE, SWAP_AB
    )

    accumulator = acc_init("dot", BLOCK_SIZE_M, n_width, SWAP_AB)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        w = stacked_gate_up_flatten(
            tl.load(b_ptrs), 2 * BLOCK_SIZE_N, BLOCK_SIZE_K, GATE, SWAP_AB
        )
        accumulator += fp8_dot(a, w, SWAP_AB, BLOCK_SIZE_K)
        a_ptrs += BLOCK_SIZE_K * stride_a_k
        b_ptrs += BLOCK_SIZE_K * stride_b_k

    if GATE:
        intermediate = split_gate_up_glu(
            accumulator,
            "dot",
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            SWAP_AB,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SIMULATE_UNFUSED,
            INTERMEDIATE_DTYPE,
        )
        store_row(C, intermediate, pid_n, stride_c_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
    else:
        accumulator = acc_finalize(accumulator, "dot", BLOCK_SIZE_N, SWAP_AB)
        store_row(C, accumulator, pid_n, stride_c_n, BLOCK_SIZE_M, BLOCK_SIZE_N)


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_block_dynamic_fp8_matmul_batched"), mutates_args=()
)
def w8a8_block_dynamic_fp8_matmul_batched(
    A: torch.Tensor,
    As: torch.Tensor | None,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int],
    gate: bool = False,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    requant: bool = False,
    simulate_unfused: bool = False,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Block-scale batched FP8 matmul: C[s] = A[s] @ B[expert_ids[s]].T; activations
    quantized offline in one pass. The ``gate``/``act_fn``/``swiglu_*``/``requant``/``output_dtype``
    flags are the flattened ``Epilogue`` (torch custom ops take only primitives —
    ``matmul_batched`` unpacks the bundle). ``gather_idx``/``scatter_idx`` map the source row of A
    / destination row of C per program (None = row s). Returns ``[C]``, or ``[C, Cs]`` under
    ``requant``.

    A:  (rows, K) raw bf16/fp16 activations — rows addressed via ``gather_idx``
    B:  (num_experts, N, K) FP8 weights; under ``gate`` the (num_experts, 2N, K) gate|up stack
    Bs: (num_experts, N // block_n, K // block_k) per-block weight scales (2N under gate)
    """
    assert A.ndim == 2, f"A must be 2D (rows, K), got ndim={A.ndim}"
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 3, f"B must be 3D (num_experts, N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"
    assert A.shape[1] == B.shape[2], (
        f"K mismatch: A has K={A.shape[1]}, B has K={B.shape[2]}"
    )

    output_dtype = resolve_output_dtype(output_dtype, A, As)
    # S is the routed-row count (one program per expert_id); A may hold fewer rows when
    # gather_idx maps many programs to one source row (gate_up reading unexpanded hidden).
    K = A.shape[1]
    S = expert_ids.shape[0]
    # Under a gate epilogue B is the (E, 2N, K) gate|up stack — N is the per-projection width.
    num_experts, n_rows, _ = B.shape
    N = n_rows // 2 if gate else n_rows

    assert len(block_size) == 2, (
        f"block_size must be [block_n, block_k], got {block_size}"
    )
    block_n, block_k = block_size[0], block_size[1]
    # MoE expert dimensions must be block-aligned; non-aligned N/K is not supported.
    assert N % block_n == 0, f"N ({N}) must be divisible by block_n ({block_n})"
    assert K % block_k == 0, f"K ({K}) must be divisible by block_k ({block_k})"
    assert Bs.shape == (num_experts, n_rows // block_n, K // block_k), (
        f"Bs shape {tuple(Bs.shape)} != expected ({num_experts}, {n_rows // block_n}, {K // block_k})"
    )

    Bs = ue8m0_as_uint8(Bs)
    # Offline quant wins here even at decode. An inline quant would rerun once per N-tile
    # of the (S x N-tiles) grid, and block-FP8 quant is an fp32 amax+div per element, so
    # the redundant work outweighs the extra launch down to T=1 (inline only edges ahead
    # near T=64). UE8M0 quant is ~free per pass, which is why the MX kernels do it inline.
    # A raw (As is None) -> quantize here (offline); else pre-quantized (As given, e.g. the
    # requantized intermediate handed to the down projection).
    if As is None:
        A_q, A_s = fp8_act_quant_block_dynamic(A, block_k)
    else:
        A_q, A_s = A, As
    if requant:
        C = A.new_empty(S, N, dtype=FP8_DTYPE)
        Cs = torch.empty(S, N // block_n, device=A.device, dtype=torch.float32)
    else:
        C = A.new_empty(S, N, dtype=output_dtype)
        Cs = expert_ids  # general dummy pointer; unread (REQUANT=False), strides literal

    grid = (S, triton.cdiv(N, block_n))

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_block_dynamic_fp8_matmul_batched_kernel)[
            grid
        ](
            A_q,
            A_s,
            B,
            Bs,
            C,
            Cs,
            expert_ids,
            gather_idx if gather_idx is not None else expert_ids,
            scatter_idx if scatter_idx is not None else expert_ids,
            S,
            N,
            K,
            A_q.stride(0),
            A_q.stride(1),
            A_s.stride(0),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            Bs.stride(0),
            Bs.stride(2),
            Bs.stride(1),
            C.stride(0),
            C.stride(1),
            Cs.stride(0) if requant else 1,
            Cs.stride(1) if requant else 1,
            expert_ids.stride(0),
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            num_experts=num_experts,
            HAS_GATHER=gather_idx is not None,
            HAS_SCATTER=scatter_idx is not None,
            GATE=gate,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            REQUANT=requant,
            SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(output_dtype),
        )

    return [C, Cs] if requant else [C]


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_tensor_dynamic_fp8_matmul_batched"), mutates_args=()
)
def w8a8_tensor_dynamic_fp8_matmul_batched(
    A: torch.Tensor,
    As: torch.Tensor | None,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tensor-scale batched FP8 matmul: C[s] = A[s] @ B[expert_ids[s]].T. ``A`` raw
    (``As`` None) -> quantized here (offline, per-token); else pre-quantized (``As`` given).
    ``gather_idx``/``scatter_idx`` map the source row of A / destination row of C per program
    (None = row s).

    A:  (rows, K) raw or pre-quantized FP8 activations — rows addressed via ``gather_idx``
    As: (rows,) per-token scales, or None when A is raw
    B:  (num_experts, N, K) FP8 expert weights
    Bs: (num_experts,) or (num_experts, 1, 1) per-expert weight scales
    """
    assert A.ndim == 2, f"A must be 2D (rows, K), got ndim={A.ndim}"
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 3, f"B must be 3D (num_experts, N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"
    assert A.shape[1] == B.shape[2], (
        f"K mismatch: A has K={A.shape[1]}, B has K={B.shape[2]}"
    )

    output_dtype = resolve_output_dtype(output_dtype, A, As)
    K = A.shape[1]
    S = expert_ids.shape[0]
    num_experts, N, _ = B.shape

    # Normalize Bs to (num_experts, 1, 1)
    if Bs.ndim == 1:
        assert Bs.shape[0] == num_experts, (
            f"Bs shape {tuple(Bs.shape)} != expected ({num_experts},)"
        )
        Bs = Bs.reshape(num_experts, 1, 1)
    else:
        assert Bs.shape == (num_experts, 1, 1), (
            f"Bs shape {tuple(Bs.shape)} != expected ({num_experts}, 1, 1)"
        )

    Bs = ue8m0_as_uint8(Bs)
    if As is None:
        qA, As = fp8_act_quant_tensor_wide(A, K)
    else:
        qA = A
    C = A.new_empty(S, N, dtype=output_dtype)

    def grid(META):
        return (S, triton.cdiv(N, META["BLOCK_SIZE_N"]))

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_tensor_dynamic_fp8_matmul_batched_kernel)[
            grid
        ](
            qA,
            As,
            B,
            Bs,
            C,
            expert_ids,
            gather_idx if gather_idx is not None else expert_ids,
            scatter_idx if scatter_idx is not None else expert_ids,
            S,
            N,
            K,
            qA.stride(0),
            qA.stride(1),
            As.stride(0),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            Bs.stride(0),
            C.stride(0),
            C.stride(1),
            expert_ids.stride(0),
            num_experts=num_experts,
            HAS_GATHER=gather_idx is not None,
            HAS_SCATTER=scatter_idx is not None,
        )

    return C


@compile_time_only_triton_op(
    add_op_namespace_prefix("mxfp_dynamic_matmul_batched"), mutates_args=()
)
def mxfp_dynamic_matmul_batched(
    A: torch.Tensor,
    As: torch.Tensor | None,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    gate: bool = False,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    requant: bool = False,
    simulate_unfused: bool = False,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Batched MX matmul ``C[s] = A[s] @ B[expert_ids[s]].T``; activations quantized
    inline in the kernel (decode: one act row per program, inline is free). The
    ``gate``/``act_fn``/``swiglu_*``/``requant``/``output_dtype`` flags are the flattened
    ``Epilogue`` (``matmul_batched`` unpacks the bundle). ``gather_idx``/``scatter_idx`` map the
    source row of A / destination row of C per program (None = row s). Returns ``[C]``, or
    ``[C, Cs]`` under ``requant``. Weight format is detected from ``B.dtype``: ``int8`` → packed
    E2M1 (MXFP4, ``B`` is ``(num_experts, N, K//2)``); ``float8_e4m3fn`` → unpacked E4M3 (MXFP8);
    both use UE8M0 group-32 scales; under ``gate`` the (num_experts, 2N, K[/2]) gate|up stack.

    A:  (rows, K) activations — raw bf16/fp16/fp32 (inline-quant) or pre-quantized E4M3
    expert_ids: (S,) which expert each routed row uses
    """
    assert A.ndim == 2 and B.ndim == 3 and Bs.ndim == 3
    assert expert_ids.ndim == 1
    # A raw (As None) -> quantized inline in the kernel (decode-free UE8M0); pre-quantized
    # (As given, e.g. the down reading a requantized intermediate) -> loaded via the kernel's
    # PRE_QUANTIZED branch.
    pre_quantized = As is not None
    assert B.dtype in (torch.int8, torch.float8_e4m3fn), (
        f"B must be int8 (packed E2M1) or float8_e4m3fn (E4M3), got {B.dtype}"
    )
    assert Bs.dtype in UE8M0_SCALE_DTYPES, (
        f"Bs must be float8_e8m0fnu or uint8 (UE8M0), got {Bs.dtype}"
    )
    VALUES_PER_BYTE = NIBBLES_PER_BYTE if B.dtype == torch.int8 else 1

    output_dtype = resolve_output_dtype(output_dtype, A, As)
    K = A.shape[1]
    S = expert_ids.shape[0]
    # Under a gate epilogue B is the (E, 2N, K) gate|up stack — N is the per-projection width.
    num_experts, n_rows, K_b = B.shape
    N = n_rows // 2 if gate else n_rows
    assert K == VALUES_PER_BYTE * K_b, (
        f"K (={K}) must equal {VALUES_PER_BYTE} * B.shape[2] (={K_b})"
    )
    assert K % MX_SCALE_GROUP_K == 0, (
        f"K (={K}) must be a multiple of {MX_SCALE_GROUP_K}"
    )
    assert Bs.shape == (num_experts, n_rows, K // MX_SCALE_GROUP_K), (
        f"Bs shape {tuple(Bs.shape)} != ({num_experts}, {n_rows}, {K // MX_SCALE_GROUP_K})"
    )

    B = e2m1_as_uint8(B)
    bs_u8 = ue8m0_as_uint8(Bs)
    if requant:
        C = A.new_empty((S, N), dtype=FP8_DTYPE)
        Cs = torch.empty(S, N // MX_SCALE_GROUP_K, device=A.device, dtype=torch.uint8)
    else:
        C = A.new_empty((S, N), dtype=output_dtype)
        Cs = expert_ids  # general dummy pointer; unread (REQUANT=False), strides literal

    def grid(META):
        return (S, triton.cdiv(N, META["BLOCK_SIZE_N"]))

    As_arg = As if pre_quantized else expert_ids  # dummy when raw (unread)
    with device_context(A.device):
        compile_time_only_triton_wrap(mxfp_dynamic_matmul_batched_kernel)[grid](
            A,
            As_arg,
            B,
            bs_u8,
            C,
            Cs,
            expert_ids,
            gather_idx if gather_idx is not None else expert_ids,
            scatter_idx if scatter_idx is not None else expert_ids,
            S,
            N,
            K,
            A.stride(0),
            A.stride(1),
            As_arg.stride(0) if pre_quantized else 1,
            B.stride(0),
            B.stride(2),
            B.stride(1),
            bs_u8.stride(0),
            bs_u8.stride(2),
            bs_u8.stride(1),
            C.stride(0),
            C.stride(1),
            Cs.stride(0) if requant else 1,
            Cs.stride(1) if requant else 1,
            expert_ids.stride(0),
            VALUES_PER_BYTE=VALUES_PER_BYTE,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
            num_experts=num_experts,
            PRE_QUANTIZED=pre_quantized,
            HAS_GATHER=gather_idx is not None,
            HAS_SCATTER=scatter_idx is not None,
            GATE=gate,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            REQUANT=requant,
            SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(output_dtype),
        )
    return [C, Cs] if requant else [C]


@compile_time_only_triton_op(
    add_op_namespace_prefix("full_precision_matmul_batched"), mutates_args=()
)
def full_precision_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    expert_ids: torch.Tensor,
    gate: bool = False,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    requant: bool = False,
    simulate_unfused: bool = False,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Full-precision (BF16/FP16) batched matmul: C[s] = A[s] @ B[expert_ids[s]].T — no
    quantization anywhere, fp32 accumulation. ``gate``/``act_fn``/``swiglu_*``/
    ``simulate_unfused`` are the flattened ``Epilogue`` (GLU only; ``requant`` is
    meaningless without a quantized recipe). ``gather_idx``/``scatter_idx`` map the source
    row of A / destination row of C per program (None = row s).

    A:  (rows, K) BF16/FP16 activations — rows addressed via ``gather_idx``
    B:  (num_experts, N, K) expert weights in A's dtype; under ``gate`` the (num_experts, 2N, K) stack
    """
    assert A.ndim == 2, f"A must be 2D (rows, K), got ndim={A.ndim}"
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 3, f"B must be 3D (num_experts, N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"
    assert A.shape[1] == B.shape[2], (
        f"K mismatch: A has K={A.shape[1]}, B has K={B.shape[2]}"
    )
    assert A.dtype == B.dtype and A.dtype in (torch.bfloat16, torch.float16), (
        f"full-precision path needs matching BF16/FP16 A and B, got {A.dtype} / {B.dtype}"
    )
    assert not requant, "requant is meaningless on the full-precision path (no quantized recipe)"

    output_dtype = resolve_output_dtype(output_dtype, A, None)
    K = A.shape[1]
    S = expert_ids.shape[0]
    # Under a gate epilogue B is the (E, 2N, K) gate|up stack — N is the per-projection width.
    num_experts, n_rows, _ = B.shape
    N = n_rows // 2 if gate else n_rows
    C = A.new_empty(S, N, dtype=output_dtype)

    def grid(META):
        return (S, triton.cdiv(N, META["BLOCK_SIZE_N"]))

    with device_context(A.device):
        compile_time_only_triton_wrap(full_precision_matmul_batched_kernel)[grid](
            A,
            B,
            C,
            expert_ids,
            gather_idx if gather_idx is not None else expert_ids,
            scatter_idx if scatter_idx is not None else expert_ids,
            S,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            expert_ids.stride(0),
            num_experts=num_experts,
            HAS_GATHER=gather_idx is not None,
            HAS_SCATTER=scatter_idx is not None,
            GATE=gate,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(output_dtype),
        )

    return [C]


def matmul_batched(
    A: torch.Tensor,
    As: torch.Tensor | None,
    B: torch.Tensor,
    Bs: torch.Tensor | None,
    expert_ids: torch.Tensor,
    block_size: list[int] | None = None,
    epilogue: Epilogue | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Batched matmul dispatcher (W8A8 FP8, W4A8 FP4, or full-precision). Routes one program per routed
    row (``expert_ids`` gives its expert). ``A`` may be raw (``As`` None) — the op quantizes it
    itself (offline for bd/tensor, inline for mx) — or pre-quantized (``As`` given, e.g. a
    requantized intermediate handed to the down projection). ``epilogue`` optionally fuses the
    gate|up projection and carries ``output_dtype`` (see ``Epilogue``); under a requant epilogue
    the return is the requantized output plus its scale tensor. Gate|up fusion is block-dynamic
    and MX only. ``gather_idx``/``scatter_idx`` (each None or a ``(S,)`` map) address the source
    row of ``A`` / destination row of ``C`` per program — None means row ``s``; the gather lets
    the gate_up read unexpanded activations (source row ``s // num_top_k``) with no copy.

    Routes by weight dtype and ``block_size``:
    - ``Bs`` None — unquantized BF16/FP16 weights (``As`` must also be None) →
      ``full_precision_matmul_batched`` (plain dot, no scales; GLU-only epilogue).
    - MX weights — ``int8`` (packed E2M1) or ``float8_e4m3fn`` (E4M3) with UE8M0
      group-32 ``Bs`` (shape ``[num_experts, N, K//32]``) → ``mxfp_dynamic_matmul_batched``
      (``block_size`` ignored; tile + dot path autotuned).
    - ``block_size`` None or full ``[N, K]`` → ``w8a8_tensor_dynamic_fp8_matmul_batched``.
    - otherwise → ``w8a8_block_dynamic_fp8_matmul_batched``.
    """
    ep = epilogue if epilogue is not None else Epilogue()
    ep_args = ep.as_args()

    if Bs is None:
        assert As is None, "full-precision path (Bs=None) takes raw activations — As must be None"
        out = full_precision_matmul_batched(
            A, B, expert_ids, *ep_args, gather_idx, scatter_idx
        )
    elif is_mxfp(B, Bs):
        out = mxfp_dynamic_matmul_batched(
            A, As, B, Bs, expert_ids, *ep_args, gather_idx, scatter_idx
        )
    elif is_tensor_wide(block_size, B):
        assert not ep.gate, "gate|up fusion is not supported for tensor-wide scales"
        out = w8a8_tensor_dynamic_fp8_matmul_batched(
            A, As, B, Bs, expert_ids, ep.output_dtype, gather_idx, scatter_idx
        )
    else:
        out = w8a8_block_dynamic_fp8_matmul_batched(
            A, As, B, Bs, expert_ids, block_size, *ep_args, gather_idx, scatter_idx
        )
    # bd/mx ops return a list ([C] or [C, Cs]); the tensor op returns a bare tensor.
    if isinstance(out, (list, tuple)):
        return out[0] if len(out) == 1 else tuple(out)
    return out
