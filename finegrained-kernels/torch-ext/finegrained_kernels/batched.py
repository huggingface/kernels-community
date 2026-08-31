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

from triton.tools.tensor_descriptor import TensorDescriptor

from .bayesian_autotuner import bayesian_autotune
from .compat import FP8_DTYPE, MX_SCALE_GROUP_K, NIBBLES_PER_BYTE, compile_time_only_triton_op, compile_time_only_triton_wrap, device_context, get_accelerator_autotuning_configs, tl_dtype
from .recipes import Epilogue, Quantization, combine_global_scales, normalize_global_scale, e2m1_as_uint8, expert_weight_shape, is_mx, mx_scale_family, normalize_per_expert_scale, resolve_input_recipe, resolve_output_dtype, resolve_output_recipe, ue8m0_as_uint8, validate_dense_operands, weight_block_size, weight_recipe
from .epilogue import fused_glu
from .quant import MX_ACT_QUANT, fp8_act_quant_block_dynamic, fp8_act_quant_tensor_wide
from .mma import block_dynamic_dot, fp8_dot, mx_compute, mx_weight_only_compute, static_dot
from .tiles import (
    advance_ptrs,
    load_act_block_dynamic,
    load_act_mx,
    load_act_plain,
    load_act_static,
    load_weight_block_dynamic,
    load_weight_mx,
    load_weight_plain,
    load_weight_static,
    operand_tile_ptrs,
    oriented_tile_ptrs,
    weight_tile_ptrs,
)
from .epilogue import acc_finalize, acc_init, add_bias, bias_strides, gemm_epilogue
from .pruners import PATH_ANCHOR_AXES, dot_scaled_staging_pruner, block_fits_dim_pruner, block_within_dim_pruner, compose_pruners, gate_tile_cap_pruner, mx_config_pruner, require_moe_dims_aligned, scale_subblock_pruner, smem_pruner, swizzled_scale_config_pruner, weight_only_swap_scope_pruner


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
    ADVANCE_BS: tl.constexpr = True,
):
    """Per-(row, expert) prologue shared by the batched kernels: read the program
    ids, look up the routed expert, and advance the A/B/C/Bs base pointers to this
    row's slice. Returns ``(batch_id, pid_n, expert_id, A, B, C, Bs, in_row, out_row)`` — the
    resolved source/destination rows fold the gather/scatter out of the kernel bodies.

    ``ADVANCE_BS=False`` leaves ``Bs`` at the buffer base (the mx scale leaf applies the expert
    offset itself — its swizzled path indexes by 128-row block, not the row-major expert stride).

    ``A``'s source row is ``GatherIdx[batch_id]`` when ``GatherIdx`` is not None (the gate_up reading
    unexpanded activations, many-to-one for top_k > 1) else ``batch_id``; ``C``'s destination
    row is ``ScatterIdx[batch_id]`` when ``ScatterIdx`` is not None else ``batch_id`` — the same virtual
    gather/scatter ``matmul_grouped`` does, so the routed rows need no materialized copy.

    The caller must early-return on the EP sentinel (``expert_id >= num_experts``)
    before any load — the pointer arithmetic itself is harmless, only the loads on a
    non-local expert would be out of bounds."""
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    # Cast to int64 to prevent overflow on expert_id * stride_b_e.
    expert_id = tl.load(ExpertIds + batch_id * stride_eid).to(tl.int64)
    in_row = tl.load(GatherIdx + batch_id).to(tl.int64) if GatherIdx is not None else batch_id
    out_row = tl.load(ScatterIdx + batch_id).to(tl.int64) if ScatterIdx is not None else batch_id
    A = A + in_row * stride_a_m
    B = B + expert_id * stride_b_e
    C = C + out_row * stride_c_m
    if ADVANCE_BS:
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
    # offs_cm[:, None] * 0: broadcast to a [BM, BN] pointer tile (all rows alias the one C row)
    # so the lane-0 mask below has a row axis to select; the M stride is deliberately 0.
    c_ptrs = C + offs_cm[:, None] * 0 + stride_c_n * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm == 0)[:, None])


@bayesian_autotune(
    get_accelerator_autotuning_configs(swap_ab=True, tune_block_n=True),
    # one winner per (shape, requant): requant narrows the legal tiles to the quant block, so a
    # shared entry could replay a sub-block winner on a launch whose output scale needs the block.
    # GATE stacks a 2*BN-wide dot — a distinct config space that must not share a winner.
    # BLOCK_N/BLOCK_K (the launch-pinned quant block) bound the legal tiles the same way —
    # two checkpoints at one shape with different blocks must not share a winner.
    ["N", "K", "S", "OUTPUT_RECIPE", "GATE", "BLOCK_N", "BLOCK_K"],
    n_trials=100,
    path_anchor_axes=PATH_ANCHOR_AXES,
    prune_configs_by={"early_config_prune": scale_subblock_pruner()},
)
@triton.jit
def w8a8_block_dynamic_fp8_matmul_batched_kernel(
    A,  # (S, K) E4M3 activations (pre-quantized once by the wrapper)
    As,  # (S, K // BLOCK_SIZE_K) fp32 per-row, per-K-block activation scales
    B,  # (num_experts, N, K) FP8 weights; under GATE the (num_experts, 2N, K) gate|up stack
    Bs,  # (num_experts, N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales (2N under GATE)
    C,  # (S, N) output; under an OUTPUT_RECIPE the FP8-requantized intermediate
    Cs,  # (S, N // BLOCK_SIZE_N) per-(row, block) output scale; written iff OUTPUT_RECIPE
    Bias,  # (E, N_out) per-expert output bias, N_out = 2N under GATE; read iff not None
    ExpertIds,  # (S,) — which expert each batch element routes to
    GatherIdx,  # (S,) int — batch_id -> source row of A; read only when not None
    ScatterIdx,  # (S,) int — batch_id -> destination row of C; read only when not None
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
    stride_bias_e,
    stride_bias_n,
    stride_eid,
    num_experts,
    # Meta-parameters. BLOCK_N/BLOCK_K are the QUANT block (one scale per (BLOCK_N, BLOCK_K) tile of
    # the weight); BLOCK_SIZE_* are the tuned COMPUTE tile. The N tile may subdivide the quant block
    # (BLOCK_SIZE_N <= BLOCK_N); the K tile is the quant block (one scale step per K iteration).
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
    # Gate|up fusion epilogue (GATE=False -> plain batched GEMM, every arm below folds out)
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    # the output recipe name, same vocabulary as Quantization (None | "fp8")
    OUTPUT_RECIPE: tl.constexpr = None,
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

    ``GATE`` fuses the gate|up projection: ``B`` is the ``(E, 2N, K)`` gate|up weight with the
    two projections INTERLEAVED per row (gate even, up odd — ``split_gate_up`` is the inverse),
    run as two dots (the decode-validated form), SwiGLU-combined, and — under
    an ``OUTPUT_RECIPE`` — FP8-requantized into ``C`` + a per-(row, block) scalar ``Cs``. Every gate arm
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
    )
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= num_experts:
        return

    # One scale per quant block, broadcast over the tile, so an N tile narrower than BLOCK_N just
    # reads its own block: tile pid_n sits in block (pid_n * BLOCK_SIZE_N) // BLOCK_N, which is pid_n
    # when the tile IS the block. Narrower tiles multiply the N grid (see scale_subblock_pruner).
    n_width: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
    offs_bn = pid_n * n_width + tl.arange(0, n_width)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = operand_tile_ptrs(A, tl.arange(0, BLOCK_SIZE_M) * 0, offs_k, stride_a_m, stride_a_k, "pointer", True)
    as_ptrs = As + in_row * stride_as_m + tl.zeros((BLOCK_SIZE_M,), tl.int32)
    # One gate|up weight tile + one block-scale pointer, like every other kernel: the n_width
    # span shares a tile + a single dot, each row's scale block following from its global row.
    b_ptrs = weight_tile_ptrs(B, offs_bn, offs_k, stride_b_n, stride_b_k, SWAP_AB)
    bs_ptr = Bs + (pid_n * n_width // BLOCK_N) * stride_bs_n
    bs_off = ((pid_n * n_width % BLOCK_N + tl.arange(0, n_width)) // BLOCK_N) * stride_bs_n
    acc = acc_init("dot", BLOCK_SIZE_M, n_width, SWAP_AB)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a, a_s = load_act_block_dynamic(
            a_ptrs, as_ptrs, None, None, 0, 0, 0, 0, 0, "pointer", False, False, False
        )
        w, b_s = load_weight_block_dynamic(
            b_ptrs, b_ptrs, bs_ptr + bs_off, None, 0, 0, 0, 0, 0, 0,
            GATE, False, "pointer", SWAP_AB, BLOCK_SIZE_N, BLOCK_SIZE_K,
        )
        acc = block_dynamic_dot(acc, a, a_s, w, b_s, BLOCK_SIZE_K, SWAP_AB, False, True)
        a_ptrs, as_ptrs, b_ptrs, bs_ptr, _, _ = advance_ptrs(
            a_ptrs, as_ptrs, b_ptrs, bs_ptr, b_ptrs, bs_ptr,
            BLOCK_SIZE_K * stride_a_k, 1, BLOCK_SIZE_K * stride_b_k, stride_bs_k,
            "pointer", "pointer", True, True, False,
        )

    gemm_epilogue(
        C, Cs, acc, out_row, pid_n, 0, out_row, 1, stride_c_n, stride_cs_m, stride_cs_n,
        BLOCK_SIZE_M, BLOCK_SIZE_N, GATE, OUTPUT_RECIPE, BLOCK_K,
        ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE,
        COMPUTE_MODE="dot", SWAP_AB=SWAP_AB, FAKE_BATCH=True,
        Bias=Bias, stride_bias_e=stride_bias_e, stride_bias_n=stride_bias_n,
        global_row=expert_id,
    )


@bayesian_autotune(
    get_accelerator_autotuning_configs(swap_ab=True, tune_block_n=True),
    # keyed like the block-dynamic sibling: requant narrows the legal tiles to the quant block
    ["N", "K", "S", "OUTPUT_RECIPE", "GATE", "BLOCK_N", "BLOCK_K"],
    n_trials=100,
    path_anchor_axes=PATH_ANCHOR_AXES,
    prune_configs_by={"early_config_prune": scale_subblock_pruner()},
)
@triton.jit
def w8a8_block_static_fp8_matmul_batched_kernel(
    A,  # (S, K) E4M3 activations (pre-quantized against the static scale by the wrapper)
    As,  # scalar — static per-tensor activation scale (calibration-time)
    B,  # (num_experts, N, K) FP8 weights; under GATE the (num_experts, 2N, K) gate|up stack
    Bs,  # (num_experts, N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales (2N under GATE)
    C,  # (S, N) output; under an OUTPUT_RECIPE the FP8-requantized intermediate
    Cs,  # (S, N // BLOCK_SIZE_N) per-(row, block) output scale; written iff OUTPUT_RECIPE
    Bias,  # (E, N_out) per-expert output bias, N_out = 2N under GATE; read iff not None
    ExpertIds,  # (S,) — which expert each batch element routes to
    GatherIdx,  # (S,) int — batch_id -> source row of A; read only when not None
    ScatterIdx,  # (S,) int — batch_id -> destination row of C; read only when not None
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
    stride_bs_e,
    stride_bs_k,
    stride_bs_n,
    stride_c_m,
    stride_c_n,
    stride_cs_m,
    stride_cs_n,
    stride_bias_e,
    stride_bias_n,
    stride_eid,
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
    # Gate|up fusion epilogue (GATE=False -> plain batched GEMM, every arm below folds out)
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    OUTPUT_RECIPE: tl.constexpr = None,  # None | "fp8" (per-(row, block) requant of the intermediate)
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
):
    """Block-scale batched FP8 expert matmul with a static (per-tensor) activation scale — the
    block-dynamic batched sibling (one program per routed token + N-tile, fake-batch decode,
    ``SWAP_AB``, ``GATE`` gate|up fusion) with the 2D ``block_static`` recipe: ``A`` arrives
    pre-quantized against the calibrated scalar, per-block weight scales apply per-K-tile
    (``accumulate`` ``"static"``, ``FAKE_BATCH``), and the scalar activation scale multiplies the
    accumulator once after the loop. bf16 GLU output only (no fused requant). GATE=False is the plain GEMM."""
    a_s_static = tl.load(As)  # per-tensor static activation scale, applied post-loop
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
    )
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= num_experts:
        return

    # the N tile may subdivide the quant block — see the dynamic sibling / scale_subblock_pruner
    n_width: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
    offs_bn = pid_n * n_width + tl.arange(0, n_width)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = operand_tile_ptrs(A, tl.arange(0, BLOCK_SIZE_M) * 0, offs_k, stride_a_m, stride_a_k, "pointer", True)
    # One gate|up weight tile over the interleaved rows + one block-scale pointer;
    # each weight row's scale block follows from its global row.
    b_ptrs = weight_tile_ptrs(B, offs_bn, offs_k, stride_b_n, stride_b_k, SWAP_AB)
    bs_ptr = Bs + (pid_n * n_width // BLOCK_N) * stride_bs_n
    bs_off = ((pid_n * n_width % BLOCK_N + tl.arange(0, n_width)) // BLOCK_N) * stride_bs_n
    acc = acc_init("dot", BLOCK_SIZE_M, n_width, SWAP_AB)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a, _ = load_act_static(a_ptrs, 0, 0, 0, None, 0, 0.0, "pointer", False)  # pre-quantized E4M3 token (fake-batch replicated)
        w, b_s = load_weight_static(
            b_ptrs, b_ptrs, bs_ptr + bs_off, None, 0, 0, 0, 0, 0, 0,
            GATE, False, "pointer", SWAP_AB, BLOCK_SIZE_N, BLOCK_SIZE_K,
        )
        acc = static_dot(acc, a, w, b_s, SWAP_AB, BLOCK_SIZE_K, True)
        a_ptrs, _, b_ptrs, bs_ptr, _, _ = advance_ptrs(
            a_ptrs, a_ptrs, b_ptrs, bs_ptr, b_ptrs, bs_ptr,
            BLOCK_SIZE_K * stride_a_k, 0, BLOCK_SIZE_K * stride_b_k, stride_bs_k,
            "pointer", "pointer", False, True, False,
        )

    acc = acc * a_s_static
    gemm_epilogue(
        C, Cs, acc, out_row, pid_n, 0, out_row, 1, stride_c_n, stride_cs_m, stride_cs_n,
        BLOCK_SIZE_M, BLOCK_SIZE_N, GATE, OUTPUT_RECIPE, BLOCK_K,
        ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE,
        COMPUTE_MODE="dot", SWAP_AB=SWAP_AB, FAKE_BATCH=True,
        Bias=Bias, stride_bias_e=stride_bias_e, stride_bias_n=stride_bias_n,
        global_row=expert_id,
    )


@bayesian_autotune(
    # S (routed rows) keyed like the block-dynamic/mx batched siblings — decode re-tunes per batch.
    get_accelerator_autotuning_configs(tune_block_nk=True, swap_ab=True),
    ["N", "K", "S"],
    n_trials=100,
    path_anchor_axes=PATH_ANCHOR_AXES,
    # BLOCK_SIZE_K/N are tuned axes; the K-loop is maskless and the N-tile store is
    # row-masked only — veto non-dividing tiles on both.
    prune_configs_by={
        "early_config_prune": compose_pruners(
            block_within_dim_pruner("K"),
            block_within_dim_pruner("N", "BLOCK_SIZE_N"),
        )
    },
)
@triton.jit
def w8a8_tensor_dynamic_fp8_matmul_batched_kernel(
    A,  # (S, K) pre-quantized FP8 activations
    As,  # (S,) per-token activation scales
    B,  # (num_experts, N, K) FP8 weight matrices
    Bs,  # (num_experts, 1, 1) per-tensor weight scales
    C,  # (S, N) output
    Bias,  # (E, N_out) per-expert output bias, N_out = 2N under GATE; read iff not None
    ExpertIds,  # (S,) — which expert each batch element routes to
    GatherIdx,  # (S,) int — batch_id -> source row of A; read only when not None
    ScatterIdx,  # (S,) int — batch_id -> destination row of C; read only when not None
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
    stride_bias_e,
    stride_bias_n,
    stride_eid,
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
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
    )
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= num_experts:
        return

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = operand_tile_ptrs(A, tl.arange(0, BLOCK_SIZE_M) * 0, offs_k, stride_a_m, stride_a_k, "pointer", True)
    b_ptrs = oriented_tile_ptrs(B, offs_bn, offs_k, stride_b_n, stride_b_k, SWAP_AB)
    b_s = tl.load(Bs)
    a_s = tl.load(As + in_row * stride_as_m)

    accumulator = acc_init("dot", BLOCK_SIZE_M, BLOCK_SIZE_N, SWAP_AB)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a, _ = load_act_plain(a_ptrs, 0, 0, 0, None, 0, "pointer", False)
        b, _ = load_weight_plain(
            b_ptrs, b_ptrs, 0, 0, 0, False, False, "pointer", SWAP_AB, BLOCK_SIZE_N, BLOCK_SIZE_K
        )
        accumulator = accumulator + fp8_dot(a, b, SWAP_AB, BLOCK_SIZE_K)
        a_ptrs, _, b_ptrs, _, _, _ = advance_ptrs(
            a_ptrs, a_ptrs, b_ptrs, b_ptrs, b_ptrs, b_ptrs,
            BLOCK_SIZE_K * stride_a_k, 0, BLOCK_SIZE_K * stride_b_k, 0,
            "pointer", "pointer", False, False, False,
        )

    accumulator = acc_finalize(accumulator, "dot", BLOCK_SIZE_N, SWAP_AB) * a_s * b_s
    # this split keeps its own store path (ungated, per-tensor dequant) but takes the same bias
    accumulator = add_bias(
        accumulator, Bias, stride_bias_e, stride_bias_n, expert_id, pid_n, BLOCK_SIZE_N
    )
    store_row(C, accumulator, pid_n, stride_c_n, BLOCK_SIZE_M, BLOCK_SIZE_N)


# The MXFP4/MXFP8 (and packed-activation) splits key themselves — the tuner appends every tensor
# arg's dtype to its cache key. BLOCK_SIZE_M is always 1 here (per-token decode), so plain `dot`
# is excluded (only scalar / dot_scaled-swap are emitted); the swapped dot helper stays
# implemented for future shapes but is not fielded. Swap verdicts are B200 (sm_100) — re-measure
# on H100 or the target device before inheriting.
def _rebind_batched_mx_bs_descriptor(nargs):
    """Per-config pre_hook: size the swizzled weight-scale descriptor box to the tile's 128-row
    blocks (doubled under GATE, whose tile spans 2*BN interleaved rows). BN<128 (fp8 scalar)
    pointer-gathers instead and never reads the descriptor. Only under SWIZZLED_SCALES; the
    un-swizzled path keeps its dummy box."""
    if not nargs.get("SWIZZLED_SCALES"):
        return
    rep = max(1, ((2 if nargs.get("GATE") else 1) * nargs["BLOCK_SIZE_N"]) // 128)
    rep_k = (nargs["BLOCK_SIZE_K"] // nargs["SCALE_GROUP_K"]) // 4
    nargs["BSDescriptor"].block_shape = [1, rep, rep_k, 2, 256]


@bayesian_autotune(
    get_accelerator_autotuning_configs(
        mx=True,
        tune_block_nk=True,
        compute_modes=("dot_scaled", "scalar"),
        swap_ab=True,
        pre_hook=_rebind_batched_mx_bs_descriptor,
    ),
    # INPUT_RECIPE keys the inline act-quant grid: A stays raw bf16 under every
    # recipe, so the tuner's dtype-appended key can't split W4A8 from W4A4 itself.
    # SWIZZLED_SCALES keys the weight-scale load: it constrains the config space (BK % 128 == 0)
    # and picks a different optimum (full-block descriptor vs the pointer scale), so the swizzled
    # and un-swizzled launches of one shape must not share a tune.
    # GATE keys the tune AND the GATE-conditional prunes (the packed-fp4 silent-zeros
    # fence in mx_config_pruner runs once per key, so a GATE=False-first tune must not
    # hand its winner to a GATE=True launch at the same shape).
    # OUTPUT_RECIPE keys the requant epilogue explicitly (the C/Cs dtype append splits it
    # incidentally today; this kernel's requant Cs is always row-major — no SWIZZLED_OUT axis).
    ["N", "K", "S", "INPUT_RECIPE", "SWIZZLED_SCALES", "GATE", "OUTPUT_RECIPE"],
    n_trials=100,
    path_anchor_axes=PATH_ANCHOR_AXES,
    # BK-within-K + the sm_10x MMA-shape guards (swapped dot_scaled needs BN >= 128 for the
    # native scaled-MMA; smaller-BN swap configs never win and mislead the TPE).
    prune_configs_by={
        "early_config_prune": compose_pruners(
            mx_config_pruner("K", "N"), swizzled_scale_config_pruner(allow_gate_subblock=True), smem_pruner(),
            gate_tile_cap_pruner(),
            dot_scaled_staging_pruner(),
        )
    },
)
@triton.jit
def mx_dynamic_matmul_batched_kernel(
    A,  # (S, K) activations: raw BF16/FP16 (inline-quant) or E4M3 (pre-quantized, As set)
    As,  # (S, K // SCALE_GROUP_K) UE8M0 act scales; None ⇒ inline-quant, read iff not None
    B,  # (num_experts, N, K[/2]); under GATE the (num_experts, 2N, K[/2]) gate|up stack
    Bs,  # (num_experts, N, K // SCALE_GROUP_K) UE8M0 weight scales (2N under GATE)
    BSDescriptor,  # host TMA descriptor over Bs when SWIZZLED (BN=128 bulk load); dummy otherwise
    C,  # (S, N[/2]) output; under an OUTPUT_RECIPE the MX-requantized intermediate
    Cs,  # (S, N // SCALE_GROUP_K) UE8M0 output scale; written iff OUTPUT_RECIPE
    Bias,  # (E, N_out) per-expert output bias, N_out = 2N under GATE; read iff not None
    AsGlobal,  # (1,) fp32 NVFP4 activation global g_a — SOLELY normalizes the inline raw-A quant (A/g_a); read iff not None
    AsBsGlobal,  # (num_experts,) fp32 NVFP4 combined global g_a·g_b — recovers on the accumulator (one multiply); read iff not None
    CsGlobal,  # (1,) fp32 NVFP4 output global (next proj's provided input_scale); normalizes the requant; read iff not None
    ExpertIds,  # (S,) — which expert each routed row uses
    GatherIdx,  # (S,) int — batch_id -> source row of A; read only when not None
    ScatterIdx,  # (S,) int — batch_id -> destination row of C; read only when not None
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
    stride_bias_e,
    stride_bias_n,
    stride_eid,
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
    # Gate|up fusion epilogue (GATE=False -> plain batched GEMM, every arm below folds out)
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    # the output recipe name, same vocabulary as Quantization (None | "mxfp8" | "mxfp4" | "nvfp4")
    OUTPUT_RECIPE: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
    INPUT_RECIPE: tl.constexpr = "mxfp8",
    # SWIZZLED_SCALES: Bs arrives pre-swizzled (SWIZZLE_32_4_4) — the checkpoint stores one layout,
    # shared with the grouped (prefill) kernel. Read via load_weight's per-expert scale leaf off the
    # single Bs pointer (+ BSDescriptor for the BN=128 bulk load); un-swizzled Bs takes the affine
    # arm in the same leaf. The op never swizzles — a 3D caller runs un-swizzled at no penalty.
    SWIZZLED_SCALES: tl.constexpr = False,
):
    """Unified batched microscaled expert matmul (MXFP8/MXFP4/NVFP4, W4A8/W4A4) with
    fused act quant.

    One routed row + one N-tile per program; expert looked up from ``ExpertIds``. ``A`` is
    quantized to E4M3 per K-group inline (UE8M0 scale). The weight dtype picks the
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
        ADVANCE_BS=False,  # scale leaf applies the per-expert offset (swizzled indexes by block)
    )
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= num_experts:
        return

    # each operand's format is its dtype: uint8 = packed E2M1 (two values per byte, W4A4
    # for A / MXFP4 for B — it also keys the autotune cache), else E4M3
    ACT_VALUES_PER_BYTE: tl.constexpr = 2 if A.dtype.element_ty == tl.uint8 else 1
    WEIGHT_VALUES_PER_BYTE: tl.constexpr = 2 if B.dtype.element_ty == tl.uint8 else 1
    n_width: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
    # Non-128 N: the partial last N-tile's pointer rows wrap into B (offs_bn % N) so the load
    # never reads past the expert's N rows; the wrapped columns' output is masked off (N_COLS)
    # in the epilogue. Inert when N % BLOCK_SIZE_N == 0 (the affine arm's BN|N veto), so it is
    # load-bearing only for the swizzled arm, whose scale rides pid_n's block index, not offs_bn.
    offs_bn = (pid_n * n_width + tl.arange(0, n_width)) % (2 * N if GATE else N)
    offs_kb = tl.arange(0, BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)
    offs_ka = tl.arange(0, BLOCK_SIZE_K // ACT_VALUES_PER_BYTE)
    a_ptrs = operand_tile_ptrs(A, tl.arange(0, BLOCK_SIZE_M) * 0, offs_ka, stride_a_m, stride_a_k, "pointer", True)
    # As is not None ⇒ pre-quantized: A is E4M3 + As UE8M0 (the down reading a requantized
    # intermediate). Else A is raw, quantized inline onto INPUT_RECIPE's grid (gate_up / plain —
    # packed E2M1 under fp4, one act row per program so the quant is decode-free); As stays None.
    if As is not None:  # build the scale pointers only when the scale is read
        as_ptrs = (
            As
            + in_row * stride_as_m
            + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0
            + offs_sf[None, :]
        )
    else:
        as_ptrs = a_ptrs  # dead placeholder so advance_ptrs can take it unconditionally
    # GATE reads one weight tile spanning the interleaved gate|up rows, oriented by
    # SWAP_AB; load_weight reads value + scale (swizzled/un-swizzled hidden) off these pointers.
    b_ptrs = weight_tile_ptrs(
        B, offs_bn, offs_kb, stride_b_n, stride_b_k, SWAP_AB
    )
    accumulator = acc_init(COMPUTE_MODE, BLOCK_SIZE_M, n_width, SWAP_AB)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a, a_scale = load_act_mx(
            a_ptrs, as_ptrs, AsGlobal, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            "pointer", False, False, False,
            BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K, INPUT_RECIPE,
        )
        b, b_s = load_weight_mx(
            b_ptrs, b_ptrs, Bs, None, BSDescriptor, Bs, 0, 0, 0, 0, expert_id, pid_n, k, N, K,
            stride_bs_e, stride_bs_n, stride_bs_k,
            GATE, False, True, "pointer", SWAP_AB, SWIZZLED_SCALES,
            BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, WEIGHT_VALUES_PER_BYTE,
        )
        accumulator = mx_compute(
            accumulator, a, a_scale, b, b_s, COMPUTE_MODE,
            BLOCK_SIZE_M, n_width, BLOCK_SIZE_K, SCALE_GROUP_K, SWAP_AB,
        )
        a_ptrs, as_ptrs, b_ptrs, _, _, _ = advance_ptrs(
            a_ptrs, as_ptrs, b_ptrs, b_ptrs, b_ptrs, b_ptrs,
            (BLOCK_SIZE_K // ACT_VALUES_PER_BYTE) * stride_a_k,
            BLOCK_SIZE_K // SCALE_GROUP_K,
            (BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE) * stride_b_k,
            0,
            "pointer", "pointer", As is not None, False, False,
        )

    # NVFP4 two-level: block e4m3 scales rode through the reduce; recover the combined per-tensor
    gemm_epilogue(
        C, Cs, accumulator, out_row, pid_n, 0, out_row, 1, stride_c_n, stride_cs_m, stride_cs_n,
        BLOCK_SIZE_M, BLOCK_SIZE_N, GATE, OUTPUT_RECIPE, SCALE_GROUP_K,
        ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE,
        COMPUTE_MODE=COMPUTE_MODE, SWAP_AB=SWAP_AB, FAKE_BATCH=True, N_COLS=N,
        CsGlobal=CsGlobal,
        GlobalScale=AsBsGlobal,
        global_row=expert_id,
        Bias=Bias,
        stride_bias_e=stride_bias_e,
        stride_bias_n=stride_bias_n,
    )


@bayesian_autotune(
    # weight-only plain bf16 dot (fp4/fp8 weight upcast per-group in-loop), plus the swapped
    # decode reduce (weight output rows lead the tile, no MMA — see the scope pruner).
    get_accelerator_autotuning_configs(
        tune_block_nk=True, compute_modes=("dot", "dot_scaled", "scalar"), swap_ab=True
    ),
    ["N", "K", "S", "GATE"],
    n_trials=100,
    path_anchor_axes=PATH_ANCHOR_AXES,
    prune_configs_by={
        # No BK-within-K veto: this kernel's tail tile slides back in bounds (see the K-loop),
        # so any BK the shape can hold is legal — that is the point, it lets a K=2880 decode
        # take a 256-byte burst instead of 32. BK must still FIT (the clamp needs K >= BK).
        "early_config_prune": compose_pruners(
            block_fits_dim_pruner("K"),
            block_within_dim_pruner("N", "BLOCK_SIZE_N"),
            mx_config_pruner("K", "N", block_within_k=False),  # dot_scaled shape gates
            weight_only_swap_scope_pruner(),
        )
    },
)
@triton.jit
def mx_weight_only_matmul_batched_kernel(
    A,  # (rows, K) raw BF16/FP16 activations — NOT quantized
    B,  # (num_experts, N, K[/2]) MXFP4/NVFP4/MXFP8 weights; 2N under GATE
    Bs,  # (num_experts, N, K // SCALE_GROUP_K) group scales — UE8M0 (MX) or E4M3 (NVFP4)
    BsGlobal,  # (num_experts,) fp32 NVFP4 per-expert global — recovers on the accumulator; read iff not None
    C,
    Bias,  # (E, N_out) per-expert output bias, N_out = 2N under GATE; read iff not None
    ExpertIds,
    GatherIdx,
    ScatterIdx,
    S,
    N,
    K,
    stride_a_m,
    stride_a_k,
    stride_b_e,
    stride_b_k,
    stride_b_n,
    stride_bs_e,
    stride_bs_n,
    stride_bs_k,
    stride_c_m,
    stride_c_n,
    stride_bias_e,
    stride_bias_n,
    stride_eid,
    num_experts,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    WEIGHT_VALUES_PER_BYTE: tl.constexpr,
    COMPUTE_MODE: tl.constexpr = "dot",
    SWAP_AB: tl.constexpr = False,
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
):
    """weight-only batched (decode) expert matmul: raw bf16 activations against MXFP4/MXFP8 weights upcast
    to bf16 in-loop (unpack + per-group group-scale), plain ``tl.dot``. One routed row + one N-tile
    per program (expert from ``ExpertIds``). Pointer/affine. ``SWAP_AB`` (tuner axis, M=1 decode):
    weight output rows lead the tile and the CUDA-core reduce replaces the MMA. ``GATE`` fuses gate|up."""
    batch_id, pid_n, expert_id, A, B, C, Bs, in_row, out_row = expert_setup(
        A, B, C, Bs, ExpertIds, GatherIdx, ScatterIdx,
        stride_a_m, stride_b_e, stride_c_m, stride_bs_e, stride_eid, ADVANCE_BS=False,
    )
    if expert_id >= num_experts:  # EP sentinel: non-local expert, output left uninit
        return
    n_width: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
    # non-128 N: the last N-tile's rows run past B (N=320, n_width=256 -> tile 2 wants rows
    # 512..767 of 640), so wrap them back into the tensor. The wrapped columns are masked off by
    # the epilogue's N_COLS. An unwrapped read is out of bounds and faults once the allocation
    # layout puts an unmapped page there — invisible in a short run, an illegal access in a long one.
    offs_bn = (pid_n * n_width + tl.arange(0, n_width)) % (2 * N if GATE else N)
    offs_kb = tl.arange(0, BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE)
    a_ptrs = operand_tile_ptrs(
        A, tl.arange(0, BLOCK_SIZE_M) * 0, tl.arange(0, BLOCK_SIZE_K),
        stride_a_m, stride_a_k, "pointer", True,
    )
    b_ptrs = weight_tile_ptrs(
        B, offs_bn, offs_kb, stride_b_n, stride_b_k, SWAP_AB
    )
    accumulator = acc_init(COMPUTE_MODE, BLOCK_SIZE_M, n_width, SWAP_AB)
    # Each trip indexes off the BASE tile by the scalar ``k_start`` rather than advancing a
    # pointer tile in place: an advanced tile stays live across the loop, and at [2*BN, BK/2]
    # int64 that is ~128 registers per thread of addresses alone — enough to halve resident
    # CTAs and starve the weight stream (measured n_regs 128 with spills vs 55 for the same
    # math indexed from a base). The clamp below already computes ``k_start``, so the walk is
    # the same addresses either way.
    # The K-loop is maskless, which normally pins BK to a divisor of K — leaving a
    # non-power-of-two K (gpt-oss 2880) with BK=64, a 32-byte burst per weight row over 45
    # trips. Instead the LAST tile slides back to end at K (``k_start`` clamp): every load
    # stays in bounds, the tile re-reads positions the previous trip already accumulated, and
    # zeroing those activation lanes cancels them for every arm (a zero operand contributes
    # nothing to the dot, the scaled MMA, or the reduce). BK | K makes the clamp a no-op, so
    # aligned shapes keep today's pointer walk bit-for-bit.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = tl.minimum(k * BLOCK_SIZE_K, K - BLOCK_SIZE_K)
        overlap = k * BLOCK_SIZE_K - k_start
        a, _ = load_act_plain(
            a_ptrs + k_start * stride_a_k, 0, 0, 0, None, 0, "pointer", False
        )
        if overlap > 0:  # only the slid tail trip pays the select
            a = tl.where(tl.arange(0, BLOCK_SIZE_K)[None, :] >= overlap, a, 0.0)
        b_k = b_ptrs + (k_start // WEIGHT_VALUES_PER_BYTE) * stride_b_k
        w, w_s = load_weight_mx(
            b_k, b_k, Bs, None, None, Bs, 0, 0, 0, 0, expert_id, pid_n, k, N, K,
            stride_bs_e, stride_bs_n, stride_bs_k,
            GATE, False, True, "pointer", SWAP_AB, False,
            BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, WEIGHT_VALUES_PER_BYTE,
            k_col_off=-(overlap // SCALE_GROUP_K),
        )
        accumulator = mx_weight_only_compute(
            accumulator, a, w, w_s, COMPUTE_MODE, BLOCK_SIZE_K, n_width, SCALE_GROUP_K,
            SWAP_AB,
        )

    gemm_epilogue(
        C, None, accumulator, out_row, pid_n, 0, out_row, 1, stride_c_n, 1, 1,
        BLOCK_SIZE_M, BLOCK_SIZE_N, GATE, None, BLOCK_SIZE_K,
        ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE,
        COMPUTE_MODE=COMPUTE_MODE, SWAP_AB=SWAP_AB, FAKE_BATCH=True, N_COLS=N,
        GlobalScale=BsGlobal,
        global_row=expert_id,
        Bias=Bias,
        stride_bias_e=stride_bias_e,
        stride_bias_n=stride_bias_n,
    )


@bayesian_autotune(
    get_accelerator_autotuning_configs(tune_block_nk=True, swap_ab=True),
    # S (routed rows) keyed like the fp8/mx batched siblings — decode re-tunes per batch;
    # GATE keys the gate|up arm separately (its stacked dot is 2*BN wide).
    ["N", "K", "S", "GATE"],
    n_trials=100,
    path_anchor_axes=PATH_ANCHOR_AXES,
    # BLOCK_SIZE_K/N are tuned axes; the K-loop is maskless and the N-tile store is
    # row-masked only — veto non-dividing tiles on both.
    prune_configs_by={
        "early_config_prune": compose_pruners(
            block_within_dim_pruner("K"),
            block_within_dim_pruner("N", "BLOCK_SIZE_N"),
        )
    },
)
@triton.jit
def full_precision_matmul_batched_kernel(
    A,  # (rows, K) BF16/FP16 activations
    B,  # (num_experts, N, K) weights in A's dtype; under GATE the (num_experts, 2N, K) gate|up stack
    C,  # (S, N) output; under GATE the GLU intermediate
    Bias,  # (E, N_out) per-expert output bias, N_out = 2N under GATE; read iff not None
    ExpertIds,  # (S,) — which expert each batch element routes to
    GatherIdx,  # (S,) int — batch_id -> source row of A; read only when not None
    ScatterIdx,  # (S,) int — batch_id -> destination row of C; read only when not None
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
    stride_bias_e,
    stride_bias_n,
    stride_eid,
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
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
        None,  # no scales on the full-precision path
        ExpertIds,
        GatherIdx,
        ScatterIdx,
        stride_a_m,
        stride_b_e,
        stride_c_m,
        0,
        stride_eid,
        ADVANCE_BS=False,
    )
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= num_experts:
        return

    n_width: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
    offs_bn = pid_n * n_width + tl.arange(0, n_width)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = operand_tile_ptrs(A, tl.arange(0, BLOCK_SIZE_M) * 0, offs_k, stride_a_m, stride_a_k, "pointer", True)
    # GATE reads one weight tile spanning the interleaved gate|up rows;
    # GATE=False -> the plain oriented tile.
    b_ptrs = weight_tile_ptrs(
        B, offs_bn, offs_k, stride_b_n, stride_b_k, SWAP_AB
    )

    accumulator = acc_init("dot", BLOCK_SIZE_M, n_width, SWAP_AB)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a, _ = load_act_plain(a_ptrs, 0, 0, 0, None, 0, "pointer", False)
        w, _ = load_weight_plain(
            b_ptrs, b_ptrs, 0, 0, 0, GATE, False, "pointer", SWAP_AB, BLOCK_SIZE_N, BLOCK_SIZE_K
        )
        accumulator = accumulator + fp8_dot(a, w, SWAP_AB, BLOCK_SIZE_K)
        a_ptrs, _, b_ptrs, _, _, _ = advance_ptrs(
            a_ptrs, a_ptrs, b_ptrs, b_ptrs, b_ptrs, b_ptrs,
            BLOCK_SIZE_K * stride_a_k, 0, BLOCK_SIZE_K * stride_b_k, 0,
            "pointer", "pointer", False, False, False,
        )

    gemm_epilogue(
        C, None, accumulator, out_row, pid_n, 0, out_row, 1, stride_c_n, 1, 1,
        BLOCK_SIZE_M, BLOCK_SIZE_N, GATE, None, BLOCK_SIZE_K,
        ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE,
        COMPUTE_MODE="dot", SWAP_AB=SWAP_AB, FAKE_BATCH=True,
        Bias=Bias, stride_bias_e=stride_bias_e, stride_bias_n=stride_bias_n,
        global_row=expert_id,
    )


GATE_UNSTACK_MAX_S = 16  # the unstacked-gate decode band (see the dispatch in the wrapper)


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_block_dynamic_fp8_matmul_batched"),
    mutates_args=(),
    opaque=True,
)
def w8a8_block_dynamic_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor | None,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int],
    gate: bool = False,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    input_recipe: str | None = None,
    output_recipe: str | None = None,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
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
    validate_dense_operands(A, B)

    output_dtype = resolve_output_dtype(output_dtype, A, As)
    # S is the routed-row count (one program per expert_id); A may hold fewer rows when
    # gather_idx maps many programs to one source row (gate_up reading unexpanded hidden).
    K = A.shape[1]
    S = expert_ids.shape[0]
    num_experts, n_rows, N = expert_weight_shape(B, gate)

    assert len(block_size) == 2, (
        f"block_size must be [block_n, block_k], got {block_size}"
    )
    block_n, block_k = block_size[0], block_size[1]
    require_moe_dims_aligned(N, K, block_n, block_k)
    assert Bs.shape == (num_experts, n_rows // block_n, K // block_k), (
        f"Bs shape {tuple(Bs.shape)} != expected ({num_experts}, {n_rows // block_n}, {K // block_k})"
    )

    bs_u8 = ue8m0_as_uint8(Bs)
    # Offline quant wins here even at decode. An inline quant would rerun once per N-tile
    # of the (S x N-tiles) grid, and block-FP8 quant is an fp32 amax+div per element, so
    # the redundant work outweighs the extra launch down to T=1 (inline only edges ahead
    # near T=64). UE8M0 quant is ~free per pass, which is why the MX kernels do it inline.
    assert input_recipe in ("weights", "fp8"), (
        f"block-dynamic activations are E4M3 ('fp8'), got {input_recipe!r}"
    )
    assert output_recipe in (None, "weights", "fp8"), (
        f"the block-dynamic recipe requantizes to 'fp8', got {output_recipe!r}"
    )
    output_recipe = "fp8" if output_recipe == "weights" else output_recipe  # this family's format
    requant = output_recipe is not None
    # the requantized intermediate's scale groups follow gate_up's block_n, and the
    # down consumes per-block_k — a non-square block recipe would misalign them
    assert not requant or block_size[0] == block_size[1], (
        f"the fused 'fp8' requant needs square quant blocks, got {block_size}"
    )
    # The decode band runs the ``gate`` contract UNSTACKED: this same op without the gate
    # (one plain GEMM over the stacked weight), then the one-kernel ``fused_glu`` — bit-
    # identical rounding, and exactly the unfused-reference order, so ``simulate_unfused``
    # needs no carve-out. Holding the gate AND up tiles per CTA doubles the smem footprint
    # and halves occupancy precisely where the launch is weight-bandwidth-bound, and the
    # fp8 ``tl.dot`` — unlike the MX/NVFP4 scaled MMA, whose wide M operand earns the
    # native instruction — gains nothing back: DSV3 shape, same 235MB weight read, stacked
    # 69.8µs vs unstacked 49.8µs at S=8, a wash by S=32. A requant is the same offline
    # quant the raw activation gets below, applied to the GLU output — quantizing the bf16
    # intermediate (the unfused-reference order) where the stacked epilogue quantizes its
    # fp32 accumulator, a sub-quantum difference consistent with the band's semantics.
    if gate and S <= GATE_UNSTACK_MAX_S:
        [gate_up] = w8a8_block_dynamic_fp8_matmul_batched(
            A, B, As, Bs, expert_ids, block_size,
            input_recipe=input_recipe, output_dtype=output_dtype,
            gather_idx=gather_idx, scatter_idx=scatter_idx,
        )
        out = fused_glu(gate_up, act_fn, swiglu_alpha, swiglu_limit,
                        quant_group=block_n if requant else None,
                        use_ue8m0=bs_u8.dtype == torch.uint8)
        return list(out) if requant else [out]
    # A raw (As is None) -> quantize here (offline); else pre-quantized (As given, e.g. the
    # requantized intermediate handed to the down projection).
    if As is None:
        A_q, A_s = fp8_act_quant_block_dynamic(
            A, block_k, use_ue8m0=bs_u8.dtype == torch.uint8
        )
    else:
        A_q, A_s = A, As
    if requant:
        C = A.new_empty(S, N, dtype=FP8_DTYPE)
        # UE8M0 model (ue8m0 weights) -> UE8M0 intermediate scales (whole-model contract);
        # the kernel infers the requant format from this dtype. fp32 weights keep fp32.
        cs_dtype = bs_u8.dtype  # uint8 (UE8M0) or float32 — the whole-model scale format
        Cs = torch.empty(S, N // block_n, device=A.device, dtype=cs_dtype)
    else:
        C = A.new_empty(S, N, dtype=output_dtype)
        Cs = None  # unread without an OUTPUT_RECIPE; strides literal below

    # the N tile is tuned (it may subdivide the block scale — see scale_subblock_pruner)
    grid = lambda meta: (S, triton.cdiv(N, meta["BLOCK_SIZE_N"]))  # noqa: E731

    with device_context(A.device):
        bias_stride_e, bias_stride_n = bias_strides(bias)
        compile_time_only_triton_wrap(w8a8_block_dynamic_fp8_matmul_batched_kernel)[
            grid
        ](
            A_q,
            A_s,
            B,
            bs_u8,
            C,
            Cs,
            bias,
            expert_ids,
            gather_idx,  # None = A is expert-sorted; read only when not None (folds at trace time)
            scatter_idx,  # None = C is expert-sorted; read only when not None (folds at trace time)
            S,
            N,
            K,
            A_q.stride(0),
            A_q.stride(1),
            A_s.stride(0),
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
            bias_stride_e,
            bias_stride_n,
            expert_ids.stride(0),
            BLOCK_SIZE_K=block_k,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_experts=num_experts,
            GATE=gate,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            OUTPUT_RECIPE=output_recipe,
            SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(output_dtype),
        )

    return [C, Cs] if requant else [C]


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_block_static_fp8_matmul_batched"),
    mutates_args=(),
    opaque=True,
)
def w8a8_block_static_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int],
    gate: bool = False,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    input_recipe: str | None = None,
    output_recipe: str | None = None,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Block-scale batched FP8 matmul with a static (per-tensor calibrated) activation scale — the
    block-dynamic batched sibling with the 2D ``block_static`` recipe. ``A`` is raw here: the op
    quantizes it against the scalar ``As`` (offline), the kernel applies the per-block weight scales
    in the K-loop and the scalar once post-loop. Returns ``[C]``, or ``[C, Cs]`` under
    ``output_recipe="fp8"`` (the per-row output scale is independent of the per-tensor input scale).

    A:  (rows, K) raw bf16/fp16 activations — rows addressed via ``gather_idx``
    B:  (num_experts, N, K) FP8 weights; under ``gate`` the (num_experts, 2N, K) gate|up stack
    As: scalar / (1,) — the calibrated per-tensor (static) activation scale
    Bs: (num_experts, N // block_n, K // block_k) per-block weight scales (2N under gate)
    """
    validate_dense_operands(A, B)

    output_dtype = resolve_output_dtype(output_dtype, A, None)
    K = A.shape[1]
    S = expert_ids.shape[0]
    num_experts, n_rows, N = expert_weight_shape(B, gate)

    assert len(block_size) == 2, (
        f"block_size must be [block_n, block_k], got {block_size}"
    )
    block_n, block_k = block_size[0], block_size[1]
    require_moe_dims_aligned(N, K, block_n, block_k)
    assert Bs.shape == (num_experts, n_rows // block_n, K // block_k), (
        f"Bs shape {tuple(Bs.shape)} != expected ({num_experts}, {n_rows // block_n}, {K // block_k})"
    )
    assert input_recipe in ("weights", "fp8"), (
        f"block-static activations are E4M3 ('fp8'), got {input_recipe!r}"
    )
    assert output_recipe in (None, "weights", "fp8"), (
        f"the block-static recipe requantizes to 'fp8', got {output_recipe!r}"
    )
    output_recipe = "fp8" if output_recipe == "weights" else output_recipe  # this family's format
    requant = output_recipe is not None
    assert not requant or block_n == block_k, (
        f"the fused 'fp8' requant needs square quant blocks, got {block_size}"
    )

    As = As.reshape(1).to(torch.float32)
    bs_u8 = ue8m0_as_uint8(Bs)
    # Pre-quantize the raw activations against the calibrated scalar (offline; the kernel folds
    # the scalar back post-loop).
    A_q = (A.to(torch.float32) / As).to(FP8_DTYPE)
    if requant:
        C = A.new_empty(S, N, dtype=FP8_DTYPE)
        Cs = torch.empty(S, N // block_n, device=A.device, dtype=bs_u8.dtype)
    else:
        C = A.new_empty(S, N, dtype=output_dtype)
        Cs = None  # unread without an OUTPUT_RECIPE; strides literal below

    # the N tile is tuned (it may subdivide the block scale — see scale_subblock_pruner)
    grid = lambda meta: (S, triton.cdiv(N, meta["BLOCK_SIZE_N"]))  # noqa: E731

    with device_context(A.device):
        bias_stride_e, bias_stride_n = bias_strides(bias)
        compile_time_only_triton_wrap(w8a8_block_static_fp8_matmul_batched_kernel)[
            grid
        ](
            A_q,
            As,
            B,
            bs_u8,
            C,
            Cs,
            bias,
            expert_ids,
            gather_idx,  # None = A is expert-sorted; read only when not None (folds at trace time)
            scatter_idx,  # None = C is expert-sorted; read only when not None (folds at trace time)
            S,
            N,
            K,
            A_q.stride(0),
            A_q.stride(1),
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
            bias_stride_e,
            bias_stride_n,
            expert_ids.stride(0),
            num_experts,
            BLOCK_SIZE_K=block_k,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GATE=gate,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            OUTPUT_RECIPE=output_recipe,
            SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(output_dtype),
        )

    return [C, Cs] if requant else [C]


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_tensor_dynamic_fp8_matmul_batched"),
    mutates_args=(),
    opaque=True,
)
def w8a8_tensor_dynamic_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor | None,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tensor-scale batched FP8 matmul: C[s] = A[s] @ B[expert_ids[s]].T. ``A`` raw
    (``As`` None) -> quantized here (offline, per-token); else pre-quantized (``As`` given).
    ``gather_idx``/``scatter_idx`` map the source row of A / destination row of C per program
    (None = row s).

    A:  (rows, K) raw or pre-quantized FP8 activations — rows addressed via ``gather_idx``
    B:  (num_experts, N, K) FP8 expert weights
    As: (rows,) per-token scales, or None when A is raw
    Bs: (num_experts,) or (num_experts, 1, 1) per-expert weight scales
    """
    validate_dense_operands(A, B)

    output_dtype = resolve_output_dtype(output_dtype, A, As)
    K = A.shape[1]
    S = expert_ids.shape[0]
    num_experts, N, _ = B.shape

    # Normalize Bs to (num_experts, 1, 1)
    Bs = normalize_per_expert_scale(Bs, num_experts)

    bs_u8 = ue8m0_as_uint8(Bs)
    if As is None:
        qA, As = fp8_act_quant_tensor_wide(A, K)
    else:
        qA = A
    C = A.new_empty(S, N, dtype=output_dtype)

    def grid(META):
        return (S, triton.cdiv(N, META["BLOCK_SIZE_N"]))

    with device_context(A.device):
        bias_stride_e, bias_stride_n = bias_strides(bias)
        compile_time_only_triton_wrap(w8a8_tensor_dynamic_fp8_matmul_batched_kernel)[
            grid
        ](
            qA,
            As,
            B,
            bs_u8,
            C,
            bias,
            expert_ids,
            gather_idx,  # None = A is expert-sorted; read only when not None (folds at trace time)
            scatter_idx,  # None = C is expert-sorted; read only when not None (folds at trace time)
            S,
            N,
            K,
            qA.stride(0),
            qA.stride(1),
            As.stride(0),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            bs_u8.stride(0),
            C.stride(0),
            C.stride(1),
            bias_stride_e,
            bias_stride_n,
            expert_ids.stride(0),
            num_experts=num_experts,
        )

    return C


@compile_time_only_triton_op(
    add_op_namespace_prefix("mx_dynamic_matmul_batched"), mutates_args=(), opaque=True
)
def mx_dynamic_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor | None,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    gate: bool = False,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    input_recipe: str | None = None,
    output_recipe: str | None = None,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    a_global_scale: torch.Tensor | None = None,
    b_global_scale: torch.Tensor | None = None,
    output_global_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
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
    assert A.ndim == 2 and B.ndim == 3 and Bs.ndim in (3, 5)  # 5D = pre-swizzled
    assert expert_ids.ndim == 1
    # A raw (As None) -> quantized inline in the kernel (decode-free UE8M0); pre-quantized
    # (As given, e.g. the down reading a requantized intermediate) -> loaded with its scales
    # (the kernel folds on As is None).
    # the kernel quantizes raw A inline on this grid (fp4 recipes pack in-register);
    # NVFP4 batched runs on the software arms — decode grid BM <= 16 < the native
    # mxf4nvf4 M=128 staging (scalar / swap-scalar column-unpack + E4M3 scale decode)
    input_recipe = resolve_input_recipe(input_recipe, output_recipe, B, Bs)
    output_recipe = resolve_output_recipe(output_recipe, B, Bs)
    requant = output_recipe is not None
    if As is not None:
        assert (As.dtype == torch.float8_e4m3fn) == (Bs.dtype == torch.float8_e4m3fn), (
            f"activation scales ({As.dtype}) must match the weight scale family ({Bs.dtype})"
        )
    pre_quantized = As is not None
    assert B.dtype in (torch.int8, torch.float8_e4m3fn), (
        f"B must be int8 (packed E2M1) or float8_e4m3fn (E4M3), got {B.dtype}"
    )
    WEIGHT_VALUES_PER_BYTE = NIBBLES_PER_BYTE if B.dtype == torch.int8 else 1
    # NVFP4 raw activations pre-quantize HERE, above the layout derivation below — the quant
    # packs A to E2M1, so ACT_VALUES_PER_BYTE and K must be read off the packed tensor. Offline
    # beats the inline arm at every M for NVFP4 (unlike the UE8M0 recipes, whose exponent-only
    # inline quant is free): its E4M3 block scale costs an amax divide plus a global normalize
    # per group, and the decode grid re-runs that in every N-tile program against the SAME
    # routed row. Measured bit-identical, GLM decode gate_up 34.2 -> 31.9us, down 18.9 -> 16.9us.
    # The act scale stays affine under a swizzled weight: load_act_mx reads it off as_ptrs
    # row-major, SWIZZLED_SCALES governs only the weight side.
    if As is None and input_recipe == "nvfp4" and A.dtype not in (torch.int8, torch.float8_e4m3fn):
        A, As = MX_ACT_QUANT["nvfp4"](A, global_scale=a_global_scale)
        pre_quantized = True
    # int8 A = caller-provided packed-E2M1 activations (W4A4, native mxf4 MMA): K is two
    # values per stored byte and the scales are mandatory (nothing left to quantize).
    ACT_VALUES_PER_BYTE = NIBBLES_PER_BYTE if A.dtype == torch.int8 else 1
    if ACT_VALUES_PER_BYTE == NIBBLES_PER_BYTE:
        assert As is not None, "packed-E2M1 activations need their UE8M0 scales (As)"

    output_dtype = resolve_output_dtype(output_dtype, A, As)
    K = A.shape[1] * ACT_VALUES_PER_BYTE
    S = expert_ids.shape[0]
    num_experts, n_rows, N = expert_weight_shape(B, gate)
    K_b = B.shape[2]
    assert K == WEIGHT_VALUES_PER_BYTE * K_b, (
        f"K (={K}) must equal {WEIGHT_VALUES_PER_BYTE} * B.shape[2] (={K_b})"
    )
    # Bs arrives either row-major (num_experts, n_rows, K // scale_group) — read affine — or
    # already SWIZZLE_32_4_4 (5D: 1, num_experts * n_rows // 128, cols // 4, 2, 256), the shared
    # checkpoint layout swizzled once at model load (the deployment contract, no per-call
    # rearrange). The recipe is the scale dtype (E4M3 = NVFP4 group-16, UE8M0 = MX group-32); the
    # swizzled cols encode (K // scale_group) // 4.
    swizzled_scales = Bs.ndim == 5
    scale_group = mx_scale_family(Bs, K)
    if not swizzled_scales:
        assert Bs.shape == (num_experts, n_rows, K // scale_group), (
            f"Bs shape {tuple(Bs.shape)} != ({num_experts}, {n_rows}, {K // scale_group})"
        )
    else:
        # the artifact's row-block count must match THIS weight stack — a wrong-layer
        # artifact with a matching K would otherwise dequantize with garbage scales silently
        expected_blocks = num_experts * triton.cdiv(n_rows, 128)
        assert Bs.shape[1] == expected_blocks, (
            f"swizzled Bs carries {Bs.shape[1]} 128-row blocks, expected {expected_blocks} "
            f"for {num_experts} experts x ({n_rows}, K) — wrong artifact"
        )

    a_u8 = e2m1_as_uint8(A)
    as_u8 = ue8m0_as_uint8(As)  # None when raw (A quantized inline)
    b_u8 = e2m1_as_uint8(B)
    bs_u8 = ue8m0_as_uint8(Bs)
    # The op never swizzles: Bs is read in whatever layout it arrives (recipe-agnostic — MX or
    # NVFP4). A pre-swizzled SWIZZLE_32_4_4 Bs (5D, the shared checkpoint layout the grouped kernel
    # also consumes) takes the fast descriptor/gather path; a row-major (3D) Bs takes the affine
    # path at no penalty. Callers swizzle once at load (public swizzle_mx_scales) to opt into perf.
    # The descriptor is built only on the swizzled path — the un-swizzled arm never reads it (None).
    bs_descriptor = (
        TensorDescriptor.from_tensor(bs_u8, [1, 1, 1, 2, 256]) if swizzled_scales else None
    )
    # Requant scales are written ROW-MAJOR (never SWIZZLE_32_4_4), unlike the grouped/2D
    # requant which fuse the swizzle in-epilogue. This is deliberate, not a gap: batched is the
    # decode kernel — one distinct routed row per program (FAKE_BATCH replicates it across the BM
    # lanes), so it never forms the 128-distinct-row MMA tile the tcgen05 swizzled scaled-MMA
    # fast path needs. Swizzled scales give decode no speedup, and the 128-row swizzle block
    # can't be written from a one-row-per-program grid without cross-program collisions. The
    # down projection reads this row-major intermediate directly (its As is row-major too).
    if output_recipe in ("mxfp4", "nvfp4"):
        # packed E2M1 intermediate (nibble pairs along N) + group scales (UE8M0 for MX,
        # E4M3 for NVFP4) — feeds a W4A4 down as-is
        assert N % (2 * scale_group) == 0, (
            f"N (={N}) must be a multiple of {2 * scale_group} to pack E2M1 pairs"
        )
        C = a_u8.new_empty((S, N // 2), dtype=torch.int8)
        Cs = torch.empty(
            S,
            N // scale_group,
            device=a_u8.device,
            dtype=bs_u8.dtype,  # UE8M0 -> uint8, NVFP4 -> e4m3 (the binder-safe weight-scale dtype)
        )
    elif requant:
        C = a_u8.new_empty((S, N), dtype=FP8_DTYPE)
        Cs = torch.empty(S, N // MX_SCALE_GROUP_K, device=a_u8.device, dtype=torch.uint8)
    else:
        C = a_u8.new_empty((S, N), dtype=output_dtype)
        Cs = None  # unread without an OUTPUT_RECIPE; strides literal below

    def grid(META):
        return (S, triton.cdiv(N, META["BLOCK_SIZE_N"]))

    # NVFP4 accumulator correction: the per-expert g_a·g_b product folded onto the fp32 accumulator.
    input_global_scale = combine_global_scales(a_global_scale, b_global_scale, B.shape[0])
    with device_context(a_u8.device):
        bias_stride_e, bias_stride_n = bias_strides(bias)
        compile_time_only_triton_wrap(mx_dynamic_matmul_batched_kernel)[grid](
            a_u8,
            as_u8,  # None when raw (A quantized inline)
            b_u8,
            bs_u8,
            bs_descriptor,
            C,
            Cs,
            bias,
            a_global_scale,  # AsGlobal (1,): g_a for the inline-quant arm (A/g_a)
            input_global_scale,  # AsBsGlobal = g_a·g_b (acc)
            output_global_scale,  # CsGlobal: requant output normalization (next proj's provided input_scale); None folds out
            expert_ids,
            gather_idx,  # None = A is expert-sorted; read only when not None (folds at trace time)
            scatter_idx,  # None = C is expert-sorted; read only when not None (folds at trace time)
            S,
            N,
            K,
            a_u8.stride(0),
            a_u8.stride(1),
            as_u8.stride(0) if pre_quantized else 1,
            b_u8.stride(0),
            b_u8.stride(2),
            b_u8.stride(1),
            bs_u8.stride(0),
            bs_u8.stride(2),
            bs_u8.stride(1),
            C.stride(0),
            C.stride(1),
            Cs.stride(0) if requant else 1,
            Cs.stride(1) if requant else 1,
            bias_stride_e,
            bias_stride_n,
            expert_ids.stride(0),
            SCALE_GROUP_K=scale_group,
            num_experts=num_experts,
            SWIZZLED_SCALES=swizzled_scales,
            INPUT_RECIPE=input_recipe,
            GATE=gate,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            OUTPUT_RECIPE=output_recipe,
            SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(output_dtype),
        )
    return [C, Cs] if requant else [C]


@compile_time_only_triton_op(
    add_op_namespace_prefix("full_precision_matmul_batched"),
    mutates_args=(),
    opaque=True,
)
def full_precision_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    expert_ids: torch.Tensor,
    gate: bool = False,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    input_recipe: str | None = None,
    output_recipe: str | None = None,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Full-precision (BF16/FP16) batched matmul: C[s] = A[s] @ B[expert_ids[s]].T — no
    quantization anywhere, fp32 accumulation. ``gate``/``act_fn``/``swiglu_*``/
    ``simulate_unfused`` are the flattened ``Epilogue`` (GLU only; ``requant`` is
    meaningless without a quantized recipe). ``gather_idx``/``scatter_idx`` map the source
    row of A / destination row of C per program (None = row s).

    A:  (rows, K) BF16/FP16 activations — rows addressed via ``gather_idx``
    B:  (num_experts, N, K) expert weights in A's dtype; under ``gate`` the (num_experts, 2N, K) stack
    """
    validate_dense_operands(A, B)
    assert A.dtype == B.dtype and A.dtype in (torch.bfloat16, torch.float16), (
        f"full-precision path needs matching BF16/FP16 A and B, got {A.dtype} / {B.dtype}"
    )
    assert input_recipe in (None, "weights") and output_recipe is None, (
        "the full-precision path quantizes nothing — no input or output recipe applies"
    )

    output_dtype = resolve_output_dtype(output_dtype, A, None)
    K = A.shape[1]
    S = expert_ids.shape[0]
    num_experts, _, N = expert_weight_shape(B, gate)
    C = A.new_empty(S, N, dtype=output_dtype)

    def grid(META):
        return (S, triton.cdiv(N, META["BLOCK_SIZE_N"]))

    with device_context(A.device):
        bias_stride_e, bias_stride_n = bias_strides(bias)
        compile_time_only_triton_wrap(full_precision_matmul_batched_kernel)[grid](
            A,
            B,
            C,
            bias,
            expert_ids,
            gather_idx,  # None = A is expert-sorted; read only when not None (folds at trace time)
            scatter_idx,  # None = C is expert-sorted; read only when not None (folds at trace time)
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
            bias_stride_e,
            bias_stride_n,
            expert_ids.stride(0),
            num_experts=num_experts,
            GATE=gate,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(output_dtype),
        )

    return [C]


@compile_time_only_triton_op(
    add_op_namespace_prefix("mx_weight_only_matmul_batched"),
    mutates_args=(),
    opaque=True,
)
def mx_weight_only_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    gate: bool = False,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    b_global_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """weight-only batched (decode) matmul: ``C[s] = A[s] @ B[expert_ids[s]].T`` with raw bf16/fp16
    activations against MXFP4/NVFP4/MXFP8 weights upcast to bf16 in-loop — the ``matmul_ogs`` recipe.
    Affine (3D) weight scales — UE8M0 group-32 (MX) or E4M3 group-16 (NVFP4, whose per-expert
    fp32 ``b_global_scale`` ``(E,)`` recovers on the accumulator; ``None`` = single-level).
    Returns ``[C]``."""
    assert A.dtype in (torch.bfloat16, torch.float16, torch.float32), (
        f"weight-only takes a raw bf16/fp16/fp32 activation, got {A.dtype}"
    )
    assert Bs.ndim == 3, f"weight-only batched takes affine (3D) weight scales, got ndim={Bs.ndim}"
    # The ``gate`` contract runs UNSTACKED at every S: this same op without the gate (one
    # plain GEMM over the stacked weight), then the one-kernel ``fused_glu`` — bit-identical
    # rounding, the unfused-reference order, so ``simulate_unfused`` rides through. The
    # upcast weight feeds a plain bf16 dot, so the stacked tile buys no native MMA and only
    # halves occupancy on a weight-bandwidth-bound loop; unlike the block-FP8 band this
    # never inverts (GPT-OSS shape: 48.0->39.7µs at S=4, still ahead at S=512; 2026-08-06).
    if gate:
        # fp32 intermediate = the exact GEMM accumulators, so the GLU keeps the gated
        # epilogue's FUSED-order rounding (bf16 operands would drift steep-sigmoid elements
        # past the weight-only tests' exact-ish tolerances). Under ``simulate_unfused`` the
        # caller is asking for the UNFUSED order instead, where the reference lands its gate_up
        # in the activation dtype before the GLU — carrying fp32 there leaves the two a bf16 ULP
        # apart (256.0 absolute at magnitude 2^15), which the parity tolerance rejects.
        inter_dtype = (
            resolve_output_dtype(output_dtype, A, None) if simulate_unfused else torch.float32
        )
        [gate_up] = mx_weight_only_matmul_batched(
            A, B, Bs, expert_ids, output_dtype=inter_dtype,
            gather_idx=gather_idx, scatter_idx=scatter_idx, b_global_scale=b_global_scale,
        )
        return [fused_glu(gate_up, act_fn, swiglu_alpha, swiglu_limit,
                          out_dtype=resolve_output_dtype(output_dtype, A, None))]
    output_dtype = resolve_output_dtype(output_dtype, A, None)
    K = A.shape[1]
    S = expert_ids.shape[0]
    WEIGHT_VALUES_PER_BYTE = 2 if B.dtype == torch.int8 else 1
    num_experts, rows, K_b = B.shape
    assert K == WEIGHT_VALUES_PER_BYTE * K_b, (
        f"K ({K}) must equal {WEIGHT_VALUES_PER_BYTE} * B.shape[2] ({K_b})"
    )
    N = rows // 2 if gate else rows
    scale_group = mx_scale_family(Bs, K)
    b_global_scale = normalize_global_scale(b_global_scale, num_experts)
    C = A.new_empty(S, N, dtype=output_dtype)
    b_u8 = e2m1_as_uint8(B)
    bs_u8 = ue8m0_as_uint8(Bs)

    def grid(META):
        return (S, triton.cdiv(N, META["BLOCK_SIZE_N"]))

    with device_context(A.device):
        bias_stride_e, bias_stride_n = bias_strides(bias)
        compile_time_only_triton_wrap(mx_weight_only_matmul_batched_kernel)[grid](
            A,
            b_u8,
            bs_u8,
            b_global_scale,
            C,
            bias,
            expert_ids,
            gather_idx,
            scatter_idx,
            S,
            N,
            K,
            A.stride(0),
            A.stride(1),
            b_u8.stride(0),
            b_u8.stride(2),
            b_u8.stride(1),
            bs_u8.stride(0),
            bs_u8.stride(1),
            bs_u8.stride(2),
            C.stride(0),
            C.stride(1),
            bias_stride_e,
            bias_stride_n,
            expert_ids.stride(0),
            num_experts=num_experts,
            SCALE_GROUP_K=scale_group,
            WEIGHT_VALUES_PER_BYTE=WEIGHT_VALUES_PER_BYTE,
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
    B: torch.Tensor,
    As: torch.Tensor | None = None,
    Bs: torch.Tensor | None = None,
    *,
    expert_ids: torch.Tensor,
    bias: torch.Tensor | None = None,  # (E, N_out) per-expert output bias; (N_out,) for 2D
    epilogue: Epilogue | None = None,
    quantization: Quantization | None = None,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    a_global_scale: torch.Tensor | None = None,
    b_global_scale: torch.Tensor | None = None,
    output_global_scale: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Batched matmul dispatcher (W8A8 FP8, W4A8/W4A4 FP4, or full-precision). Routes one
    program per routed row (``expert_ids`` gives its expert).

    ``As`` marks ``A`` as already quantized (framework-precomputed scales, or a requantized
    intermediate handed to the down projection); a per-tensor scalar ``As`` is instead the static
    (calibrated) activation scale for block-scale FP8 weights — the op quantizes raw ``A`` against
    it; ``None`` = raw ``A``, quantized dynamically by the op per ``quantization`` (see
    ``Quantization`` — recipe-default fp8/E4M3, offline for bd/tensor and inline for MX, or packed
    E2M1 under ``input_recipe="mxfp4"``). ``Bs`` ``None`` =
    unquantized BF16/FP16 weights. ``quantization.output_recipe`` requantizes the output into
    the recipe's format — the return is then ``(C, Cs)``. ``epilogue`` is the fused output
    transform (gate|up + GLU). ``As``/``Bs`` are each a bare block-scale tensor; the two-level NVFP4
    second-level scales ride the separate ``a_global_scale``/``b_global_scale`` (fp32 per-tensor,
    weights per-expert ``(E,)``; from ``nvfp4_quantize_two_level``), and the op folds ``g_a · g_b``
    onto the accumulator. The activation global ``g_a`` is CALIBRATED (the checkpoint's
    ``input_scale``): ``a_global_scale=g_a`` with a raw ``A`` has the op quantize ``A / g_a`` per
    block, and rides a pre-quantized ``As`` the same way. Under NVFP4 ``output_recipe`` the
    fused requant normalizes the GLU intermediate by the PROVIDED ``output_global_scale`` (the next
    proj's calibrated ``input_scale``) before the block quant and returns ``[C, Cs]``; the down
    consumes it as ``As=Cs, a_global_scale=output_global_scale``. ``gather_idx``/``scatter_idx`` (each None or a ``(S,)`` map)
    address the source row of ``A`` / destination row of ``C`` per program — None means row
    ``s``; the gather lets the gate_up read unexpanded activations (source row
    ``s // num_top_k``) with no copy.

    Routes by what the weight tensors themselves say (there is no ``block_size``
    parameter — the quantization block is derived from the scale shape,
    ``weight_block_size``):
    - ``Bs`` None → ``full_precision_matmul_batched`` (plain dot, no scales anywhere).
    - MX weights — ``int8`` (packed E2M1) or ``float8_e4m3fn`` (E4M3) with UE8M0
      group-32 ``Bs`` → ``mx_dynamic_matmul_batched``.
    - one scale per expert (``Bs`` ``(E,)``/``(E, 1, 1)``) →
      ``w8a8_tensor_dynamic_fp8_matmul_batched``.
    - block scales (``Bs`` ``(E, N/bn, K/bk)``) → ``w8a8_block_dynamic_fp8_matmul_batched``.
    """
    ep = epilogue if epilogue is not None else Epilogue()
    q = quantization if quantization is not None else Quantization()
    assert (a_global_scale is None and b_global_scale is None) or (
        Bs is not None and weight_recipe(B, Bs) == "nvfp4"
    ), "two-level globals (a_global_scale / b_global_scale) are NVFP4-only"
    assert output_global_scale is None or q.output_recipe == "nvfp4", (
        "output_global_scale is the NVFP4 requant second level — it requires output_recipe='nvfp4' "
        "(the epilogue would otherwise normalize by it with nothing downstream to compensate)"
    )
    if As is not None and As.numel() == 1:
        # static (per-tensor calibrated) activation quant: a per-tensor scalar As for block-scale FP8
        # weights — the caller hands raw A, the op quantizes it against the scalar (As IS the scale).
        assert Bs is not None and not is_mx(B, Bs) and weight_block_size(B, Bs) is not None, (
            "a per-tensor scalar As (static activation scale) needs block-scale FP8 weights"
        )
        out = w8a8_block_static_fp8_matmul_batched(
            A,
            B,
            As,
            Bs,
            expert_ids,
            weight_block_size(B, Bs),
            *ep.as_args(),
            *q.as_args(),
            output_dtype,
            gather_idx,
            scatter_idx,
            bias=bias,
        )
        return out[0] if len(out) == 1 else tuple(out)

    if Bs is None:
        assert As is None, (
            "the full-precision path (Bs=None) takes no activation scales"
        )
        out = full_precision_matmul_batched(
            A,
            B,
            expert_ids,
            *ep.as_args(),
            *q.as_args(),
            output_dtype,
            gather_idx,
            scatter_idx,
            bias=bias,
        )
    elif is_mx(B, Bs) and q.input_recipe is None:  # weight-only: raw bf16 acts, MX weight upcast in-MMA
        assert As is None and a_global_scale is None and q.output_recipe is None, (
            "weight-only (input_recipe=None) takes a raw activation, no As/global/requant"
        )
        assert output_global_scale is None, (
            "weight-only has no requant epilogue for output_global_scale to normalize"
        )
        out = mx_weight_only_matmul_batched(
            A,
            B,
            Bs,
            expert_ids,
            *ep.as_args(),
            output_dtype,
            gather_idx,
            scatter_idx,
            b_global_scale=b_global_scale,
            bias=bias,
        )
    elif is_mx(B, Bs):
        out = mx_dynamic_matmul_batched(
            A,
            B,
            As,
            Bs,
            expert_ids,
            *ep.as_args(),
            *q.as_args(),
            output_dtype,
            gather_idx,
            scatter_idx,
            a_global_scale,
            b_global_scale,
            output_global_scale,
            bias=bias,
        )
    elif (block_size := weight_block_size(B, Bs)) is None:
        assert not ep.gate, (
            "the batched op has no tensor-wide gate|up fusion (grouped and 2D support it)"
        )
        assert q.input_recipe in ("weights", "fp8") and q.output_recipe is None, (
            "tensor-wide supports neither packed activations nor a fused requant"
        )
        out = w8a8_tensor_dynamic_fp8_matmul_batched(
            A, B, As, Bs, expert_ids, output_dtype, gather_idx, scatter_idx, bias=bias
        )
    else:
        out = w8a8_block_dynamic_fp8_matmul_batched(
            A,
            B,
            As,
            Bs,
            expert_ids,
            block_size,
            *ep.as_args(),
            *q.as_args(),
            output_dtype,
            gather_idx,
            scatter_idx,
            bias=bias,
        )
    # bd/mx/full-precision ops return a list ([C] or [C, Cs]); the tensor op returns a bare tensor.
    if isinstance(out, (list, tuple)):
        return out[0] if len(out) == 1 else tuple(out)
    return out
