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
from .utils import (
    compile_time_only_triton_op,
    compile_time_only_triton_wrap,
    Epilogue,
    Quantization,
    FP8_DTYPE,
    MX_SCALE_GROUP_K,
    NIBBLES_PER_BYTE,
    device_context,
    tl_dtype,
    resolve_input_recipe,
    resolve_output_dtype,
    accumulate,
    oriented_tile_ptrs,
    operand_tile_ptrs,
    advance_ptrs,
    gemm_epilogue,
    acc_init,
    mx_config_pruner,
    swizzled_scale_config_pruner,
    smem_pruner,
    block_within_dim_pruner,
    require_moe_dims_aligned,
    compose_pruners,
    acc_finalize,
    weight_tile_ptrs,
    load_act,
    load_weight,
    expert_weight_shape,
    mx_scale_family,
    normalize_per_expert_scale,
    validate_dense_operands,
    fp8_act_quant_tensor_wide,
    fp8_act_quant_block_dynamic,
    get_accelerator_autotuning_configs,
    is_mx,
    combine_global_scales,
    split_scale,
    weight_block_size,
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
    C,  # (S, N) output; under an OUTPUT_RECIPE the FP8-requantized intermediate
    Cs,  # (S, N // BLOCK_SIZE_N) per-(row, block) output scale; written iff OUTPUT_RECIPE
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
    stride_eid,
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
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

    ``GATE`` fuses the gate|up projection: ``B`` is the ``(E, 2N, K)`` stack (gate rows [0,N),
    up rows [N,2N)), run as two dots (the decode-validated form), SwiGLU-combined, and — under
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

    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    n_width: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = operand_tile_ptrs(A, tl.arange(0, BLOCK_SIZE_M) * 0, offs_k, stride_a_m, stride_a_k, "pointer", True)
    as_ptrs = As + in_row * stride_as_m + tl.zeros((BLOCK_SIZE_M,), tl.int32)
    # One stacked gate|up weight tile (gate rows [0,N), up rows [N,2N)) + one block-scale pointer,
    # like every other kernel. The up block scale sits num_n_tiles blocks after gate; folding that
    # into the per-weight-row load offset (tl.where) lets gate|up share a tile + a single dot.
    b_ptrs = weight_tile_ptrs(B, offs_bn, offs_k, N * stride_b_n, stride_b_n, stride_b_k, GATE, SWAP_AB)
    bs_ptr = Bs + pid_n * stride_bs_n
    bs_off = tl.where(tl.arange(0, n_width) < BLOCK_SIZE_N, 0, num_n_tiles * stride_bs_n)
    acc = acc_init("dot", BLOCK_SIZE_M, n_width, SWAP_AB)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a, a_s = load_act("block_dynamic", a_ptrs, as_ptrs, None, None, None, 0, 0, 0, "pointer")
        w, b_s = load_weight(
            "block_dynamic", b_ptrs, bs_ptr + bs_off, None, b_ptrs, 0, 0, "pointer", SWAP_AB, GATE,
            BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        acc = accumulate(
            acc, a, a_s, w, b_s, "block_dynamic", "dot", SWAP_AB, False,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, FAKE_BATCH=True,
        )
        a_ptrs, as_ptrs, b_ptrs, bs_ptr, _, _ = advance_ptrs(
            a_ptrs, as_ptrs, b_ptrs, bs_ptr, b_ptrs, bs_ptr,
            BLOCK_SIZE_K * stride_a_k, 1, BLOCK_SIZE_K * stride_b_k, stride_bs_k,
            "pointer", "pointer", True, True, False,
        )

    gemm_epilogue(
        C, Cs, acc, out_row, pid_n, 0, out_row, 1, stride_c_n, stride_cs_m, stride_cs_n,
        BLOCK_SIZE_M, BLOCK_SIZE_N, GATE, OUTPUT_RECIPE, BLOCK_SIZE_K,
        ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE,
        COMPUTE_MODE="dot", SWAP_AB=SWAP_AB, FAKE_BATCH=True,
    )


@bayesian_autotune(
    get_accelerator_autotuning_configs(swap_ab=True),
    ["N", "K", "S"],
    n_trials=100,
)
@triton.jit
def w8a8_block_static_fp8_matmul_batched_kernel(
    A,  # (S, K) E4M3 activations (pre-quantized against the static scale by the wrapper)
    As,  # scalar — static per-tensor activation scale (calibration-time)
    B,  # (num_experts, N, K) FP8 weights; under GATE the (num_experts, 2N, K) gate|up stack
    Bs,  # (num_experts, N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales (2N under GATE)
    C,  # (S, N) output; under an OUTPUT_RECIPE the FP8-requantized intermediate
    Cs,  # (S, N // BLOCK_SIZE_N) per-(row, block) output scale; written iff OUTPUT_RECIPE
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
    stride_eid,
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
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

    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    n_width: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = operand_tile_ptrs(A, tl.arange(0, BLOCK_SIZE_M) * 0, offs_k, stride_a_m, stride_a_k, "pointer", True)
    # One stacked gate|up weight tile (gate rows [0,N), up rows [N,2N)) + one block-scale pointer;
    # the up block scale sits num_n_tiles blocks after gate (tl.where on the load offset).
    b_ptrs = weight_tile_ptrs(B, offs_bn, offs_k, N * stride_b_n, stride_b_n, stride_b_k, GATE, SWAP_AB)
    bs_ptr = Bs + pid_n * stride_bs_n
    bs_off = tl.where(tl.arange(0, n_width) < BLOCK_SIZE_N, 0, num_n_tiles * stride_bs_n)
    acc = acc_init("dot", BLOCK_SIZE_M, n_width, SWAP_AB)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a, _ = load_act("static", a_ptrs, a_ptrs, None, None, None, 0, 0, 0, "pointer")  # pre-quantized E4M3 token (fake-batch replicated)
        w, b_s = load_weight(
            "static", b_ptrs, bs_ptr + bs_off, None, b_ptrs, 0, 0, "pointer", SWAP_AB, GATE,
            BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        acc = accumulate(
            acc, a, a_s_static, w, b_s, "static", "dot", SWAP_AB, False,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, FAKE_BATCH=True,
        )
        a_ptrs, _, b_ptrs, bs_ptr, _, _ = advance_ptrs(
            a_ptrs, a_ptrs, b_ptrs, bs_ptr, b_ptrs, bs_ptr,
            BLOCK_SIZE_K * stride_a_k, 0, BLOCK_SIZE_K * stride_b_k, stride_bs_k,
            "pointer", "pointer", False, True, False,
        )

    acc = acc * a_s_static
    gemm_epilogue(
        C, Cs, acc, out_row, pid_n, 0, out_row, 1, stride_c_n, stride_cs_m, stride_cs_n,
        BLOCK_SIZE_M, BLOCK_SIZE_N, GATE, OUTPUT_RECIPE, BLOCK_SIZE_K,
        ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE,
        COMPUTE_MODE="dot", SWAP_AB=SWAP_AB, FAKE_BATCH=True,
    )


@bayesian_autotune(
    # S (routed rows) keyed like the block-dynamic/mx batched siblings — decode re-tunes per batch.
    get_accelerator_autotuning_configs(tune_block_nk=True, swap_ab=True),
    ["N", "K", "S"],
    n_trials=100,
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
        a, _ = load_act("tensor", a_ptrs, a_ptrs, None, None, None, 0, 0, 0, "pointer")
        b, _ = load_weight(
            "tensor", b_ptrs, b_ptrs, None, b_ptrs, 0, 0, "pointer", SWAP_AB,
            BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        accumulator = accumulate(
            accumulator, a, a, b, b, "tensor", "dot", SWAP_AB, False,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        )
        a_ptrs, _, b_ptrs, _, _, _ = advance_ptrs(
            a_ptrs, a_ptrs, b_ptrs, b_ptrs, b_ptrs, b_ptrs,
            BLOCK_SIZE_K * stride_a_k, 0, BLOCK_SIZE_K * stride_b_k, 0,
            "pointer", "pointer", False, False, False,
        )

    accumulator = acc_finalize(accumulator, "dot", BLOCK_SIZE_N, SWAP_AB) * a_s * b_s
    store_row(C, accumulator, pid_n, stride_c_n, BLOCK_SIZE_M, BLOCK_SIZE_N)


# The MXFP4/MXFP8 (and packed-activation) splits key themselves — the tuner appends every tensor
# arg's dtype to its cache key. BLOCK_SIZE_M is always 1 here (per-token decode), so plain `dot`
# is excluded (only scalar / dot_scaled-swap are emitted); the swapped dot helper stays
# implemented for future shapes but is not fielded. Swap verdicts are B200 (sm_100) — re-measure
# on H100 or the target device before inheriting.
def _rebind_batched_mx_bs_descriptor(nargs):
    """Per-config pre_hook: size the swizzled weight-scale descriptor box to one 128-row block
    (the BN=128 fp4 dot_scaled bulk load). BN<128 (fp8 scalar) pointer-gathers instead and never
    reads the descriptor. Only under SWIZZLED_SCALES; the un-swizzled path keeps its dummy box."""
    if not nargs.get("SWIZZLED_SCALES"):
        return
    rep = max(1, nargs["BLOCK_SIZE_N"] // 128)
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
    ["N", "K", "S", "INPUT_RECIPE", "SWIZZLED_SCALES"],
    n_trials=100,
    # BK-within-K + the sm_10x MMA-shape guards (swapped dot_scaled needs BN >= 128 for the
    # native scaled-MMA; smaller-BN swap configs never win and mislead the TPE).
    prune_configs_by={
        "early_config_prune": compose_pruners(
            mx_config_pruner("K", "N"), swizzled_scale_config_pruner(), smem_pruner()
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
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
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
    # GATE stacks gate|up into one weight tile (the up block sits N rows away), oriented by
    # SWAP_AB; load_weight reads value + scale (swizzled/un-swizzled hidden) off these pointers.
    b_ptrs = weight_tile_ptrs(
        B, offs_bn, offs_kb, N * stride_b_n, stride_b_n, stride_b_k, GATE, SWAP_AB
    )
    accumulator = acc_init(COMPUTE_MODE, BLOCK_SIZE_M, n_width, SWAP_AB)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a, a_scale = load_act(
            "mx", a_ptrs, as_ptrs, AsGlobal, None, None, 0, 0, 0, "pointer",
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_K=BLOCK_SIZE_K, SCALE_GROUP_K=SCALE_GROUP_K,
            INPUT_RECIPE=INPUT_RECIPE,
        )
        b, b_s = load_weight(
            "mx", b_ptrs, Bs, None, b_ptrs, 0, 0, "pointer", SWAP_AB, GATE, PER_EXPERT=True,
            bs_descriptor=BSDescriptor, bs_ptr=Bs, expert_id=expert_id, pid_n=pid_n, k=k, N=N, K=K,
            stride_bs_e=stride_bs_e, stride_bs_n=stride_bs_n, stride_bs_k=stride_bs_k,
            BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, SCALE_GROUP_K=SCALE_GROUP_K,
            SWIZZLED_SCALES=SWIZZLED_SCALES, WEIGHT_VALUES_PER_BYTE=WEIGHT_VALUES_PER_BYTE,
        )
        accumulator = accumulate(
            accumulator, a, a_scale, b, b_s, "mx", COMPUTE_MODE, SWAP_AB, False,
            BLOCK_SIZE_M, n_width, BLOCK_SIZE_K, SCALE_GROUP_K,
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
    # global g_a·g_b on the accumulator — one multiply (only the product matters here; g_a alone is
    # used solely by the inline-quant arm). None folds out at trace time.
    if AsBsGlobal is not None:
        accumulator = accumulator * tl.load(AsBsGlobal + expert_id).to(tl.float32)

    gemm_epilogue(
        C, Cs, accumulator, out_row, pid_n, 0, out_row, 1, stride_c_n, stride_cs_m, stride_cs_n,
        BLOCK_SIZE_M, BLOCK_SIZE_N, GATE, OUTPUT_RECIPE, SCALE_GROUP_K,
        ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE,
        COMPUTE_MODE=COMPUTE_MODE, SWAP_AB=SWAP_AB, FAKE_BATCH=True, CsGlobal=CsGlobal,
    )


@bayesian_autotune(
    get_accelerator_autotuning_configs(tune_block_nk=True, swap_ab=True),
    # S (routed rows) keyed like the fp8/mx batched siblings — decode re-tunes per batch;
    # GATE keys the gate|up arm separately (its stacked dot is 2*BN wide).
    ["N", "K", "S", "GATE"],
    n_trials=100,
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
        B,  # dummy Bs (no scales); stride 0 keeps the advance a no-op
        ExpertIds,
        GatherIdx,
        ScatterIdx,
        stride_a_m,
        stride_b_e,
        stride_c_m,
        0,
        stride_eid,
    )
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= num_experts:
        return

    n_width: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = operand_tile_ptrs(A, tl.arange(0, BLOCK_SIZE_M) * 0, offs_k, stride_a_m, stride_a_k, "pointer", True)
    # GATE stacks gate|up into one weight tile (the up block sits N rows away);
    # GATE=False -> the plain oriented tile.
    b_ptrs = weight_tile_ptrs(
        B, offs_bn, offs_k, N * stride_b_n, stride_b_n, stride_b_k, GATE, SWAP_AB
    )

    accumulator = acc_init("dot", BLOCK_SIZE_M, n_width, SWAP_AB)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a, _ = load_act("full_precision", a_ptrs, a_ptrs, None, None, None, 0, 0, 0, "pointer")
        w, _ = load_weight(
            "full_precision", b_ptrs, b_ptrs, None, b_ptrs, 0, 0, "pointer", SWAP_AB, GATE,
            BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        accumulator = accumulate(
            accumulator, a, a, w, w, "full_precision", "dot", SWAP_AB, False,
            BLOCK_SIZE_M, n_width, BLOCK_SIZE_K,
        )
        a_ptrs, _, b_ptrs, _, _, _ = advance_ptrs(
            a_ptrs, a_ptrs, b_ptrs, b_ptrs, b_ptrs, b_ptrs,
            BLOCK_SIZE_K * stride_a_k, 0, BLOCK_SIZE_K * stride_b_k, 0,
            "pointer", "pointer", False, False, False,
        )

    gemm_epilogue(
        C, C, accumulator, out_row, pid_n, 0, out_row, 1, stride_c_n, 1, 1,
        BLOCK_SIZE_M, BLOCK_SIZE_N, GATE, None, BLOCK_SIZE_K,
        ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE,
        COMPUTE_MODE="dot", SWAP_AB=SWAP_AB, FAKE_BATCH=True,
    )


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_block_dynamic_fp8_matmul_batched"),
    mutates_args=(),
    opaque=True,
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
    simulate_unfused: bool = False,
    input_recipe: str | None = None,
    output_recipe: str | None = None,
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
    assert input_recipe in (None, "fp8"), (
        f"block-dynamic activations are E4M3 ('fp8'), got {input_recipe!r}"
    )
    assert output_recipe in (None, "fp8"), (
        f"the block-dynamic recipe requantizes to 'fp8', got {output_recipe!r}"
    )
    requant = output_recipe is not None
    # the requantized intermediate's scale groups follow gate_up's block_n, and the
    # down consumes per-block_k — a non-square block recipe would misalign them
    assert not requant or block_size[0] == block_size[1], (
        f"the fused 'fp8' requant needs square quant blocks, got {block_size}"
    )
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
        Cs = expert_ids  # general dummy pointer; unread (no OUTPUT_RECIPE), strides literal

    grid = (S, triton.cdiv(N, block_n))

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_block_dynamic_fp8_matmul_batched_kernel)[
            grid
        ](
            A_q,
            A_s,
            B,
            bs_u8,
            C,
            Cs,
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
            expert_ids.stride(0),
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
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
    activation_scale: torch.Tensor,
    B: torch.Tensor,
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
) -> list[torch.Tensor]:
    """Block-scale batched FP8 matmul with a static (per-tensor calibrated) activation scale — the
    block-dynamic batched sibling with the 2D ``block_static`` recipe. ``A`` is raw here: the op
    quantizes it against the scalar ``activation_scale`` (offline), the kernel applies the per-block
    weight scales in the K-loop and the scalar once post-loop. Returns ``[C]``, or ``[C, Cs]`` under
    ``output_recipe="fp8"`` (the per-row output scale is independent of the per-tensor input scale).

    A:  (rows, K) raw bf16/fp16 activations — rows addressed via ``gather_idx``
    activation_scale: scalar / (1,) — the calibrated per-tensor activation scale
    B:  (num_experts, N, K) FP8 weights; under ``gate`` the (num_experts, 2N, K) gate|up stack
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
    assert input_recipe in (None, "fp8"), (
        f"block-static activations are E4M3 ('fp8'), got {input_recipe!r}"
    )
    assert output_recipe in (None, "fp8"), (
        f"the block-static recipe requantizes to 'fp8', got {output_recipe!r}"
    )
    requant = output_recipe is not None
    assert not requant or block_n == block_k, (
        f"the fused 'fp8' requant needs square quant blocks, got {block_size}"
    )

    As = activation_scale.reshape(1).to(torch.float32)
    bs_u8 = ue8m0_as_uint8(Bs)
    # Pre-quantize the raw activations against the calibrated scalar (offline; the kernel folds
    # the scalar back post-loop).
    A_q = (A.to(torch.float32) / As).to(FP8_DTYPE)
    if requant:
        C = A.new_empty(S, N, dtype=FP8_DTYPE)
        Cs = torch.empty(S, N // block_n, device=A.device, dtype=bs_u8.dtype)
    else:
        C = A.new_empty(S, N, dtype=output_dtype)
        Cs = expert_ids  # dummy pointer; unread (no OUTPUT_RECIPE), strides literal

    grid = (S, triton.cdiv(N, block_n))

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_block_static_fp8_matmul_batched_kernel)[
            grid
        ](
            A_q,
            As,
            B,
            bs_u8,
            C,
            Cs,
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
            expert_ids.stride(0),
            num_experts,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
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
        compile_time_only_triton_wrap(w8a8_tensor_dynamic_fp8_matmul_batched_kernel)[
            grid
        ](
            qA,
            As,
            B,
            bs_u8,
            C,
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
            expert_ids.stride(0),
            num_experts=num_experts,
        )

    return C


@compile_time_only_triton_op(
    add_op_namespace_prefix("mx_dynamic_matmul_batched"), mutates_args=(), opaque=True
)
def mx_dynamic_matmul_batched(
    A: torch.Tensor,
    As: torch.Tensor | None,
    B: torch.Tensor,
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
    assert A.ndim == 2 and B.ndim == 3 and Bs.ndim in (3, 5)  # 5D = pre-swizzled SWIZZLE_32_4_4
    assert expert_ids.ndim == 1
    # A raw (As None) -> quantized inline in the kernel (decode-free UE8M0); pre-quantized
    # (As given, e.g. the down reading a requantized intermediate) -> loaded with its scales
    # (the kernel folds on As is None).
    # the kernel quantizes raw A inline on this grid (fp4 recipes pack in-register);
    # NVFP4 batched runs on the software arms — decode grid BM <= 16 < the native
    # mxf4nvf4 M=128 staging (scalar / swap-scalar column-unpack + E4M3 scale decode)
    input_recipe = resolve_input_recipe(input_recipe, output_recipe, Bs)
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
        Cs = expert_ids  # general dummy pointer; unread (no OUTPUT_RECIPE), strides literal

    def grid(META):
        return (S, triton.cdiv(N, META["BLOCK_SIZE_N"]))

    # NVFP4 accumulator correction: the per-expert g_a·g_b product folded onto the fp32 accumulator.
    input_global_scale = combine_global_scales(a_global_scale, b_global_scale, B.shape[0])
    with device_context(a_u8.device):
        compile_time_only_triton_wrap(mx_dynamic_matmul_batched_kernel)[grid](
            a_u8,
            as_u8,  # None when raw (A quantized inline)
            b_u8,
            bs_u8,
            bs_descriptor,
            C,
            Cs,
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
    assert input_recipe is None and output_recipe is None, (
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
        compile_time_only_triton_wrap(full_precision_matmul_batched_kernel)[grid](
            A,
            B,
            C,
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


def matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor | list[torch.Tensor] | None = None,
    Bs: torch.Tensor | list[torch.Tensor] | None = None,
    *,
    expert_ids: torch.Tensor,
    epilogue: Epilogue | None = None,
    quantization: Quantization | None = None,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
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
    transform (gate|up + GLU). ``As``/``Bs`` are each a bare block-scale tensor or — for
    two-level NVFP4 (always the canonical form, ``nvfp4_quantize_two_level``) — a
    ``[block, global]`` pair, where ``global`` is the fp32 per-tensor (weights: per-expert
    ``(E,)``) second-level scale; the op folds ``g_a · g_b`` onto the accumulator. The
    activation global ``g_a`` is CALIBRATED (the checkpoint's ``input_scale``):
    ``As = [None, g_a]`` with a raw ``A`` has the op quantize ``A / g_a`` per block;
    ``As = [block, g_a]`` is the matching pre-quantized form. Under NVFP4 ``output_recipe`` the
    fused requant normalizes the GLU intermediate by the PROVIDED ``output_global_scale`` (the next
    proj's calibrated ``input_scale``) before the block quant and returns ``[C, Cs]``; the down
    consumes it as ``As = [Cs, output_global_scale]``. ``gather_idx``/``scatter_idx`` (each None or a ``(S,)`` map)
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
    # scales may arrive as [block, global] pairs (two-level NVFP4); split them apart.
    # As=[None, g_a] is the canonical raw-A form: the op quantizes A/g_a per block.
    As, a_global_scale = split_scale(As)
    Bs, b_global_scale = split_scale(Bs)
    assert (a_global_scale is None and b_global_scale is None) or (Bs is not None and is_mx(B, Bs)), (
        "a [block, global] two-level scale pair is NVFP4-only (MX weights)"
    )
    if As is not None and As.numel() == 1:
        # static (per-tensor calibrated) activation quant: a per-tensor scalar As for block-scale FP8
        # weights — the caller hands raw A, the op quantizes it against the scalar (As IS the scale).
        assert Bs is not None and not is_mx(B, Bs) and weight_block_size(B, Bs) is not None, (
            "a per-tensor scalar As (static activation scale) needs block-scale FP8 weights"
        )
        out = w8a8_block_static_fp8_matmul_batched(
            A,
            As,
            B,
            Bs,
            expert_ids,
            weight_block_size(B, Bs),
            *ep.as_args(),
            *q.as_args(),
            output_dtype,
            gather_idx,
            scatter_idx,
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
        )
    elif is_mx(B, Bs):
        out = mx_dynamic_matmul_batched(
            A,
            As,
            B,
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
        )
    elif (block_size := weight_block_size(B, Bs)) is None:
        assert not ep.gate, "gate|up fusion is not supported for tensor-wide scales"
        assert q.input_recipe is None and q.output_recipe is None, (
            "tensor-wide supports neither packed activations nor a fused requant"
        )
        out = w8a8_tensor_dynamic_fp8_matmul_batched(
            A, As, B, Bs, expert_ids, output_dtype, gather_idx, scatter_idx
        )
    else:
        out = w8a8_block_dynamic_fp8_matmul_batched(
            A,
            As,
            B,
            Bs,
            expert_ids,
            block_size,
            *ep.as_args(),
            *q.as_args(),
            output_dtype,
            gather_idx,
            scatter_idx,
        )
    # bd/mx/full-precision ops return a list ([C] or [C, Cs]); the tensor op returns a bare tensor.
    if isinstance(out, (list, tuple)):
        return out[0] if len(out) == 1 else tuple(out)
    return out
