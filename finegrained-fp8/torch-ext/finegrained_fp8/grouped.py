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
from .compat import FP8_DTYPE, NIBBLES_PER_BYTE, compile_time_only_triton_op, compile_time_only_triton_wrap, device_context, get_accelerator_autotuning_configs, sm_count, tl_dtype
from .recipes import Epilogue, Quantization, combine_global_scales, e2m1_as_uint8, expert_weight_shape, is_mx, mx_scale_family, normalize_per_expert_scale, resolve_input_recipe, resolve_output_dtype, routed_rows, tokens_per_expert_bucket, ue8m0_as_uint8, validate_dense_operands, weight_block_size
from .tile_layout import build_tile_layout
from .quant import MX_ACT_QUANT, fp8_act_quant_block_dynamic, fp8_act_quant_tensor_wide, mx_act_quant_swizzled_grouped, swizzle_grouped_mx_scales
from .mma import block_dynamic_dot, fp8_dot, mx_compute, static_dot
from .scheduling import resolve_grouped_tile
from .tiles import (
    load_act_block_dynamic,
    load_act_mx,
    load_act_plain,
    load_act_static,
    load_weight_block_dynamic,
    load_weight_mx,
    load_weight_plain,
    load_weight_static,
    operand_tile_ptrs,
    weight_tile_ptrs,
)
from .epilogue import acc_init, gemm_epilogue
from .pruners import affine_scale_warp_spec_pruner, block_dynamic_grouped_matmul_pruner, block_within_dim_pruner, compose_pruners, descriptor_box_pruner, mx_config_pruner, require_moe_dims_aligned, smem_pruner, swizzled_scales_bm_pruner, warp_spec_compile_guard_pruner


def _rebind_grouped_weight_descriptor(nargs):
    """Per-config pre_hook: set the MX weight descriptor box to the tuned
    ``[(2 if GATE else 1), BLOCK_SIZE_N, BLOCK_SIZE_K // values_per_byte]`` over the
    ``(2E|E, N, K_bytes)`` weight view (one box holds both gate|up projections — the
    fused-era TMA form). MUST mutate ``block_shape`` in place — a rebind never reaches
    the launch. No-op for pointer configs (they never read the descriptor)."""
    if nargs.get("B_MEMORY_MODE", "pointer") == "pointer" or isinstance(
        nargs["BDescriptor"], int
    ):
        return
    values_per_byte = 2 if nargs["B"].dtype == torch.uint8 else 1
    nargs["BDescriptor"].block_shape = [
        2 if nargs.get("GATE") else 1,
        nargs["BLOCK_SIZE_N"],
        nargs["BLOCK_SIZE_K"] // values_per_byte,
    ]


def _rebind_grouped_act_descriptor(nargs):
    """Per-config pre_hook: set the activation descriptor box to the tuned
    ``[BLOCK_SIZE_M, BLOCK_SIZE_K // act_values_per_byte]`` over the ``(rows, K_bytes)``
    activation matrix. In-place mutate; no-op for pointer-A configs."""
    if nargs.get("A_MEMORY_MODE", "pointer") == "pointer" or isinstance(
        nargs["ADescriptor"], int
    ):
        return
    act_values_per_byte = 2 if nargs["A"].dtype == torch.uint8 else 1
    # tma gather4 loads N independent rows: descriptor_gather requires a 1-row box;
    # the contiguous (no-gather) arm loads the whole [BM, BK_bytes] tile in one box
    nargs["ADescriptor"].block_shape = [
        1 if nargs.get("GatherIdx") is not None else nargs["BLOCK_SIZE_M"],
        nargs["BLOCK_SIZE_K"] // act_values_per_byte,
    ]


def _rebind_grouped_descriptors(nargs):
    """Composite pre_hook: both weight and activation descriptor boxes."""
    _rebind_grouped_weight_descriptor(nargs)
    _rebind_grouped_act_descriptor(nargs)


def build_grouped_operand_descriptors(a_operand, b_operand):
    """Operand host-TMA descriptors for a grouped launch: A box ``[16, 64]``, B box
    ``[1, 128, 64]`` over the ``(2E|E, N, K_bytes)`` weight view. Placeholder boxes, re-bound to
    the tuned tile per config by ``_rebind_grouped_descriptors``."""
    return (
        TensorDescriptor.from_tensor(a_operand, block_shape=[16, 64]),
        TensorDescriptor.from_tensor(b_operand, block_shape=[1, 128, 64]),
    )


def _rebind_grouped_mx_descriptors(nargs):
    """MX composite pre_hook: the operand boxes plus the two SWIZZLE_32_4_4 scale boxes
    ``[1, BLOCK // 128, (BK // SCALE_GROUP_K) // 4, 2, 256]`` over the swizzled
    ``(1, rows // 128, cols // 4, 2, 256)`` views. Activation box is BM // 128 (BM pinned 128);
    the weight box is ``(2 if GATE else 1) * BN // 128`` — the stacked gate|up tile is one 2*BN
    block (BN pinned 128 under GATE by ``swizzled_scales_bm_pruner``). Mutate in place. Both scale
    descriptors are None on the un-swizzled arm (affine read — one SWIZZLED_SCALES flag governs both
    operands), so their boxes are skipped."""
    _rebind_grouped_descriptors(nargs)
    rep_k = (nargs["BLOCK_SIZE_K"] // nargs["SCALE_GROUP_K"]) // 4
    if nargs["ASDescriptor"] is not None:
        nargs["ASDescriptor"].block_shape = [1, nargs["BLOCK_SIZE_M"] // 128, rep_k, 2, 256]
    if nargs["BSDescriptor"] is not None:
        # One bulk-load spans (2 if GATE) * BN//128 blocks: GATE reads the block-interleaved gate|up
        # pair ([g,u] adjacent) as one 2*BN tile. BN<128 (non-gate, non-128 N) reads via the per-row
        # pointer gather instead — clamp the box to one block so the (unread) descriptor keeps a valid,
        # non-degenerate shape (a 0-block box traps the descriptor-encoding pass).
        bn_blocks = (2 if nargs.get("GATE") else 1) * max(nargs["BLOCK_SIZE_N"] // 128, 1)
        nargs["BSDescriptor"].block_shape = [1, bn_blocks, rep_k, 2, 256]
    # Swizzled requant output (Cs is a descriptor): the store tile is [1, 1, rep_n, 2, 256], and
    # rep_n = (BN // SCALE_GROUP_K) // 4 depends on the tuned BLOCK_SIZE_N and the group size — so
    # rebind per config, else nvfp4 (group-16 -> rep_n=2) mismatches the [.,.,1,.,.] build default.
    if nargs["CSDescriptor"] is not None:
        rep_n = (nargs["BLOCK_SIZE_N"] // nargs["SCALE_GROUP_K"]) // 4
        nargs["CSDescriptor"].block_shape = [1, 1, rep_n, 2, 256]


@bayesian_autotune(
    # No SWAP_AB axis: the descriptor loads the natural-orientation ((2|1), BN, BK) gate|up box
    # and transposes once to the same K-major tile the pointer arm builds. Both memory axes are
    # emitted; the tuner routes per key. Swap-coupling verdicts are B200 (sm_100) — re-chart on
    # H100 or the target device before reusing SWAP_AB here.
    get_accelerator_autotuning_configs(
        warp_spec=True,
        tune_block_m=True,
        a_memory_modes=("descriptor", "pointer"),
        b_memory_modes=("descriptor", "pointer"),
        pre_hook=_rebind_grouped_descriptors,
    ),
    # GATE keys the gate|up arm separately: its dot is 2*BN wide, a different tile optimum.
    ["N", "K", "tokens_per_expert_bit_length", "GATE"],
    n_trials=100,
    # Pipeliner-race guard: per launch-BM, WS-only at BM >= 64 and non-WS below (see the pruner).
    prune_configs_by={
        "early_config_prune": compose_pruners(
            block_dynamic_grouped_matmul_pruner(),
            descriptor_box_pruner(),
        )
    },
)
@triton.jit
def w8a8_block_dynamic_fp8_matmul_grouped_kernel(
    A,  # (num_tokens, K) E4M3 activations (pre-quantized once by the wrapper), any row order
    ADescriptor,  # host TMA descriptor over A (rows, K), box (BM, BK); read iff A_MEMORY_MODE != "pointer"
    As,  # (S, K // BLOCK_SIZE_K) fp32 per-row, per-K-block activation scales
    B,  # (num_experts, N, K) FP8 weights; under GATE the (num_experts, 2N, K) gate|up stack
    BDescriptor,  # host TMA descriptor over B viewed (2E|E, N, K), box ((2|1), BN, BK); read iff B_MEMORY_MODE != "pointer"
    Bs,  # (num_experts, N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales (2N under GATE)
    C,  # (S, N) output; under an OUTPUT_RECIPE the FP8-requantized intermediate
    Cs,  # (S, N // BLOCK_SIZE_N) per-row, per-block output scale; written iff OUTPUT_RECIPE
    GatherIdx,  # (S,) int32 — sorted position -> source row of A; read only when not None
    ScatterIdx,  # (S,) int32 — sorted position -> destination row of C; read only when not None
    ExpertStart,  # (NUM_EXPERTS_POW2 + 1,) int32 — cumulative row starts, S sentinel
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
    num_experts,
    tokens_per_expert_bit_length,  # autotune key only (log2 avg-tokens bucket); unused in body
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS_POW2: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
    A_MEMORY_MODE: tl.constexpr = "pointer",
    B_MEMORY_MODE: tl.constexpr = "pointer",
    # Gate|up fusion epilogue (GATE=False -> plain grouped GEMM, every arm below folds out)
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    # the output recipe name, same vocabulary as Quantization (None | "fp8")
    OUTPUT_RECIPE: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
):
    """Block-scale grouped FP8 expert matmul kernel — persistent grid-stride over tiles.

    Each M-tile maps to its owning expert via ``ExpertStart`` and gathers its rows
    through ``GatherIdx`` — the expert sort is virtual, ``A`` arrives in any row order.
    Activations arrive pre-quantized (one pass in the wrapper — the
    inline per-N-tile quant would repeat N//BN times per element; see the fused kernels' log).

    ``GATE`` fuses the MoE gate|up projection: ``B`` is the ``(E, 2N, K)`` gate|up stack
    (``N`` = per-projection width), each tile loads gate + up as one ``[BK, 2*BN]`` dot, and
    the epilogue splits, applies the ``ACT_FN``/SwiGLU ``glu``, and — under an ``OUTPUT_RECIPE`` — FP8-
    requantizes the intermediate into ``C`` + per-row ``Cs``. ``GATE=False`` is the plain
    grouped GEMM (down projection = plain GEMM with an output scatter); every gate arm folds
    out at compile time, leaving the plain path bit-identical.

    UE8M0 scales (activations power-of-two, weights UE8M0) on a native-M tile
    (``BLOCK_SIZE_M >= 128``) fold the 128-group scales into a tcgen05 ``dot_scaled`` MMA —
    the same broadcast trick as the 2D block-dynamic kernel (``block_dynamic_dot``), covering
    the single (down) and stacked 2*BN (gate|up) tiles alike. fp32 scales, or a narrow M
    tile, keep the plain ``tl.dot`` + software rescale."""
    USE_DOT_SCALED: tl.constexpr = (As.dtype.element_ty == tl.uint8) and (
        BLOCK_SIZE_M >= 128
    )
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
        ExpertStart, NUM_EXPERTS_POW2, BLOCK_SIZE_M
    )
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn, row0, n_off, m_start = (
            resolve_grouped_tile(
                tile_id,
                num_n_tiles,
                exp_start,
                freqs,
                tile_start_excl,
                e_offs,
                GatherIdx,
                ScatterIdx,
                BLOCK_SIZE_N,
                BLOCK_SIZE_M,
                GATE,
            )
        )
        a_ptrs = operand_tile_ptrs(A, in_row, offs_k, stride_a_m, stride_a_k, A_MEMORY_MODE, True)
        as_ptrs = As + in_row * stride_as_m
        # GATE stacks gate (rows [0, N)) and up (rows [N, 2N)) into one [BK, 2*BN] tile — the
        # up block sits N rows away (N = per-projection width). GATE=False -> plain [BK, BN].
        b_ptrs = weight_tile_ptrs(
            B + expert_id64 * stride_b_e,
            offs_bn,
            offs_k,
            N * stride_b_n,
            stride_b_n,
            stride_b_k,
            GATE,
            False,
        )
        # gate scale block, up scale block (N tiles away); non-GATE reads the single block
        # (up_s_ptr dead, == gate_s_ptr).
        gate_s_ptr = Bs + expert_id64 * stride_bs_e + pid_n * stride_bs_n
        up_s_ptr = Bs + expert_id64 * stride_bs_e + (num_n_tiles + pid_n) * stride_bs_n

        acc = acc_init("dot", BLOCK_SIZE_M, (2 if GATE else 1) * BLOCK_SIZE_N, False)
        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
            a, a_s = load_act_block_dynamic(
                a_ptrs, as_ptrs, row_mask, row_mask, ADescriptor, m_start, k * BLOCK_SIZE_K,
                in_row, k, A_MEMORY_MODE, GatherIdx is not None, True, False,
            )
            w, w_s = load_weight_block_dynamic(
                b_ptrs, BDescriptor, gate_s_ptr, None, up_s_ptr, row0, n_off, k * BLOCK_SIZE_K, k,
                stride_bs_k, GATE, True, B_MEMORY_MODE, False, BLOCK_SIZE_N, BLOCK_SIZE_K,
            )
            acc = block_dynamic_dot(acc, a, a_s, w, w_s, BLOCK_SIZE_K, False, USE_DOT_SCALED, False)
            a_ptrs += BLOCK_SIZE_K * stride_a_k
            b_ptrs += BLOCK_SIZE_K * stride_b_k

        gemm_epilogue(
            C,
            Cs,
            acc,
            out_row,
            pid_n,
            tile_id // num_n_tiles,
            row_mask,
            stride_c_m,
            stride_c_n,
            stride_cs_m,
            stride_cs_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GATE,
            OUTPUT_RECIPE,
            1,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SIMULATE_UNFUSED,
            INTERMEDIATE_DTYPE,
        )


@bayesian_autotune(
    # Mirrors the block-dynamic grouped decorator (same (2|1, BN, BK) gate|up descriptor box, no
    # SWAP_AB axis, both memory axes emitted). No dot_scaled here — the static activation scale is
    # a per-tensor scalar applied post-loop (see the kernel), so the K-loop is a plain dot.
    get_accelerator_autotuning_configs(
        warp_spec=True,
        tune_block_m=True,
        a_memory_modes=("descriptor", "pointer"),
        b_memory_modes=("descriptor", "pointer"),
        pre_hook=_rebind_grouped_descriptors,
    ),
    ["N", "K", "tokens_per_expert_bit_length", "GATE"],
    n_trials=100,
    # Pipeliner-race WS guard (per launch-BM: forces WS where the default pipeliner races at
    # BM>=64, keeps non-WS below) + descriptor-box limits.
    prune_configs_by={
        "early_config_prune": compose_pruners(
            block_dynamic_grouped_matmul_pruner(),
            descriptor_box_pruner(),
        )
    },
)
@triton.jit
def w8a8_block_static_fp8_matmul_grouped_kernel(
    A,  # (num_tokens, K) E4M3 activations (pre-quantized against the static scale by the wrapper)
    ADescriptor,  # host TMA descriptor over A (rows, K), box (BM, BK); read iff A_MEMORY_MODE != "pointer"
    As,  # scalar — static per-tensor activation scale (calibration-time)
    B,  # (num_experts, N, K) FP8 weights; under GATE the (num_experts, 2N, K) gate|up stack
    BDescriptor,  # host TMA descriptor over B viewed (2E|E, N, K), box ((2|1), BN, BK); read iff B_MEMORY_MODE != "pointer"
    Bs,  # (num_experts, N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales (2N under GATE)
    C,  # (S, N) output; under an OUTPUT_RECIPE the FP8-requantized intermediate
    Cs,  # (S, N // BLOCK_SIZE_N) per-(row, block) output scale; written iff OUTPUT_RECIPE
    GatherIdx,  # (S,) int32 — sorted position -> source row of A; read only when not None
    ScatterIdx,  # (S,) int32 — sorted position -> destination row of C; read only when not None
    ExpertStart,  # (NUM_EXPERTS_POW2 + 1,) int32 — cumulative row starts, S sentinel
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
    num_experts,
    tokens_per_expert_bit_length,  # autotune key only (log2 avg-tokens bucket); unused in body
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS_POW2: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
    A_MEMORY_MODE: tl.constexpr = "pointer",
    B_MEMORY_MODE: tl.constexpr = "pointer",
    # Gate|up fusion epilogue (GATE=False -> plain grouped GEMM, every arm below folds out)
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    OUTPUT_RECIPE: tl.constexpr = None,  # None | "fp8" (per-(row, block) requant of the intermediate)
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
):
    """Block-scale grouped FP8 expert matmul with a static (per-tensor) activation scale —
    persistent grid-stride, the block-dynamic sibling's structure (virtual expert sort via
    ``ExpertStart``/``GatherIdx``, ``GATE`` gate|up fusion, host-TMA memory modes) with the 2D
    ``block_static`` recipe: ``A`` arrives pre-quantized against the calibrated scalar, per-block
    weight scales apply per-K-tile (plain ``tl.dot`` + software rescale, ``accumulate("static")``),
    and the scalar activation scale multiplies the accumulator once after the loop. GATE=False is
    the plain grouped GEMM (down projection), bit-identical."""
    a_s_static = tl.load(As)  # per-tensor static activation scale, applied post-loop
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
        ExpertStart, NUM_EXPERTS_POW2, BLOCK_SIZE_M
    )
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn, row0, n_off, m_start = (
            resolve_grouped_tile(
                tile_id,
                num_n_tiles,
                exp_start,
                freqs,
                tile_start_excl,
                e_offs,
                GatherIdx,
                ScatterIdx,
                BLOCK_SIZE_N,
                BLOCK_SIZE_M,
                GATE,
            )
        )
        a_ptrs = operand_tile_ptrs(A, in_row, offs_k, stride_a_m, stride_a_k, A_MEMORY_MODE, True)
        b_ptrs = weight_tile_ptrs(
            B + expert_id64 * stride_b_e,
            offs_bn,
            offs_k,
            N * stride_b_n,
            stride_b_n,
            stride_b_k,
            GATE,
            False,
        )
        gate_s_ptr = Bs + expert_id64 * stride_bs_e + pid_n * stride_bs_n
        up_s_ptr = Bs + expert_id64 * stride_bs_e + (num_n_tiles + pid_n) * stride_bs_n

        acc = acc_init("dot", BLOCK_SIZE_M, (2 if GATE else 1) * BLOCK_SIZE_N, False)
        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
            a, a_dead = load_act_static(
                a_ptrs, ADescriptor, m_start, k * BLOCK_SIZE_K, row_mask, in_row, 0.0,
                A_MEMORY_MODE, GatherIdx is not None,
            )
            w, w_s = load_weight_static(
                b_ptrs, BDescriptor, gate_s_ptr, None, up_s_ptr, row0, n_off, k * BLOCK_SIZE_K, k,
                stride_bs_k, GATE, True, B_MEMORY_MODE, False, BLOCK_SIZE_N, BLOCK_SIZE_K,
            )
            acc = static_dot(acc, a, w, w_s, False, BLOCK_SIZE_K, False)
            # Explicit advance like the block-dynamic grouped sibling — advance_ptrs does not
            # compile in the grouped loop (the weight block scale is read via k inside the leaf,
            # and the static act scale is a scalar; only the operand pointers move).
            a_ptrs += BLOCK_SIZE_K * stride_a_k
            b_ptrs += BLOCK_SIZE_K * stride_b_k

        acc = acc * a_s_static
        gemm_epilogue(
            C,
            Cs,
            acc,
            out_row,
            pid_n,
            tile_id // num_n_tiles,
            row_mask,
            stride_c_m,
            stride_c_n,
            stride_cs_m,
            stride_cs_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GATE,
            OUTPUT_RECIPE,
            1,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SIMULATE_UNFUSED,
            INTERMEDIATE_DTYPE,
        )


@bayesian_autotune(
    get_accelerator_autotuning_configs(
        tune_block_nk=True,
        warp_spec=True,
        tune_block_m=True,
        a_memory_modes=("descriptor", "pointer"),
        b_memory_modes=("descriptor", "pointer"),
        pre_hook=_rebind_grouped_descriptors,
    ),
    # GATE keys the gate|up arm separately (its dot is 2*BN wide, a different tile optimum).
    ["N", "K", "tokens_per_expert_bit_length", "GATE"],
    n_trials=100,
    # BLOCK_SIZE_K/N are tuned axes; the K-loop is maskless and the N-tile store is
    # row-masked only — veto non-dividing tiles on both. WS is a pure perf axis here
    # (non-WS is the validated state), compile-guarded. The GATE arm's stacked width-512
    # dots (BN=256) are clean — probed bit-exact 2026-07-14; the oversized-smem ones fail
    # benignly at launch and self-prune as inf.
    prune_configs_by={
        "early_config_prune": compose_pruners(
            block_within_dim_pruner("K"),
            block_within_dim_pruner("N", "BLOCK_SIZE_N"),
            warp_spec_compile_guard_pruner(),
            descriptor_box_pruner(),
            smem_pruner(),
        )
    },
)
@triton.jit
def w8a8_tensor_dynamic_fp8_matmul_grouped_kernel(
    A,  # (num_tokens, K) pre-quantized FP8 activations, any row order
    ADescriptor,  # host TMA descriptor over A (rows, K), box (BM, BK); read iff A_MEMORY_MODE != "pointer"
    As,  # (S,) per-token activation scales
    B,  # (num_experts, N, K) FP8 weights; under GATE the (num_experts, 2N, K) gate|up stack
    BDescriptor,  # host TMA descriptor over B viewed (2E|E, N, K), box ((2|1), BN, BK); read iff B_MEMORY_MODE != "pointer"
    Bs,  # (num_experts, 1, 1) per-tensor weight scales (one scalar covers the gate|up stack)
    C,  # (S, N) output; under GATE the bf16 GLU intermediate
    Cs,  # unused dummy (tensor-wide has no fused requant); kept for the shared epilogue signature
    GatherIdx,  # (S,) int32 — sorted position -> source row of A; read only when not None
    ScatterIdx,  # (S,) int32 — sorted position -> destination row of C; read only when not None
    ExpertStart,  # (NUM_EXPERTS_POW2 + 1,) int32 — cumulative row starts, S sentinel
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
    stride_cs_m,
    stride_cs_n,
    num_experts,
    tokens_per_expert_bit_length,  # autotune key only (log2 avg-tokens bucket); unused in body
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS_POW2: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
    A_MEMORY_MODE: tl.constexpr = "pointer",
    B_MEMORY_MODE: tl.constexpr = "pointer",
    # Gate|up fusion epilogue (GATE=False -> plain grouped GEMM). No fused requant here
    # (tensor-wide down needs a per-token whole-row scale a per-tile epilogue can't form).
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    OUTPUT_RECIPE: tl.constexpr = None,  # tensor-wide has no fused requant (kept None)
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
):
    """Tensor-scale grouped FP8 expert matmul kernel — persistent grid-stride over tiles.

    Grouped expert scheduling with pre-quantized activations plus per-token activation scales
    and one per-expert tensor weight scale. ``GATE`` fuses the gate|up projection (``B`` the
    ``(E, 2N, K)`` stack, one scale for both) into a ``[BK, 2*BN]`` dot + SwiGLU ``glu``,
    emitting the bf16 intermediate; ``GATE=False`` is the plain GEMM (bit-identical)."""
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
        ExpertStart, NUM_EXPERTS_POW2, BLOCK_SIZE_M
    )
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn, row0, n_off, m_start = (
            resolve_grouped_tile(
                tile_id,
                num_n_tiles,
                exp_start,
                freqs,
                tile_start_excl,
                e_offs,
                GatherIdx,
                ScatterIdx,
                BLOCK_SIZE_N,
                BLOCK_SIZE_M,
                GATE,
            )
        )
        a_ptrs = operand_tile_ptrs(A, in_row, offs_k, stride_a_m, stride_a_k, A_MEMORY_MODE, True)
        # GATE stacks gate|up into one [BK, 2*BN] tile (one per-tensor scale covers both);
        # the up block sits N rows away. GATE=False -> the plain [BK, BN] tile.
        b_ptrs = weight_tile_ptrs(
            B + expert_id64 * stride_b_e,
            offs_bn,
            offs_k,
            N * stride_b_n,
            stride_b_n,
            stride_b_k,
            GATE,
            False,
        )
        a_s = tl.load(As + in_row * stride_as_m, mask=row_mask, other=0.0)
        b_s = tl.load(Bs + expert_id64 * stride_bs_e)

        acc = acc_init("dot", BLOCK_SIZE_M, (2 if GATE else 1) * BLOCK_SIZE_N, False)
        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
            a, _as = load_act_plain(
                a_ptrs, ADescriptor, m_start, k * BLOCK_SIZE_K, row_mask, in_row,
                A_MEMORY_MODE, GatherIdx is not None,
            )
            w, _ws = load_weight_plain(
                b_ptrs, BDescriptor, row0, n_off, k * BLOCK_SIZE_K,
                GATE, True, B_MEMORY_MODE, False, BLOCK_SIZE_N, BLOCK_SIZE_K,
            )
            acc = acc + fp8_dot(a, w, False, BLOCK_SIZE_K)
            a_ptrs += BLOCK_SIZE_K * stride_a_k
            b_ptrs += BLOCK_SIZE_K * stride_b_k
        # per-token activation scale and per-expert tensor weight scale, applied once post-loop
        acc = acc * a_s[:, None] * b_s

        gemm_epilogue(
            C,
            Cs,
            acc,
            out_row,
            pid_n,
            tile_id // num_n_tiles,
            row_mask,
            stride_c_m,
            stride_c_n,
            stride_cs_m,
            stride_cs_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GATE,
            OUTPUT_RECIPE,
            1,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SIMULATE_UNFUSED,
            INTERMEDIATE_DTYPE,
        )


@bayesian_autotune(
    get_accelerator_autotuning_configs(
        mx=True,
        tune_block_nk=True,
        tune_block_m=True,
        compute_modes=("dot_scaled", "dot"),
        a_memory_modes=("descriptor", "pointer"),
        b_memory_modes=("descriptor", "pointer"),
        pre_hook=_rebind_grouped_mx_descriptors,
        warp_spec=True,  # dot_scaled+WS+TMA compiles+wins on a clean context (was a num_warps=2 misread)
    ),  # prefill: no scalar branch; TMA descriptor vs pointer loads on both operands
    # the MXFP4/MXFP8 (and packed-activation) splits key themselves — the tuner appends
    # every tensor arg's dtype to its cache key (memory and disk);
    # GATE keys the gate|up arm separately (its stacked dot is 2*BN wide, a different tile optimum).
    # SWIZZLED_SCALES splits the pre-swizzled (descriptor) and row-major (affine) arms — they take
    # different memory-mode optima and must not share a tuned config.
    ["N", "K", "tokens_per_expert_bit_length", "GATE", "SWIZZLED_SCALES"],
    n_trials=100,
    # BK-within-K veto + the sm_10x dot_scaled shape/trap gates (this kernel had no
    # pruner while its BK span was {128,256} — the union span's BK=64 rows made the
    # gates load-bearing).
    prune_configs_by={
        "early_config_prune": compose_pruners(
            mx_config_pruner("K", "N"),
            swizzled_scales_bm_pruner(),
            descriptor_box_pruner(),
            smem_pruner(),
            warp_spec_compile_guard_pruner(),
            affine_scale_warp_spec_pruner(),
        )
    },
)
@triton.jit
def mx_dynamic_matmul_grouped_kernel(
    A,  # (num_tokens, K) E4M3 activations (pre-quantized once by the wrapper), any row order
    ADescriptor,  # host TMA descriptor over A (rows, K_bytes), box (BM, BK_bytes); read iff A_MEMORY_MODE != "pointer"
    As,  # activation scales: row-major group scales read affine (gathered per row) iff not SWIZZLED_SCALES; else dummy (swizzled via ASDescriptor)
    ASDescriptor,  # host TMA descriptor over the SWIZZLE_32_4_4, expert-sorted/128-padded A scales; read iff SWIZZLED_SCALES
    B,  # (num_experts, N, K) E4M3 (MXFP8) or (num_experts, N, K // 2) packed E2M1 (MXFP4); 2N under GATE
    BDescriptor,  # host TMA descriptor over B viewed (2E|E, N, K_bytes), box ((2|1), BN, BK_bytes); read iff B_MEMORY_MODE != "pointer"
    Bs,  # (num_experts, N, K // SCALE_GROUP_K) UE8M0 weight scales (2N under GATE)
    BSDescriptor,  # host TMA descriptor over the SWIZZLE_32_4_4 per-expert B scales; read iff SWIZZLED_SCALES
    C,  # (S, N[/2]) output; under an OUTPUT_RECIPE the MX-requantized intermediate
    Cs,  # (S, N // SCALE_GROUP_K) row-major output scale; written iff OUTPUT_RECIPE and not SWIZZLED_OUT
    CSDescriptor,  # SWIZZLE_32_4_4 output-scale descriptor; written iff SWIZZLED_OUT (dummy else), like AS/BS
    AsBsGlobal,  # (num_experts,) fp32 NVFP4 combined global g_a·g_b — recovers on the accumulator (grouped A is pre-quantized by the wrapper, so no in-kernel g_a); read iff not None
    CsGlobal,  # (1,) fp32 NVFP4 output global (next proj's provided input_scale); normalizes the requant; read iff not None
    GatherIdx,  # (S,) int32 — sorted position -> source row of A; read only when not None
    ScatterIdx,  # (S,) int32 — sorted position -> destination row of C; read only when not None
    ExpertStart,  # (NUM_EXPERTS_POW2 + 1,) int32 — cumulative row starts, S sentinel
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
    num_experts,
    tokens_per_expert_bit_length,  # autotune key only (log2 avg-tokens bucket); unused in body
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS_POW2: tl.constexpr,
    NUM_SMS: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
    B_MEMORY_MODE: tl.constexpr = "pointer",
    A_MEMORY_MODE: tl.constexpr = "pointer",
    # Gate|up fusion epilogue (GATE=False -> plain grouped GEMM, every arm below folds out)
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    # the output recipe name, same vocabulary as Quantization (None | "mxfp8" | "mxfp4" | "nvfp4")
    OUTPUT_RECIPE: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
    SWIZZLED_SCALES: tl.constexpr = True,  # scales pre-swizzled (5D weight + matching acts); else affine
    SWIZZLED_OUT: tl.constexpr = False,  # requant emits Cs in SWIZZLE_32_4_4 (descriptor); single source: the wrapper
    WARP_SPEC: tl.constexpr = False,  # tuner axis; dot_scaled+WS+TMA compiles+wins (num_warps%4, BM>=64)
):
    """Unified grouped microscaled expert matmul (MXFP8/MXFP4/NVFP4; a ``uint8`` ``A``
    is caller-packed E2M1 — W4A4 on the native fp4 MMA) — persistent grid-stride.

    Each M-tile maps to its expert via ``ExpertStart`` and gathers its rows through
    ``PermToken`` (virtual sort — ``A`` in any row order). ``A``
    arrives pre-quantized (E4M3 + UE8M0 group-32 scales, one pass in the wrapper — the
    inline per-N-tile quant would repeat N//BN times per element). Each operand's format is
    its dtype (``uint8`` = packed E2M1, two values per byte; else E4M3); ``COMPUTE_MODE``
    picks ``tl.dot_scaled`` vs fp8 ``tl.dot`` + per-group rescale (decode; FP4 unpacks
    E2M1->E4M3 first, lossless).
    """
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
        ExpertStart, NUM_EXPERTS_POW2, BLOCK_SIZE_M
    )
    # uint8 A = caller-provided packed-E2M1 activations (W4A4, native mxf4 MMA — the
    # dtype IS the activation format and keys the autotune cache); else one value per byte.
    ACT_VALUES_PER_BYTE: tl.constexpr = 2 if A.dtype.element_ty == tl.uint8 else 1
    WEIGHT_VALUES_PER_BYTE: tl.constexpr = 2 if B.dtype.element_ty == tl.uint8 else 1
    # Scales are read swizzled (SWIZZLE_32_4_4 bulk off the SA/SB descriptors — the tcgen05 fast
    # path, BM pinned 128, BN pinned 128 under GATE) or affine (row-major, gathered per row), per
    # SWIZZLED_SCALES — one flag for both operands (a swizzled weight is paired with swizzled acts).
    # The offline act-quant emits its scales in the matching layout; the down projection's
    # pre-quantized As arrives in it too.
    #
    # gate|up loads two whole 128-row blocks (gate + up) stacked into a 2*BN weight tile
    # (weight_tile_ptrs' gate arm + the two-slab scale below); the dot is 2*BN wide and the epilogue
    # splits the accumulator into its gate/up halves for SiLU*up. Only the load (which two blocks) and
    # the epilogue (split) depend on GATE — the K-loop body is otherwise uniform.
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    offs_ka = tl.arange(0, BLOCK_SIZE_K // ACT_VALUES_PER_BYTE)
    offs_kb = tl.arange(0, BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE)

    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn, row0, n_off, m_start = (
            resolve_grouped_tile(
                tile_id,
                num_n_tiles,
                exp_start,
                freqs,
                tile_start_excl,
                e_offs,
                GatherIdx,
                ScatterIdx,
                BLOCK_SIZE_N,
                BLOCK_SIZE_M,
                GATE,
            )
        )
        a_ptrs = operand_tile_ptrs(A, in_row, offs_ka, stride_a_m, stride_a_k, A_MEMORY_MODE, True)
        # Non-128 N: the partial last N-tile's pointer-arm rows wrap into B (offs_bn % N) so the load
        # never reads past the expert's N rows; the wrapped columns' output is masked off (N_COLS) in
        # the epilogue. Inert when N % BLOCK_SIZE_N == 0. The descriptor arm uses n_off (OOB-clamped),
        # and the scale rides blk_idx off the padded swizzled block — neither depends on offs_bn.
        offs_bn = offs_bn % N
        kb_off = 0
        ka_off = 0
        # GATE stacks the gate + up 128-blocks (the up block sits N rows away) into a 2*BN tile; a
        # plain tile is the single BN block.
        b_ptrs = weight_tile_ptrs(
            B + expert_id64 * stride_b_e,
            offs_bn,
            offs_kb,
            N * stride_b_n,
            stride_b_n,
            stride_b_k,
            GATE,
            False,
        )

        # Accumulator is 2*BN under GATE (the stacked gate|up dot), BN otherwise — split in the
        # epilogue for GATE.
        acc = acc_init("dot", BLOCK_SIZE_M, (2 if GATE else 1) * BLOCK_SIZE_N, False)
        # Pre-swizzled scales (SWIZZLE_32_4_4, emitted by the offline act-quant / requant): the M
        # tile's expert-sorted, 128-padded scale block is the flat tile index pid_m; the weight scale
        # block (descriptor bulk-load, BN=128) is expert*num_n_tiles + pid_n. Each expert's swizzled
        # slab is num_n_tiles 128-row blocks (non-gate); under GATE it is 2*num_n_tiles, block-interleaved
        # ([g,u] per tile adjacent) so this same block index bulk-loads the stacked gate|up scale.
        pid_m = tile_id // num_n_tiles
        weight_blk = (expert_id64 * num_n_tiles + pid_n).to(tl.int32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
            a, a_s = load_act_mx(
                a_ptrs, As, None, row_mask, row_mask, ADescriptor, m_start, ka_off,
                ASDescriptor, As, in_row, stride_as_m, pid_m, k, 0, K,
                A_MEMORY_MODE, GatherIdx is not None, True, SWIZZLED_SCALES,
                BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K, "mxfp8",
            )
            w, w_s = load_weight_mx(
                b_ptrs, BDescriptor, Bs, None, BSDescriptor, 0, row0, n_off, kb_off,
                weight_blk, expert_id64, pid_n, k, N, K,
                stride_bs_e, stride_bs_n, stride_bs_k,
                GATE, True, False, B_MEMORY_MODE, False, SWIZZLED_SCALES,
                BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, WEIGHT_VALUES_PER_BYTE,
            )
            acc = mx_compute(
                acc, a, a_s, w, w_s, COMPUTE_MODE,
                BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, False,
            )
            a_ptrs += (BLOCK_SIZE_K // ACT_VALUES_PER_BYTE) * stride_a_k
            ka_off += BLOCK_SIZE_K // ACT_VALUES_PER_BYTE
            b_ptrs += (BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE) * stride_b_k
            kb_off += BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE

        # NVFP4 two-level: block e4m3 scales rode through dot_scaled; recover the combined per-tensor
        # global g_a·g_b on the accumulator — one multiply. None folds out at trace time.
        if AsBsGlobal is not None:
            acc = acc * tl.load(AsBsGlobal + expert_id64).to(tl.float32)

        gemm_epilogue(
            C,
            Cs,
            acc,
            out_row,
            pid_n,
            pid_m,
            row_mask,
            stride_c_m,
            stride_c_n,
            stride_cs_m,
            stride_cs_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GATE,
            OUTPUT_RECIPE,
            SCALE_GROUP_K,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SIMULATE_UNFUSED,
            INTERMEDIATE_DTYPE,
            SWIZZLED_OUT=SWIZZLED_OUT,  # single source: the wrapper decided it and built Cs to match
            CSDescriptor=CSDescriptor,
            CsGlobal=CsGlobal,
            N_COLS=N,  # mask the partial last N-tile's column tail (non-128 N; inert when N % BN == 0)
        )


@bayesian_autotune(
    get_accelerator_autotuning_configs(
        tune_block_nk=True,
        warp_spec=True,
        tune_block_m=True,
        a_memory_modes=("descriptor", "pointer"),
        b_memory_modes=("descriptor", "pointer"),
        pre_hook=_rebind_grouped_descriptors,
    ),
    # GATE keys the gate|up arm separately (its dot is 2*BN wide, a different tile optimum).
    ["N", "K", "tokens_per_expert_bit_length", "GATE"],
    n_trials=100,
    # BLOCK_SIZE_K is a tuned axis and the K-loop is maskless — veto non-dividing BKs.
    prune_configs_by={
        "early_config_prune": compose_pruners(
            block_within_dim_pruner("K"),
            block_within_dim_pruner("N", "BLOCK_SIZE_N"),
            warp_spec_compile_guard_pruner(),
            descriptor_box_pruner(),
            smem_pruner(),
        )
    },
)
@triton.jit
def full_precision_matmul_grouped_kernel(
    A,  # (num_tokens, K) BF16/FP16 activations, any row order
    ADescriptor,  # host TMA descriptor over A (rows, K), box (BM, BK); read iff A_MEMORY_MODE != "pointer"
    B,  # (num_experts, N, K) weights in A's dtype; under GATE the (num_experts, 2N, K) gate|up stack
    BDescriptor,  # host TMA descriptor over B viewed (2E|E, N, K), box ((2|1), BN, BK); read iff B_MEMORY_MODE != "pointer"
    C,  # (S, N) output; under GATE the GLU intermediate
    GatherIdx,  # (S,) int32 — sorted position -> source row of A; read only when not None
    ScatterIdx,  # (S,) int32 — sorted position -> destination row of C; read only when not None
    ExpertStart,  # (NUM_EXPERTS_POW2 + 1,) int32 — cumulative row starts, S sentinel
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
    num_experts,
    tokens_per_expert_bit_length,  # autotune key only (log2 avg-tokens bucket); unused in body
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS_POW2: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
    # descriptor modes (host-built TMA / in-kernel tensormap) load weight tiles via the
    # descriptor and run the swapped (weights-in-M) loop; "pointer" is the natural loop.
    B_MEMORY_MODE: tl.constexpr = "pointer",
    A_MEMORY_MODE: tl.constexpr = "pointer",
    # Gate|up fusion epilogue (GATE=False -> plain grouped GEMM). No requant arm: the
    # full-precision chain has no quantized intermediate — down consumes the GLU output as is.
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
):
    """Full-precision grouped expert matmul kernel — persistent grid-stride over tiles.

    Grouped expert scheduling over unquantized BF16/FP16 activations and weights: plain
    ``tl.dot`` with fp32 accumulation, no scales anywhere. ``GATE`` fuses the gate|up
    projection (``B`` the ``(E, 2N, K)`` stack) into a ``[BK, 2*BN]`` dot + SwiGLU ``glu``,
    emitting the intermediate; ``GATE=False`` is the plain GEMM (bit-identical)."""
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
        ExpertStart, NUM_EXPERTS_POW2, BLOCK_SIZE_M
    )
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn, row0, n_off, m_start = (
            resolve_grouped_tile(
                tile_id,
                num_n_tiles,
                exp_start,
                freqs,
                tile_start_excl,
                e_offs,
                GatherIdx,
                ScatterIdx,
                BLOCK_SIZE_N,
                BLOCK_SIZE_M,
                GATE,
            )
        )
        a_ptrs = operand_tile_ptrs(A, in_row, offs_k, stride_a_m, stride_a_k, A_MEMORY_MODE, True)
        # GATE stacks gate|up into one [BK, 2*BN] tile; the up block sits N rows away.
        # GATE=False -> the plain [BK, BN] tile. Pointer arm — the descriptor modes
        # fetch the [BN, BK] box at row0 and transpose it instead.
        b_ptrs = weight_tile_ptrs(
            B + expert_id64 * stride_b_e,
            offs_bn,
            offs_k,
            N * stride_b_n,
            stride_b_n,
            stride_b_k,
            GATE,
            False,
        )

        acc = acc_init("dot", BLOCK_SIZE_M, (2 if GATE else 1) * BLOCK_SIZE_N, False)
        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
            a, _as = load_act_plain(
                a_ptrs, ADescriptor, m_start, k * BLOCK_SIZE_K, row_mask, in_row,
                A_MEMORY_MODE, GatherIdx is not None,
            )
            w, _ws = load_weight_plain(
                b_ptrs, BDescriptor, row0, n_off, k * BLOCK_SIZE_K,
                GATE, True, B_MEMORY_MODE, False, BLOCK_SIZE_N, BLOCK_SIZE_K,
            )
            acc = acc + fp8_dot(a, w, False, BLOCK_SIZE_K)
            a_ptrs += BLOCK_SIZE_K * stride_a_k
            b_ptrs += BLOCK_SIZE_K * stride_b_k

        gemm_epilogue(
            C,
            C,  # dummy Cs (no requant arm)
            acc,
            out_row,
            pid_n,
            tile_id // num_n_tiles,
            row_mask,
            stride_c_m,
            stride_c_n,
            1,  # dummy Cs strides
            1,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GATE,
            None,
            1,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SIMULATE_UNFUSED,
            INTERMEDIATE_DTYPE,
        )


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_block_dynamic_fp8_matmul_grouped"),
    mutates_args=(),
    opaque=True,
)
def w8a8_block_dynamic_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor | None,
    Bs: torch.Tensor,
    expert_start: torch.Tensor,
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
    """Block-scale grouped FP8 matmul over expert-sorted positions (per-tile
    gather/scatter, the sort is virtual — see ``compute_grouped_scheduling`` for the maps).
    Activations arrive pre-quantized: the caller owns the act-quant (``fp8_act_quant_block_dynamic``),
    the op is a pure GEMM. The ``gate``/``act_fn``/``swiglu_*``/``requant``/``simulate_unfused``
    flags are the flattened ``Epilogue`` (torch custom ops take only primitive params — the
    ``matmul_grouped`` dispatcher unpacks the bundle here).

    A:  (S, K) pre-quantized FP8 activations — rows addressed via ``gather_idx``
    B:  (num_experts, N, K) FP8 weights; under ``gate`` the (num_experts, 2N, K) gate|up stack
    As: (S, K // block_k) per-row, per-K-block activation scales
    Bs: (num_experts, N // block_n, K // block_k) per-block weight scales (2N under gate)
    expert_start: (num_experts_pow2 + 1,) int32 — cumulative sorted-row starts, S sentinel
    gate: fuse the gate|up projection (SwiGLU ``act_fn``); ``requant`` FP8-requantizes the result
    gather_idx: optional (S,) — sorted position -> source row of A; None = A is expert-sorted
    scatter_idx: optional (S,) — sorted position -> destination row of C; None = C stays expert-sorted

    Returns the ``(S, N)`` output, or — under ``requant`` — the FP8 output plus its
    ``(S, N // block_n)`` per-row, per-N-tile scale tensor.
    """
    validate_dense_operands(A, B)

    _, K = A.shape

    num_experts, n_rows, N = expert_weight_shape(B, gate)
    S = routed_rows(A, gather_idx, scatter_idx, expert_start, num_experts)

    assert len(block_size) == 2, (
        f"block_size must be [block_n, block_k], got {block_size}"
    )
    block_n, block_k = block_size[0], block_size[1]
    require_moe_dims_aligned(N, K, block_n, block_k)
    assert Bs.shape == (num_experts, n_rows // block_n, K // block_k), (
        f"Bs shape {tuple(Bs.shape)} != expected ({num_experts}, {n_rows // block_n}, {K // block_k})"
    )

    output_dtype = resolve_output_dtype(output_dtype, A, As)
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
    # A may arrive raw (As is None) or pre-quantized (As given, e.g. a requantized
    # intermediate handed over between the fused GEMMs). Raw -> quantize here (offline).
    bs_u8 = ue8m0_as_uint8(Bs)
    # UE8M0 weight scales (DeepGEMM-Blackwell) -> quantize the raw activations to UE8M0 too so
    # the kernel folds both group scales into the tcgen05 dot_scaled MMA (else fp32 software).
    if As is None:
        A, As = fp8_act_quant_block_dynamic(
            A, block_k, use_ue8m0=bs_u8.dtype == torch.uint8
        )
    if requant:
        C = A.new_empty(S, N, dtype=FP8_DTYPE)
        # UE8M0 model (ue8m0 weights) -> UE8M0 intermediate scales so the down proj reads
        # power-of-two activation scales and takes its dot_scaled arm (the epilogue infers
        # the format from this dtype); fp32 weights keep fp32 (software) as before.
        cs_dtype = bs_u8.dtype
        Cs = torch.empty(S, N // block_n, device=A.device, dtype=cs_dtype)
    else:
        C = A.new_empty(S, N, dtype=output_dtype)
        Cs = expert_start  # general dummy pointer; unread (no OUTPUT_RECIPE), strides literal
    num_sms = sm_count(A.device.index)
    a_descriptor, b_descriptor = build_grouped_operand_descriptors(
        A, B.view(2 * num_experts if gate else num_experts, N, K)
    )

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_block_dynamic_fp8_matmul_grouped_kernel)[
            (num_sms,)
        ](
            A,
            a_descriptor,
            As,
            B,
            b_descriptor,
            bs_u8,
            C,
            Cs,
            gather_idx,  # None = A is expert-sorted; read only when not None (folds at trace time)
            scatter_idx,  # None = C is expert-sorted; read only when not None (folds at trace time)
            expert_start,
            S,
            N,
            K,
            A.stride(0),
            A.stride(1),
            As.stride(0),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            bs_u8.stride(0),
            bs_u8.stride(2),
            bs_u8.stride(1),
            C.stride(0),
            C.stride(1),
            Cs.stride(0) if requant else 1,  # dummy stride when unread
            Cs.stride(1) if requant else 1,
            # Meta-parameters
            num_experts=num_experts,
            tokens_per_expert_bit_length=tokens_per_expert_bucket(S, num_experts),
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            NUM_EXPERTS_POW2=triton.next_power_of_2(num_experts),
            NUM_SMS=num_sms,
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
    add_op_namespace_prefix("w8a8_block_static_fp8_matmul_grouped"),
    mutates_args=(),
    opaque=True,
)
def w8a8_block_static_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    expert_start: torch.Tensor,
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
    """Block-scale grouped FP8 matmul with a static (per-tensor calibrated) activation scale —
    the block-dynamic sibling with the 2D ``block_static`` recipe. ``A`` is raw here: the op
    quantizes it against the scalar ``As`` (offline, ``(A / As).to(fp8)``), the kernel applies the
    per-block weight scales in the K-loop and the scalar once post-loop. Returns the ``[C]`` GLU
    intermediate, or — under ``output_recipe="fp8"`` — the FP8-requantized ``[C, Cs]`` (the per-row
    output scale is independent of the per-tensor input scale).

    A:  (S, K) raw bf16/fp16 activations — rows addressed via ``gather_idx``
    B:  (num_experts, N, K) FP8 weights; under ``gate`` the (num_experts, 2N, K) gate|up stack
    As: scalar / (1,) — the calibrated per-tensor (static) activation scale
    Bs: (num_experts, N // block_n, K // block_k) per-block weight scales (2N under gate)
    """
    validate_dense_operands(A, B)

    _, K = A.shape
    num_experts, n_rows, N = expert_weight_shape(B, gate)
    S = routed_rows(A, gather_idx, scatter_idx, expert_start, num_experts)

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

    output_dtype = resolve_output_dtype(output_dtype, A, None)
    As = As.reshape(1).to(torch.float32)
    bs_u8 = ue8m0_as_uint8(Bs)
    # Pre-quantize the raw activations against the calibrated scalar (offline — MoE always
    # pre-quants; the kernel folds the scalar back post-loop).
    A_q = (A.to(torch.float32) / As).to(FP8_DTYPE)
    if requant:
        C = A.new_empty(S, N, dtype=FP8_DTYPE)
        # UE8M0 model (ue8m0 weights) -> UE8M0 intermediate scales; the epilogue infers the format
        # from this dtype (fp32 weights keep fp32), matching the block-dynamic requant.
        Cs = torch.empty(S, N // block_n, device=A.device, dtype=bs_u8.dtype)
    else:
        C = A.new_empty(S, N, dtype=output_dtype)
        Cs = expert_start  # dummy pointer; unread (no OUTPUT_RECIPE), strides literal
    num_sms = sm_count(A.device.index)
    a_descriptor, b_descriptor = build_grouped_operand_descriptors(
        A_q, B.view(2 * num_experts if gate else num_experts, N, K)
    )

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_block_static_fp8_matmul_grouped_kernel)[
            (num_sms,)
        ](
            A_q,
            a_descriptor,
            As,
            B,
            b_descriptor,
            bs_u8,
            C,
            Cs,
            gather_idx,  # None = A is expert-sorted; read only when not None (folds at trace time)
            scatter_idx,  # None = C is expert-sorted; read only when not None (folds at trace time)
            expert_start,
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
            Cs.stride(0) if requant else 1,  # dummy stride when unread
            Cs.stride(1) if requant else 1,
            num_experts=num_experts,
            tokens_per_expert_bit_length=tokens_per_expert_bucket(S, num_experts),
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            NUM_EXPERTS_POW2=triton.next_power_of_2(num_experts),
            NUM_SMS=num_sms,
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
    add_op_namespace_prefix("w8a8_tensor_dynamic_fp8_matmul_grouped"),
    mutates_args=(),
    opaque=True,
)
def w8a8_tensor_dynamic_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor | None,
    Bs: torch.Tensor,
    expert_start: torch.Tensor,
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
    """Tensor-scale grouped FP8 matmul over expert-sorted positions (per-tile
    gather/scatter, the sort is virtual — see ``compute_grouped_scheduling`` for the maps).
    Activations arrive pre-quantized: the caller owns the act-quant (``fp8_act_quant_tensor_wide``).
    ``gate``/``act_fn``/``swiglu_*``/``simulate_unfused`` are the flattened ``Epilogue`` (GLU only;
    ``requant`` is unsupported here — the dispatcher unpacks the bundle).

    A:  (S, K) pre-quantized FP8 activations — rows addressed via ``gather_idx``
    B:  (num_experts, N, K) FP8 expert weights; under ``gate`` the (num_experts, 2N, K) stack
    As: (S,) per-token activation scales
    Bs: (num_experts,) or (num_experts, 1, 1) per-expert weight scales
    expert_start: (num_experts_pow2 + 1,) int32 — cumulative sorted-row starts, S sentinel
    gather_idx: optional (S,) — sorted position -> source row of A; None = A is expert-sorted
    scatter_idx: optional (S,) — sorted position -> destination row of C; None = C stays expert-sorted
    """
    validate_dense_operands(A, B)

    _, K = A.shape

    # Under a gate epilogue B is the (E, 2N, K) gate|up stack — N is the per-projection width.
    assert input_recipe in (None, "fp8"), (
        f"tensor-wide activations are E4M3 ('fp8'), got {input_recipe!r}"
    )
    assert output_recipe is None, (
        "requant is unsupported for tensor-wide gate_up (its down needs a per-token whole-row "
        "scale a per-tile epilogue can't form); use a plain gate epilogue + external quant"
    )
    num_experts, _, N = expert_weight_shape(B, gate)
    S = routed_rows(A, gather_idx, scatter_idx, expert_start, num_experts)

    # Normalize Bs to (num_experts, 1, 1) — one per-tensor scale (covers the gate|up stack)
    Bs = normalize_per_expert_scale(Bs, num_experts)

    # A raw (As is None) -> quantize here (offline, per-token); else pre-quantized.
    output_dtype = resolve_output_dtype(output_dtype, A, As)
    if As is None:
        A, As = fp8_act_quant_tensor_wide(A, K)
    C = A.new_empty(S, N, dtype=output_dtype)
    num_sms = sm_count(A.device.index)
    a_descriptor, b_descriptor = build_grouped_operand_descriptors(
        A, B.view(2 * num_experts if gate else num_experts, N, K)
    )

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_tensor_dynamic_fp8_matmul_grouped_kernel)[
            (num_sms,)
        ](
            A,
            a_descriptor,
            As,
            B,
            b_descriptor,
            Bs,
            C,
            expert_start,  # dummy Cs (no fused requant for tensor-wide)
            gather_idx,  # None = A is expert-sorted; read only when not None (folds at trace time)
            scatter_idx,  # None = C is expert-sorted; read only when not None (folds at trace time)
            expert_start,
            S,
            N,
            K,
            A.stride(0),
            A.stride(1),
            As.stride(0),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            Bs.stride(0),
            C.stride(0),
            C.stride(1),
            1,  # dummy Cs strides
            1,
            num_experts=num_experts,
            tokens_per_expert_bit_length=tokens_per_expert_bucket(S, num_experts),
            NUM_EXPERTS_POW2=triton.next_power_of_2(num_experts),
            NUM_SMS=num_sms,
            GATE=gate,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(output_dtype),
        )

    return [C]


@compile_time_only_triton_op(
    add_op_namespace_prefix("mx_dynamic_matmul_grouped"), mutates_args=(), opaque=True
)
def mx_dynamic_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor | None,
    Bs: torch.Tensor,
    expert_start: torch.Tensor,
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
    """Grouped MX matmul over expert-sorted positions (per-tile gather/scatter, the
    sort is virtual — see ``compute_grouped_scheduling`` for the maps). Activations arrive
    pre-quantized: the caller owns the act-quant (``mxfp8_act_quant``). The
    ``gate``/``act_fn``/``swiglu_*``/``requant``/``simulate_unfused`` flags are the flattened
    ``Epilogue`` (the dispatcher unpacks the bundle).
    Weight format detected from ``B.dtype``: ``int8`` →
    packed E2M1 (MXFP4, ``B`` is ``(num_experts, N, K//2)``); ``float8_e4m3fn`` → unpacked E4M3
    (MXFP8, ``(num_experts, N, K)``). UE8M0 group-32 scales ``(num_experts, N, K//32)``; tile + dot autotuned.

    A:  (S, K) pre-quantized E4M3 activations — rows addressed via ``gather_idx``
    As: (S, K // 32) UE8M0 group-32 activation scales
    expert_start: (num_experts_pow2 + 1,) int32 — cumulative sorted-row starts, S sentinel
    gather_idx: optional (S,) — sorted position -> source row of A; None = A is expert-sorted
    scatter_idx: optional (S,) — sorted position -> destination row of C; None = C stays expert-sorted
    """
    assert A.ndim == 2 and B.ndim == 3 and Bs.ndim in (3, 5)  # 5D = pre-swizzled SWIZZLE_32_4_4
    assert B.dtype in (torch.int8, torch.float8_e4m3fn), (
        f"B must be int8 (packed E2M1) or float8_e4m3fn (E4M3), got {B.dtype}"
    )
    WEIGHT_VALUES_PER_BYTE = NIBBLES_PER_BYTE if B.dtype == torch.int8 else 1
    # int8 A = caller-provided packed-E2M1 activations (native fp4 MMA): K is two values
    # per stored byte and the scales are mandatory (nothing left to quantize).
    ACT_VALUES_PER_BYTE = NIBBLES_PER_BYTE if A.dtype == torch.int8 else 1
    if ACT_VALUES_PER_BYTE == NIBBLES_PER_BYTE:
        assert As is not None, "packed-E2M1 activations need their group scales (As)"

    K = A.shape[1] * ACT_VALUES_PER_BYTE
    num_experts, n_rows, N = expert_weight_shape(B, gate)
    K_b = B.shape[2]
    S = routed_rows(A, gather_idx, scatter_idx, expert_start, num_experts)
    assert K == WEIGHT_VALUES_PER_BYTE * K_b, (
        f"K (={K}) must equal {WEIGHT_VALUES_PER_BYTE} * B.shape[2] (={K_b})"
    )
    # Bs is either row-major (num_experts, n_rows, K // scale_group) — the kernel reads it affine —
    # or already SWIZZLE_32_4_4 (5D), swizzled once at model load: the deployment contract, one
    # checkpoint shared with batched decode (is_preswizzled_mx). The recipe is the scale dtype
    # (E4M3 = NVFP4 group-16, UE8M0 = MX group-32).
    swizzled_scales = Bs.ndim == 5
    scale_group = mx_scale_family(Bs, K)
    if not swizzled_scales:
        assert Bs.shape == (num_experts, n_rows, K // scale_group), (
            f"Bs shape {tuple(Bs.shape)} != ({num_experts}, {n_rows}, {K // scale_group})"
        )

    output_dtype = resolve_output_dtype(output_dtype, A, As)
    input_recipe = resolve_input_recipe(input_recipe, output_recipe, Bs)
    requant = output_recipe is not None
    # Non-128 N on the swizzled arm (bf16, non-gate): each expert's weight-scale slab pads to
    # cdiv(N,128) whole 128-row SWIZZLE_32_4_4 blocks, so the partial last N-tile reads a full (padded)
    # scale block off the descriptor and a TMA-clamped (zero) weight tile; the epilogue masks the
    # column tail (N_COLS) and %-wraps the pointer-arm rows. Two non-128 cases stay unsupported:
    #   - GATE: the gate|up split at row N lands mid-128-block unless N % 128 == 0 (gate tail and up
    #     head share a swizzle block), so the block-level gate/up interleave needs each projection
    #     padded to a 128 multiple first — a weight-layout change, not done here.
    #   - requant: the swizzled Cs store is correct per band, but the round-trip is not yet right for
    #     a non-multiple-of-4 N-group count (N//group not a multiple of 4) — under investigation.
    if swizzled_scales and N % 128 != 0 and (gate or requant):
        raise ValueError(
            f"the swizzled routed MX path supports non-128 N ({N}) only for the plain (non-gate) "
            f"bf16 GEMM; gate|up needs N a multiple of 128, requant needs the un-swizzled arm or the "
            f"dense matmul_2d op."
        )
    scale_dtype = ue8m0_as_uint8(Bs).dtype  # UE8M0 -> uint8, NVFP4 -> e4m3 (binder-safe)
    # Activation scales track the weight layout (SWIZZLED_SCALES, one flag): swizzled weight ->
    # swizzled acts (the tcgen05 fast path); un-swizzled weight -> affine acts (no gain swizzling
    # only one operand). Raw A (As is None) is quantized here — the swizzled arm emits SWIZZLE_32_4_4
    # scales directly (the act-quant kernel's fused expert-sorted, 128-padded store, no post-quant
    # pass), the affine arm a plain row-major grid the kernel gathers per row. Pre-quantized As
    # (given, e.g. the down projection's intermediate) arrives in the matching layout (5D swizzled /
    # row-major affine). Each branch yields ``(a_vals, act scales, n_m_tiles)``; ``as_u8`` and its
    # descriptor are built below, symmetric with the weight scales.
    if As is not None:
        assert (As.dtype == torch.float8_e4m3fn) == (Bs.dtype == torch.float8_e4m3fn), (
            f"activation scales ({As.dtype}) must match the weight scale family ({Bs.dtype})"
        )
    # g_a normalizes the act quant here in the wrapper (the raw-A arm below, or applied offline for a
    # pre-quantized As); the kernel only ever sees the combined g_a·g_b via AsBsGlobal (grouped A is
    # pre-quantized, so there's no in-kernel inline-quant that would need g_a alone).
    if a_global_scale is not None:
        assert input_recipe == "nvfp4", "an activation global is NVFP4-only"
    if swizzled_scales:
        if As is None:
            a_vals, act_scales, n_m_tiles = mx_act_quant_swizzled_grouped(
                A, input_recipe, scale_group, scale_dtype, gather_idx, expert_start, a_global_scale
            )
        elif As.ndim == 5:  # pre-swizzled by the gate_up requant epilogue (fused down) — read as is
            a_vals, act_scales, n_m_tiles = A, As, As.shape[1]
        else:  # given row-major scales -> gather+swizzle into the tcgen05 layout
            a_vals = A
            act_scales, n_m_tiles = swizzle_grouped_mx_scales(
                ue8m0_as_uint8(As), expert_start, gather_idx
            )
    else:  # un-swizzled weight -> affine acts (no gain swizzling only one operand)
        assert As is None or As.ndim != 5, (
            "un-swizzled weights pair with affine (row-major) activation scales, got 5D As"
        )
        if As is None:
            a_vals, act_scales = (
                MX_ACT_QUANT[input_recipe](A, global_scale=a_global_scale)
                if a_global_scale is not None
                else MX_ACT_QUANT[input_recipe](A)
            )
        else:
            a_vals, act_scales = A, As
        n_rows = gather_idx.numel() if gather_idx is not None else A.shape[0]
        n_m_tiles = n_rows // 128 + num_experts
    # uint8 aliases for the binder — never clobber the caller's A/B/Bs dtype/view. Each operand's
    # scale is one pointer (``*_u8``) read affine in-kernel, plus a descriptor built only on the
    # swizzled path (the affine arm reads the pointer and never touches the descriptor — None).
    # SWIZZLED_SCALES governs both operands: a swizzled weight is paired with swizzled acts.
    a_u8 = e2m1_as_uint8(a_vals)
    b_u8 = e2m1_as_uint8(B)
    as_u8 = ue8m0_as_uint8(act_scales)
    bs_u8 = ue8m0_as_uint8(Bs)
    box = [1, 1, 1, 2, 256]
    as_descriptor = TensorDescriptor.from_tensor(as_u8, box) if swizzled_scales else None
    bs_descriptor = TensorDescriptor.from_tensor(bs_u8, box) if swizzled_scales else None
    # mxfp8 requant writes Cs straight into the down proj's SWIZZLE_32_4_4 layout (the epilogue
    # gets a descriptor, not a pointer) — the down reads it affine, no post-requant swizzle pass.
    # Only when the output stays expert-sorted (scatter_idx None, the fused gate_up convention):
    # the swizzle needs contiguous 128-row blocks, which a scattered output can't provide.
    # Swizzled in -> swizzled out: when the block runs swizzled (swizzled_scales), the requant
    # emits Cs straight into the down proj's SWIZZLE_32_4_4 layout (a TMA descriptor), so the fused
    # down reads it on the fast path (the As.ndim == 5 arm above). Recipe-general — the swizzle is a
    # byte-tiling over the group-scale grid (N // scale_group), so every MX family qualifies (UE8M0
    # group-32, E4M3 group-16 NVFP4); only the column count (cb_cs) and scale byte dtype differ. The
    # swizzle needs contiguous 128-row blocks, i.e. an expert-sorted output (scatter_idx None — the
    # fused gate_up convention); a scattered requant keeps row-major Cs.
    swizzled_out = requant and scatter_idx is None and swizzled_scales
    if output_recipe in ("mxfp4", "nvfp4"):  # packed E2M1 intermediate, feeds a W4A4 down as-is
        assert N % (2 * scale_group) == 0, (
            f"N (={N}) must be a multiple of {2 * scale_group} to pack E2M1 pairs"
        )
        C = A.new_empty((S, N // 2), dtype=torch.int8)
    elif requant:
        C = A.new_empty((S, N), dtype=FP8_DTYPE)
    else:
        C = A.new_empty((S, N), dtype=output_dtype)
    if swizzled_out:
        cb_cs = triton.cdiv(N // scale_group, 4)
        cs_ret = torch.empty(1, n_m_tiles, cb_cs, 2, 256, device=A.device, dtype=scale_dtype)
        CSDescriptor = TensorDescriptor.from_tensor(cs_ret, [1, 1, 1, 2, 256])
        Cs = None  # row-major pointer unread here; CSDescriptor does the store
    elif requant:  # un-swizzled block (or scattered output) -> row-major Cs pointer
        cs_ret = torch.empty(S, N // scale_group, device=A.device, dtype=scale_dtype)
        Cs, CSDescriptor = cs_ret, None
    else:
        cs_ret, Cs, CSDescriptor = None, None, None  # unread (no OUTPUT_RECIPE)
    num_sms = sm_count(A.device.index)
    # NVFP4 accumulator correction: the per-expert g_a·g_b product folded onto the fp32 accumulator
    # (grouped A is pre-quantized, so the kernel needs only this product, never g_a alone).
    input_global_scale = combine_global_scales(a_global_scale, b_global_scale, B.shape[0])
    # host TMA descriptor over the (2E|E, N, K_bytes) view — one box holds both gate|up
    # projections; the placeholder box is re-bound per tuned config by the pre_hook
    rows0 = 2 * num_experts if gate else num_experts
    a_descriptor, b_descriptor = build_grouped_operand_descriptors(
        a_u8, b_u8.view(rows0, N, b_u8.shape[2])
    )
    # (the SA/SB swizzled-scale descriptors and their placeholder boxes — re-bound per tuned
    # config by _rebind_grouped_mx_descriptors — are built above with the scales.)

    with device_context(A.device):
        compile_time_only_triton_wrap(mx_dynamic_matmul_grouped_kernel)[(num_sms,)](
            a_u8,
            a_descriptor,
            as_u8,  # act scales; read affine iff not SWIZZLED_SCALES (else via ASDescriptor)
            as_descriptor,
            b_u8,
            b_descriptor,
            bs_u8,
            bs_descriptor,
            C,
            Cs,
            CSDescriptor,
            input_global_scale,  # AsBsGlobal = g_a·g_b (acc); grouped A pre-quantized so no in-kernel g_a
            output_global_scale,  # CsGlobal: requant output normalization (next proj's provided input_scale); None folds out
            gather_idx,  # None = A is expert-sorted; read only when not None (folds at trace time)
            scatter_idx,  # None = C is expert-sorted; read only when not None (folds at trace time)
            expert_start,
            S,
            N,
            K,
            a_u8.stride(0),
            a_u8.stride(1),
            as_u8.stride(0),  # As row stride (affine act-scale read; dead on the swizzled arm)
            b_u8.stride(0),
            b_u8.stride(2),
            b_u8.stride(1),
            bs_u8.stride(0),
            bs_u8.stride(2),
            bs_u8.stride(1),
            C.stride(0),
            C.stride(1),
            # a swizzled Cs is a descriptor (no strides); a row-major requant keeps pointer strides
            cs_ret.stride(0) if (requant and not swizzled_out) else 1,
            cs_ret.stride(1) if (requant and not swizzled_out) else 1,
            num_experts=num_experts,
            tokens_per_expert_bit_length=tokens_per_expert_bucket(S, num_experts),
            NUM_EXPERTS_POW2=triton.next_power_of_2(num_experts),
            NUM_SMS=num_sms,
            SCALE_GROUP_K=scale_group,
            GATE=gate,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            OUTPUT_RECIPE=output_recipe,
            SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(output_dtype),
            SWIZZLED_SCALES=swizzled_scales,
            SWIZZLED_OUT=swizzled_out,
        )
    return [C, cs_ret] if requant else [C]


@compile_time_only_triton_op(
    add_op_namespace_prefix("full_precision_matmul_grouped"),
    mutates_args=(),
    opaque=True,
)
def full_precision_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    expert_start: torch.Tensor,
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
    """Full-precision (BF16/FP16) grouped matmul over expert-sorted positions (per-tile
    gather/scatter, the sort is virtual — see ``compute_grouped_scheduling`` for the maps).
    No quantization anywhere: activations and weights share one high-precision dtype and the
    dot accumulates in fp32. ``gate``/``act_fn``/``swiglu_*``/``simulate_unfused`` are the
    flattened ``Epilogue`` (GLU only; ``requant`` is meaningless without a quantized recipe).

    A:  (S, K) BF16/FP16 activations — rows addressed via ``gather_idx``
    B:  (num_experts, N, K) expert weights in A's dtype; under ``gate`` the (num_experts, 2N, K) stack
    expert_start: (num_experts_pow2 + 1,) int32 — cumulative sorted-row starts, S sentinel
    gather_idx: optional (S,) — sorted position -> source row of A; None = A is expert-sorted
    scatter_idx: optional (S,) — sorted position -> destination row of C; None = C stays expert-sorted
    """
    validate_dense_operands(A, B)
    assert A.dtype == B.dtype and A.dtype in (torch.bfloat16, torch.float16), (
        f"full-precision path needs matching BF16/FP16 A and B, got {A.dtype} / {B.dtype}"
    )
    assert input_recipe is None and output_recipe is None, (
        "the full-precision path quantizes nothing — no input or output recipe applies"
    )

    _, K = A.shape

    num_experts, _, N = expert_weight_shape(B, gate)
    S = routed_rows(A, gather_idx, scatter_idx, expert_start, num_experts)

    output_dtype = resolve_output_dtype(output_dtype, A, None)
    C = A.new_empty(S, N, dtype=output_dtype)
    num_sms = sm_count(A.device.index)
    a_descriptor, b_descriptor = build_grouped_operand_descriptors(
        A, B.view(2 * num_experts if gate else num_experts, N, K)
    )

    with device_context(A.device):
        compile_time_only_triton_wrap(full_precision_matmul_grouped_kernel)[(num_sms,)](
            A,
            a_descriptor,
            B,
            b_descriptor,
            C,
            gather_idx,  # None = A is expert-sorted; read only when not None (folds at trace time)
            scatter_idx,  # None = C is expert-sorted; read only when not None (folds at trace time)
            expert_start,
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
            num_experts=num_experts,
            tokens_per_expert_bit_length=tokens_per_expert_bucket(S, num_experts),
            NUM_EXPERTS_POW2=triton.next_power_of_2(num_experts),
            NUM_SMS=num_sms,
            GATE=gate,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(output_dtype),
        )

    return [C]


def matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor | None = None,
    Bs: torch.Tensor | None = None,
    *,
    expert_start: torch.Tensor,
    epilogue: Epilogue | None = None,
    quantization: Quantization | None = None,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    a_global_scale: torch.Tensor | None = None,
    b_global_scale: torch.Tensor | None = None,
    output_global_scale: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Grouped matmul dispatcher (W8A8 FP8, W4A8/W4A4 FP4, or full-precision).
    ``expert_start`` is the ``(E+1,)`` tiling schedule from one ``compute_grouped_scheduling``
    pass, shared by every grouped GEMM of the layer (the expert sort is virtual — nothing is
    physically permuted).

    ``As`` marks ``A`` as already quantized (framework-precomputed scales, or a requantized
    intermediate handed to the down projection); a per-tensor scalar ``As`` is instead the static
    (calibrated) activation scale for block-scale FP8 weights — the op quantizes raw ``A`` against
    it; ``None`` = raw ``A``, quantized dynamically by the op per ``quantization`` (see
    ``Quantization`` — recipe-default fp8/E4M3, or packed E2M1 under ``input_recipe="mxfp4"``).
    ``Bs`` ``None`` = unquantized BF16/FP16 weights.
    ``quantization.output_recipe`` requantizes the output into the recipe's format — the
    return is then ``(C, Cs)``. ``epilogue`` is the fused output transform (gate|up + GLU).
    ``As``/``Bs`` are each a bare block-scale tensor; the two-level NVFP4 second-level scales ride
    the separate ``a_global_scale``/``b_global_scale`` (fp32 per-tensor, weights per-expert ``(E,)``;
    from ``nvfp4_quantize_two_level``), and the op folds ``g_a · g_b`` onto the accumulator. The
    activation global ``g_a`` is CALIBRATED (the checkpoint's ``input_scale``): ``a_global_scale=g_a``
    with a raw ``A`` has the op quantize ``A / g_a`` per block, and rides a pre-quantized ``As`` the
    same way. Under NVFP4 ``output_recipe`` the fused requant normalizes the GLU intermediate by the
    PROVIDED ``output_global_scale`` (the next proj's calibrated ``input_scale``) before the block
    quant and returns ``[C, Cs]``; the down consumes it as ``As=Cs, a_global_scale=output_global_scale``.
    Row order is carried by the standalone maps: ``gather_idx`` gathers ``A`` (``None`` ->
    already expert-ordered), ``scatter_idx`` scatters the output. The fused MoE chain is one
    scheduling pass: gate_up with ``scatter_idx=None`` + ``Epilogue(gate=True)`` +
    ``Quantization(output_recipe=...)``, then down with ``gather_idx=None`` and the
    intermediate's scales as ``As``. EP-sentinel routes fall past ``expert_start[-1]`` and
    are never touched.

    Routes by what the weight tensors themselves say (there is no ``block_size``
    parameter — the quantization block is derived from the scale shape,
    ``weight_block_size``):
    - ``Bs`` None → ``full_precision_matmul_grouped`` (plain dot, no scales anywhere).
    - MX weights — ``int8`` (packed E2M1) or ``float8_e4m3fn`` (E4M3) with UE8M0
      group-32 ``Bs`` → ``mx_dynamic_matmul_grouped``.
    - one scale per expert (``Bs`` ``(E,)``/``(E, 1, 1)``) →
      ``w8a8_tensor_dynamic_fp8_matmul_grouped``.
    - block scales (``Bs`` ``(E, N/bn, K/bk)``) → ``w8a8_block_dynamic_fp8_matmul_grouped``.
    """
    ep = epilogue if epilogue is not None else Epilogue()
    q = quantization if quantization is not None else Quantization()
    assert (a_global_scale is None and b_global_scale is None) or (Bs is not None and is_mx(B, Bs)), (
        "two-level globals (a_global_scale / b_global_scale) are NVFP4-only (MX weights)"
    )
    if As is not None and As.numel() == 1:
        # static (per-tensor calibrated) activation quant: a per-tensor scalar As for block-scale FP8
        # weights — the caller hands raw A, the op quantizes it against the scalar (As IS the scale).
        assert Bs is not None and not is_mx(B, Bs) and weight_block_size(B, Bs) is not None, (
            "a per-tensor scalar As (static activation scale) needs block-scale FP8 weights"
        )
        out = w8a8_block_static_fp8_matmul_grouped(
            A,
            B,
            As,
            Bs,
            expert_start,
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
        out = full_precision_matmul_grouped(
            A,
            B,
            expert_start,
            *ep.as_args(),
            *q.as_args(),
            output_dtype,
            gather_idx,
            scatter_idx,
        )
    elif is_mx(B, Bs):
        out = mx_dynamic_matmul_grouped(
            A,
            B,
            As,
            Bs,
            expert_start,
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
        out = w8a8_tensor_dynamic_fp8_matmul_grouped(
            A,
            B,
            As,
            Bs,
            expert_start,
            *ep.as_args(),
            *q.as_args(),
            output_dtype,
            gather_idx,
            scatter_idx,
        )
    else:
        out = w8a8_block_dynamic_fp8_matmul_grouped(
            A,
            B,
            As,
            Bs,
            expert_start,
            block_size,
            *ep.as_args(),
            *q.as_args(),
            output_dtype,
            gather_idx,
            scatter_idx,
        )
    # The ops return a list (torch custom ops can't return a Tensor-or-tuple union): [C] plain,
    # [C, Cs] under an output_recipe. Unwrap to the documented Tensor / (Tensor, Tensor) return.
    return out[0] if len(out) == 1 else tuple(out)
