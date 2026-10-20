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


from typing import List

import torch
import triton
import triton.language as tl


from triton.tools.tensor_descriptor import TensorDescriptor

from .bayesian_autotuner import bayesian_autotune
from .bench import weight_input
from .compat import add_op_namespace_prefix, FP8_DTYPE, NIBBLES_PER_BYTE, compile_time_only_triton_op, compile_time_only_triton_wrap, device_context, get_accelerator_autotuning_configs, persistent_program_count, tl_dtype, MX_SCALE_GROUP_K, NVFP4_SCALE_GROUP_K, pipeline_autotuning_configs, persistent_programs_per_sm, is_sm10x
from .descriptors import build_grouped_operand_descriptors, rebind_grouped_descriptors, rebind_grouped_mx_descriptors
from .formats import FP8_WEIGHT_ONLY_DEQUANT_MIN_M, block_fp8_weight_scale_grid, _validate_quantization_args, global_scale_stride, gate_out_cols, is_per_expert_global, normalize_global_scale, e2m1_as_uint8, expert_weight_shape, is_mx, mx_scale_family, normalize_per_expert_scale, resolve_activation_format, resolve_output_dtype, routed_rows, row_scale_stride, tokens_per_expert_bucket, ue8m0_as_uint8, validate_dense_operands, weight_block_size, validate_int_operands, INT_WEIGHT_DTYPES
from .quant import fp8_act_quant_block_dynamic, MX_ACT_QUANT, mx_act_quant_grouped, mx_act_quant_routed_rows, quantize_routed_rows_per_expert, static_expert_act_operands, swizzle_grouped_mx_scales, tensor_wide_act_operands, prepare_fp8_mx_inputs, permute_routed_rows
from .swizzle import swizzled_scale_descriptor
from .loading.scales import apply_row_scale, dequant_block_fp8_weight
from .mma import block_dynamic_dot, fp8_dot, plain_dot, mx_compute, mx_weight_only_compute, static_dot, int_weight_only_compute
from .scheduling import build_tile_layout, expand_gather_below_parity, expand_regime, load_packed_schedule, grouped_tile_order, prefetch_packed_entry, resolve_grouped_tile, resolve_grouped_tile_packed, fill_packed_schedule, packed_schedule_buffer
from .loading.tiles import (
    load_act_block_dynamic,
    load_act_mx,
    load_act_plain,
    load_act_static,
    load_weight_block_dynamic,
    load_weight_block_fp8,
    load_weight_int_group,
    load_weight_mx,
    load_weight_plain,
    load_weight_static,
    operand_tile_descriptor,
    operand_tile_ptrs,
    weight_tile_ptrs,
)
from .epilogue import acc_init, bias_strides, gemm_epilogue
from .pruners import PATH_ANCHOR_AXES, int_scale_group_tile_pruner, fp8_dot_warp_pruner, raw_activation_pointer_pruner, warp_spec_memory_mode_pruner, packed_schedule_scope_pruner, affine_scale_warp_spec_pruner, block_dynamic_grouped_matmul_pruner, block_fits_dim_pruner, block_within_dim_pruner, compose_pruners, descriptor_box_pruner, gate_stacked_tmem_trap_pruner, gated_pointer_weight_warp_spec_pruner, mx_config_pruner, require_moe_dims_aligned, smem_pruner, swizzled_out_bm_pruner, swizzled_scale_config_pruner, swizzled_scales_bm_pruner, warp_spec_compile_guard_pruner


def grouped_pre_hook(nargs):
    """Per-config pre_hook of the grouped kernels: the operand descriptor boxes, and the packed schedule for the
    configs that read it."""
    rebind_grouped_descriptors(nargs)
    fill_packed_schedule(nargs)


def grouped_mx_pre_hook(nargs):
    """``grouped_pre_hook`` plus the MX scale boxes."""
    rebind_grouped_mx_descriptors(nargs)
    fill_packed_schedule(nargs)


def _grouped_dot_configs():
    return get_accelerator_autotuning_configs(
        packed_schedule=True,
        tune_block_nk=True,
        warp_spec=True,
        tune_block_m=True,
        a_memory_modes=("descriptor", "pointer"),
        b_memory_modes=("descriptor", "pointer"),
        pre_hook=grouped_pre_hook,
    )


@bayesian_autotune(
    # No SWAP_AB axis: the descriptor loads the natural-orientation ((2|1), BN, BK) gate|up box
    # and transposes once to the same K-major tile the pointer arm builds. Both memory axes are
    # emitted; the tuner routes per key. Swap-coupling verdicts are B200 (sm_100) — re-chart on
    # H100 or the target device before reusing SWAP_AB here.
    get_accelerator_autotuning_configs(
        packed_schedule=True,
        warp_spec=True,
        tune_block_m=True,
        a_memory_modes=("descriptor", "pointer"),
        b_memory_modes=("descriptor", "pointer"),
        pre_hook=grouped_pre_hook,
    ),
    # GATE keys the gate|up arm separately: its dot is 2*BN wide, a different tile optimum.
    # BLOCK_SIZE_N/BLOCK_SIZE_K here are the launch-pinned quant block (not tuned) — they gate
    # descriptor-box legality, so checkpoints with different blocks must not share a winner.
    ["N", "K", "tokens_per_expert_bit_length", "GATE", "BLOCK_SIZE_N", "BLOCK_SIZE_K"],
    n_trials=100,
    benchmark_input=weight_input,
    path_anchor_axes=PATH_ANCHOR_AXES,
    # Pipeliner-race guard: per launch-BM, WS-only at BM >= 64 and non-WS below (see the pruner).
    prune_configs_by={
        "early_config_prune": compose_pruners(
            fp8_dot_warp_pruner(),
            packed_schedule_scope_pruner(),
            block_dynamic_grouped_matmul_pruner(),
            warp_spec_memory_mode_pruner(),
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
    BDescriptor,  # host TMA descriptor over B viewed (E, 2N|N, K), box (1, (2|1)*BN, BK); read iff B_MEMORY_MODE != "pointer"
    Bs,  # (num_experts, N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales (2N under GATE)
    C,  # (S, N) output; under an OUTPUT_FORMAT the FP8-requantized intermediate
    Cs,  # (S, N // BLOCK_SIZE_N) per-row, per-block output scale; written iff OUTPUT_FORMAT
    Bias,  # (E, N_out) per-expert output bias, N_out = 2N under GATE; read iff not None
    GatherIdx,  # (S,) int32 — sorted position -> source row of A; read only when not None
    ScatterIdx,  # (S,) int32 — sorted position -> destination row of C; read only when not None
    ExpertStart,  # (NUM_EXPERTS_POW2 + 1,) int32 — cumulative row starts, S sentinel
    Schedule,  # packed tile schedule (fill_packed_schedule); read iff PACKED_SCHEDULE
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
    num_experts,
    tokens_per_expert_bit_length,  # autotune key only (log2 avg-tokens bucket); unused in body
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS_POW2: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
    # tuner axis: resolve M-tiles from the packed schedule (3 scalar loads) instead of the
    # E-wide register-resident layout — register relief, bit-identical tiles; scope pruner
    # admits it at BM>=128 only (see packed_schedule_scope_pruner)
    PACKED_SCHEDULE: tl.constexpr = False,
    A_MEMORY_MODE: tl.constexpr = "pointer",
    B_MEMORY_MODE: tl.constexpr = "pointer",
    # Gate|up fusion epilogue (GATE=False -> plain grouped GEMM, every arm below folds out)
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    # the output format name (None | "fp8")
    OUTPUT_FORMAT: tl.constexpr = None,
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
    the epilogue splits, applies the ``ACT_FN``/SwiGLU ``glu``, and — under an ``OUTPUT_FORMAT`` — FP8-
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
    if PACKED_SCHEDULE:
        total_m_tiles = load_packed_schedule(Schedule, BLOCK_SIZE_M)
        prefetched = prefetch_packed_entry(
            Schedule, start_pid // tl.cdiv(N, BLOCK_SIZE_N), total_m_tiles
        )
    else:
        exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
            ExpertStart, NUM_EXPERTS_POW2, BLOCK_SIZE_M
        )
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn, row0, n_off, m_start = (
            resolve_grouped_tile_packed(
                tile_id, num_n_tiles, prefetched,
                GatherIdx, ScatterIdx, BLOCK_SIZE_N, BLOCK_SIZE_M, GATE,
            ) if PACKED_SCHEDULE else resolve_grouped_tile(
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
        if PACKED_SCHEDULE:
            prefetched = prefetch_packed_entry(
                Schedule, (tile_id + NUM_SMS) // num_n_tiles, total_m_tiles
            )
        a_ptrs = operand_tile_ptrs(A, in_row, offs_k, stride_a_m, stride_a_k, A_MEMORY_MODE, True)
        as_ptrs = As + in_row * stride_as_m
        # GATE stacks gate (rows [0, N)) and up (rows [N, 2N)) into one [BK, 2*BN] tile — the
        # rows are interleaved, so the tile is one 2*BN span. GATE=False -> plain [BK, BN].
        b_ptrs = weight_tile_ptrs(
            B + expert_id64 * stride_b_e,
            offs_bn,
            offs_k,
            stride_b_n,
            stride_b_k,
            False,
        )
        # under GATE the tile's two scale blocks are the adjacent (gate, up) pair at 2*pid_n; non-GATE
        # reads the single block (up_s_ptr dead).
        gate_s_ptr = Bs + expert_id64 * stride_bs_e + ((2 if GATE else 1) * pid_n) * stride_bs_n
        up_s_ptr = gate_s_ptr + stride_bs_n

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
            OUTPUT_FORMAT,
            1,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SIMULATE_UNFUSED,
            INTERMEDIATE_DTYPE,
            Bias=Bias,
            stride_bias_e=stride_bias_e,
            stride_bias_n=stride_bias_n,
            global_row=expert_id64,
        )


@bayesian_autotune(
    # Mirrors the block-dynamic grouped decorator (same (2|1, BN, BK) gate|up descriptor box, no
    # SWAP_AB axis, both memory axes emitted). No dot_scaled here — the static activation scale is
    # a per-tensor scalar applied post-loop (see the kernel), so the K-loop is a plain dot.
    get_accelerator_autotuning_configs(
        packed_schedule=True,
        warp_spec=True,
        tune_block_m=True,
        a_memory_modes=("descriptor", "pointer"),
        b_memory_modes=("descriptor", "pointer"),
        pre_hook=grouped_pre_hook,
    ),
    ["N", "K", "tokens_per_expert_bit_length", "GATE", "BLOCK_SIZE_N", "BLOCK_SIZE_K"],
    n_trials=100,
    benchmark_input=weight_input,
    path_anchor_axes=PATH_ANCHOR_AXES,
    # Pipeliner-race WS guard (per launch-BM: forces WS where the default pipeliner races at
    # BM>=64, keeps non-WS below) + descriptor-box limits.
    prune_configs_by={
        "early_config_prune": compose_pruners(
            fp8_dot_warp_pruner(),
            packed_schedule_scope_pruner(),
            block_dynamic_grouped_matmul_pruner(),
            descriptor_box_pruner(),
            raw_activation_pointer_pruner(),
        )
    },
)
@triton.jit
def w8a8_block_static_fp8_matmul_grouped_kernel(
    A,  # (num_tokens, K) E4M3 activations (pre-quantized against the static scale by the wrapper)
    ADescriptor,  # host TMA descriptor over A (rows, K), box (BM, BK); read iff A_MEMORY_MODE != "pointer"
    As,  # calibrated (static) activation scale: one value, or one per expert
    B,  # (num_experts, N, K) FP8 weights; under GATE the (num_experts, 2N, K) gate|up stack
    BDescriptor,  # host TMA descriptor over B viewed (E, 2N|N, K), box (1, (2|1)*BN, BK); read iff B_MEMORY_MODE != "pointer"
    Bs,  # (num_experts, N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales (2N under GATE)
    C,  # (S, N) output; under an OUTPUT_FORMAT the FP8-requantized intermediate
    Cs,  # (S, N // BLOCK_SIZE_N) per-(row, block) output scale; written iff OUTPUT_FORMAT
    Bias,  # (E, N_out) per-expert output bias, N_out = 2N under GATE; read iff not None
    GatherIdx,  # (S,) int32 — sorted position -> source row of A; read only when not None
    ScatterIdx,  # (S,) int32 — sorted position -> destination row of C; read only when not None
    ExpertStart,  # (NUM_EXPERTS_POW2 + 1,) int32 — cumulative row starts, S sentinel
    Schedule,  # packed tile schedule (fill_packed_schedule); read iff PACKED_SCHEDULE
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
    stride_as_e,  # 0 = one calibrated scale shared by every expert
    stride_bs_e,
    stride_bs_k,
    stride_bs_n,
    stride_c_m,
    stride_c_n,
    stride_cs_m,
    stride_cs_n,
    stride_bias_e,
    stride_bias_n,
    num_experts,
    tokens_per_expert_bit_length,  # autotune key only (log2 avg-tokens bucket); unused in body
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS_POW2: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
    # tuner axis: resolve M-tiles from the packed schedule (3 scalar loads) instead of the
    # E-wide register-resident layout — register relief, bit-identical tiles; scope pruner
    # admits it at BM>=128 only (see packed_schedule_scope_pruner)
    PACKED_SCHEDULE: tl.constexpr = False,
    A_MEMORY_MODE: tl.constexpr = "pointer",
    B_MEMORY_MODE: tl.constexpr = "pointer",
    # Gate|up fusion epilogue (GATE=False -> plain grouped GEMM, every arm below folds out)
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    OUTPUT_FORMAT: tl.constexpr = None,  # None | "fp8" (per-(row, block) requant of the intermediate)
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
):
    """Block-scale grouped FP8 expert matmul with a static (per-tensor) activation scale —
    persistent grid-stride, the block-dynamic sibling's structure (virtual expert sort via
    ``ExpertStart``/``GatherIdx``, ``GATE`` gate|up fusion, host-TMA memory modes) with the 2D
    ``block_static`` format: ``A`` arrives pre-quantized against the calibrated scalar, per-block
    weight scales apply per-K-tile (plain ``tl.dot`` + software rescale, ``accumulate("static")``),
    and the scalar activation scale multiplies the accumulator once after the loop. GATE=False is
    the plain grouped GEMM (down projection), bit-identical."""
    start_pid = tl.program_id(axis=0)
    if PACKED_SCHEDULE:
        total_m_tiles = load_packed_schedule(Schedule, BLOCK_SIZE_M)
        prefetched = prefetch_packed_entry(
            Schedule, start_pid // tl.cdiv(N, BLOCK_SIZE_N), total_m_tiles
        )
    else:
        exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
            ExpertStart, NUM_EXPERTS_POW2, BLOCK_SIZE_M
        )
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn, row0, n_off, m_start = (
            resolve_grouped_tile_packed(
                tile_id, num_n_tiles, prefetched,
                GatherIdx, ScatterIdx, BLOCK_SIZE_N, BLOCK_SIZE_M, GATE,
            ) if PACKED_SCHEDULE else resolve_grouped_tile(
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
        if PACKED_SCHEDULE:
            prefetched = prefetch_packed_entry(
                Schedule, (tile_id + NUM_SMS) // num_n_tiles, total_m_tiles
            )
        a_ptrs = operand_tile_ptrs(A, in_row, offs_k, stride_a_m, stride_a_k, A_MEMORY_MODE, True)
        b_ptrs = weight_tile_ptrs(
            B + expert_id64 * stride_b_e,
            offs_bn,
            offs_k,
            stride_b_n,
            stride_b_k,
            False,
        )
        gate_s_ptr = Bs + expert_id64 * stride_bs_e + ((2 if GATE else 1) * pid_n) * stride_bs_n
        up_s_ptr = gate_s_ptr + stride_bs_n

        # this tile's expert scale: stride 0 means one calibrated scale for every expert.
        # It feeds the inline quant arm (raw A) and folds back onto the accumulator post-loop.
        a_s_static = tl.load(As + expert_id64.to(tl.int32) * stride_as_e)
        acc = acc_init("dot", BLOCK_SIZE_M, (2 if GATE else 1) * BLOCK_SIZE_N, False)
        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
            a, a_dead = load_act_static(
                a_ptrs, ADescriptor, m_start, k * BLOCK_SIZE_K, row_mask, in_row, a_s_static,
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
            OUTPUT_FORMAT,
            1,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SIMULATE_UNFUSED,
            INTERMEDIATE_DTYPE,
            Bias=Bias,
            stride_bias_e=stride_bias_e,
            stride_bias_n=stride_bias_n,
            global_row=expert_id64,
        )


@bayesian_autotune(
    _grouped_dot_configs(),
    # GATE keys the gate|up arm separately (its dot is 2*BN wide, a different tile optimum).
    ["N", "K", "tokens_per_expert_bit_length", "GATE"],
    n_trials=100,
    benchmark_input=weight_input,
    path_anchor_axes=PATH_ANCHOR_AXES,
    # BLOCK_SIZE_K/N are tuned axes; the K-loop is maskless and the N-tile store is
    # row-masked only — veto non-dividing tiles on both. WS is a pure perf axis here
    # (non-WS is the validated state), compile-guarded. The GATE arm's stacked width-512
    # dots (BN=256) are clean — probed bit-exact 2026-07-14; the oversized-smem ones fail
    # benignly at launch and self-prune as inf.
    prune_configs_by={
        "early_config_prune": compose_pruners(
            fp8_dot_warp_pruner(),
            packed_schedule_scope_pruner(),
            block_within_dim_pruner("K"),
            block_within_dim_pruner("N", "BLOCK_SIZE_N"),
            warp_spec_compile_guard_pruner(),
            warp_spec_memory_mode_pruner(),
            descriptor_box_pruner(),
            smem_pruner(),
            raw_activation_pointer_pruner(),
        )
    },
)
@triton.jit
def w8a8_tensor_dynamic_fp8_matmul_grouped_kernel(
    A,  # (num_tokens, K) pre-quantized FP8 activations, any row order
    ADescriptor,  # host TMA descriptor over A (rows, K), box (BM, BK); read iff A_MEMORY_MODE != "pointer"
    As,  # per-token activation scales (S,), or a calibrated (static) one: shared, or per expert
    B,  # (num_experts, N, K) FP8 weights; under GATE the (num_experts, 2N, K) gate|up stack
    BDescriptor,  # host TMA descriptor over B viewed (E, 2N|N, K), box (1, (2|1)*BN, BK); read iff B_MEMORY_MODE != "pointer"
    Bs,  # (num_experts, N, 1) row-wise weight scales, or (num_experts, 1, 1) per tensor (stride_bs_n 0)
    C,  # (S, N) output; under GATE the bf16 GLU intermediate
    Cs,  # None (tensor-wide has no fused requant); kept for the shared epilogue signature
    Bias,  # (E, N_out) per-expert output bias, N_out = 2N under GATE; read iff not None
    GatherIdx,  # (S,) int32 — sorted position -> source row of A; read only when not None
    ScatterIdx,  # (S,) int32 — sorted position -> destination row of C; read only when not None
    ExpertStart,  # (NUM_EXPERTS_POW2 + 1,) int32 — cumulative row starts, S sentinel
    Schedule,  # packed tile schedule (fill_packed_schedule); read iff PACKED_SCHEDULE
    # Shape
    S,
    N,
    K,
    # Strides
    stride_a_m,
    stride_a_k,
    stride_as_m,
    stride_as_e,  # 0 = the scale is per token; 1 = one calibrated scale per expert
    stride_b_e,
    stride_b_k,
    stride_b_n,
    stride_bs_e,
    stride_bs_n,
    stride_c_m,
    stride_c_n,
    stride_cs_m,
    stride_cs_n,
    stride_bias_e,
    stride_bias_n,
    num_experts,
    tokens_per_expert_bit_length,  # autotune key only (log2 avg-tokens bucket); unused in body
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS_POW2: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
    # tuner axis: resolve M-tiles from the packed schedule (3 scalar loads) instead of the
    # E-wide register-resident layout — register relief, bit-identical tiles; scope pruner
    # admits it at BM>=128 only (see packed_schedule_scope_pruner)
    PACKED_SCHEDULE: tl.constexpr = False,
    A_MEMORY_MODE: tl.constexpr = "pointer",
    B_MEMORY_MODE: tl.constexpr = "pointer",
    # Gate|up fusion epilogue (GATE=False -> plain grouped GEMM). No fused requant here
    # (tensor-wide down needs a per-token whole-row scale a per-tile epilogue can't form).
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    OUTPUT_FORMAT: tl.constexpr = None,  # tensor-wide has no fused requant (kept None)
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
):
    """Tensor-scale / row-wise grouped FP8 expert matmul kernel — persistent grid-stride over tiles.

    Grouped expert scheduling with pre-quantized activations plus per-token activation scales
    and a per-expert weight scale constant along K (per tensor, or per output row), both folded
    after the loop. ``GATE`` fuses the gate|up projection (``B`` the ``(E, 2N, K)`` stack, its
    scales interleaved alike) into a ``[BK, 2*BN]`` dot + SwiGLU ``glu``, emitting the bf16
    intermediate; ``GATE=False`` is the plain GEMM (bit-identical)."""
    NUM_PROGRAMS = tl.num_programs(axis=0)
    start_pid = tl.program_id(axis=0)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    if PACKED_SCHEDULE:
        total_m_tiles = load_packed_schedule(Schedule, BLOCK_SIZE_M)
        prefetched = prefetch_packed_entry(
            Schedule, grouped_tile_order(start_pid, total_m_tiles, num_n_tiles) // num_n_tiles, total_m_tiles
        )
    else:
        exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
            ExpertStart, NUM_EXPERTS_POW2, BLOCK_SIZE_M
        )
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for logical_tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_PROGRAMS):
        tile_id = grouped_tile_order(logical_tile_id, total_m_tiles, num_n_tiles)
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn, row0, n_off, m_start = (
            resolve_grouped_tile_packed(
                tile_id, num_n_tiles, prefetched,
                GatherIdx, ScatterIdx, BLOCK_SIZE_N, BLOCK_SIZE_M, GATE,
            ) if PACKED_SCHEDULE else resolve_grouped_tile(
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
        if PACKED_SCHEDULE:
            prefetched = prefetch_packed_entry(
                Schedule, grouped_tile_order(logical_tile_id + NUM_PROGRAMS, total_m_tiles, num_n_tiles) // num_n_tiles, total_m_tiles
            )
        a_ptrs = operand_tile_ptrs(A, in_row, offs_k, stride_a_m, stride_a_k, A_MEMORY_MODE, True)
        # GATE stacks gate|up into one [BK, 2*BN] tile; rows are interleaved, so the tile is
        # one 2*BN span. GATE=False -> plain [BK, BN].
        b_ptrs = weight_tile_ptrs(
            B + expert_id64 * stride_b_e,
            offs_bn,
            offs_k,
            stride_b_n,
            stride_b_k,
            False,
        )
        # per token, or the tile's own expert under a calibrated scale (stride_as_m 0); 1.0 on
        # the masked rows so the static arm's in-register divide never sees a zero
        a_s = tl.load(
            As + in_row * stride_as_m + expert_id64.to(tl.int32) * stride_as_e, mask=row_mask, other=1.0
        )

        acc = acc_init("dot", BLOCK_SIZE_M, (2 if GATE else 1) * BLOCK_SIZE_N, False)
        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
            a, _as = load_act_static(
                a_ptrs, ADescriptor, m_start, k * BLOCK_SIZE_K, row_mask, in_row, a_s[:, None],
                A_MEMORY_MODE, GatherIdx is not None,
            )
            w, _ws = load_weight_plain(
                b_ptrs, BDescriptor, row0, n_off, k * BLOCK_SIZE_K,
                GATE, True, B_MEMORY_MODE, False, BLOCK_SIZE_N, BLOCK_SIZE_K,
            )
            acc = acc + fp8_dot(a, w, False, BLOCK_SIZE_K)
            a_ptrs += BLOCK_SIZE_K * stride_a_k
            b_ptrs += BLOCK_SIZE_K * stride_b_k
        # per-token activation scale and the expert's weight scale, applied once post-loop
        acc = apply_row_scale(acc * a_s[:, None], Bs + expert_id64 * stride_bs_e, offs_bn, stride_bs_n)

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
            OUTPUT_FORMAT,
            1,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SIMULATE_UNFUSED,
            INTERMEDIATE_DTYPE,
            Bias=Bias,
            stride_bias_e=stride_bias_e,
            stride_bias_n=stride_bias_n,
            global_row=expert_id64,
        )


@bayesian_autotune(
    get_accelerator_autotuning_configs(
        packed_schedule=True,
        mx=True,
        tune_block_nk=True,
        tune_block_m=True,
        compute_modes=("dot_scaled", "dot"),
        a_memory_modes=("descriptor", "pointer"),
        b_memory_modes=("descriptor", "pointer"),
        pre_hook=grouped_mx_pre_hook,
        warp_spec=True,  # dot_scaled+WS+TMA compiles+wins on a clean context (was a num_warps=2 misread)
    ),  # prefill: no scalar branch; TMA descriptor vs pointer loads on both operands
    # the MXFP4/MXFP8 (and packed-activation) splits key themselves — the tuner appends
    # every tensor arg's dtype to its cache key (memory and disk);
    # GATE keys the gate|up arm separately (its stacked dot is 2*BN wide, a different tile optimum).
    # SWIZZLED_SCALES splits the pre-swizzled (descriptor) and row-major (affine) arms — they take
    # different memory-mode optima and must not share a tuned config.
    # OUTPUT_FORMAT/SWIZZLED_OUT key the requant epilogue explicitly (the C/Cs dtype append
    # splits them incidentally today; a descriptor Cs has no dtype token).
    ["N", "K", "tokens_per_expert_bit_length", "GATE", "SWIZZLED_SCALES", "OUTPUT_FORMAT", "SWIZZLED_OUT"],
    n_trials=100,
    benchmark_input=weight_input,
    path_anchor_axes=PATH_ANCHOR_AXES,
    # BK-within-K veto + the sm_10x dot_scaled shape/trap gates (this kernel had no
    # pruner while its BK span was {128,256} — the union span's BK=64 rows made the
    # gates load-bearing). swizzled_scale_config_pruner: the swizzled scale read needs
    # BK % 128 == 0 (REP_K collapses to 0 below it — a degenerate descriptor box).
    prune_configs_by={
        "early_config_prune": compose_pruners(
            swizzled_out_bm_pruner(),
            packed_schedule_scope_pruner(),
            mx_config_pruner("K", "N"),
            swizzled_scales_bm_pruner(),
            swizzled_scale_config_pruner(),
            descriptor_box_pruner(),
            smem_pruner(min_sets=2),  # two full buffer sets: the grouped pipeline's floor (see smem_pruner)
            warp_spec_compile_guard_pruner(),
            affine_scale_warp_spec_pruner(),
            warp_spec_memory_mode_pruner(),
        )
    },
)
@triton.jit
def mx_dynamic_matmul_grouped_kernel(
    A,  # (num_tokens, K) E4M3 activations (pre-quantized once by the wrapper), any row order
    ADescriptor,  # host TMA descriptor over A (rows, K_bytes), box (BM, BK_bytes); read iff A_MEMORY_MODE != "pointer"
    As,  # activation scales: row-major group scales read affine (gathered per row) iff not SWIZZLED_SCALES; else None (swizzled via ASDescriptor)
    ASDescriptor,  # host TMA descriptor over the SWIZZLE_32_4_4, expert-sorted/128-padded A scales; read iff SWIZZLED_SCALES
    B,  # (num_experts, N, K) E4M3 (MXFP8) or (num_experts, N, K // 2) packed E2M1 (MXFP4); 2N under GATE
    BDescriptor,  # host TMA descriptor over B viewed (E, 2N|N, K_bytes), box (1, (2|1)*BN, BK_bytes); read iff B_MEMORY_MODE != "pointer"
    Bs,  # (num_experts, N, K // SCALE_GROUP_K) UE8M0 weight scales (2N under GATE)
    BSDescriptor,  # host TMA descriptor over the SWIZZLE_32_4_4 per-expert B scales; read iff SWIZZLED_SCALES
    C,  # (S, N[/2]) output; under an OUTPUT_FORMAT the MX-requantized intermediate
    Cs,  # (S, N // SCALE_GROUP_K) row-major output scale; written iff OUTPUT_FORMAT and not SWIZZLED_OUT
    Bias,  # (E, N_out) per-expert output bias, N_out = 2N under GATE; read iff not None
    CSDescriptor,  # SWIZZLE_32_4_4 output-scale descriptor; written iff SWIZZLED_OUT (dummy else), like AS/BS
    AsGlobal,  # (1,) fp32 NVFP4 activation global g_a — multiplied onto the accumulator in-register (A is pre-quantized by the wrapper against it); read iff not None
    AsBsGlobal,  # (num_experts,) fp32 NVFP4 WEIGHT global g_b — recovers on the accumulator together with AsGlobal; read iff not None
    CsGlobal,  # (1,) fp32 NVFP4 output global (next proj's provided input_scale); normalizes the requant; read iff not None
    GatherIdx,  # (S,) int32 — sorted position -> source row of A; read only when not None
    ScatterIdx,  # (S,) int32 — sorted position -> destination row of C; read only when not None
    ExpertStart,  # (NUM_EXPERTS_POW2 + 1,) int32 — cumulative row starts, S sentinel
    Schedule,  # packed tile schedule (fill_packed_schedule); read iff PACKED_SCHEDULE
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
    stride_as_global,  # expert stride of AsGlobal (0 broadcasts a shared scalar)
    stride_bs_global,  # expert stride of AsBsGlobal (``global_scale_stride``)
    stride_cs_global,  # expert stride of CsGlobal (0 broadcasts a shared scalar)
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
    # the output format name (None | "mxfp8" | "mxfp4" | "nvfp4")
    OUTPUT_FORMAT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
    SWIZZLED_SCALES: tl.constexpr = True,  # scales pre-swizzled (5D weight + matching acts); else affine
    SWIZZLED_OUT: tl.constexpr = False,  # requant emits Cs in SWIZZLE_32_4_4 (descriptor); single source: the wrapper
    WARP_SPEC: tl.constexpr = False,  # tuner axis; dot_scaled+WS+TMA compiles+wins (num_warps%4, BM>=64)
    # tuner axis: resolve M-tiles from the packed schedule (3 scalar loads) instead of the
    # E-wide register-resident layout — register relief, bit-identical tiles; scope pruner
    # admits it at BM>=128 only (see packed_schedule_scope_pruner)
    PACKED_SCHEDULE: tl.constexpr = False,
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
    if PACKED_SCHEDULE:
        total_m_tiles = load_packed_schedule(Schedule, BLOCK_SIZE_M)
        prefetched = prefetch_packed_entry(
            Schedule, start_pid // tl.cdiv(N, BLOCK_SIZE_N), total_m_tiles
        )
    else:
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
            resolve_grouped_tile_packed(
                tile_id, num_n_tiles, prefetched,
                GatherIdx, ScatterIdx, BLOCK_SIZE_N, BLOCK_SIZE_M, GATE,
            ) if PACKED_SCHEDULE else resolve_grouped_tile(
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
        if PACKED_SCHEDULE:
            prefetched = prefetch_packed_entry(
                Schedule, (tile_id + NUM_SMS) // num_n_tiles, total_m_tiles
            )
        a_ptrs = operand_tile_ptrs(A, in_row, offs_ka, stride_a_m, stride_a_k, A_MEMORY_MODE, True)
        # Non-128 N: the partial last N-tile's pointer-arm rows wrap into B (offs_bn % N) so the load
        # never reads past the expert's N rows; the wrapped columns' output is masked off (N_COLS) in
        # the epilogue. Inert when N % BLOCK_SIZE_N == 0. The descriptor arm uses n_off (OOB-clamped),
        # and the scale rides blk_idx off the padded swizzled block — neither depends on offs_bn.
        offs_bn = offs_bn % (2 * N if GATE else N)
        kb_off = 0
        ka_off = 0
        # GATE reads the interleaved gate|up rows as one contiguous 2*BN tile; a
        # plain tile is the single BN block.
        b_ptrs = weight_tile_ptrs(
            B + expert_id64 * stride_b_e,
            offs_bn,
            offs_kb,
            stride_b_n,
            stride_b_k,
            False,
        )

        # Accumulator is 2*BN under GATE (the stacked gate|up dot), BN otherwise — split in the
        # epilogue for GATE.
        acc = acc_init("dot", BLOCK_SIZE_M, (2 if GATE else 1) * BLOCK_SIZE_N, False)
        # Pre-swizzled scales (SWIZZLE_32_4_4, emitted by the offline act-quant / requant): the M
        # tile's expert-sorted, 128-padded scale block is the flat tile index pid_m; the weight scale
        # block (descriptor bulk-load, BN=128) is expert*num_n_tiles + pid_n. Each expert's swizzled
        # slab is num_n_tiles REP-blocks (REP=2 under GATE, whose tile spans both).
        pid_m = tile_id // num_n_tiles
        weight_blk = (expert_id64 * num_n_tiles + pid_n).to(tl.int32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
            a, a_s = load_act_mx(
                a_ptrs, As, None, 0, row_mask, row_mask, ADescriptor, m_start, ka_off,
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
                (2 if GATE else 1) * BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, False,
            )
            a_ptrs += (BLOCK_SIZE_K // ACT_VALUES_PER_BYTE) * stride_a_k
            ka_off += BLOCK_SIZE_K // ACT_VALUES_PER_BYTE
            b_ptrs += (BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE) * stride_b_k
            kb_off += BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE

        # NVFP4 two-level: block e4m3 scales rode through dot_scaled; recover the combined per-tensor
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
            OUTPUT_FORMAT,
            SCALE_GROUP_K,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SIMULATE_UNFUSED,
            INTERMEDIATE_DTYPE,
            SWIZZLED_OUT=SWIZZLED_OUT,  # single source: the wrapper decided it and built Cs to match
            CSDescriptor=CSDescriptor,
            CsGlobal=CsGlobal,
            stride_cs_global=stride_cs_global,
            N_COLS=N,  # mask the partial last N-tile's column tail (non-128 N; inert when N % BN == 0)
            GlobalScale=AsBsGlobal,
                stride_global_e=stride_bs_global,
            GlobalScaleA=AsGlobal,
            stride_global_a=stride_as_global,
            global_row=expert_id64,
            Bias=Bias,
            stride_bias_e=stride_bias_e,
            stride_bias_n=stride_bias_n,
        )


def _mx_weight_only_grouped_configs():
    configs = get_accelerator_autotuning_configs(
        packed_schedule=True,
        swap_ab_prefill=True,
        tune_block_nk=True,
        tune_block_m=True,
        warp_spec=True,
        compute_modes=("dot", "dot_scaled"),
        a_memory_modes=("descriptor", "pointer"),
        b_memory_modes=("descriptor", "pointer"),
        pre_hook=grouped_pre_hook,
    )
    for config in configs:
        config.kwargs["PIPELINE"] = "k_loop"
    # Overlap consecutive output tiles behind the computed-lhs prefill seeds.
    for config in [c for c in configs if c.kwargs["SWAP_AB"] and c.num_stages == 2]:
        for packed in (False, True):
            seed = triton.Config(
                dict(config.kwargs, PACKED_SCHEDULE=packed), num_warps=config.num_warps,
                num_stages=config.num_stages, pre_hook=config.pre_hook,
            )
            configs.extend(pipeline_autotuning_configs([seed], ("tile_loop", "flattened")))
    return configs


@bayesian_autotune(
    # weight-only is a plain bf16 dot (fp4/fp8 weight upcast per-group in-loop). Host-TMA memory-mode axis
    # on both operands + WS. weight-only uses NO swizzled scales (those are a tcgen05 dot_scaled layout;
    # this kernel decodes the group scale in registers) — the grouped weight-scale affine load is
    # UNMASKED, so it never blocks WS. The one WS blocker is the masked gathered/padded ACTIVATION
    # value load (`other=0.0`); the descriptor arm's tma gather4 has no such constant, so WS pairs
    # with A_MEMORY_MODE=descriptor. warp_spec_compile_guard_pruner owns the num_warps%4 / BM>=64
    # WS-compile region; WS+pointer-A (masked load) is scored out.
    _mx_weight_only_grouped_configs(),
    ["N", "K", "tokens_per_expert_bit_length", "GATE"],
    n_trials=100,
    benchmark_input=weight_input,
    path_anchor_axes=PATH_ANCHOR_AXES,
    prune_configs_by={
        "early_config_prune": compose_pruners(
            warp_spec_memory_mode_pruner(weight_only=True),
            packed_schedule_scope_pruner(),
            # the tail tile slides back in bounds (see the K-loop) — BK only has to fit
            block_fits_dim_pruner("K"),
            # no N veto: this kernel wraps its N tail (``offs_bn % N`` above the loop, wrapped
            # columns masked by N_COLS in the epilogue), so non-dividing BN is correct — the
            # veto belongs to the maskless-store kernels (see OPTIMIZATION_LOG: n-veto-lift)
            mx_config_pruner("K", "N", block_within_k=False),  # dot_scaled shape gates
            gate_stacked_tmem_trap_pruner(),
            descriptor_box_pruner(),
            smem_pruner(),
            warp_spec_compile_guard_pruner(),
            gated_pointer_weight_warp_spec_pruner(),
        )
    },
)
@triton.jit
def mx_weight_only_matmul_grouped_kernel(
    A,  # (num_tokens, K) raw BF16/FP16 activations — NOT quantized
    ADescriptor,  # host TMA descriptor over A (rows, K), box (BM, BK); read iff A_MEMORY_MODE != "pointer"
    AsGlobal,
    B,  # (num_experts, N, K[/2]) MXFP4/NVFP4 (packed E2M1) / MXFP8 (E4M3) weights; 2N under GATE
    BDescriptor,  # host TMA descriptor over B viewed (E, 2N|N, K_bytes), box (1, (2|1)*BN, BK_bytes); read iff B_MEMORY_MODE != "pointer"
    Bs,  # (num_experts, N, K // SCALE_GROUP_K) group scales — UE8M0 (MX) or E4M3 (NVFP4)
    BsGlobal,  # (num_experts,) fp32 NVFP4 per-expert global — recovers on the accumulator; read iff not None
    C,
    Bias,  # (E, N_out) per-expert output bias, N_out = 2N under GATE; read iff not None
    GatherIdx,
    ScatterIdx,
    ExpertStart,
    Schedule,  # packed tile schedule (fill_packed_schedule); read iff PACKED_SCHEDULE
    S,
    N,
    K,
    stride_a_m,
    stride_a_k,
    stride_as_global: tl.constexpr,
    stride_b_e,
    stride_b_k,
    stride_b_n,
    stride_bs_e,
    stride_bs_n,
    stride_bs_k,
    stride_bs_global,  # expert stride of BsGlobal (``global_scale_stride``)
    stride_c_m,
    stride_c_n,
    stride_bias_e,
    stride_bias_n,
    num_experts,
    tokens_per_expert_bit_length,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS_POW2: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    WEIGHT_VALUES_PER_BYTE: tl.constexpr,
    A_MEMORY_MODE: tl.constexpr = "pointer",
    B_MEMORY_MODE: tl.constexpr = "pointer",
    COMPUTE_MODE: tl.constexpr = "dot",
    WARP_SPEC: tl.constexpr = False,
    # tuner axis: resolve M-tiles from the packed schedule (3 scalar loads) instead of the
    # E-wide register-resident layout — register relief, bit-identical tiles; scope pruner
    # admits it at BM>=128 only (see packed_schedule_scope_pruner)
    PACKED_SCHEDULE: tl.constexpr = False,
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
    PIPELINE: tl.constexpr = "k_loop",
    SWAP_AB: tl.constexpr = False,
):
    """weight-only grouped expert matmul: raw bf16 activations against MXFP4/MXFP8 weights upcast to bf16
    in-loop (unpack + per-group group-scale), plain ``tl.dot``. The ``matmul_ogs`` format. Persistent
    grid-stride; pointer or host-TMA operands, affine scales. ``GATE`` fuses the (E, 2N, K) gate|up
    stack + SwiGLU."""
    n_width: tl.constexpr = (2 if GATE else 1) * BLOCK_SIZE_N
    NUM_PROGRAMS = tl.num_programs(axis=0)
    start_pid = tl.program_id(axis=0)
    if PACKED_SCHEDULE:
        total_m_tiles = load_packed_schedule(Schedule, BLOCK_SIZE_M)
        if PIPELINE == "k_loop":
            prefetched = prefetch_packed_entry(
                Schedule, start_pid // tl.cdiv(N, BLOCK_SIZE_N), total_m_tiles
            )
    else:
        exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
            ExpertStart, NUM_EXPERTS_POW2, BLOCK_SIZE_M
        )
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_kb = tl.arange(0, BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE)
    for tile_id in tl.range(
        start_pid, total_m_tiles * num_n_tiles, NUM_PROGRAMS,
        flatten=PIPELINE == "flattened", warp_specialize=WARP_SPEC and PIPELINE != "k_loop",
    ):
        if PACKED_SCHEDULE and PIPELINE != "k_loop":
            prefetched = prefetch_packed_entry(Schedule, tile_id // num_n_tiles, total_m_tiles)
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn, row0, n_off, m_start = (
            resolve_grouped_tile_packed(
                tile_id, num_n_tiles, prefetched,
                GatherIdx, ScatterIdx, BLOCK_SIZE_N, BLOCK_SIZE_M, GATE,
            ) if PACKED_SCHEDULE else resolve_grouped_tile(
                tile_id, num_n_tiles, exp_start, freqs, tile_start_excl, e_offs,
                GatherIdx, ScatterIdx, BLOCK_SIZE_N, BLOCK_SIZE_M, GATE,
            )
        )
        if PACKED_SCHEDULE and PIPELINE == "k_loop":
            prefetched = prefetch_packed_entry(
                Schedule, (tile_id + NUM_PROGRAMS) // num_n_tiles, total_m_tiles
            )
        offs_bn = offs_bn % (2 * N if GATE else N)  # non-128 N: pointer rows wrap; wrapped cols masked (N_COLS) in epilogue
        a_ptrs = operand_tile_ptrs(A, in_row, offs_k, stride_a_m, stride_a_k, A_MEMORY_MODE, True)
        b_ptrs = weight_tile_ptrs(
            B + expert_id64 * stride_b_e, offs_bn, offs_kb,
            stride_b_n, stride_b_k, SWAP_AB,
        )
        acc = acc_init("dot", BLOCK_SIZE_M, n_width, SWAP_AB)
        kb_off = 0
        # The tail K-tile slides back to end at K instead of running past the row, so BK need
        # only FIT K (see the batched sibling): the overlapped lanes are re-read and cancelled
        # by zeroing their activation. BK | K leaves every offset untouched.
        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC and PIPELINE == "k_loop"):
            k_start = tl.minimum(k * BLOCK_SIZE_K, K - BLOCK_SIZE_K)
            overlap = k * BLOCK_SIZE_K - k_start
            a, _as = load_act_plain(
                a_ptrs, ADescriptor, m_start, k_start, row_mask, in_row,
                A_MEMORY_MODE, GatherIdx is not None,
            )
            if overlap > 0:  # only the slid tail trip pays the select
                a = tl.where(tl.arange(0, BLOCK_SIZE_K)[None, :] >= overlap, a, 0.0)
            w, w_s = load_weight_mx(
                b_ptrs, BDescriptor, Bs, None, None, 0, row0, n_off, kb_off, 0, expert_id64, pid_n, k, N, K,
                stride_bs_e, stride_bs_n, stride_bs_k,
                GATE, True, False, B_MEMORY_MODE, SWAP_AB, False,
                BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, WEIGHT_VALUES_PER_BYTE,
                k_col_off=-(overlap // SCALE_GROUP_K),
            )
            acc = mx_weight_only_compute(
                acc, a, w, w_s, COMPUTE_MODE, BLOCK_SIZE_K, n_width, SCALE_GROUP_K,
                SWAP_AB=SWAP_AB,
            )
            step = tl.minimum((k + 1) * BLOCK_SIZE_K, K - BLOCK_SIZE_K) - k_start
            a_ptrs += step * stride_a_k
            b_ptrs += (step // WEIGHT_VALUES_PER_BYTE) * stride_b_k
            kb_off = tl.minimum((k + 1) * BLOCK_SIZE_K, K - BLOCK_SIZE_K) // WEIGHT_VALUES_PER_BYTE

        if SWAP_AB:
            acc = tl.trans(acc)
        gemm_epilogue(
            C, C, acc, out_row, pid_n, tile_id // num_n_tiles, row_mask,
            stride_c_m, stride_c_n, 1, 1,
            BLOCK_SIZE_M, BLOCK_SIZE_N, GATE, None, 1,
            ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE,
            N_COLS=N,
            GlobalScale=BsGlobal, stride_global_e=stride_bs_global,
            GlobalScaleA=AsGlobal, stride_global_a=stride_as_global,
            global_row=expert_id64,
            Bias=Bias,
            stride_bias_e=stride_bias_e,
            stride_bias_n=stride_bias_n,
        )


def _int_grouped_configs():
    configs = _grouped_dot_configs()
    for config in configs:
        # the small-tile oversubscription was measured on SM10x only; elsewhere one program per SM
        config.kwargs["PROGRAMS_PER_SM"] = (
            persistent_programs_per_sm(config.kwargs["BLOCK_SIZE_M"]) if is_sm10x() else 1
        )
        config.kwargs["SWAP_AB"] = False
    # These narrower tiles fit twice per SM when the initial WS register reservation is capped.
    # Without the cap the vector partition reserves the whole register file, even with no spills.
    if is_sm10x():
        for bn in (64, 128):
            for packed in (False, True):
                for stages in (2, 3):
                    configs.append(triton.Config(
                        dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=128, WARP_SPEC=True,
                             A_MEMORY_MODE="host_descriptor", B_MEMORY_MODE="host_descriptor",
                             PACKED_SCHEDULE=packed, PROGRAMS_PER_SM=2, SWAP_AB=False),
                        num_warps=4, num_stages=stages, maxnreg=128, pre_hook=grouped_pre_hook,
                    ))
        # A computed MMA lhs can stay in tensor memory instead of staging decoded weights through shared memory.
        for bn in (64, 128):
            for packed in (False, True):
                for stages in (2, 3):
                    configs.append(triton.Config(
                        dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=256, WARP_SPEC=True,
                             A_MEMORY_MODE="host_descriptor", B_MEMORY_MODE="host_descriptor",
                             PACKED_SCHEDULE=packed, PROGRAMS_PER_SM=1, SWAP_AB=True),
                        num_warps=4, num_stages=stages, pre_hook=grouped_pre_hook,
                    ))
    for config in configs:
        config.kwargs["PIPELINE"] = "k_loop"
    # Tile overlap and flattening are separate choices: GLU register pressure can favor preserving the K loop.
    for config in list(configs):
        if config.kwargs["SWAP_AB"] and config.num_stages == 2:
            configs.extend(pipeline_autotuning_configs([config], ("flattened",)))
            if config.kwargs["BLOCK_SIZE_N"] == 64:
                pipelines = ("tile_loop", "flattened") if config.kwargs["PACKED_SCHEDULE"] else ("tile_loop",)
                configs.extend(pipeline_autotuning_configs([config], pipelines, num_warps=8))
            elif config.kwargs["PACKED_SCHEDULE"]:
                # More epilogue warps relieve FP32 register pressure when the down projection adds bias.
                configs.extend(pipeline_autotuning_configs([config], ("flattened",), num_warps=8))
    return configs


@bayesian_autotune(
    # the MX weight-only grouped kernel's space minus dot_scaled (the scaled MMA takes float formats only)
    _int_grouped_configs(),
    # the width and the scale group decide which tiles fit and compile: part of the key
    ["N", "K", "tokens_per_expert_bit_length", "GATE", "INT_BITS", "SCALE_GROUP_K", "SIMULATE_UNFUSED"],
    n_trials=100,
    benchmark_input=weight_input,
    path_anchor_axes=PATH_ANCHOR_AXES,
    prune_configs_by={
        "early_config_prune": compose_pruners(
            warp_spec_memory_mode_pruner(weight_only=True),
            packed_schedule_scope_pruner(),
            # the tail tile slides back in bounds (see the K-loop) — BK only has to fit; no N veto, the N tail
            # wraps and the epilogue masks it
            block_fits_dim_pruner("K"),
            int_scale_group_tile_pruner(),
            gate_stacked_tmem_trap_pruner(),
            descriptor_box_pruner(),
            smem_pruner(),
            warp_spec_compile_guard_pruner(),
            gated_pointer_weight_warp_spec_pruner(),
        )
    },
)
@triton.jit
def int_weight_only_matmul_grouped_kernel(
    A,  # (num_tokens, K) prepared bf16/fp16, FP8 (W4A8), or INT8 (SmoothQuant) activations
    ADescriptor,  # host TMA descriptor over A (rows, K), box (BM, BK); read iff A_MEMORY_MODE != "pointer"
    AsGlobal,  # fp32 calibrated input scale; None for weight-only
    B,  # (num_experts, N, K // VPW) interleaved uint32 words of VPW = 32 // INT_BITS values; 2N under GATE
    BDescriptor,  # host TMA descriptor over B viewed (E, 2N|N, K // VPW), box (1, (2|1)*BN, BK // VPW); read iff B_MEMORY_MODE != "pointer"
    Bs,  # (num_experts, N, K // SCALE_GROUP_K) float group scales
    Bz,  # (num_experts, N, K // SCALE_GROUP_K) integer zero points; None for a symmetric weight
    BsGlobal,  # fp32 W4A8 second-level weight scale; None otherwise
    C,
    Bias,  # (E, N_out) per-expert output bias, N_out = 2N under GATE; read iff not None
    GatherIdx,
    ScatterIdx,
    ExpertStart,
    Schedule,  # packed tile schedule (fill_packed_schedule); read iff PACKED_SCHEDULE
    S,
    N,
    K,
    stride_a_m,
    stride_a_k,
    stride_asglobal_e,
    stride_b_e,
    stride_b_k,
    stride_b_n,
    stride_bs_e,
    stride_bs_n,
    stride_bs_k,
    stride_bz_e,
    stride_bz_n,
    stride_bz_k,
    stride_bsglobal_e,
    stride_c_m,
    stride_c_n,
    stride_bias_e,
    stride_bias_n,
    num_experts,
    tokens_per_expert_bit_length,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS_POW2: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    A_MEMORY_MODE: tl.constexpr = "pointer",
    B_MEMORY_MODE: tl.constexpr = "pointer",
    WARP_SPEC: tl.constexpr = False,
    PACKED_SCHEDULE: tl.constexpr = False,
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
    INT_BITS: tl.constexpr = 4,
    PROGRAMS_PER_SM: tl.constexpr = 1,
    SWAP_AB: tl.constexpr = False,
    PIPELINE: tl.constexpr = "k_loop",
):
    """Integer grouped expert matmul: bf16/fp16 weight-only, calibrated INT4/FP8, or INT8 SmoothQuant.
    Persistent grid-stride; pointer or host-TMA operands, affine scales.
    ``GATE`` fuses the (E, 2N, K) gate|up stack + SwiGLU."""
    n_width: tl.constexpr = (2 if GATE else 1) * BLOCK_SIZE_N
    # output-rows-major for the mma.sync tiles (BM < 64), whose rhs stays in registers; K-major for the tcgen05 tiles,
    # whose rhs is staged through shared memory (int_dequant_tile)
    ROWS_MAJOR: tl.constexpr = SWAP_AB or BLOCK_SIZE_M < 64
    NUM_PROGRAMS = tl.num_programs(axis=0)  # persistent_programs_per_sm per SM
    start_pid = tl.program_id(axis=0)
    if PACKED_SCHEDULE:
        total_m_tiles = load_packed_schedule(Schedule, BLOCK_SIZE_M)
        if PIPELINE == "k_loop":
            prefetched = prefetch_packed_entry(
                Schedule, start_pid // tl.cdiv(N, BLOCK_SIZE_N), total_m_tiles
            )
    else:
        exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
            ExpertStart, NUM_EXPERTS_POW2, BLOCK_SIZE_M
        )
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    VPW: tl.constexpr = 32 // INT_BITS  # values per uint32 word
    offs_kb = tl.arange(0, BLOCK_SIZE_K // VPW)
    for tile_id in tl.range(
        start_pid, total_m_tiles * num_n_tiles, NUM_PROGRAMS,
        flatten=PIPELINE == "flattened", warp_specialize=WARP_SPEC and PIPELINE != "k_loop",
    ):
        if PACKED_SCHEDULE and PIPELINE != "k_loop":
            # Let the outer pipeline schedule this tile's entry instead of carrying a second prefetch across it.
            prefetched = prefetch_packed_entry(Schedule, tile_id // num_n_tiles, total_m_tiles)
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn, row0, n_off, m_start = (
            resolve_grouped_tile_packed(
                tile_id, num_n_tiles, prefetched,
                GatherIdx, ScatterIdx, BLOCK_SIZE_N, BLOCK_SIZE_M, GATE,
            ) if PACKED_SCHEDULE else resolve_grouped_tile(
                tile_id, num_n_tiles, exp_start, freqs, tile_start_excl, e_offs,
                GatherIdx, ScatterIdx, BLOCK_SIZE_N, BLOCK_SIZE_M, GATE,
            )
        )
        if PACKED_SCHEDULE and PIPELINE == "k_loop":
            prefetched = prefetch_packed_entry(
                Schedule, (tile_id + NUM_PROGRAMS) // num_n_tiles, total_m_tiles
            )
        offs_bn = offs_bn % (2 * N if GATE else N)  # non-128 N: pointer rows wrap; wrapped cols masked (N_COLS) in epilogue
        a_ptrs = operand_tile_ptrs(A, in_row, offs_k, stride_a_m, stride_a_k, A_MEMORY_MODE, True)
        b_ptrs = weight_tile_ptrs(
            B + expert_id64 * stride_b_e, offs_bn, offs_kb,
            stride_b_n, stride_b_k, ROWS_MAJOR,
        )
        acc = acc_init("dot", BLOCK_SIZE_M, n_width, SWAP_AB, tl.int32 if A.dtype.element_ty == tl.int8 else tl.float32)
        # zero points run one K step ahead (the 2D kernel's K-loop)
        k_last = K // BLOCK_SIZE_K - 1
        w_s_next = load_weight_int_group(
            Bs, Bs, expert_id64, pid_n, 0, N, K, stride_bs_e, stride_bs_n, stride_bs_k,
            GATE, False, BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, GROUPED=True,
        )
        w_z_next = None
        if Bz is not None:
            w_z_next = load_weight_int_group(
                Bz, Bz, expert_id64, pid_n, 0, N, K, stride_bz_e, stride_bz_n, stride_bz_k,
                GATE, False, BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, GROUPED=True,
            )
        # BK divides K (int_scale_group_tile_pruner): no tail tile, and a constant pointer step keeps the weight
        # rows' 16-byte alignment provable (a runtime step drops the async copies to 8 bytes)
        for k in tl.range(0, K // BLOCK_SIZE_K, warp_specialize=WARP_SPEC and PIPELINE == "k_loop"):
            k_start = k * BLOCK_SIZE_K
            kb_off = k_start // VPW
            a, _as = load_act_plain(
                a_ptrs, ADescriptor, m_start, k_start, row_mask, in_row,
                A_MEMORY_MODE, GatherIdx is not None,
            )
            w_s = w_s_next
            w_s_next = load_weight_int_group(
                Bs, Bs, expert_id64, pid_n, tl.minimum(k + 1, k_last), N, K,
                stride_bs_e, stride_bs_n, stride_bs_k, GATE, False,
                BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, GROUPED=True,
            )
            w, _w_unused = load_weight_plain(
                b_ptrs, BDescriptor, row0, n_off, kb_off, GATE, True, B_MEMORY_MODE, ROWS_MAJOR,
                BLOCK_SIZE_N, BLOCK_SIZE_K, VPW,
            )
            w_z = w_z_next
            if Bz is not None:
                w_z_next = load_weight_int_group(
                    Bz, Bz, expert_id64, pid_n, tl.minimum(k + 1, k_last), N, K, stride_bz_e, stride_bz_n,
                    stride_bz_k, GATE, False, BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, GROUPED=True,
                )
            acc = int_weight_only_compute(
                acc, a, w, w_s, w_z, BLOCK_SIZE_K, n_width, SCALE_GROUP_K, "dot", ROWS_MAJOR, INT_BITS, SWAP_AB,
                weight_global=None if BsGlobal is None else tl.load(BsGlobal + expert_id64 * stride_bsglobal_e),
            )
            a_ptrs += BLOCK_SIZE_K * stride_a_k
            b_ptrs += (BLOCK_SIZE_K // VPW) * stride_b_k

        if SWAP_AB:
            acc = tl.trans(acc)
        output_scale = BsGlobal
        output_scale_stride = stride_bsglobal_e
        if A.dtype.element_ty == tl.int8:
            output_scale = Bs + offs_bn * stride_bs_n
            output_scale_stride = stride_bs_e
        gemm_epilogue(
            C, C, acc, out_row, pid_n, tile_id // num_n_tiles, row_mask,
            stride_c_m, stride_c_n, 1, 1,
            BLOCK_SIZE_M, BLOCK_SIZE_N, GATE, None, 1,
            ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE,
            N_COLS=N,
            global_row=expert_id64,
            GlobalScale=output_scale, stride_global_e=output_scale_stride,
            GlobalScaleA=AsGlobal, stride_global_a=stride_asglobal_e,
            Bias=Bias,
            stride_bias_e=stride_bias_e,
            stride_bias_n=stride_bias_n,
        )


def _full_precision_grouped_configs():
    configs = _grouped_dot_configs()
    for config in configs:
        config.kwargs["PIPELINE"] = "k_loop"
    if is_sm10x():
        seeds = [
            c for c in configs
            if c.kwargs["WARP_SPEC"] and c.kwargs["A_MEMORY_MODE"] == c.kwargs["B_MEMORY_MODE"] == "host_descriptor"
            and c.kwargs["BLOCK_SIZE_M"] == 128 and c.kwargs["BLOCK_SIZE_K"] == 64 and c.num_stages == 4
            and (c.kwargs["BLOCK_SIZE_N"], c.num_warps) in ((128, 8), (256, 4))
        ]
        configs.extend(pipeline_autotuning_configs(seeds, ("tile_loop", "flattened")))
    return configs


@bayesian_autotune(
    _full_precision_grouped_configs(),
    # GATE keys the gate|up arm separately (its dot is 2*BN wide, a different tile optimum).
    ["N", "K", "tokens_per_expert_bit_length", "GATE"],
    n_trials=100,
    benchmark_input=weight_input,
    path_anchor_axes=PATH_ANCHOR_AXES,
    # BLOCK_SIZE_K is a tuned axis and the K-loop is maskless — veto non-dividing BKs.
    prune_configs_by={
        "early_config_prune": compose_pruners(
            block_within_dim_pruner("K"),
            block_within_dim_pruner("N", "BLOCK_SIZE_N"),
            warp_spec_compile_guard_pruner(),
            packed_schedule_scope_pruner(),
            descriptor_box_pruner(),
            gate_stacked_tmem_trap_pruner(),
            smem_pruner(),
        )
    },
)
@triton.jit
def full_precision_matmul_grouped_kernel(
    A,  # (num_tokens, K) BF16/FP16 activations, any row order
    ADescriptor,  # host TMA descriptor over A (rows, K), box (BM, BK); read iff A_MEMORY_MODE != "pointer"
    B,  # (num_experts, N, K) weights in A's dtype; under GATE the (num_experts, 2N, K) gate|up stack
    BDescriptor,  # host TMA descriptor over B viewed (E, 2N|N, K), box (1, (2|1)*BN, BK); read iff B_MEMORY_MODE != "pointer"
    C,  # (S, N) output; under GATE the GLU intermediate
    Bias,  # (E, N_out) per-expert output bias, N_out = 2N under GATE; read iff not None
    GatherIdx,  # (S,) int32 — sorted position -> source row of A; read only when not None
    ScatterIdx,  # (S,) int32 — sorted position -> destination row of C; read only when not None
    ExpertStart,  # (NUM_EXPERTS_POW2 + 1,) int32 — cumulative row starts, S sentinel
    Schedule,  # packed tile schedule (fill_packed_schedule); read iff PACKED_SCHEDULE
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
    num_experts,
    tokens_per_expert_bit_length,  # autotune key only (log2 avg-tokens bucket); unused in body
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS_POW2: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
    # tuner axis: resolve M-tiles from the packed schedule (3 scalar loads) instead of the
    # E-wide register-resident layout — pure register relief, bit-identical tiles; wins only
    # at pressure-walled big-tile configs (dsv4 BM128/BK256: regs 244->187, +8%)
    PACKED_SCHEDULE: tl.constexpr = False,
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
    PIPELINE: tl.constexpr = "k_loop",
):
    """Full-precision grouped expert matmul kernel — persistent grid-stride over tiles.

    Grouped expert scheduling over unquantized BF16/FP16 activations and weights: plain
    ``tl.dot`` with fp32 accumulation, no scales anywhere. ``GATE`` fuses the gate|up
    projection (``B`` the ``(E, 2N, K)`` stack) into a ``[BK, 2*BN]`` dot + SwiGLU ``glu``,
    emitting the intermediate; ``GATE=False`` is the plain GEMM (bit-identical)."""
    NUM_PROGRAMS = tl.num_programs(axis=0)
    start_pid = tl.program_id(axis=0)
    if PACKED_SCHEDULE:
        total_m_tiles = load_packed_schedule(Schedule, BLOCK_SIZE_M)
        if PIPELINE == "k_loop":
            prefetched = prefetch_packed_entry(
                Schedule, start_pid // tl.cdiv(N, BLOCK_SIZE_N), total_m_tiles
            )
    else:
        exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
            ExpertStart, NUM_EXPERTS_POW2, BLOCK_SIZE_M
        )
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    for tile_id in tl.range(
        start_pid, total_m_tiles * num_n_tiles, NUM_PROGRAMS,
        flatten=PIPELINE == "flattened", warp_specialize=WARP_SPEC and PIPELINE != "k_loop",
    ):
        if PACKED_SCHEDULE and PIPELINE != "k_loop":
            prefetched = prefetch_packed_entry(Schedule, tile_id // num_n_tiles, total_m_tiles)
        if PACKED_SCHEDULE:
            pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn, row0, n_off, m_start = (
                resolve_grouped_tile_packed(
                    tile_id, num_n_tiles, prefetched,
                    GatherIdx, ScatterIdx, BLOCK_SIZE_N, BLOCK_SIZE_M, GATE,
                )
            )
        else:
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
        if PACKED_SCHEDULE and PIPELINE == "k_loop":
            prefetched = prefetch_packed_entry(
                Schedule, (tile_id + NUM_PROGRAMS) // num_n_tiles, total_m_tiles
            )
        a_ptrs = operand_tile_ptrs(A, in_row, offs_k, stride_a_m, stride_a_k, A_MEMORY_MODE, True)
        # GATE reads the interleaved gate|up rows as one [BK, 2*BN] tile.
        # GATE=False -> the plain [BK, BN] tile. Pointer arm — the descriptor modes
        # fetch the [BN, BK] box at row0 and transpose it instead.
        b_ptrs = weight_tile_ptrs(
            B + expert_id64 * stride_b_e,
            offs_bn,
            offs_k,
            stride_b_n,
            stride_b_k,
            False,
        )

        acc = acc_init("dot", BLOCK_SIZE_M, (2 if GATE else 1) * BLOCK_SIZE_N, False)
        # Resolve each descriptor operand once per tile: the host-built box as passed, or an
        # in-kernel tensormap where the host cannot build one (a host descriptor IS a TMA
        # descriptor, and Xe has no TMA). The weight box is 2D over THIS expert's slab rather
        # than 3D over the stack, for two reasons: a device-built 3D box silently drops its
        # outermost offset on Xe (measured — every expert reads expert 0), and the slab's own
        # K-contiguous axes are the orientation the Xe 2D block load needs to reach rate (a
        # (K, N) view, what the pointer arm effectively does, falls well off it). A needs no
        # such care — its box is 2D either way, and no gather reaches the descriptor arm here
        # (``descriptor_box_pruner`` fences gathered descriptor A to sm_100), so the rows are
        # the contiguous span at ``m_start``, tails zero-fill like the affine arm's mask, and
        # rows outside the expert are dropped by the epilogue's row_mask.
        a_desc = operand_tile_descriptor(
            ADescriptor, A, S, K, stride_a_m, 1,  # K is the contiguous dim of (S, K)
            BLOCK_SIZE_M, BLOCK_SIZE_K, A_MEMORY_MODE,
        )
        b_desc = operand_tile_descriptor(
            BDescriptor, B + expert_id64 * stride_b_e, N, K, stride_b_n, 1,  # and of the slab
            BLOCK_SIZE_N, BLOCK_SIZE_K, B_MEMORY_MODE, GATE,
        )
        # The box's rank, not the backend: only this kernel resolves its own 2D per-expert box
        # (above). The other grouped kernels run the same mode on Xe against host-built 3D boxes.
        B_DESCRIPTOR_IS_2D: tl.constexpr = B_MEMORY_MODE == "device_descriptor"
        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC and PIPELINE == "k_loop"):
            a, _as = load_act_plain(
                a_ptrs, a_desc, m_start, k * BLOCK_SIZE_K, row_mask, in_row,
                A_MEMORY_MODE, GatherIdx is not None,
            )
            w, _ws = load_weight_plain(
                b_ptrs, b_desc, row0, n_off, k * BLOCK_SIZE_K,
                GATE, True, B_MEMORY_MODE, False, BLOCK_SIZE_N, BLOCK_SIZE_K,
                B_DESCRIPTOR_IS_2D=B_DESCRIPTOR_IS_2D,
            )
            acc = acc + fp8_dot(a, w, False, BLOCK_SIZE_K)
            if A_MEMORY_MODE == "pointer":
                a_ptrs += BLOCK_SIZE_K * stride_a_k
            if B_MEMORY_MODE == "pointer":
                b_ptrs += BLOCK_SIZE_K * stride_b_k

        gemm_epilogue(
            C,
            None,  # no requant arm here
            acc,
            out_row,
            pid_n,
            tile_id // num_n_tiles,
            row_mask,
            stride_c_m,
            stride_c_n,
            1,  # Cs strides: unread, Cs is None
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
            Bias=Bias,
            stride_bias_e=stride_bias_e,
            stride_bias_n=stride_bias_n,
            global_row=expert_id64,
        )


def _block_fp8_weight_only_grouped_configs():
    configs = _grouped_dot_configs()
    for config in configs:
        config.kwargs.update(SWAP_AB=False, PIPELINE="k_loop")
    # Stage decoded weights on the MMA lhs and overlap consecutive output tiles.
    if is_sm10x():
        for packed in (False, True):
            seed = triton.Config(
                dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=256,
                     WARP_SPEC=True, PACKED_SCHEDULE=packed, SWAP_AB=True,
                     A_MEMORY_MODE="host_descriptor", B_MEMORY_MODE="host_descriptor"),
                num_warps=8, num_stages=2, pre_hook=grouped_pre_hook,
            )
            configs.extend(pipeline_autotuning_configs([seed], ("tile_loop", "flattened")))
    return configs


@bayesian_autotune(
    _block_fp8_weight_only_grouped_configs(),
    # GATE keys the gate|up arm separately (its dot is 2*BN wide, a different tile optimum).
    ["N", "K", "tokens_per_expert_bit_length", "GATE", "BLOCK_N", "BLOCK_K", "GATE_SCALES"],
    n_trials=100,
    benchmark_input=weight_input,
    path_anchor_axes=PATH_ANCHOR_AXES,
    # BLOCK_SIZE_K is a tuned axis and the K-loop is maskless — veto non-dividing BKs.
    prune_configs_by={
        "early_config_prune": compose_pruners(
            block_within_dim_pruner("K"),
            block_within_dim_pruner("N", "BLOCK_SIZE_N"),
            warp_spec_compile_guard_pruner(),
            warp_spec_memory_mode_pruner(gather_only=True),
            packed_schedule_scope_pruner(),
            descriptor_box_pruner(),
            gate_stacked_tmem_trap_pruner(),
            smem_pruner(),
        )
    },
)
@triton.jit
def block_fp8_weight_only_matmul_grouped_kernel(
    A,  # (num_tokens, K) BF16/FP16 activations, any row order
    ADescriptor,  # host TMA descriptor over A (rows, K), box (BM, BK); read iff A_MEMORY_MODE != "pointer"
    B,  # (num_experts, N, K) FP8 weights; under GATE the (num_experts, 2N, K) gate|up stack
    BDescriptor,  # host TMA descriptor over B viewed (E, 2N|N, K), box (1, (2|1)*BN, BK); read iff B_MEMORY_MODE != "pointer"
    Bs,  # coarse FP8 weight scales
    C,  # (S, N) output; under GATE the GLU intermediate
    Bias,  # (E, N_out) per-expert output bias, N_out = 2N under GATE; read iff not None
    GatherIdx,  # (S,) int32 — sorted position -> source row of A; read only when not None
    ScatterIdx,  # (S,) int32 — sorted position -> destination row of C; read only when not None
    ExpertStart,  # (NUM_EXPERTS_POW2 + 1,) int32 — cumulative row starts, S sentinel
    Schedule,  # packed tile schedule; read iff PACKED_SCHEDULE
    # Shape
    S,
    N,
    K,
    num_experts,
    tokens_per_expert_bit_length,  # autotune key only
    # Strides
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
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_EXPERTS_POW2: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
    PACKED_SCHEDULE: tl.constexpr = False,
    B_MEMORY_MODE: tl.constexpr = "pointer",
    A_MEMORY_MODE: tl.constexpr = "pointer",
    PIPELINE: tl.constexpr = "k_loop",
    SWAP_AB: tl.constexpr = False,
    GATE: tl.constexpr = False,
    GATE_SCALES: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
):
    """Persistent FP8 W8A16 GEMM with optional fused gate/up activation."""
    start_pid = tl.program_id(axis=0)
    if PACKED_SCHEDULE:
        total_m_tiles = load_packed_schedule(Schedule, BLOCK_SIZE_M)
        if PIPELINE == "k_loop":
            prefetched = prefetch_packed_entry(
                Schedule, start_pid // tl.cdiv(N, BLOCK_SIZE_N), total_m_tiles
            )
    else:
        exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
            ExpertStart, NUM_EXPERTS_POW2, BLOCK_SIZE_M
        )
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    for tile_id in tl.range(
        start_pid, total_m_tiles * num_n_tiles, NUM_SMS,
        flatten=PIPELINE == "flattened", warp_specialize=WARP_SPEC and PIPELINE != "k_loop",
    ):
        if PACKED_SCHEDULE and PIPELINE != "k_loop":
            prefetched = prefetch_packed_entry(Schedule, tile_id // num_n_tiles, total_m_tiles)
        if PACKED_SCHEDULE:
            pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn, row0, n_off, m_start = (
                resolve_grouped_tile_packed(
                    tile_id, num_n_tiles, prefetched,
                    GatherIdx, ScatterIdx, BLOCK_SIZE_N, BLOCK_SIZE_M, GATE,
                )
            )
        else:
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
        if PACKED_SCHEDULE and PIPELINE == "k_loop":
            prefetched = prefetch_packed_entry(
                Schedule, (tile_id + NUM_SMS) // num_n_tiles, total_m_tiles
            )
        a_ptrs = operand_tile_ptrs(A, in_row, offs_k, stride_a_m, stride_a_k, A_MEMORY_MODE, True)
        b_ptrs = weight_tile_ptrs(
            B + expert_id64 * stride_b_e,
            offs_bn,
            offs_k,
            stride_b_n,
            stride_b_k,
            SWAP_AB,
        )

        acc = acc_init("dot", BLOCK_SIZE_M, (2 if GATE else 1) * BLOCK_SIZE_N, False)
        # Device-built weight descriptors address a 2D slab for this expert.
        a_desc = operand_tile_descriptor(
            ADescriptor, A, S, K, stride_a_m, 1,  # K is the contiguous dim of (S, K)
            BLOCK_SIZE_M, BLOCK_SIZE_K, A_MEMORY_MODE,
        )
        b_desc = operand_tile_descriptor(
            BDescriptor, B + expert_id64 * stride_b_e, N, K, stride_b_n, 1,  # and of the slab
            BLOCK_SIZE_N, BLOCK_SIZE_K, B_MEMORY_MODE, GATE,
        )
        B_DESCRIPTOR_IS_2D: tl.constexpr = B_MEMORY_MODE == "device_descriptor"
        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC and PIPELINE == "k_loop"):
            a, _as = load_act_plain(
                a_ptrs, a_desc, m_start, k * BLOCK_SIZE_K, row_mask, in_row,
                A_MEMORY_MODE, GatherIdx is not None,
            )
            w = load_weight_block_fp8(
                b_ptrs, b_desc, Bs, row0, n_off, k * BLOCK_SIZE_K,
                expert_id64, offs_bn, k * BLOCK_SIZE_K + offs_k,
                stride_bs_e, stride_bs_n, stride_bs_k,
                GATE, True, B_MEMORY_MODE, SWAP_AB, BLOCK_SIZE_N, BLOCK_SIZE_K,
                BLOCK_N, BLOCK_K, GATE_SCALES, a.dtype,
                B_DESCRIPTOR_IS_2D=B_DESCRIPTOR_IS_2D,
            )
            acc = plain_dot(acc, a, w, SWAP_AB)
            if A_MEMORY_MODE == "pointer":
                a_ptrs += BLOCK_SIZE_K * stride_a_k
            if B_MEMORY_MODE == "pointer":
                b_ptrs += BLOCK_SIZE_K * stride_b_k

        gemm_epilogue(
            C,
            None,  # no requant arm here
            acc,
            out_row,
            pid_n,
            tile_id // num_n_tiles,
            row_mask,
            stride_c_m,
            stride_c_n,
            1,  # Cs strides: unread, Cs is None
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
            Bias=Bias,
            stride_bias_e=stride_bias_e,
            stride_bias_n=stride_bias_n,
            global_row=expert_id64,
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
    block_size: List[int],
    gate: bool = False,
    act_fn: str | None = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    activation_format: str | None = None,
    quantize_output: bool = False,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> List[torch.Tensor]:
    """Block-scale grouped FP8 matmul over expert-sorted positions (per-tile
    gather/scatter, the sort is virtual — see ``compute_grouped_scheduling`` for the maps).
    Raw activations are quantized here (``fp8_act_quant_block_dynamic``, one offline pass);
    pre-quantized ``A`` + ``As`` (e.g. a requantized intermediate) is read as-is. The ``gate``/``act_fn``/``swiglu_*``/``requant``/``simulate_unfused``
    flags are the dispatcher's gate|up fusion kwargs (torch custom ops take only primitive params — the
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
    assert activation_format in (None, "fp8"), (
        f"block-dynamic activations are E4M3 ('fp8'), got {activation_format!r}"
    )
    output_format = "fp8" if quantize_output else None
    requant = output_format is not None
    # the requantized intermediate's scale groups follow gate_up's block_n, and the
    # down consumes per-block_k — a non-square block format would misalign them
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
        # post-quant: trade the in-kernel gather for one packed-row copy where that wins
        A, As, gather_idx = expand_gather_below_parity(A, As, gather_idx, num_experts, expert_start)
    if requant:
        C = A.new_empty(S, N, dtype=FP8_DTYPE)
        # UE8M0 model (ue8m0 weights) -> UE8M0 intermediate scales so the down proj reads
        # power-of-two activation scales and takes its dot_scaled arm (the epilogue infers
        # the format from this dtype); fp32 weights keep fp32 (software) as before.
        cs_dtype = bs_u8.dtype
        Cs = torch.empty(S, N // block_n, device=A.device, dtype=cs_dtype)
    else:
        C = A.new_empty(S, gate_out_cols(N, gate, act_fn), dtype=output_dtype)
        Cs = None  # unread without an OUTPUT_FORMAT; strides literal below
    num_sms = persistent_program_count(A.device.index)
    a_descriptor, b_descriptor = build_grouped_operand_descriptors(
        A, B.view(num_experts, 2 * N if gate else N, K)
    )

    schedule = packed_schedule_buffer(S, num_experts, expert_start.device)
    with device_context(A.device):
        bias_stride_e, bias_stride_n = bias_strides(bias)
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
            bias,
            gather_idx,  # None = A is expert-sorted; read only when not None (folds at trace time)
            scatter_idx,  # None = C is expert-sorted; read only when not None (folds at trace time)
            expert_start,
            schedule,
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
            Cs.stride(0) if requant else 1,  # literal when Cs is None
            Cs.stride(1) if requant else 1,
            bias_stride_e,
            bias_stride_n,
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
            OUTPUT_FORMAT=output_format,
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
    block_size: List[int],
    gate: bool = False,
    act_fn: str | None = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    activation_format: str | None = None,
    quantize_output: bool = False,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> List[torch.Tensor]:
    """Block-scale grouped FP8 matmul with a static (per-tensor calibrated) activation scale —
    the block-dynamic sibling with the 2D ``block_static`` format. ``A`` is raw here: the op
    quantizes it against the scalar ``As`` (offline, ``(A / As).to(fp8)``), the kernel applies the
    per-block weight scales in the K-loop and the scalar once post-loop. Returns the ``[C]`` GLU
    intermediate, or — under ``quantize_output`` — the FP8-requantized ``[C, Cs]`` (the per-row
    output scale is independent of the per-tensor input scale).

    A:  (S, K) raw bf16/fp16 activations — rows addressed via ``gather_idx``
    B:  (num_experts, N, K) FP8 weights; under ``gate`` the (num_experts, 2N, K) gate|up stack
    As: scalar / (1,) for one calibrated scale, or (num_experts,) for a MoE that
        calibrates each expert separately
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
    assert activation_format in (None, "fp8"), (
        f"block-static activations are E4M3 ('fp8'), got {activation_format!r}"
    )
    output_format = "fp8" if quantize_output else None
    requant = output_format is not None
    assert not requant or block_n == block_k, (
        f"the fused 'fp8' requant needs square quant blocks, got {block_size}"
    )

    output_dtype = resolve_output_dtype(output_dtype, A, None)
    A_q, As, as_stride = static_expert_act_operands(A, As, num_experts)
    bs_u8 = ue8m0_as_uint8(Bs)
    if requant:
        C = A.new_empty(S, N, dtype=FP8_DTYPE)
        # UE8M0 model (ue8m0 weights) -> UE8M0 intermediate scales; the epilogue infers the format
        # from this dtype (fp32 weights keep fp32), matching the block-dynamic requant.
        Cs = torch.empty(S, N // block_n, device=A.device, dtype=bs_u8.dtype)
    else:
        C = A.new_empty(S, gate_out_cols(N, gate, act_fn), dtype=output_dtype)
        Cs = None  # unread without an OUTPUT_FORMAT; strides literal below
    num_sms = persistent_program_count(A.device.index)
    a_descriptor, b_descriptor = build_grouped_operand_descriptors(
        A_q, B.view(num_experts, 2 * N if gate else N, K)
    )

    schedule = packed_schedule_buffer(S, num_experts, expert_start.device)
    with device_context(A.device):
        bias_stride_e, bias_stride_n = bias_strides(bias)
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
            bias,
            gather_idx,  # None = A is expert-sorted; read only when not None (folds at trace time)
            scatter_idx,  # None = C is expert-sorted; read only when not None (folds at trace time)
            expert_start,
            schedule,
            S,
            N,
            K,
            A_q.stride(0),
            A_q.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            as_stride,
            bs_u8.stride(0),
            bs_u8.stride(2),
            bs_u8.stride(1),
            C.stride(0),
            C.stride(1),
            Cs.stride(0) if requant else 1,  # literal when Cs is None
            Cs.stride(1) if requant else 1,
            bias_stride_e,
            bias_stride_n,
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
            OUTPUT_FORMAT=output_format,
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
    act_fn: str | None = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    activation_format: str | None = None,
    quantize_output: bool = False,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    activation_scale_ub: float | None = None,
) -> List[torch.Tensor]:
    """Tensor-scale / row-wise grouped FP8 matmul over expert-sorted positions (per-tile
    gather/scatter, the sort is virtual — see ``compute_grouped_scheduling`` for the maps).
    Raw activations are quantized here (``fp8_act_quant_tensor_wide``, one offline pass);
    pre-quantized ``A`` + ``As`` is read as-is.
    ``gate``/``act_fn``/``swiglu_*``/``simulate_unfused`` are the dispatcher's gate|up fusion kwargs (GLU only;
    ``requant`` is unsupported here — the dispatcher unpacks the bundle).

    A:  (S, K) pre-quantized FP8 activations — rows addressed via ``gather_idx``
    B:  (num_experts, N, K) FP8 expert weights; under ``gate`` the (num_experts, 2N, K) stack
    As: (S,) per-token scales alongside a pre-quantized A, or — on a raw A — the calibrated
        (static) activation scale: one value, or one per expert
    Bs: (num_experts,) or (num_experts, 1, 1) per-expert weight scales, or (num_experts, N, 1)
        row-wise (2N rows under ``gate``)
    expert_start: (num_experts_pow2 + 1,) int32 — cumulative sorted-row starts, S sentinel
    gather_idx: optional (S,) — sorted position -> source row of A; None = A is expert-sorted
    scatter_idx: optional (S,) — sorted position -> destination row of C; None = C stays expert-sorted
    """
    validate_dense_operands(A, B)

    _, K = A.shape

    # Under a gate epilogue B is the (E, 2N, K) gate|up stack — N is the per-projection width.
    assert activation_format in (None, "fp8"), (
        f"tensor-wide activations are E4M3 ('fp8'), got {activation_format!r}"
    )
    assert not quantize_output, (
        "requant is unsupported for tensor-wide gate_up (its down needs a per-token whole-row "
        "scale a per-tile epilogue can't form); use a plain gate epilogue + external quant"
    )
    num_experts, _, N = expert_weight_shape(B, gate)
    S = routed_rows(A, gather_idx, scatter_idx, expert_start, num_experts)

    Bs = normalize_per_expert_scale(Bs, num_experts)

    output_dtype = resolve_output_dtype(output_dtype, A, As)
    A, As, as_stride_m, as_stride_e = tensor_wide_act_operands(
        A, As, num_experts, activation_scale_ub=activation_scale_ub
    )
    if as_stride_e and expand_regime(S, num_experts):
        # a per-expert calibrated scale: lay the routed rows out quantized rather than read
        # them raw through every N-tile — the same copy-amortizes-at-prefill law below
        A = quantize_routed_rows_per_expert(A, As, gather_idx, expert_start, num_experts)
        gather_idx = None  # spent: the rows are laid out now
    if A.dtype == FP8_DTYPE:
        # post-quant: trade the in-kernel gather for one packed-row copy where that wins. Keyed
        # on A being quantized, which the dynamic quant, a shared calibrated scale and the
        # per-expert layout above all reach; raw rows (decode) keep the gather. Per-token scales
        # (stride_m 1) follow their rows into the copy.
        row_scales = As.view(-1, 1) if as_stride_m else None
        A, row_scales, gather_idx = expand_gather_below_parity(A, row_scales, gather_idx, num_experts, expert_start)
        if as_stride_m:
            As = row_scales.view(-1)
            as_stride_m = As.stride(0)
    C = A.new_empty(S, gate_out_cols(N, gate, act_fn), dtype=output_dtype)
    a_descriptor, b_descriptor = build_grouped_operand_descriptors(
        A, B.view(num_experts, 2 * N if gate else N, K)
    )

    schedule = packed_schedule_buffer(S, num_experts, expert_start.device)
    with device_context(A.device):
        bias_stride_e, bias_stride_n = bias_strides(bias)
        compile_time_only_triton_wrap(w8a8_tensor_dynamic_fp8_matmul_grouped_kernel)[
            lambda meta: (persistent_program_count(A.device.index, meta["BLOCK_SIZE_M"]),)
        ](
            A,
            a_descriptor,
            As,
            B,
            b_descriptor,
            Bs,
            C,
            None,
            bias,  # tensor-wide has no fused requant
            gather_idx,  # None = A is expert-sorted; read only when not None (folds at trace time)
            scatter_idx,  # None = C is expert-sorted; read only when not None (folds at trace time)
            expert_start,
            schedule,
            S,
            N,
            K,
            A.stride(0),
            A.stride(1),
            as_stride_m,
            as_stride_e,
            B.stride(0),
            B.stride(2),
            B.stride(1),
            Bs.stride(0),
            row_scale_stride(Bs),
            C.stride(0),
            C.stride(1),
            1,  # Cs strides: unread, Cs is None
            1,
            bias_stride_e,
            bias_stride_n,
            num_experts=num_experts,
            tokens_per_expert_bit_length=tokens_per_expert_bucket(S, num_experts),
            NUM_EXPERTS_POW2=triton.next_power_of_2(num_experts),
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
    act_fn: str | None = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    activation_format: str | None = None,
    quantize_output: bool = False,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    a_global_scale: torch.Tensor | None = None,
    b_global_scale: torch.Tensor | None = None,
    output_global_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    a_pre_quant_scale: torch.Tensor | None = None,
) -> List[torch.Tensor]:
    """Grouped MX matmul over expert-sorted positions (per-tile gather/scatter, the
    sort is virtual — see ``compute_grouped_scheduling`` for the maps). Activations arrive
    pre-quantized: the caller owns the act-quant (``mxfp8_act_quant``). The
    ``gate``/``act_fn``/``swiglu_*``/``requant``/``simulate_unfused`` flags are the dispatcher's
    gate|up fusion kwargs.
    Weight format detected from ``B.dtype``: ``int8`` →
    packed E2M1 (MXFP4, ``B`` is ``(num_experts, N, K//2)``); ``float8_e4m3fn`` → unpacked E4M3
    (MXFP8, ``(num_experts, N, K)``). UE8M0 group-32 scales ``(num_experts, N, K//32)``; tile + dot autotuned.

    A:  (S, K) pre-quantized E4M3 activations — rows addressed via ``gather_idx``
    As: (S, K // 32) UE8M0 group-32 activation scales
    expert_start: (num_experts_pow2 + 1,) int32 — cumulative sorted-row starts, S sentinel
    gather_idx: optional (S,) — sorted position -> source row of A; None = A is expert-sorted
    scatter_idx: optional (S,) — sorted position -> destination row of C; None = C stays expert-sorted
    """
    assert A.ndim == 2 and B.ndim == 3 and Bs.ndim in (3, 5)  # 5D = pre-swizzled
    if a_pre_quant_scale is not None and activation_format != "fp8":
        assert As is None, "pre_quant_scale applies before activation quantization; pass raw inputs"
        num_rows = routed_rows(A, gather_idx, scatter_idx, expert_start, B.shape[0])
        A = permute_routed_rows(
            A, None, num_rows, gather_idx, expert_start=expert_start,
            pre_quant_scale=a_pre_quant_scale, num_experts=B.shape[0],
        )
        gather_idx = None
    if activation_format == "fp8":
        assert not quantize_output and a_global_scale is None, "ordinary FP8 inputs use As=input_scale and no fused requant"
        output_dtype = resolve_output_dtype(output_dtype, A, None)
        group = mx_scale_family(Bs, A.shape[-1])
        num_rows = routed_rows(A, gather_idx, scatter_idx, expert_start, B.shape[0])
        weight_upcast = Bs.ndim == 3 and Bs.dtype == torch.float8_e4m3fn and group == MX_SCALE_GROUP_K
        A, As, a_global_scale = prepare_fp8_mx_inputs(
            A, As, ue8m0_as_uint8(Bs).dtype, group, B.shape[0], num_rows,
            gather_idx, expert_start=expert_start, pre_quant_scale=a_pre_quant_scale, storage_dtype=torch.bfloat16 if weight_upcast else FP8_DTYPE,
        )
        gather_idx = None
        if weight_upcast:
            return mx_weight_only_matmul_grouped(
                A, B, Bs, expert_start, output_dtype=output_dtype, gate=gate, act_fn=act_fn,
                swiglu_alpha=swiglu_alpha, swiglu_limit=swiglu_limit, simulate_unfused=simulate_unfused,
                scatter_idx=scatter_idx,
                b_global_scale=b_global_scale, a_global_scale=a_global_scale,
                bias=bias,
            )
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
    # checkpoint shared with batched decode (is_preswizzled_mx). The format is the scale dtype
    # (E4M3 = NVFP4 group-16, UE8M0 = MX group-32).
    swizzled_scales = Bs.ndim == 5
    scale_group = mx_scale_family(Bs, K)
    if not swizzled_scales:
        assert Bs.shape == (num_experts, n_rows, K // scale_group), (
            f"Bs shape {tuple(Bs.shape)} != ({num_experts}, {n_rows}, {K // scale_group})"
        )
    else:
        # the artifact's expert count and per-expert row blocks must match THIS weight stack — a
        # wrong-layer artifact with a matching K would otherwise dequantize with garbage scales silently
        assert Bs.shape[:2] == (num_experts, triton.cdiv(n_rows, 128)), (
            f"swizzled Bs carries {tuple(Bs.shape[:2])} (experts, 128-row blocks), expected "
            f"{(num_experts, triton.cdiv(n_rows, 128))} for ({n_rows}, K) weights — wrong artifact"
        )

    output_dtype = resolve_output_dtype(output_dtype, A, As)
    activation_format = resolve_activation_format(activation_format, B, Bs)
    output_format = activation_format if quantize_output else None
    requant = quantize_output
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
    # g_a normalizes the act quant here in the wrapper (the raw-A arms below, or offline for a
    # pre-quantized As) and rides down as AsGlobal for the accumulator to multiply back — grouped A
    # is always quantized before the GEMM, so no in-kernel inline quant reads it.
    act_global_scale = normalize_global_scale(a_global_scale, num_experts)
    # A per-expert g_a quantizes each row against ITS expert's global. A gathered A holds one row
    # per token, routed to top_k experts at once, so it quantizes per routed row, into expert-sorted
    # rows the GEMM then reads without a gather.
    quantize_per_routed_row = is_per_expert_global(a_global_scale) and gather_idx is not None
    if a_global_scale is not None:
        assert activation_format in ("nvfp4", "fp8"), "an activation global requires NVFP4 or calibrated tensor-FP8"
        assert not quantize_per_routed_row or As is None, (
            "a per-expert a_global_scale under a gather needs a raw A"
        )
    if quantize_per_routed_row:
        a_vals, act_scales, n_m_tiles = mx_act_quant_routed_rows(
            A, activation_format, scale_group, scale_dtype, gather_idx, expert_start, act_global_scale,
            swizzled=swizzled_scales,
        )
        gather_idx = None
    elif swizzled_scales:
        if As is None and gather_idx is not None:
            # Quantize ONCE at (num_tokens, K) and let the kernel gather the packed rows:
            # the fused sorted-quant below re-reads and re-quantizes each row per routed
            # copy (top_k times). A/B at GLM nvfp4 / dsv4 mxfp8 prefill shapes: 1697 ->
            # 1531us / 1180 -> 1159us on the gate_up GEMM, bit-identical — the (T, K)
            # packed buffer is L2-sized, so the descriptor gather4 reads hit cache while
            # any materialized (S, K) layout streams DRAM (a fused quant+scatter form
            # measured SLOWER everywhere — see OPTIMIZATION_LOG: fused-quant-scatter).
            # The scales take the same gather+swizzle pass a caller-provided row-major
            # As takes.
            A, As = (
                MX_ACT_QUANT[activation_format](A, global_scale=act_global_scale)
                if a_global_scale is not None
                else MX_ACT_QUANT[activation_format](A)
            )
        if As is None:
            a_vals, act_scales, n_m_tiles = mx_act_quant_grouped(
                A, activation_format, scale_group, scale_dtype, gather_idx, expert_start,
                act_global_scale,
            )
        elif As.ndim == 5:  # pre-swizzled (requant epilogue, a forward's routed quant) — read as is
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
        if As is None and is_per_expert_global(act_global_scale):
            # each row divides by ITS expert's global, so quantize on the expert-sorted grid,
            # whose tiles carry an expert — the dense grid would need that map as a tensor
            a_vals, act_scales, _ = mx_act_quant_grouped(
                A, activation_format, scale_group, scale_dtype, gather_idx, expert_start,
                act_global_scale, swizzled=False,
            )
        elif As is None:
            a_vals, act_scales = (
                MX_ACT_QUANT[activation_format](A, global_scale=act_global_scale)
                if a_global_scale is not None
                else MX_ACT_QUANT[activation_format](A)
            )
        else:
            a_vals, act_scales = A, As
        # post-quant: trade the in-kernel gather for one packed-row copy where that wins.
        # Unlike the swizzled arm, the affine gather LOSES to the copy (2109 vs 1993us at
        # GLM shape): the affine scale gather forfeits WS and the descriptor-A configs, so
        # materialized rows keep the faster GEMM.
        a_vals, act_scales, gather_idx = expand_gather_below_parity(
            a_vals, act_scales, gather_idx, num_experts, expert_start
        )
        n_rows = gather_idx.numel() if gather_idx is not None else a_vals.shape[0]
        n_m_tiles = n_rows // 128 + num_experts
    # uint8 aliases for the binder — never clobber the caller's A/B/Bs dtype/view. Each operand's
    # scale is one pointer (``*_u8``) read affine in-kernel, plus a descriptor built only on the
    # swizzled path (the affine arm reads the pointer and never touches the descriptor — None).
    # SWIZZLED_SCALES governs both operands: a swizzled weight is paired with swizzled acts.
    a_u8 = e2m1_as_uint8(a_vals)
    b_u8 = e2m1_as_uint8(B)
    as_u8 = ue8m0_as_uint8(act_scales)
    bs_u8 = ue8m0_as_uint8(Bs)
    as_descriptor = swizzled_scale_descriptor(as_u8) if swizzled_scales else None
    bs_descriptor = swizzled_scale_descriptor(bs_u8) if swizzled_scales else None
    # mxfp8 requant writes Cs straight into the down proj's SWIZZLE_32_4_4 layout (the epilogue
    # gets a descriptor, not a pointer) — the down reads it affine, no post-requant swizzle pass.
    # Only when the output stays expert-sorted (scatter_idx None, the fused gate_up convention):
    # the swizzle needs contiguous 128-row blocks, which a scattered output can't provide.
    # Swizzled in -> swizzled out: when the block runs swizzled (swizzled_scales), the requant
    # emits Cs straight into the down proj's SWIZZLE_32_4_4 layout (a TMA descriptor), so the fused
    # down reads it on the fast path (the As.ndim == 5 arm above). Format-general — the swizzle is a
    # byte-tiling over the group-scale grid (N // scale_group), so every MX family qualifies (UE8M0
    # group-32, E4M3 group-16 NVFP4); only the column count (cb_cs) and scale byte dtype differ. The
    # swizzle needs contiguous 128-row blocks, i.e. an expert-sorted output (scatter_idx None — the
    # fused gate_up convention); a scattered requant keeps row-major Cs.
    swizzled_out = requant and scatter_idx is None and swizzled_scales
    if output_format in ("mxfp4", "nvfp4"):  # packed E2M1 intermediate, feeds a W4A4 down as-is
        assert N % (2 * scale_group) == 0, (
            f"N (={N}) must be a multiple of {2 * scale_group} to pack E2M1 pairs"
        )
        C = A.new_empty((S, N // 2), dtype=torch.int8)
    elif requant:
        C = A.new_empty((S, N), dtype=FP8_DTYPE)
    else:
        C = A.new_empty((S, gate_out_cols(N, gate, act_fn)), dtype=output_dtype)
    if swizzled_out:
        cb_cs = triton.cdiv(N // scale_group, 4)
        cs_ret = torch.empty(1, n_m_tiles, cb_cs, 2, 256, device=A.device, dtype=scale_dtype)
        CSDescriptor = TensorDescriptor.from_tensor(cs_ret, [1, 1, 1, 2, 256])
        Cs = None  # row-major pointer unread here; CSDescriptor does the store
    elif requant:  # un-swizzled block (or scattered output) -> row-major Cs pointer
        cs_ret = torch.empty(S, N // scale_group, device=A.device, dtype=scale_dtype)
        Cs, CSDescriptor = cs_ret, None
    else:
        cs_ret, Cs, CSDescriptor = None, None, None  # unread (no OUTPUT_FORMAT)
    num_sms = persistent_program_count(A.device.index)
    # NVFP4 accumulator correction: the per-expert g_a·g_b product folded onto the fp32 accumulator
    # (grouped A is pre-quantized, so the kernel needs only this product, never g_a alone).
    # g_b per expert and g_a scalar go down SEPARATELY; the kernel multiplies them in-register
    # (the old host-side g_a*g_b product was a torch launch per GEMM per call)
    input_global_scale = normalize_global_scale(b_global_scale, B.shape[0])
    out_global_scale = normalize_global_scale(output_global_scale, num_experts)
    # host TMA descriptor over the (E, 2N|N, K_bytes) view — a gate tile is one contiguous
    # 2*BN row span; the placeholder box is re-bound per tuned config by the pre_hook
    a_descriptor, b_descriptor = build_grouped_operand_descriptors(
        a_u8, b_u8.view(num_experts, 2 * N if gate else N, b_u8.shape[2])
    )
    # (the SA/SB swizzled-scale descriptors and their placeholder boxes — re-bound per tuned
    # config by rebind_grouped_mx_descriptors — are built above with the scales.)

    schedule = packed_schedule_buffer(S, num_experts, expert_start.device)
    with device_context(A.device):
        bias_stride_e, bias_stride_n = bias_strides(bias)
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
            bias,
            CSDescriptor,
            act_global_scale,  # AsGlobal = g_a (acc, in-register)
            input_global_scale,  # AsBsGlobal = g_b per expert (acc)
            out_global_scale,  # CsGlobal: requant output normalization (next proj's provided input_scale); None folds out
            gather_idx,  # None = A is expert-sorted; read only when not None (folds at trace time)
            scatter_idx,  # None = C is expert-sorted; read only when not None (folds at trace time)
            expert_start,
            schedule,
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
            bias_stride_e,
            bias_stride_n,
            global_scale_stride(act_global_scale),
            global_scale_stride(input_global_scale),
            global_scale_stride(out_global_scale),
            num_experts=num_experts,
            tokens_per_expert_bit_length=tokens_per_expert_bucket(S, num_experts),
            NUM_EXPERTS_POW2=triton.next_power_of_2(num_experts),
            NUM_SMS=num_sms,
            SCALE_GROUP_K=scale_group,
            GATE=gate,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            OUTPUT_FORMAT=output_format,
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
    act_fn: str | None = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    activation_format: str | None = None,
    quantize_output: bool = False,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> List[torch.Tensor]:
    """Full-precision (BF16/FP16) grouped matmul over expert-sorted positions (per-tile
    gather/scatter, the sort is virtual — see ``compute_grouped_scheduling`` for the maps).
    No quantization anywhere: activations and weights share one high-precision dtype and the
    dot accumulates in fp32. ``gate``/``act_fn``/``swiglu_*``/``simulate_unfused`` are the
    dispatcher's gate|up fusion kwargs (GLU only; ``requant`` is meaningless without a quantized format).

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
    assert activation_format in (None, "bf16") and not quantize_output, (
        "the full-precision path quantizes nothing — no quantized activation format or requant applies"
    )

    _, K = A.shape

    num_experts, _, N = expert_weight_shape(B, gate)
    S = routed_rows(A, gather_idx, scatter_idx, expert_start, num_experts)

    output_dtype = resolve_output_dtype(output_dtype, A, None)
    C = A.new_empty(S, gate_out_cols(N, gate, act_fn), dtype=output_dtype)
    a_descriptor, b_descriptor = build_grouped_operand_descriptors(
        A, B.view(num_experts, 2 * N if gate else N, K)
    )
    schedule = packed_schedule_buffer(S, num_experts, expert_start.device)

    with device_context(A.device):
        bias_stride_e, bias_stride_n = bias_strides(bias)
        compile_time_only_triton_wrap(full_precision_matmul_grouped_kernel)[
            lambda meta: (persistent_program_count(A.device.index, meta["BLOCK_SIZE_M"]),)
        ](
            A,
            a_descriptor,
            B,
            b_descriptor,
            C,
            bias,
            gather_idx,  # None = A is expert-sorted; read only when not None (folds at trace time)
            scatter_idx,  # None = C is expert-sorted; read only when not None (folds at trace time)
            expert_start,
            schedule,
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
            num_experts=num_experts,
            tokens_per_expert_bit_length=tokens_per_expert_bucket(S, num_experts),
            NUM_EXPERTS_POW2=triton.next_power_of_2(num_experts),
            GATE=gate,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(output_dtype),
        )

    return [C]


@compile_time_only_triton_op(
    add_op_namespace_prefix("block_fp8_weight_only_matmul_grouped"),
    mutates_args=(),
    opaque=True,
)
def block_fp8_weight_only_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_start: torch.Tensor,
    gate: bool = False,
    act_fn: str | None = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    activation_format: str | None = None,
    quantize_output: bool = False,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> List[torch.Tensor]:
    """FP8 W8A16 grouped GEMM; stage weights only when routed reuse amortizes conversion."""
    validate_dense_operands(A, B)
    assert A.dtype in (torch.bfloat16, torch.float16), "FP8 weight-only takes raw BF16/FP16 activations"
    assert activation_format in (None, "bf16") and not quantize_output, (
        "FP8 weight-only requires unquantized activations and output"
    )

    _, K = A.shape

    num_experts, _, N = expert_weight_shape(B, gate)
    S = routed_rows(A, gather_idx, scatter_idx, expert_start, num_experts)
    if S >= FP8_WEIGHT_ONLY_DEQUANT_MIN_M * num_experts:
        weights = dequant_block_fp8_weight(
            B, Bs, A.dtype, gate=gate, expert_start=expert_start,
        )
        return full_precision_matmul_grouped(
            A, weights, expert_start, gate, act_fn, swiglu_alpha, swiglu_limit,
            simulate_unfused, activation_format, quantize_output, output_dtype,
            gather_idx, scatter_idx, bias=bias,
        )
    Bs, scale_block_n, scale_block_k = block_fp8_weight_scale_grid(B, Bs, gate)
    Bs = ue8m0_as_uint8(Bs)

    output_dtype = resolve_output_dtype(output_dtype, A, None)
    C = A.new_empty(S, gate_out_cols(N, gate, act_fn), dtype=output_dtype)
    num_sms = persistent_program_count(A.device.index)
    a_descriptor, b_descriptor = build_grouped_operand_descriptors(
        A, B.view(num_experts, 2 * N if gate else N, K)
    )
    schedule = packed_schedule_buffer(S, num_experts, A.device)

    with device_context(A.device):
        bias_stride_e, bias_stride_n = bias_strides(bias)
        compile_time_only_triton_wrap(block_fp8_weight_only_matmul_grouped_kernel)[(num_sms,)](
            A,
            a_descriptor,
            B,
            b_descriptor,
            Bs,
            C,
            bias,
            gather_idx,  # None = A is expert-sorted; read only when not None (folds at trace time)
            scatter_idx,  # None = C is expert-sorted; read only when not None (folds at trace time)
            expert_start,
            schedule,
            S,
            N,
            K,
            num_experts,
            tokens_per_expert_bucket(S, num_experts),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            Bs.stride(0),
            Bs.stride(1),
            Bs.stride(2),
            C.stride(0),
            C.stride(1),
            bias_stride_e,
            bias_stride_n,
            NUM_EXPERTS_POW2=triton.next_power_of_2(num_experts),
            NUM_SMS=num_sms,
            BLOCK_N=scale_block_n,
            BLOCK_K=scale_block_k,
            GATE=gate,
            GATE_SCALES=gate and Bs.shape[1] > 1,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(output_dtype),
        )

    return [C]


@compile_time_only_triton_op(
    add_op_namespace_prefix("mx_weight_only_matmul_grouped"),
    mutates_args=(),
    opaque=True,
)
def mx_weight_only_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_start: torch.Tensor,
    gate: bool = False,
    act_fn: str | None = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    b_global_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    a_global_scale: torch.Tensor | None = None,
) -> List[torch.Tensor]:
    """weight-only grouped matmul: raw bf16/fp16 activations against MXFP4/NVFP4/MXFP8 weights upcast
    to bf16 in-loop — the ``matmul_ogs`` format (fp4/fp8 weight bytes, bf16 acts, weight upcast in-MMA).
    Affine (3D) weight scales — UE8M0 group-32 (MX) or E4M3 group-16 (NVFP4, whose per-expert fp32
    ``b_global_scale`` ``(E,)`` recovers on the accumulator; ``None`` = single-level). Same
    gather/scatter expert scheduling as the other grouped ops. Returns ``[C]`` (no requant — the
    bf16 intermediate needs none)."""
    assert A.dtype in (torch.bfloat16, torch.float16, torch.float32), (
        f"weight-only takes a raw bf16/fp16/fp32 activation, got {A.dtype}"
    )
    assert Bs.ndim == 3, f"weight-only grouped takes affine (3D) weight scales, got ndim={Bs.ndim}"
    _, K = A.shape
    WEIGHT_VALUES_PER_BYTE = 2 if B.dtype == torch.int8 else 1
    num_experts, rows, K_b = B.shape
    assert K == WEIGHT_VALUES_PER_BYTE * K_b, (
        f"K ({K}) must equal {WEIGHT_VALUES_PER_BYTE} * B.shape[2] ({K_b})"
    )
    N = rows // 2 if gate else rows
    scale_group = mx_scale_family(Bs, K)
    assert Bs.dtype != torch.float8_e4m3fn or scale_group == NVFP4_SCALE_GROUP_K or a_global_scale is not None, (
        "group-32 E4M3 FP4 weights require calibrated FP8 activations"
    )
    b_global_scale = normalize_global_scale(b_global_scale, num_experts)
    a_global_scale = normalize_global_scale(a_global_scale, num_experts)
    S = routed_rows(A, gather_idx, scatter_idx, expert_start, num_experts)
    output_dtype = resolve_output_dtype(output_dtype, A, None)
    C = A.new_empty(S, gate_out_cols(N, gate, act_fn), dtype=output_dtype)
    b_u8 = e2m1_as_uint8(B)
    bs_u8 = ue8m0_as_uint8(Bs)
    # Operand host-TMA descriptors (A over (S, K), B over the (E, 2N|N, K_bytes) weight view);
    # placeholder boxes rebound per config by rebind_grouped_descriptors, read only by descriptor
    # configs the tuner picks. Scales stay affine (3D pointer) — no scale descriptor.
    a_descriptor, b_descriptor = build_grouped_operand_descriptors(
        A, b_u8.view(num_experts, 2 * N if gate else N, K_b)
    )

    schedule = packed_schedule_buffer(S, num_experts, expert_start.device)
    with device_context(A.device):
        bias_stride_e, bias_stride_n = bias_strides(bias)
        compile_time_only_triton_wrap(mx_weight_only_matmul_grouped_kernel)[
            lambda meta: (persistent_program_count(A.device.index, meta["BLOCK_SIZE_M"]),)
        ](
            A,
            a_descriptor,
            a_global_scale,
            b_u8,
            b_descriptor,
            bs_u8,
            b_global_scale,
            C,
            bias,
            gather_idx,
            scatter_idx,
            expert_start,
            schedule,
            S,
            N,
            K,
            A.stride(0),
            A.stride(1),
            global_scale_stride(a_global_scale),
            b_u8.stride(0),
            b_u8.stride(2),
            b_u8.stride(1),
            bs_u8.stride(0),
            bs_u8.stride(1),
            bs_u8.stride(2),
            global_scale_stride(b_global_scale),
            C.stride(0),
            C.stride(1),
            bias_stride_e,
            bias_stride_n,
            num_experts=num_experts,
            tokens_per_expert_bit_length=tokens_per_expert_bucket(S, num_experts),
            NUM_EXPERTS_POW2=triton.next_power_of_2(num_experts),
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


@compile_time_only_triton_op(
    add_op_namespace_prefix("int_weight_only_matmul_grouped"),
    mutates_args=(),
    opaque=True,
)
def int_weight_only_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_start: torch.Tensor,
    gate: bool = False,
    act_fn: str | None = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    b_zero: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    b_perm: torch.Tensor | None = None,
    a_pre_quant_scale: torch.Tensor | None = None,
    a_scale: torch.Tensor | None = None,
    b_scale2: torch.Tensor | None = None,
) -> List[torch.Tensor]:
    """Integer weight-only grouped matmul: raw bf16/fp16 activations against 2/4/8-bit integer weights, ``B``
    ``(E, N, K // VPW)`` interleaved uint32 words, ``Bs`` ``(E, N, K // group)`` float group scales, ``b_zero`` the
    matching integer zero points (``None``: symmetric) and ``b_perm`` ``(E, K)`` int32 the act-order K orders
    (``None``: natural), read by one ``permute_routed_rows`` launch that lays the routed rows out contiguous and in
    their experts' orders ahead of the natural-order kernel (B200, Qwen3-30B-A3B gate|up, T=1-512: 1.1-1.6x faster
    than an in-kernel column gather). Same gather/scatter expert scheduling as the other grouped ops. Returns
    ``[C]``. The weight is dequantized to the activation dtype (``((q - zero) * scale)`` rounded in the scale dtype, then cast), as the torch
    "dequantize, then matmul" reference does, so results match it up to fp32 summation order."""
    bits = validate_int_operands(A, B, Bs, b_zero, b_perm, a_scale, b_scale2)
    smoothquant = a_scale is not None and b_scale2 is None
    a_scale = normalize_global_scale(a_scale, B.shape[0])
    b_scale2 = normalize_global_scale(b_scale2, B.shape[0])
    _, K = A.shape
    num_experts, rows, K_b = B.shape
    N = rows // 2 if gate else rows
    S = routed_rows(A, gather_idx, scatter_idx, expert_start, num_experts)
    output_dtype = resolve_output_dtype(output_dtype, A, None)
    if b_perm is not None or a_pre_quant_scale is not None or a_scale is not None:
        A = permute_routed_rows(
            A, b_perm, S, gather_idx, expert_start=expert_start,
            pre_quant_scale=a_pre_quant_scale, num_experts=num_experts, quant_scale=a_scale,
            quant_dtype=torch.int8 if smoothquant else FP8_DTYPE,
        )
        gather_idx = None
    C = A.new_empty(S, gate_out_cols(N, gate, act_fn), dtype=output_dtype)
    num_sms = persistent_program_count(A.device.index)
    a_descriptor, b_descriptor = build_grouped_operand_descriptors(A, B)
    schedule = packed_schedule_buffer(S, num_experts, expert_start.device)
    with device_context(A.device):
        bias_stride_e, bias_stride_n = bias_strides(bias)
        compile_time_only_triton_wrap(int_weight_only_matmul_grouped_kernel)[
            lambda meta: (num_sms * meta["PROGRAMS_PER_SM"],)
        ](
            A,
            a_descriptor,
            a_scale,
            B,
            b_descriptor,
            Bs,
            b_zero,
            b_scale2,
            C,
            bias,
            gather_idx,
            scatter_idx,
            expert_start,
            schedule,
            S,
            N,
            K,
            A.stride(0),
            A.stride(1),
            global_scale_stride(a_scale),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            Bs.stride(0),
            Bs.stride(1),
            Bs.stride(2),
            *(b_zero.stride() if b_zero is not None else (0, 0, 0)),
            global_scale_stride(b_scale2),
            C.stride(0),
            C.stride(1),
            bias_stride_e,
            bias_stride_n,
            num_experts=num_experts,
            tokens_per_expert_bit_length=tokens_per_expert_bucket(S, num_experts),
            NUM_EXPERTS_POW2=triton.next_power_of_2(num_experts),
            SCALE_GROUP_K=K // Bs.shape[2],
            INT_BITS=bits,
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
    bias: torch.Tensor | None = None,  # (E, N_out) per-expert output bias; (N_out,) for 2D
    activation_format: str | None = None,
    activation_scale_ub: float | None = None,
    gate: bool = False,
    act_fn: str | None = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    quantize_output: bool = False,
    simulate_unfused: bool = False,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    a_global_scale: torch.Tensor | None = None,
    b_global_scale: torch.Tensor | None = None,
    output_global_scale: torch.Tensor | None = None,
    b_zero: torch.Tensor | None = None,
    b_perm: torch.Tensor | None = None,
    a_pre_quant_scale: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Grouped matmul dispatcher (W8A8 FP8, W4A8/W4A4 FP4, or full-precision).
    ``expert_start`` is the ``(E+1,)`` tiling schedule from one ``compute_grouped_scheduling``
    pass, shared by every grouped GEMM of the layer (the expert sort is virtual — nothing is
    physically permuted).

    ``As`` marks ``A`` as already quantized (framework-precomputed scales, or a requantized
    intermediate handed to the down projection); a per-tensor scalar ``As`` is instead the static
    (calibrated) activation scale for block-scale FP8 weights — the op quantizes raw ``A`` against
    it; ``None`` = raw ``A``, quantized dynamically by the op in ``activation_format`` (``None`` =
    the weights' own format — fp8/E4M3, or the fp4 chains; ``"bf16"`` = weight-only; ``"mxfp4"`` =
    packed E2M1 activations). ``Bs`` ``None`` = unquantized BF16/FP16 weights.
    ``quantize_output`` requantizes the output into ``activation_format`` — the return is then
    ``(C, Cs)``. ``gate``/``act_fn``/``swiglu_alpha``/``swiglu_limit`` fuse the gate|up GLU into the
    epilogue; ``act_fn=None`` with ``gate`` skips it, returning the raw interleaved gate|up
    output for a host-side activation.
    ``As``/``Bs`` are each a bare block-scale tensor; the two-level NVFP4 second-level scales ride
    the separate ``a_global_scale``/``b_global_scale`` (fp32 per-tensor, weights per-expert ``(E,)``;
    from ``nvfp4_quantize_two_level``), and the op folds ``g_a · g_b`` onto the accumulator. The
    activation global ``g_a`` is CALIBRATED (the checkpoint's ``input_scale``): ``a_global_scale=g_a``
    with a raw ``A`` has the op quantize ``A / g_a`` per block, and rides a pre-quantized ``As`` the
    same way. Under an NVFP4 requant the fused requant normalizes the GLU intermediate by the
    PROVIDED ``output_global_scale`` (the next proj's calibrated ``input_scale``) before the block
    quant and returns ``[C, Cs]``; the down consumes it as ``As=Cs, a_global_scale=output_global_scale``.
    Row order is carried by the standalone maps: ``gather_idx`` gathers ``A`` (``None`` ->
    already expert-ordered), ``scatter_idx`` scatters the output. The fused MoE chain is one
    scheduling pass: gate_up with ``scatter_idx=None`` + ``gate=True`` + ``quantize_output=True``,
    then down with ``gather_idx=None`` and the
    intermediate's scales as ``As``. EP-sentinel routes sit past ``expert_start[-1]``, which
    no tile reaches.

    Routes by what the weight tensors themselves say (there is no ``block_size``
    parameter — the quantization block is derived from the scale shape,
    ``weight_block_size``):
    - ``Bs`` None → ``full_precision_matmul_grouped`` (plain dot, no scales anywhere).
    - MX weights — ``int8`` (packed E2M1) or ``float8_e4m3fn`` (E4M3) with UE8M0
      group-32 ``Bs`` → ``mx_dynamic_matmul_grouped``.
    - one scale per expert (``Bs`` ``(E,)``/``(E, 1, 1)``) or per output row (``(E, N, 1)``) →
      ``w8a8_tensor_dynamic_fp8_matmul_grouped``.
    - block scales (``Bs`` ``(E, N/bn, K/bk)``) → ``w8a8_block_dynamic_fp8_matmul_grouped``.

    ``activation_scale_ub`` caps the dynamic per-token amax for row-wise FP8;
    it requires raw activations and applies before conversion to E4M3.
    - uint32 integer weights → ``int_weight_only_matmul_grouped`` (weight-only), ``b_zero`` their zero points
      (``None``: symmetric), ``b_perm`` ``(E, K)`` their act-order K orders (``None``: natural).
    """
    _validate_quantization_args(
        A, As, B, Bs, a_global_scale, b_global_scale, output_global_scale,
        b_zero, b_perm, a_pre_quant_scale, activation_format, quantize_output, activation_scale_ub,
    )
    assert act_fn is not None or not quantize_output, "the raw gate|up output (act_fn=None) is not requantized"
    if gather_idx is not None and activation_format == "bf16" and As is None and a_pre_quant_scale is None:
        # weight-only: raw bf16 rows are what the kernel consumes, so the trade is decided
        # here; the QUANTIZED formats decide it in their family wrappers AFTER the act quant
        # (expanding packed rows + scales — see ``expand_gather_below_parity``). An AWQ input
        # transform combines the sorted copy and column scaling in its preparation pass below.
        A, _, gather_idx = expand_gather_below_parity(A, None, gather_idx, B.shape[0], expert_start)
    if B.dtype == FP8_DTYPE and Bs is not None and not is_mx(B, Bs) and activation_format == "bf16":
        assert As is None and not quantize_output, "weight-only takes raw activations without requantization"
        out = block_fp8_weight_only_matmul_grouped(
            A, B, Bs, expert_start, gate, act_fn, swiglu_alpha, swiglu_limit,
            simulate_unfused, activation_format, quantize_output, output_dtype,
            gather_idx, scatter_idx, bias=bias,
        )
        return out[0]
    # a calibrated (static) scale on a raw A (see `tensor_wide_act_operands`): block-scale weights
    # have a dedicated static kernel, per-tensor ones read the same As on the tensor-wide arm
    static_act = B.dtype not in INT_WEIGHT_DTYPES and not is_mx(B, Bs) and As is not None and As.ndim <= 1 and As.numel() in (1, B.shape[0]) and A.dtype != FP8_DTYPE
    if static_act:
        assert Bs is not None and not is_mx(B, Bs), (
            "a calibrated (static) activation scale is an FP8 form — MX activations carry a scale "
            "per group, derived per call"
        )
    if static_act and weight_block_size(B, Bs) is not None:
        out = w8a8_block_static_fp8_matmul_grouped(
            A,
            B,
            As,
            Bs,
            expert_start,
            weight_block_size(B, Bs),
            gate,
            act_fn,
            swiglu_alpha,
            swiglu_limit,
            simulate_unfused,
            activation_format,
            quantize_output,
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
        out = full_precision_matmul_grouped(
            A,
            B,
            expert_start,
            gate,
            act_fn,
            swiglu_alpha,
            swiglu_limit,
            simulate_unfused,
            activation_format,
            quantize_output,
            output_dtype,
            gather_idx,
            scatter_idx,
            bias=bias,
        )
    elif B.dtype in INT_WEIGHT_DTYPES:  # packed integer weights
        out = int_weight_only_matmul_grouped(
            A, B, Bs, expert_start, gate, act_fn, swiglu_alpha, swiglu_limit, simulate_unfused, output_dtype,
            gather_idx, scatter_idx, b_zero=b_zero, bias=bias, b_perm=b_perm, a_pre_quant_scale=a_pre_quant_scale, a_scale=As, b_scale2=b_global_scale,
        )
    elif is_mx(B, Bs) and activation_format == "bf16":  # weight-only: raw bf16 acts, MX weight upcast in-MMA
        assert As is None and a_global_scale is None and not quantize_output, (
            "weight-only (activation_format='bf16') takes a raw activation, no As/global/requant"
        )
        assert output_global_scale is None, (
            "weight-only has no requant epilogue for output_global_scale to normalize"
        )
        out = mx_weight_only_matmul_grouped(
            A,
            B,
            Bs,
            expert_start,
            gate,
            act_fn,
            swiglu_alpha,
            swiglu_limit,
            simulate_unfused,
            output_dtype,
            gather_idx,
            scatter_idx,
            b_global_scale=b_global_scale,
            bias=bias,
        )
    elif is_mx(B, Bs):
        out = mx_dynamic_matmul_grouped(
            A,
            B,
            As,
            Bs,
            expert_start,
            gate,
            act_fn,
            swiglu_alpha,
            swiglu_limit,
            simulate_unfused,
            activation_format,
            quantize_output,
            output_dtype,
            gather_idx,
            scatter_idx,
            a_global_scale,
            b_global_scale,
            output_global_scale,
            bias=bias, a_pre_quant_scale=a_pre_quant_scale,
        )
    elif (block_size := weight_block_size(B, Bs)) is None:
        out = w8a8_tensor_dynamic_fp8_matmul_grouped(
            A,
            B,
            As,
            Bs,
            expert_start,
            gate,
            act_fn,
            swiglu_alpha,
            swiglu_limit,
            simulate_unfused,
            activation_format,
            quantize_output,
            output_dtype,
            gather_idx,
            scatter_idx,
            bias=bias,
            activation_scale_ub=activation_scale_ub,
        )
    else:
        out = w8a8_block_dynamic_fp8_matmul_grouped(
            A,
            B,
            As,
            Bs,
            expert_start,
            block_size,
            gate,
            act_fn,
            swiglu_alpha,
            swiglu_limit,
            simulate_unfused,
            activation_format,
            quantize_output,
            output_dtype,
            gather_idx,
            scatter_idx,
            bias=bias,
        )
    # The ops return a list (torch custom ops can't return a Tensor-or-tuple union): [C] plain,
    # [C, Cs] under ``quantize_output``. Unwrap to the documented Tensor / (Tensor, Tensor) return.
    return out[0] if len(out) == 1 else tuple(out)
