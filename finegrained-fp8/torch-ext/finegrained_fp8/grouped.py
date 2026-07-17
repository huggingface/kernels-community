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
    resolve_output_dtype,
    FP8_DTYPE,
    MX_SCALE_GROUP_K,
    NIBBLES_PER_BYTE,
    block_dynamic_grouped_matmul_pruner,
    block_dynamic_dot_scaled_ws_pruner,
    mx_config_pruner,
    build_tile_layout,
    resolve_grouped_tile,
    block_within_dim_pruner,
    compose_pruners,
    device_context,
    sm_count,
    tl_dtype,
    fp8_act_quant_block_dynamic,
    fp8_act_quant_tensor_wide,
    expert_weight_shape,
    mx_scale_family,
    normalize_per_expert_scale,
    validate_dense_operands,
    routed_rows,
    tokens_per_expert_bucket,
    resolve_input_recipe,
    load_grouped_weight_tile,
    load_grouped_act_tile,
    load_mx_grouped_act,
    load_mx_grouped_weight,
    descriptor_box_pruner,
    stacked_gate_up_ptrs,
    grouped_gemm_epilogue,
    get_accelerator_autotuning_configs,
    warp_spec_compile_guard_pruner,
    smem_pruner,
    is_mx,
    weight_block_size,
    e2m1_as_uint8,
    ue8m0_as_uint8,
    block_dynamic_dot,
    swizzle_mx_scales,
    swizzle_grouped_mx_scales,
    swizzle_gateup_weight_scales,
    mx_act_quant_swizzled_grouped,
    swizzled_scales_bm_pruner,
    mx_compute,
)


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
        1 if nargs.get("HAS_GATHER") else nargs["BLOCK_SIZE_M"],
        nargs["BLOCK_SIZE_K"] // act_values_per_byte,
    ]


def _rebind_grouped_descriptors(nargs):
    """Composite pre_hook: both weight and activation descriptor boxes."""
    _rebind_grouped_weight_descriptor(nargs)
    _rebind_grouped_act_descriptor(nargs)


def _rebind_grouped_mx_descriptors(nargs):
    """MX composite pre_hook: the operand boxes plus the two SWIZZLE_32_4_4 scale boxes
    ``[1, BLOCK // 128, (BK // SCALE_GROUP_K) // 4, 2, 256]`` over the swizzled
    ``(1, rows // 128, cols // 4, 2, 256)`` views. Activation box is BM // 128 (BM pinned 128);
    the weight box is ``(2 if GATE else 1) * BN // 128`` — the stacked gate|up tile is one 2*BN
    block (BN pinned 128 under GATE by ``swizzled_scales_bm_pruner``). Mutate in place."""
    _rebind_grouped_descriptors(nargs)
    rep_k = (nargs["BLOCK_SIZE_K"] // nargs["SCALE_GROUP_K"]) // 4
    bn_blocks = (2 if nargs.get("GATE") else 1) * nargs["BLOCK_SIZE_N"] // 128
    nargs["ASDescriptor"].block_shape = [1, nargs["BLOCK_SIZE_M"] // 128, rep_k, 2, 256]
    nargs["BSDescriptor"].block_shape = [1, bn_blocks, rep_k, 2, 256]


@bayesian_autotune(
    # TMA history: a first descriptor port FORCED SWAP_AB alongside it (weights in the
    # MMA M operand — a coupling inherited from the bd-2D verdict), measured a loser
    # (1944us vs WS-pointer 1796us, dsv4 E=256, 2026-07-14) and was removed. The current
    # form has NO swap and no SWAP_AB axis: the descriptor loads the natural-orientation
    # ((2|1), BN, BK) gate|up box and transposes once to the same K-major tile the
    # pointer arm builds. That form wins outright: gate_up 948->455us (-52%), down
    # 256->212us desc/desc (-17%) at dense E=8 (2026-07-16) — the old loss was the swap
    # coupling, not TMA. Both memory axes are emitted; the tuner routes per key.
    # All verdicts B200 (sm_100); re-chart the swap coupling on H100 or the target
    # device before reusing them there.
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
            block_dynamic_dot_scaled_ws_pruner(),
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
    GatherIdx,  # (S,) int32 — sorted position -> source row of A; read iff HAS_GATHER
    ScatterIdx,  # (S,) int32 — sorted position -> destination row of C; read iff HAS_SCATTER
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
    HAS_GATHER: tl.constexpr,
    HAS_SCATTER: tl.constexpr,
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
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn = (
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
                HAS_GATHER,
                HAS_SCATTER,
            )
        )
        a_ptrs = A + in_row[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
        as_ptrs = As + in_row * stride_as_m
        # GATE stacks gate (rows [0, N)) and up (rows [N, 2N)) into one [BK, 2*BN] tile — the
        # up block sits N rows away (N = per-projection width). GATE=False -> plain [BK, BN].
        b_ptrs = stacked_gate_up_ptrs(
            B + expert_id64 * stride_b_e,
            offs_bn,
            offs_k,
            N * stride_b_n,
            stride_b_n,
            stride_b_k,
            GATE,
            False,
        )
        if GATE:
            gate_s_ptr = Bs + expert_id64 * stride_bs_e + pid_n * stride_bs_n
            up_s_ptr = (
                Bs + expert_id64 * stride_bs_e + (num_n_tiles + pid_n) * stride_bs_n
            )
        else:
            bs_ptrs = Bs + expert_id64 * stride_bs_e + pid_n * stride_bs_n

        # descriptor box coordinates: expert slab (x2 under GATE), N offset; without a
        # gather the A rows are contiguous, so their min IS the box row start
        row0 = (expert_id64 * (2 if GATE else 1)).to(tl.int32)
        n_off = pid_n * BLOCK_SIZE_N
        m_start = tl.min(in_row).to(tl.int32)

        acc = tl.zeros(
            (BLOCK_SIZE_M, (2 if GATE else 1) * BLOCK_SIZE_N), dtype=tl.float32
        )
        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
            a = load_grouped_act_tile(
                a_ptrs,
                ADescriptor,
                m_start,
                k * BLOCK_SIZE_K,
                row_mask,
                in_row,
                A_MEMORY_MODE,
                HAS_GATHER,
            )
            a_s = tl.load(as_ptrs, mask=row_mask, other=0.0)
            w = load_grouped_weight_tile(
                b_ptrs,
                BDescriptor,
                row0,
                n_off,
                k * BLOCK_SIZE_K,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GATE,
                B_MEMORY_MODE,
            )
            if GATE:
                # gate scale on the first BN columns, up scale on the rest (raw — the
                # block_dynamic_dot arm decodes fp32/UE8M0 or broadcasts for dot_scaled)
                w_s = tl.where(
                    tl.arange(0, 2 * BLOCK_SIZE_N) < BLOCK_SIZE_N,
                    tl.load(gate_s_ptr),
                    tl.load(up_s_ptr),
                )
                gate_s_ptr += stride_bs_k
                up_s_ptr += stride_bs_k
            else:
                w_s = tl.load(bs_ptrs)
                bs_ptrs += stride_bs_k
            acc = block_dynamic_dot(
                acc, a, a_s, w, w_s, BLOCK_SIZE_K, False, USE_DOT_SCALED
            )
            a_ptrs += BLOCK_SIZE_K * stride_a_k
            as_ptrs += 1
            b_ptrs += BLOCK_SIZE_K * stride_b_k

        grouped_gemm_epilogue(
            C,
            Cs,
            acc,
            out_row,
            offs_bn,
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
    GatherIdx,  # (S,) int32 — sorted position -> source row of A; read iff HAS_GATHER
    ScatterIdx,  # (S,) int32 — sorted position -> destination row of C; read iff HAS_SCATTER
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
    HAS_GATHER: tl.constexpr,
    HAS_SCATTER: tl.constexpr,
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
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn = (
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
                HAS_GATHER,
                HAS_SCATTER,
            )
        )
        a_ptrs = A + in_row[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
        # GATE stacks gate|up into one [BK, 2*BN] tile (one per-tensor scale covers both);
        # the up block sits N rows away. GATE=False -> the plain [BK, BN] tile.
        b_ptrs = stacked_gate_up_ptrs(
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
        # descriptor box coordinates: expert slab (x2 under GATE), N offset; without a
        # gather the A rows are contiguous, so their min IS the box row start
        row0 = (expert_id64 * (2 if GATE else 1)).to(tl.int32)
        n_off = pid_n * BLOCK_SIZE_N
        m_start = tl.min(in_row).to(tl.int32)

        acc = tl.zeros(
            (BLOCK_SIZE_M, (2 if GATE else 1) * BLOCK_SIZE_N), dtype=tl.float32
        )
        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
            a = load_grouped_act_tile(
                a_ptrs,
                ADescriptor,
                m_start,
                k * BLOCK_SIZE_K,
                row_mask,
                in_row,
                A_MEMORY_MODE,
                HAS_GATHER,
            )
            w = load_grouped_weight_tile(
                b_ptrs,
                BDescriptor,
                row0,
                n_off,
                k * BLOCK_SIZE_K,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GATE,
                B_MEMORY_MODE,
            )
            acc += tl.dot(a, w)
            a_ptrs += BLOCK_SIZE_K * stride_a_k
            b_ptrs += BLOCK_SIZE_K * stride_b_k
        acc = acc * a_s[:, None] * b_s

        grouped_gemm_epilogue(
            C,
            Cs,
            acc,
            out_row,
            offs_bn,
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
    ),  # prefill: no scalar branch; TMA descriptor vs pointer loads on both operands
    # the MXFP4/MXFP8 (and packed-activation) splits key themselves — the tuner appends
    # every tensor arg's dtype to its cache key (memory and disk);
    # GATE keys the gate|up arm separately (its stacked dot is 2*BN wide, a different tile optimum).
    ["N", "K", "tokens_per_expert_bit_length", "GATE"],
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
        )
    },
)
@triton.jit
def mx_dynamic_matmul_grouped_kernel(
    A,  # (num_tokens, K) E4M3 activations (pre-quantized once by the wrapper), any row order
    ADescriptor,  # host TMA descriptor over A (rows, K_bytes), box (BM, BK_bytes); read iff A_MEMORY_MODE != "pointer"
    As,  # (S, K // 32) UE8M0 group-32 activation scales
    ASDescriptor,  # host TMA descriptor over the SWIZZLE_32_4_4, expert-sorted/128-padded A scales; read iff SWIZZLED_SCALES
    B,  # (num_experts, N, K) E4M3 (MXFP8) or (num_experts, N, K // 2) packed E2M1 (MXFP4); 2N under GATE
    BDescriptor,  # host TMA descriptor over B viewed (2E|E, N, K_bytes), box ((2|1), BN, BK_bytes); read iff B_MEMORY_MODE != "pointer"
    Bs,  # (num_experts, N, K // SCALE_GROUP_K) UE8M0 weight scales (2N under GATE)
    BSDescriptor,  # host TMA descriptor over the SWIZZLE_32_4_4 per-expert B scales; read iff SWIZZLED_SCALES
    C,  # (S, N[/2]) output; under an OUTPUT_RECIPE the MX-requantized intermediate
    Cs,  # (S, N // SCALE_GROUP_K) UE8M0 output scale; written iff OUTPUT_RECIPE
    GatherIdx,  # (S,) int32 — sorted position -> source row of A; read iff HAS_GATHER
    ScatterIdx,  # (S,) int32 — sorted position -> destination row of C; read iff HAS_SCATTER
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
    HAS_GATHER: tl.constexpr,
    HAS_SCATTER: tl.constexpr,
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
    # Grouped MX always reads pre-swizzled SWIZZLE_32_4_4 scales affine off the SA/SB descriptors
    # — the tcgen05 fast path (BM pinned 128, BN pinned 128 under GATE; the wrapper builds the
    # descriptors and asserts N % 128). Activations arrive pre-quantized: the offline act-quant
    # emits swizzled scales, and the down projection's pre-quantized As is swizzled by its
    # producer (the requant epilogue). Under GATE the weight scale is the stacked 2*BN block.
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    offs_ka = tl.arange(0, BLOCK_SIZE_K // ACT_VALUES_PER_BYTE)
    offs_kb = tl.arange(0, BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE)

    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn = (
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
                HAS_GATHER,
                HAS_SCATTER,
            )
        )
        a_ptrs = A + in_row[:, None] * stride_a_m + offs_ka[None, :] * stride_a_k
        # descriptor box coordinates: expert slab (x2 under GATE), N offset, K-byte offsets per
        # operand; without a gather the A rows are the contiguous sorted positions, so their min
        # IS the box row start
        row0 = (expert_id64 * (2 if GATE else 1)).to(tl.int32)
        n_off = pid_n * BLOCK_SIZE_N
        m_start = tl.min(in_row).to(tl.int32)
        kb_off = 0
        ka_off = 0
        # GATE stacks gate|up into the weight (K-major); the up block sits N rows away.
        b_ptrs = stacked_gate_up_ptrs(
            B + expert_id64 * stride_b_e,
            offs_bn,
            offs_kb,
            N * stride_b_n,
            stride_b_n,
            stride_b_k,
            GATE,
            False,
        )

        acc = tl.zeros(
            (BLOCK_SIZE_M, (2 if GATE else 1) * BLOCK_SIZE_N), dtype=tl.float32
        )
        # Pre-swizzled scales (SWIZZLE_32_4_4, emitted by the offline act-quant / requant): the
        # M tile's expert-sorted, 128-padded scale block is the flat tile index pid_m; the weight
        # scale block is expert*(N//BN) + pid_n (num_n_tiles == N // BN).
        pid_m = tile_id // num_n_tiles
        weight_blk = (expert_id64 * num_n_tiles + pid_n).to(tl.int32)
        SCALE_COLS: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
        REP_K: tl.constexpr = SCALE_COLS // 4
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a, a_scale = load_mx_grouped_act(
                a_ptrs,
                ADescriptor,
                ASDescriptor,
                m_start,
                ka_off,
                pid_m,
                k,
                row_mask,
                in_row,
                BLOCK_SIZE_M,
                REP_K,
                SCALE_COLS,
                A_MEMORY_MODE,
                HAS_GATHER,
            )
            b, b_s = load_mx_grouped_weight(
                b_ptrs,
                BDescriptor,
                BSDescriptor,
                row0,
                n_off,
                kb_off,
                weight_blk,
                k,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE,
                REP_K,
                SCALE_COLS,
                GATE,
                B_MEMORY_MODE,
            )
            acc = mx_compute(
                acc,
                a,
                a_scale,
                b,
                b_s,
                COMPUTE_MODE,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                SCALE_GROUP_K,
            )
            a_ptrs += (BLOCK_SIZE_K // ACT_VALUES_PER_BYTE) * stride_a_k
            ka_off += BLOCK_SIZE_K // ACT_VALUES_PER_BYTE
            b_ptrs += (BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE) * stride_b_k
            kb_off += BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE

        grouped_gemm_epilogue(
            C,
            Cs,
            acc,
            out_row,
            offs_bn,
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
            SCALE_GROUP_K,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SIMULATE_UNFUSED,
            INTERMEDIATE_DTYPE,
            # mxfp8 requant writes Cs straight into the down proj's SWIZZLE_32_4_4 layout (Cs is
            # a descriptor). Only when the output stays expert-sorted (no scatter) — the swizzle
            # needs contiguous 128-row blocks, which the fused gate_up (scatter_idx=None)
            # guarantees; a scattered output and the fp4 requants keep row-major Cs.
            OUTPUT_RECIPE == "mxfp8" and not HAS_SCATTER,
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
    GatherIdx,  # (S,) int32 — sorted position -> source row of A; read iff HAS_GATHER
    ScatterIdx,  # (S,) int32 — sorted position -> destination row of C; read iff HAS_SCATTER
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
    HAS_GATHER: tl.constexpr,
    HAS_SCATTER: tl.constexpr,
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
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn = (
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
                HAS_GATHER,
                HAS_SCATTER,
            )
        )
        a_ptrs = A + in_row[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
        # GATE stacks gate|up into one [BK, 2*BN] tile; the up block sits N rows away.
        # GATE=False -> the plain [BK, BN] tile. Pointer arm — the descriptor modes
        # fetch the [BN, BK] box at row0 and transpose it instead.
        b_ptrs = stacked_gate_up_ptrs(
            B + expert_id64 * stride_b_e,
            offs_bn,
            offs_k,
            N * stride_b_n,
            stride_b_n,
            stride_b_k,
            GATE,
            False,
        )
        # descriptor box coordinates: expert slab (x2 under GATE), N offset, K offset;
        # without a gather the A rows are contiguous, so their min IS the box row start
        row0 = (expert_id64 * (2 if GATE else 1)).to(tl.int32)
        n_off = pid_n * BLOCK_SIZE_N
        m_start = tl.min(in_row).to(tl.int32)

        acc = tl.zeros(
            (BLOCK_SIZE_M, (2 if GATE else 1) * BLOCK_SIZE_N), dtype=tl.float32
        )
        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
            a = load_grouped_act_tile(
                a_ptrs,
                ADescriptor,
                m_start,
                k * BLOCK_SIZE_K,
                row_mask,
                in_row,
                A_MEMORY_MODE,
                HAS_GATHER,
            )
            w = load_grouped_weight_tile(
                b_ptrs,
                BDescriptor,
                row0,
                n_off,
                k * BLOCK_SIZE_K,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GATE,
                B_MEMORY_MODE,
            )
            acc += tl.dot(a, w)
            a_ptrs += BLOCK_SIZE_K * stride_a_k
            b_ptrs += BLOCK_SIZE_K * stride_b_k

        grouped_gemm_epilogue(
            C,
            C,  # dummy Cs (no requant arm)
            acc,
            out_row,
            offs_bn,
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
    As: torch.Tensor | None,
    B: torch.Tensor,
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
    As: (S, K // block_k) per-row, per-K-block activation scales
    B:  (num_experts, N, K) FP8 weights; under ``gate`` the (num_experts, 2N, K) gate|up stack
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
    # MoE expert dimensions must be block-aligned; non-aligned N/K is not supported.
    assert N % block_n == 0, f"N ({N}) must be divisible by block_n ({block_n})"
    assert K % block_k == 0, f"K ({K}) must be divisible by block_k ({block_k})"
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

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_block_dynamic_fp8_matmul_grouped_kernel)[
            (num_sms,)
        ](
            A,
            TensorDescriptor.from_tensor(A, block_shape=[16, 64]),
            As,
            B,
            TensorDescriptor.from_tensor(
                B.view(2 * num_experts if gate else num_experts, N, K),
                block_shape=[1, 128, 64],
            ),
            bs_u8,
            C,
            Cs,
            gather_idx if gather_idx is not None else expert_start,  # dummy ptr
            scatter_idx if scatter_idx is not None else expert_start,  # dummy ptr
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
            HAS_GATHER=gather_idx is not None,
            HAS_SCATTER=scatter_idx is not None,
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
    As: torch.Tensor | None,
    B: torch.Tensor,
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
    As: (S,) per-token activation scales
    B:  (num_experts, N, K) FP8 expert weights; under ``gate`` the (num_experts, 2N, K) stack
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

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_tensor_dynamic_fp8_matmul_grouped_kernel)[
            (num_sms,)
        ](
            A,
            TensorDescriptor.from_tensor(A, block_shape=[16, 64]),
            As,
            B,
            TensorDescriptor.from_tensor(
                B.view(2 * num_experts if gate else num_experts, N, K),
                block_shape=[1, 128, 64],
            ),
            Bs,
            C,
            expert_start,  # dummy Cs (no fused requant for tensor-wide)
            gather_idx if gather_idx is not None else expert_start,  # dummy ptr
            scatter_idx if scatter_idx is not None else expert_start,  # dummy ptr
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
            HAS_GATHER=gather_idx is not None,
            HAS_SCATTER=scatter_idx is not None,
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
    As: torch.Tensor | None,
    B: torch.Tensor,
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
    assert A.ndim == 2 and B.ndim == 3 and Bs.ndim == 3
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
    nvfp4, scale_group = mx_scale_family(Bs, K)
    assert Bs.shape == (num_experts, n_rows, K // scale_group), (
        f"Bs shape {tuple(Bs.shape)} != ({num_experts}, {n_rows}, {K // scale_group})"
    )

    output_dtype = resolve_output_dtype(output_dtype, A, As)
    input_recipe = resolve_input_recipe(input_recipe, output_recipe, Bs)
    requant = output_recipe is not None
    # SWIZZLE_32_4_4 scales — the tcgen05 scaled-MMA fast path — for the whole grouped MX
    # kernel (down projection, gate_up, and plain grouped GEMM alike). BM is pinned to 128, and
    # BN to 128 under GATE (swizzled_scales_bm_pruner), matching the 128-padded scale layouts.
    assert N % 128 == 0, f"the grouped MX path needs N ({N}) a multiple of 128"
    scale_dtype = torch.float8_e4m3fn if nvfp4 else torch.uint8
    cb = triton.cdiv(K // scale_group, 4)
    C_cols = K // scale_group
    # A raw (As is None) -> the act-quant kernel emits its scales directly in the expert-sorted,
    # 128-padded SWIZZLE_32_4_4 layout (no post-quant pass); else pre-quantized (As given, e.g.
    # the down projection's intermediate) -> gather+swizzle it. The tensor dtypes say the format.
    if As is None:
        a_vals, as_sw, n_m_tiles = mx_act_quant_swizzled_grouped(
            A, input_recipe, scale_group, scale_dtype, gather_idx, expert_start
        )
        as_descriptor = TensorDescriptor.from_tensor(as_sw, [1, 1, 1, 2, 256])
    elif As.ndim == 5:
        # pre-swizzled by the gate_up requant epilogue (the fused down projection) — read as is
        a_vals = A
        n_m_tiles = As.shape[1]
        as_descriptor = TensorDescriptor.from_tensor(As, [1, 1, 1, 2, 256])
    else:
        assert (As.dtype == torch.float8_e4m3fn) == nvfp4, (
            f"activation scales ({As.dtype}) must match the weight scale family ({Bs.dtype})"
        )
        a_vals = A
        as_sw, n_m_tiles = swizzle_grouped_mx_scales(
            ue8m0_as_uint8(As), expert_start, gather_idx
        )
        as_descriptor = TensorDescriptor.from_tensor(as_sw, [1, 1, 1, 2, 256])
    # uint8 aliases for the binder — never clobber the caller's A/B/Bs dtype/view. The
    # swizzled arm reads scales via SA/SB descriptors, so the As positional is a dead dummy.
    a_u8 = e2m1_as_uint8(a_vals)
    b_u8 = e2m1_as_uint8(B)
    bs_u8 = ue8m0_as_uint8(Bs)
    # static weight scales -> SWIZZLE_32_4_4, indexed expert*(N//BN) + pid_n. Under GATE the
    # gate|up rows are interleaved per pid_n tile (each tile's gate-128 then up-128) into one
    # contiguous 2*BN swizzled block (BN pinned 128), the interleave computed in-kernel (no
    # torch index / gather). Both are single fused triton launches.
    if gate:
        bs_sw = swizzle_gateup_weight_scales(bs_u8, num_experts, N)
    else:
        bs_sw = swizzle_mx_scales(bs_u8.reshape(num_experts * N, C_cols)).reshape(
            1, num_experts * (N // 128), cb, 2, 256
        )
    bs_descriptor = TensorDescriptor.from_tensor(bs_sw, [1, 1, 1, 2, 256])
    # mxfp8 requant writes Cs straight into the down proj's SWIZZLE_32_4_4 layout (the epilogue
    # gets a descriptor, not a pointer) — the down reads it affine, no post-requant swizzle pass.
    # Only when the output stays expert-sorted (scatter_idx None, the fused gate_up convention):
    # the swizzle needs contiguous 128-row blocks, which a scattered output can't provide.
    mxfp8_swizzled_out = (
        requant and output_recipe == "mxfp8" and scatter_idx is None
    )
    if output_recipe in ("mxfp4", "nvfp4"):
        # packed E2M1 intermediate (nibble pairs along N) + group scales (UE8M0 for MX,
        # E4M3 for NVFP4) — feeds a W4A4 down as-is
        assert N % (2 * scale_group) == 0, (
            f"N (={N}) must be a multiple of {2 * scale_group} to pack E2M1 pairs"
        )
        C = A.new_empty((S, N // 2), dtype=torch.int8)
        cs_ret = torch.empty(
            S,
            N // scale_group,
            device=A.device,
            dtype=torch.float8_e4m3fn if nvfp4 else torch.uint8,
        )
        Cs = cs_ret
    elif mxfp8_swizzled_out:
        C = A.new_empty((S, N), dtype=FP8_DTYPE)
        cb_cs = triton.cdiv(N // MX_SCALE_GROUP_K, 4)
        cs_ret = torch.empty(
            1, n_m_tiles, cb_cs, 2, 256, device=A.device, dtype=torch.uint8
        )
        Cs = TensorDescriptor.from_tensor(cs_ret, [1, 1, 1, 2, 256])
    elif requant:
        # mxfp8 requant that can't swizzle (scattered output) — row-major Cs
        C = A.new_empty((S, N), dtype=FP8_DTYPE)
        cs_ret = torch.empty(S, N // MX_SCALE_GROUP_K, device=A.device, dtype=torch.uint8)
        Cs = cs_ret
    else:
        C = A.new_empty((S, N), dtype=output_dtype)
        cs_ret = None
        Cs = expert_start  # general dummy pointer; unread (no OUTPUT_RECIPE), strides literal
    num_sms = sm_count(A.device.index)
    # host TMA descriptor over the (2E|E, N, K_bytes) view — one box holds both gate|up
    # projections; the placeholder box is re-bound per tuned config by the pre_hook
    rows0 = 2 * num_experts if gate else num_experts
    b_descriptor = TensorDescriptor.from_tensor(
        b_u8.view(rows0, N, b_u8.shape[2]), block_shape=[1, 128, 64]
    )
    a_descriptor = TensorDescriptor.from_tensor(a_u8, block_shape=[16, 64])
    # (the SA/SB swizzled-scale descriptors and their placeholder boxes — re-bound per tuned
    # config by _rebind_grouped_mx_descriptors — are built above with the scales.)

    with device_context(A.device):
        compile_time_only_triton_wrap(mx_dynamic_matmul_grouped_kernel)[(num_sms,)](
            a_u8,
            a_descriptor,
            a_u8,  # As positional is a dead dummy on the swizzled arm (scales via ASDescriptor)
            as_descriptor,
            b_u8,
            b_descriptor,
            bs_u8,
            bs_descriptor,
            C,
            Cs,
            gather_idx if gather_idx is not None else expert_start,  # dummy ptr
            scatter_idx if scatter_idx is not None else expert_start,  # dummy ptr
            expert_start,
            S,
            N,
            K,
            a_u8.stride(0),
            a_u8.stride(1),
            a_u8.stride(0),  # dummy As stride
            b_u8.stride(0),
            b_u8.stride(2),
            b_u8.stride(1),
            bs_u8.stride(0),
            bs_u8.stride(2),
            bs_u8.stride(1),
            C.stride(0),
            C.stride(1),
            # mxfp8 swizzled Cs is a descriptor (no strides); the fp4 requants keep pointer strides
            cs_ret.stride(0) if (requant and not mxfp8_swizzled_out) else 1,
            cs_ret.stride(1) if (requant and not mxfp8_swizzled_out) else 1,
            num_experts=num_experts,
            tokens_per_expert_bit_length=tokens_per_expert_bucket(S, num_experts),
            HAS_GATHER=gather_idx is not None,
            HAS_SCATTER=scatter_idx is not None,
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

    with device_context(A.device):
        compile_time_only_triton_wrap(full_precision_matmul_grouped_kernel)[(num_sms,)](
            A,
            TensorDescriptor.from_tensor(A, block_shape=[16, 64]),
            B,
            TensorDescriptor.from_tensor(
                B.view(2 * num_experts if gate else num_experts, N, K),
                block_shape=[1, 128, 64],
            ),  # boxes re-bound to the tuned config by _rebind_grouped_descriptors
            C,
            gather_idx if gather_idx is not None else expert_start,  # dummy ptr
            scatter_idx if scatter_idx is not None else expert_start,  # dummy ptr
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
            HAS_GATHER=gather_idx is not None,
            HAS_SCATTER=scatter_idx is not None,
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
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Grouped matmul dispatcher (W8A8 FP8, W4A8/W4A4 FP4, or full-precision).
    ``expert_start`` is the ``(E+1,)`` tiling schedule from one ``compute_grouped_scheduling``
    pass, shared by every grouped GEMM of the layer (the expert sort is virtual — nothing is
    physically permuted).

    ``As`` marks ``A`` as already quantized (framework-precomputed scales, or a requantized
    intermediate handed to the down projection); ``None`` = raw ``A``, quantized by the op per
    ``quantization`` (see ``Quantization`` — recipe-default fp8/E4M3, or packed E2M1 under
    ``input_recipe="mxfp4"``). ``Bs`` ``None`` = unquantized BF16/FP16 weights.
    ``quantization.output_recipe`` requantizes the output into the recipe's format — the
    return is then ``(C, Cs)``. ``epilogue`` is the fused output transform (gate|up + GLU).
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
            As,
            B,
            Bs,
            expert_start,
            *ep.as_args(),
            *q.as_args(),
            output_dtype,
            gather_idx,
            scatter_idx,
        )
    elif (block_size := weight_block_size(B, Bs)) is None:
        out = w8a8_tensor_dynamic_fp8_matmul_grouped(
            A,
            As,
            B,
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
            As,
            B,
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
