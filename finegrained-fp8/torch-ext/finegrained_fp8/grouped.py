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
    resolve_output_dtype,
    FP8_DTYPE,
    MX_SCALE_GROUP_K,
    NIBBLES_PER_BYTE,
    UE8M0_SCALE_DTYPES,
    block_dynamic_grouped_matmul_pruner,
    mx_config_pruner,
    build_tile_layout,
    resolve_grouped_tile,
    block_k_within_k_pruner,
    compose_pruners,
    device_context,
    sm_count,
    tl_dtype,
    fp8_act_quant_block_dynamic,
    fp8_act_quant_tensor_wide,
    mxfp_act_quant,
    load_mx_act_tile,
    stacked_gate_up_ptrs,
    stacked_gate_up_flatten,
    grouped_gemm_epilogue,
    get_accelerator_autotuning_configs,
    warp_spec_compile_guard_pruner,
    is_mxfp,
    is_tensor_wide,
    e2m1_as_uint8,
    ue8m0_as_uint8,
    decode_ue8m0_scale,
    mx_compute,
)


@bayesian_autotune(
    # SWAP_AB/MEMORY_MODE (TMA) arms were ported here, measured, and REMOVED: at dsv4-like
    # prefill (S=8192, E=256, N=4096, K=7168) WS-pointer BM=64 w4 s4 = 1796us beats
    # descriptor+swap (1944us, its best) and pointer+swap (2088us); descriptor+swap was
    # also numerically WRONG at BM=16 at large K. The dormant reference implementation
    # lives on the 2D kernel (w8a8_block_dynamic_fp8_matmul_kernel); see OPTIMIZATION_LOG.
    get_accelerator_autotuning_configs(warp_spec=True, tune_block_m=True),
    # GATE keys the gate|up arm separately: its dot is 2*BN wide, a different tile optimum.
    ["N", "K", "tokens_per_expert_bit_length", "GATE"],
    n_trials=100,
    # Pipeliner-race guard: per launch-BM, WS-only at BM >= 64 and non-WS below (see the pruner).
    prune_configs_by={"early_config_prune": block_dynamic_grouped_matmul_pruner()},
)
@triton.jit
def w8a8_block_dynamic_fp8_matmul_grouped_kernel(
    A,  # (num_tokens, K) E4M3 activations (pre-quantized once by the wrapper), any row order
    As,  # (S, K // BLOCK_SIZE_K) fp32 per-row, per-K-block activation scales
    B,  # (num_experts, N, K) FP8 weights; under GATE the (num_experts, 2N, K) gate|up stack
    Bs,  # (num_experts, N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales (2N under GATE)
    C,  # (S, N) output; under REQUANT the FP8-requantized intermediate
    Cs,  # (S, N // BLOCK_SIZE_N) per-row, per-N-tile output scale; written iff REQUANT
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
    # Gate|up fusion epilogue (GATE=False -> plain grouped GEMM, every arm below folds out)
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    REQUANT: tl.constexpr = False,
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
    the epilogue splits, applies the ``ACT_FN``/SwiGLU ``glu``, and — under ``REQUANT`` — FP8-
    requantizes the intermediate into ``C`` + per-row ``Cs``. ``GATE=False`` is the plain
    grouped GEMM (down projection = plain GEMM with an output scatter); every gate arm folds
    out at compile time, leaving the plain path bit-identical."""
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
        ExpertStart, NUM_EXPERTS_POW2, BLOCK_SIZE_M
    )
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn = resolve_grouped_tile(
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

        acc = tl.zeros(
            (BLOCK_SIZE_M, (2 if GATE else 1) * BLOCK_SIZE_N), dtype=tl.float32
        )
        for _ in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
            a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
            a_s = tl.load(as_ptrs, mask=row_mask, other=0.0)
            w = stacked_gate_up_flatten(
                tl.load(b_ptrs), 2 * BLOCK_SIZE_N, BLOCK_SIZE_K, GATE, False
            )
            if GATE:
                # gate scale on the first BN columns, up scale on the rest
                w_s = tl.where(
                    tl.arange(0, 2 * BLOCK_SIZE_N) < BLOCK_SIZE_N,
                    decode_ue8m0_scale(tl.load(gate_s_ptr)),
                    decode_ue8m0_scale(tl.load(up_s_ptr)),
                )
                gate_s_ptr += stride_bs_k
                up_s_ptr += stride_bs_k
            else:
                w_s = decode_ue8m0_scale(tl.load(bs_ptrs))
                bs_ptrs += stride_bs_k
            acc += tl.dot(a, w) * a_s[:, None] * w_s[None, :]
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
            row_mask,
            stride_c_m,
            stride_c_n,
            stride_cs_m,
            stride_cs_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GATE,
            REQUANT,
            "fp8",
            1,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SIMULATE_UNFUSED,
            INTERMEDIATE_DTYPE,
        )


@bayesian_autotune(
    get_accelerator_autotuning_configs(
        tune_block_nk=True, warp_spec=True, tune_block_m=True
    ),
    # GATE keys the gate|up arm separately (its dot is 2*BN wide, a different tile optimum).
    ["N", "K", "tokens_per_expert_bit_length", "GATE"],
    n_trials=100,
    # BLOCK_SIZE_K is a tuned axis and the K-loop is maskless — veto non-dividing BKs;
    # WS is a pure perf axis here (non-WS is the validated state), compile-guarded.
    # The GATE arm's stacked width-512 dots (BN=256) are clean — probed bit-exact
    # 2026-07-14; the oversized-smem ones fail benignly at launch and self-prune as inf.
    prune_configs_by={
        "early_config_prune": compose_pruners(
            block_k_within_k_pruner("K"), warp_spec_compile_guard_pruner()
        )
    },
)
@triton.jit
def w8a8_tensor_dynamic_fp8_matmul_grouped_kernel(
    A,  # (num_tokens, K) pre-quantized FP8 activations, any row order
    As,  # (S,) per-token activation scales
    B,  # (num_experts, N, K) FP8 weights; under GATE the (num_experts, 2N, K) gate|up stack
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
    # Gate|up fusion epilogue (GATE=False -> plain grouped GEMM). REQUANT unsupported here
    # (tensor-wide down needs a per-token whole-row scale a per-tile epilogue can't form).
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    REQUANT: tl.constexpr = False,
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
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn = resolve_grouped_tile(
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

        acc = tl.zeros(
            (BLOCK_SIZE_M, (2 if GATE else 1) * BLOCK_SIZE_N), dtype=tl.float32
        )
        for _ in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
            a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
            w = stacked_gate_up_flatten(
                tl.load(b_ptrs), 2 * BLOCK_SIZE_N, BLOCK_SIZE_K, GATE, False
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
            row_mask,
            stride_c_m,
            stride_c_n,
            stride_cs_m,
            stride_cs_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GATE,
            REQUANT,
            "fp8",
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
    ),  # prefill: no scalar branch
    # VALUES_PER_BYTE keys the MXFP4/MXFP8 split so a cached winner is only reused for its packing;
    # GATE keys the gate|up arm separately (its stacked dot is 2*BN wide, a different tile optimum).
    ["N", "K", "tokens_per_expert_bit_length", "VALUES_PER_BYTE", "GATE"],
    n_trials=100,
    # BK-within-K veto + the sm_10x dot_scaled shape/trap gates (this kernel had no
    # pruner while its BK span was {128,256} — the union span's BK=64 rows made the
    # gates load-bearing).
    prune_configs_by={"early_config_prune": mx_config_pruner("K")},
)
@triton.jit
def mxfp_dynamic_matmul_grouped_kernel(
    A,  # (num_tokens, K) E4M3 activations (pre-quantized once by the wrapper), any row order
    As,  # (S, K // 32) UE8M0 group-32 activation scales
    B,  # (num_experts, N, K) E4M3 (MXFP8) or (num_experts, N, K // 2) packed E2M1 (MXFP4); 2N under GATE
    Bs,  # (num_experts, N, K // SCALE_GROUP_K) UE8M0 weight scales (2N under GATE)
    C,  # (S, N) output; under REQUANT the MX-requantized intermediate
    Cs,  # (S, N // SCALE_GROUP_K) UE8M0 output scale; written iff REQUANT
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
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
    # Gate|up fusion epilogue (GATE=False -> plain grouped GEMM, every arm below folds out)
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    REQUANT: tl.constexpr = False,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
):
    """Unified grouped MXFP4/MXFP8 (W4A8/W8A8) expert matmul — persistent grid-stride.

    Each M-tile maps to its expert via ``ExpertStart`` and gathers its rows through
    ``PermToken`` (virtual sort — ``A`` in any row order). ``A``
    arrives pre-quantized (E4M3 + UE8M0 group-32 scales, one pass in the wrapper — the
    inline per-N-tile quant would repeat N//BN times per element). ``VALUES_PER_BYTE`` picks the
    weight format (2 = packed E2M1 / MXFP4, 1 = unpacked E4M3 / MXFP8); ``COMPUTE_MODE``
    picks ``tl.dot_scaled`` vs fp8 ``tl.dot`` + per-group rescale (decode; FP4 unpacks
    E2M1->E4M3 first, lossless).
    """
    start_pid = tl.program_id(axis=0)
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
        ExpertStart, NUM_EXPERTS_POW2, BLOCK_SIZE_M
    )
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_kb = tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)

    for tile_id in tl.range(start_pid, total_m_tiles * num_n_tiles, NUM_SMS):
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn = resolve_grouped_tile(
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
        a_ptrs = A + in_row[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
        as_ptrs = As + in_row[:, None] * stride_as_m + offs_sf[None, :]
        # GATE stacks gate|up into the weight (K-major) and its UE8M0 scale (N-major, SWAP_AB);
        # the up block sits N rows away. GATE=False -> the plain single tile / scale.
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
        bs_ptrs = stacked_gate_up_ptrs(
            Bs + expert_id64 * stride_bs_e,
            offs_bn,
            offs_sf,
            N * stride_bs_n,
            stride_bs_n,
            stride_bs_k,
            GATE,
            True,
        )

        acc = tl.zeros(
            (BLOCK_SIZE_M, (2 if GATE else 1) * BLOCK_SIZE_N), dtype=tl.float32
        )
        for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a, a_scale = load_mx_act_tile(
                a_ptrs, as_ptrs, row_mask, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K
            )
            as_ptrs += BLOCK_SIZE_K // SCALE_GROUP_K
            b = stacked_gate_up_flatten(
                tl.load(b_ptrs),
                2 * BLOCK_SIZE_N,
                BLOCK_SIZE_K // VALUES_PER_BYTE,
                GATE,
                False,
            )
            b_s = stacked_gate_up_flatten(
                tl.load(bs_ptrs),
                2 * BLOCK_SIZE_N,
                BLOCK_SIZE_K // SCALE_GROUP_K,
                GATE,
                True,
            ).to(tl.uint8)
            acc = mx_compute(
                acc,
                a,
                a_scale,
                b,
                b_s,
                COMPUTE_MODE,
                VALUES_PER_BYTE,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                SCALE_GROUP_K,
            )
            a_ptrs += BLOCK_SIZE_K * stride_a_k
            b_ptrs += (BLOCK_SIZE_K // VALUES_PER_BYTE) * stride_b_k
            bs_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_bs_k

        grouped_gemm_epilogue(
            C,
            Cs,
            acc,
            out_row,
            offs_bn,
            pid_n,
            row_mask,
            stride_c_m,
            stride_c_n,
            stride_cs_m,
            stride_cs_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GATE,
            REQUANT,
            "mx",
            SCALE_GROUP_K,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SIMULATE_UNFUSED,
            INTERMEDIATE_DTYPE,
        )


@bayesian_autotune(
    get_accelerator_autotuning_configs(
        tune_block_nk=True, warp_spec=True, tune_block_m=True
    ),
    # GATE keys the gate|up arm separately (its dot is 2*BN wide, a different tile optimum).
    ["N", "K", "tokens_per_expert_bit_length", "GATE"],
    n_trials=100,
    # BLOCK_SIZE_K is a tuned axis and the K-loop is maskless — veto non-dividing BKs;
    # WS compile-guarded (same plain-dot loop family as the tensor kernel).
    prune_configs_by={
        "early_config_prune": compose_pruners(
            block_k_within_k_pruner("K"), warp_spec_compile_guard_pruner()
        )
    },
)
@triton.jit
def full_precision_matmul_grouped_kernel(
    A,  # (num_tokens, K) BF16/FP16 activations, any row order
    B,  # (num_experts, N, K) weights in A's dtype; under GATE the (num_experts, 2N, K) gate|up stack
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
        pid_n, _, expert_id64, in_row, out_row, row_mask, offs_bn = resolve_grouped_tile(
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
        a_ptrs = A + in_row[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
        # GATE stacks gate|up into one [BK, 2*BN] tile; the up block sits N rows away.
        # GATE=False -> the plain [BK, BN] tile.
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

        acc = tl.zeros(
            (BLOCK_SIZE_M, (2 if GATE else 1) * BLOCK_SIZE_N), dtype=tl.float32
        )
        for _ in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
            a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
            w = stacked_gate_up_flatten(
                tl.load(b_ptrs), 2 * BLOCK_SIZE_N, BLOCK_SIZE_K, GATE, False
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
            row_mask,
            stride_c_m,
            stride_c_n,
            1,  # dummy Cs strides
            1,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GATE,
            False,
            "fp8",
            1,
            ACT_FN,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SIMULATE_UNFUSED,
            INTERMEDIATE_DTYPE,
        )


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_block_dynamic_fp8_matmul_grouped"), mutates_args=()
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
    requant: bool = False,
    simulate_unfused: bool = False,
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
    assert A.ndim == 2, f"A must be 2D (S, K), got ndim={A.ndim}"
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 3, f"B must be 3D (num_experts, N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"
    assert A.shape[1] == B.shape[2], (
        f"K mismatch: A has K={A.shape[1]}, B has K={B.shape[2]}"
    )

    _, K = A.shape
    # S = routed rows (num_tokens * top_k), carried by the (S,) perms — A's rows are
    # gather SOURCES and under-count S whenever top_k > 1 (gate_up reading raw hidden).
    # Only with no perms at all is A itself the expert-sorted (S, K) matrix.
    if gather_idx is not None:
        S = gather_idx.numel()
    elif scatter_idx is not None:
        S = scatter_idx.numel()
    else:
        S = A.shape[0]

    for perm_map in (gather_idx, scatter_idx):
        assert perm_map is None or (perm_map.numel() == S and perm_map.is_contiguous())

    # Under a gate epilogue B is the (E, 2N, K) gate|up stack — N is the per-projection width.
    num_experts, n_rows, _ = B.shape
    N = n_rows // 2 if gate else n_rows
    assert (
        expert_start.is_contiguous()
        and expert_start.numel() == triton.next_power_of_2(num_experts) + 1
    ), "expert_start must be contiguous (next_power_of_2(num_experts) + 1,)"

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
    # A may arrive raw (As is None) or pre-quantized (As given, e.g. a requantized
    # intermediate handed over between the fused GEMMs). Raw -> quantize here (offline).
    if As is None:
        A, As = fp8_act_quant_block_dynamic(A, block_k)
    Bs = ue8m0_as_uint8(Bs)
    if requant:
        C = A.new_empty(S, N, dtype=FP8_DTYPE)
        Cs = torch.empty(S, N // block_n, device=A.device, dtype=torch.float32)
    else:
        C = A.new_empty(S, N, dtype=output_dtype)
        Cs = expert_start  # general dummy pointer; unread (REQUANT=False), strides literal
    num_sms = sm_count(A.device.index)

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_block_dynamic_fp8_matmul_grouped_kernel)[
            (num_sms,)
        ](
            A,
            As,
            B,
            Bs,
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
            Bs.stride(0),
            Bs.stride(2),
            Bs.stride(1),
            C.stride(0),
            C.stride(1),
            Cs.stride(0) if requant else 1,  # dummy stride when unread
            Cs.stride(1) if requant else 1,
            # Meta-parameters
            num_experts=num_experts,
            tokens_per_expert_bit_length=int(
                (S + num_experts - 1) // num_experts
            ).bit_length(),
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
            REQUANT=requant,
            SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(output_dtype),
        )

    return [C, Cs] if requant else [C]


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_tensor_dynamic_fp8_matmul_grouped"), mutates_args=()
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
    requant: bool = False,
    simulate_unfused: bool = False,
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
    assert A.ndim == 2, f"A must be 2D (S, K), got ndim={A.ndim}"
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 3, f"B must be 3D (num_experts, N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"
    assert A.shape[1] == B.shape[2], (
        f"K mismatch: A has K={A.shape[1]}, B has K={B.shape[2]}"
    )

    _, K = A.shape
    # S = routed rows (num_tokens * top_k), carried by the (S,) perms — A's rows are
    # gather SOURCES and under-count S whenever top_k > 1 (gate_up reading raw hidden).
    # Only with no perms at all is A itself the expert-sorted (S, K) matrix.
    if gather_idx is not None:
        S = gather_idx.numel()
    elif scatter_idx is not None:
        S = scatter_idx.numel()
    else:
        S = A.shape[0]
    for perm_map in (gather_idx, scatter_idx):
        assert perm_map is None or (perm_map.numel() == S and perm_map.is_contiguous())

    # Under a gate epilogue B is the (E, 2N, K) gate|up stack — N is the per-projection width.
    assert not requant, (
        "requant is unsupported for tensor-wide gate_up (its down needs a per-token whole-row "
        "scale a per-tile epilogue can't form); use a plain gate epilogue + external quant"
    )
    num_experts, n_rows, _ = B.shape
    N = n_rows // 2 if gate else n_rows
    assert (
        expert_start.is_contiguous()
        and expert_start.numel() == triton.next_power_of_2(num_experts) + 1
    ), "expert_start must be contiguous (next_power_of_2(num_experts) + 1,)"

    # Normalize Bs to (num_experts, 1, 1) — one per-tensor scale (covers the gate|up stack)
    if Bs.ndim == 1:
        assert Bs.shape[0] == num_experts, (
            f"Bs shape {tuple(Bs.shape)} != expected ({num_experts},)"
        )
        Bs = Bs.reshape(num_experts, 1, 1)
    else:
        assert Bs.shape == (num_experts, 1, 1), (
            f"Bs shape {tuple(Bs.shape)} != expected ({num_experts}, 1, 1)"
        )

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
            As,
            B,
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
            tokens_per_expert_bit_length=int(
                (S + num_experts - 1) // num_experts
            ).bit_length(),
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
    add_op_namespace_prefix("mxfp_dynamic_matmul_grouped"), mutates_args=()
)
def mxfp_dynamic_matmul_grouped(
    A: torch.Tensor,
    As: torch.Tensor | None,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_start: torch.Tensor,
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
    """Grouped MX matmul over expert-sorted positions (per-tile gather/scatter, the
    sort is virtual — see ``compute_grouped_scheduling`` for the maps). Activations arrive
    pre-quantized: the caller owns the act-quant (``mxfp_act_quant``). The
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
    assert Bs.dtype in UE8M0_SCALE_DTYPES, (
        f"Bs must be float8_e8m0fnu or uint8 (UE8M0), got {Bs.dtype}"
    )
    VALUES_PER_BYTE = NIBBLES_PER_BYTE if B.dtype == torch.int8 else 1

    _, K = A.shape
    # S = routed rows (num_tokens * top_k), carried by the (S,) perms — A's rows are
    # gather SOURCES and under-count S whenever top_k > 1 (gate_up reading raw hidden).
    # Only with no perms at all is A itself the expert-sorted (S, K) matrix.
    if gather_idx is not None:
        S = gather_idx.numel()
    elif scatter_idx is not None:
        S = scatter_idx.numel()
    else:
        S = A.shape[0]
    for perm_map in (gather_idx, scatter_idx):
        assert perm_map is None or (perm_map.numel() == S and perm_map.is_contiguous())
    # Under a gate epilogue B is the (E, 2N, K) gate|up stack — N is the per-projection width.
    num_experts, n_rows, K_b = B.shape
    N = n_rows // 2 if gate else n_rows
    assert (
        expert_start.is_contiguous()
        and expert_start.numel() == triton.next_power_of_2(num_experts) + 1
    ), "expert_start must be contiguous (next_power_of_2(num_experts) + 1,)"
    assert K == VALUES_PER_BYTE * K_b, (
        f"K (={K}) must equal {VALUES_PER_BYTE} * B.shape[2] (={K_b})"
    )
    assert K % MX_SCALE_GROUP_K == 0, (
        f"K (={K}) must be a multiple of {MX_SCALE_GROUP_K}"
    )
    assert Bs.shape == (num_experts, n_rows, K // MX_SCALE_GROUP_K), (
        f"Bs shape {tuple(Bs.shape)} != ({num_experts}, {n_rows}, {K // MX_SCALE_GROUP_K})"
    )

    output_dtype = resolve_output_dtype(output_dtype, A, As)
    # A raw (As is None) -> quantize here (offline MX); else pre-quantized (As given).
    if As is None:
        A, As = mxfp_act_quant(A)
    B = e2m1_as_uint8(B)
    bs_u8 = ue8m0_as_uint8(Bs)
    if requant:
        C = A.new_empty((S, N), dtype=FP8_DTYPE)
        Cs = torch.empty(
            S, N // MX_SCALE_GROUP_K, device=A.device, dtype=torch.uint8
        )
    else:
        C = A.new_empty((S, N), dtype=output_dtype)
        Cs = expert_start  # general dummy pointer; unread (REQUANT=False), strides literal
    num_sms = sm_count(A.device.index)

    with device_context(A.device):
        compile_time_only_triton_wrap(mxfp_dynamic_matmul_grouped_kernel)[(num_sms,)](
            A,
            As,
            B,
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
            num_experts=num_experts,
            tokens_per_expert_bit_length=int(
                (S + num_experts - 1) // num_experts
            ).bit_length(),
            HAS_GATHER=gather_idx is not None,
            HAS_SCATTER=scatter_idx is not None,
            NUM_EXPERTS_POW2=triton.next_power_of_2(num_experts),
            NUM_SMS=num_sms,
            VALUES_PER_BYTE=VALUES_PER_BYTE,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
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
    add_op_namespace_prefix("full_precision_matmul_grouped"), mutates_args=()
)
def full_precision_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    expert_start: torch.Tensor,
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
    assert A.ndim == 2, f"A must be 2D (S, K), got ndim={A.ndim}"
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

    _, K = A.shape
    # S = routed rows (num_tokens * top_k), carried by the (S,) perms — A's rows are
    # gather SOURCES and under-count S whenever top_k > 1 (gate_up reading raw hidden).
    # Only with no perms at all is A itself the expert-sorted (S, K) matrix.
    if gather_idx is not None:
        S = gather_idx.numel()
    elif scatter_idx is not None:
        S = scatter_idx.numel()
    else:
        S = A.shape[0]
    for perm_map in (gather_idx, scatter_idx):
        assert perm_map is None or (perm_map.numel() == S and perm_map.is_contiguous())

    # Under a gate epilogue B is the (E, 2N, K) gate|up stack — N is the per-projection width.
    num_experts, n_rows, _ = B.shape
    N = n_rows // 2 if gate else n_rows
    assert (
        expert_start.is_contiguous()
        and expert_start.numel() == triton.next_power_of_2(num_experts) + 1
    ), "expert_start must be contiguous (next_power_of_2(num_experts) + 1,)"

    output_dtype = resolve_output_dtype(output_dtype, A, None)
    C = A.new_empty(S, N, dtype=output_dtype)
    num_sms = sm_count(A.device.index)

    with device_context(A.device):
        compile_time_only_triton_wrap(full_precision_matmul_grouped_kernel)[(num_sms,)](
            A,
            B,
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
            tokens_per_expert_bit_length=int(
                (S + num_experts - 1) // num_experts
            ).bit_length(),
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
    As: torch.Tensor | None,
    B: torch.Tensor,
    Bs: torch.Tensor | None,
    expert_start: torch.Tensor,
    block_size: list[int] | None = None,
    epilogue: Epilogue | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Grouped matmul dispatcher (W8A8 FP8, W4A8 FP4, or full-precision). ``A`` arrives
    pre-quantized with its scales ``As`` — the caller owns the act-quant (``fp8_act_quant_block_dynamic`` /
    ``fp8_act_quant_tensor_wide`` / ``mxfp_act_quant`` per recipe); the op is a pure GEMM.
    ``expert_start`` is the ``(E+1,)`` tiling schedule from one ``compute_grouped_scheduling`` pass,
    shared by every grouped GEMM of the layer (the expert sort is virtual — nothing is physically
    permuted).

    Row order is carried by the standalone maps: ``gather_idx`` gathers ``A`` (``None`` -> already
    expert-ordered, no gather), ``scatter_idx`` scatters the output (``None`` -> stays
    expert-ordered, no scatter). ``epilogue`` is the fused output transform (gate|up + optional
    requant, and it carries ``output_dtype``); under a requant epilogue the return is the
    requantized output plus its scale tensor, and gate|up fusion supports the block-dynamic and MX
    recipes (tensor-wide gate|up is GLU-only — no fused requant). The unfused MoE chain is one
    scheduling pass: gate_up with ``scatter_idx=None`` (intermediate stays expert-ordered) +
    ``Epilogue(gate=True, requant=True)``, then down with ``gather_idx=None``. EP-sentinel routes
    fall past ``expert_start[-1]`` and are never touched (their output rows are uninitialized).

    Routes by weight dtype and ``block_size``:
    - ``Bs`` None — unquantized BF16/FP16 weights (``As`` must also be None) →
      ``full_precision_matmul_grouped`` (plain dot, no scales; GLU-only epilogue).
    - MX weights — ``int8`` (packed E2M1) or ``float8_e4m3fn`` (E4M3) with UE8M0
      group-32 ``Bs`` (shape ``[num_experts, N, K//32]``) → ``mxfp_dynamic_matmul_grouped``
      (``block_size`` ignored; tile + dot path autotuned).
    - ``block_size`` None or full ``[N, K]`` → ``w8a8_tensor_dynamic_fp8_matmul_grouped``.
    - otherwise → ``w8a8_block_dynamic_fp8_matmul_grouped``.
    """
    ep = epilogue if epilogue is not None else Epilogue()
    ep_args = ep.as_args()

    if Bs is None:
        assert As is None, "full-precision path (Bs=None) takes raw activations — As must be None"
        out = full_precision_matmul_grouped(
            A, B, expert_start, *ep_args, gather_idx, scatter_idx
        )
    elif is_mxfp(B, Bs):
        out = mxfp_dynamic_matmul_grouped(
            A, As, B, Bs, expert_start, *ep_args, gather_idx, scatter_idx
        )
    elif is_tensor_wide(block_size, B):
        out = w8a8_tensor_dynamic_fp8_matmul_grouped(
            A, As, B, Bs, expert_start, *ep_args, gather_idx, scatter_idx
        )
    else:
        out = w8a8_block_dynamic_fp8_matmul_grouped(
            A, As, B, Bs, expert_start, block_size, *ep_args, gather_idx, scatter_idx
        )
    # The ops return a list (torch custom ops can't return a Tensor-or-tuple union): [C] plain,
    # [C, Cs] under requant. Unwrap to the documented Tensor / (Tensor, Tensor) return.
    return out[0] if len(out) == 1 else tuple(out)
