# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from triton.tools.tensor_descriptor import TensorDescriptor

from ._ops import add_op_namespace_prefix
from .bayesian_autotuner import bayesian_autotune
from .utils import (
    Epilogue,
    Quantization,
    resolve_output_dtype,
    load_act,
    load_weight,
    accumulate,
    advance_ptrs,
    FP8_DTYPE,
    compile_time_only_triton_op,
    compile_time_only_triton_wrap,
    NIBBLES_PER_BYTE,
    acc_init,
    mx_config_pruner,
    smem_pruner,
    scalar_max_m_pruner,
    gate_pointer_only_pruner,
    block_within_dim_pruner,
    compose_pruners,
    descriptor_box_pruner,
    descriptor_needs_prequant_pruner,
    matched_memory_modes_pruner,
    device_context,
    fp8_act_quant_tensor_wide,
    fp8_act_quant_block_dynamic,
    operand_tile_ptrs,
    mx_2d_scale_ptrs,
    gate_stacked_block_scale_ptrs,
    matmul_weight_ptrs,
    gemm_epilogue,
    tl_dtype,
    operand_tile_descriptor,
    get_accelerator_autotuning_configs,
    warp_spec_compile_guard_pruner,
    is_mx,
    combine_global_scales,
    maybe_act_quant,
    MX_ACT_QUANT,
    mx_scale_family,
    resolve_input_recipe,
    store_masked,
    store_masked_oriented,
    swizzle_offsets,
    validate_dense_2d_operands,
    e2m1_as_uint8,
    ue8m0_as_uint8,
)

# The 2D-grid kernels' L2-locality swizzle depth is derived per-tile inside
# ``swizzle_offsets`` (see SWIZZLE_GROUP_A_BYTES there) — no per-kernel constant here.

# maybe_act_quant crossovers (min rows for offline pre-quant). MX: MEASURED — B200
# 2D-matmul sweep, graph-timed, H=6144 MXFP8 / H=4096 MXFP4: inline wins only at M=1
# (33 vs 44us / 20 vs 30us), offline from M=16 for MXFP8 (22 vs 33us), 2-3x by M>=64
# (MXFP4's M=16 cell marginally favors inline — outweighed); NVFP4 inherits the
# family gate (not swept — same pack/amax structure as MXFP4). STATIC: inherited
# estimate, not swept — its inline arm is cheaper elementwise work, so the true
# crossover is at or above the MX one; the M=1 decode case (the one that matters) is
# inline either way.
MX_MATMUL_ACT_PREQUANT_MIN_M = 16
STATIC_MATMUL_ACT_PREQUANT_MIN_M = 16
# Block-dynamic has NO gate: offline wins at EVERY M incl. M=1 (24.7 vs 31.0us, isolated
# arm A/B, graph-timed, H=6144 b128) — the inline arm pays a per-tile fp32 amax+div. The
# always-offline paths (bd 2D/batched/grouped, fused gate_ups, MX grouped) quantize
# unconditionally; only the two gates above have a real M=1 inline win.


def _rebind_operand_box(nargs, mode_key, desc_key, rows, cols):
    """Set one operand's host-TMA box to ``[rows, cols]`` in place — MUST mutate (a rebind never
    reaches the launch). No-op for a pointer config, whose descriptor is a dead int placeholder."""
    if nargs.get(mode_key, "pointer") != "pointer" and not isinstance(nargs[desc_key], int):
        nargs[desc_key].block_shape = [rows, cols]


def _rebind_bd_descriptors(nargs):
    """Per-config pre_hook: set the A and B host-TMA boxes to the tuned tile over the
    ``(rows, K)`` matrices — ``[BLOCK_SIZE_M, block_k]`` and ``[BLOCK_SIZE_N, block_k]``."""
    _rebind_operand_box(nargs, "A_MEMORY_MODE", "ADescriptor", nargs["BLOCK_SIZE_M"], nargs["block_k"])
    _rebind_operand_box(nargs, "B_MEMORY_MODE", "BDescriptor", nargs["BLOCK_SIZE_N"], nargs["block_k"])


def _rebind_mx_descriptors(nargs):
    """Per-config pre_hook for the MX kernel — set the A/B host-TMA boxes to the tuned tile
    in BYTES over the (rows, K_bytes) packed matrices: ``[BM, BK // ACT_VALUES_PER_BYTE]`` and
    ``[BN, BK // WEIGHT_VALUES_PER_BYTE]`` (values-per-byte read off the operand dtype; uint8 =
    packed E2M1), plus the SWIZZLE_32_4_4 scale boxes on the offline (pre-quantized A) path."""
    avpb = 2 if nargs["A"].dtype == torch.uint8 else 1
    wvpb = 2 if nargs["B"].dtype == torch.uint8 else 1
    bk = nargs["BLOCK_SIZE_K"]
    _rebind_operand_box(nargs, "A_MEMORY_MODE", "ADescriptor", nargs["BLOCK_SIZE_M"], bk // avpb)
    _rebind_operand_box(nargs, "B_MEMORY_MODE", "BDescriptor", nargs["BLOCK_SIZE_N"], bk // wvpb)
    # SWIZZLE_32_4_4 scale boxes: [1, BLOCK//128, (BK // SCALE_GROUP_K) // 4, 2, 256]. Only where the
    # scale is actually swizzled — else the SA/SB descriptor is a dummy aliased to the operand
    # descriptor, and stamping a scale box would clobber its [BM, BK] operand box. The weight is
    # swizzled iff SWIZZLED_SCALES; the act only when it was also offline-quantized (E4M3 / packed
    # E2M1) — inline (raw bf16 A) stays affine even under a swizzled weight.
    if nargs["SWIZZLED_SCALES"]:
        rep_k = (nargs["BLOCK_SIZE_K"] // nargs["SCALE_GROUP_K"]) // 4
        nargs["BSDescriptor"].block_shape = [1, nargs["BLOCK_SIZE_N"] // 128, rep_k, 2, 256]
        if nargs["A"].dtype in (torch.float8_e4m3fn, torch.uint8):
            nargs["ASDescriptor"].block_shape = [1, nargs["BLOCK_SIZE_M"] // 128, rep_k, 2, 256]
    # swizzled requant output (Cs is a descriptor): store tile [1, 1, rep_n, 2, 256], rep_n per config
    if nargs["CSDescriptor"] is not None:
        rep_n = (nargs["BLOCK_SIZE_N"] // nargs["SCALE_GROUP_K"]) // 4
        nargs["CSDescriptor"].block_shape = [1, 1, rep_n, 2, 256]


@bayesian_autotune(
    # tune_block_m: BLOCK_SIZE_M is a config axis. tune_block_n: the N tile is DECOUPLED
    # from the caller's scale granularity (block_n) — a BN=256 tile over 128-wide scale
    # columns halves activation re-reads; any BN is numerically fine (the offs_bn //
    # block_n gather spans or splits scale columns), while BK stays pinned to block_k
    # (activation scale groups are per block_k).
    #
    # Two memory regimes, tuner-routed per (N, K, M) key:
    #  - pointer: wins decode / short-K (big tiles starve here — the loop is load-bound).
    #  - host-TMA on BOTH operands (A_MEMORY_MODE / B_MEMORY_MODE = descriptor) in the NATURAL
    #    orientation (no SWAP_AB — the descriptor box transposes once, in load_weight_tile, to the
    #    pointer arm's K-major rhs): wins wide-N compute prefill, feeding the big tiles the pointer
    #    arm cannot. WS is load-bearing here: the per-iteration transpose races without it.
    # Emitting both axes lets the tuner route per regime. Verdicts B200 (sm_100) — re-chart on H100.
    get_accelerator_autotuning_configs(
        warp_spec=True,
        tune_block_m=True,
        tune_block_n=True,
        a_memory_modes=("descriptor", "pointer"),
        b_memory_modes=("descriptor", "pointer"),
        pre_hook=_rebind_bd_descriptors,
    ),
    # m_bit_length (log2 M bucket) keys the M tile, mirroring mx_dynamic_matmul_kernel; GATE keys the
    # gate|up arm separately (its stacked dot is 2*BN wide, a distinct config space).
    ["N", "K", "m_bit_length", "GATE"],
    n_trials=100,
    # WS compile guard + descriptor TMA-box limits (256/dim) + shared-memory fit (the big
    # TMA prefill tiles reach the smem ceiling; BM256 BN256 OOMs). The gate|up stack reads via the
    # pointer arm (a contiguous descriptor box can't span the N-apart gate/up rows), so prune the
    # B-descriptor configs under GATE.
    prune_configs_by={
        "early_config_prune": compose_pruners(
            warp_spec_compile_guard_pruner(),
            matched_memory_modes_pruner(),
            descriptor_box_pruner("block_k"),
            smem_pruner("block_k"),
            gate_pointer_only_pruner(),
        )
    },
)
@triton.jit
def w8a8_block_dynamic_fp8_matmul_kernel(
    A,  # (M, K) E4M3 activations (pre-quantized once by the wrapper)
    ADescriptor,  # host TMA descriptor over A (M, K), box (BLOCK_SIZE_M, block_k); read iff A_MEMORY_MODE != "pointer"
    As,  # (M, K // block_k) per-row, per-K-block activation scales (fp32 or uint8/UE8M0)
    B,  # (N, K) FP8 weights
    BDescriptor,  # host TMA descriptor over B (N, K), box (BLOCK_SIZE_N, block_k); read iff B_MEMORY_MODE != "pointer"
    Bs,  # (N // block_n, K // block_k) weight scales (fp32 or uint8/UE8M0)
    C,  # (M, N) output
    # Shape
    M,
    N,
    K,
    m_bit_length,  # autotune key only (log2 M bucket); unused in body
    # Strides
    stride_a_m,
    stride_a_k,
    stride_as_m,
    stride_b_k,
    stride_b_n,
    stride_bs_k,
    stride_bs_n,
    stride_c_m,
    stride_c_n,
    # Weight-quantization blocks (the caller's block_size); block_k is also the K tile
    # (the activation scale groups are per block_k)
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
    B_MEMORY_MODE: tl.constexpr = "pointer",
    A_MEMORY_MODE: tl.constexpr = "pointer",
    SWAP_AB: tl.constexpr = False,
    # Dense gate|up fusion: B is the (2N, K) gate|up stack, C the [M, N] GLU output (bf16 — the
    # requant output recipes live on the MX kernel). Every arm folds out at compile time when
    # GATE=False (the plain dense GEMM, unchanged).
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
):
    """Block-scale FP8 GEMM kernel.

    Computes ``C = A @ B.T``. Activations arrive pre-quantized (one pass in the
    wrapper — the inline per-N-tile quant would repeat N//BN times per element; see the
    grouped kernels). 2D grid with swizzle for L2 cache locality on B.

    Two operand-feed regimes, tuner-routed (see the decorator): explicit pointers, or
    host-TMA descriptors on both operands. Under the descriptor modes the natural
    orientation loads the ``(BN, BK)`` weight box and transposes it once (in
    ``load_weight_tile``) to the same K-major rhs the pointer arm builds — no ``SWAP_AB``
    (that trans is race-free only under ``WARP_SPEC``). ``SWAP_AB`` remains an
    independent pointer-arm orientation knob (weight tile in the MMA M dim, activation
    loaded transposed via strides); it is implemented but not currently emitted.

    UE8M0 scales (activations quantized to power-of-two, weights UE8M0) on a native-M tile
    (the MMA M operand is 128-wide: BLOCK_SIZE_M no-swap, BLOCK_SIZE_N swapped) fold the
    128-group scales into a tcgen05 ``dot_scaled`` MMA — see ``block_dynamic_dot``. fp32
    scales, or a narrow M tile (decode), keep the plain ``tl.dot`` + software rescale.
    """
    USE_DOT_SCALED: tl.constexpr = (As.dtype.element_ty == tl.uint8) and (
        (BLOCK_SIZE_N if SWAP_AB else BLOCK_SIZE_M) >= 128
    )
    pid_m, pid_n, offs_am, offs_bn, offs_k = swizzle_offsets(
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, block_k
    )
    # Scale pointers index off AFFINE row/col offsets + a bounds mask; the %-wrapped forms
    # (offs_am/offs_bn from swizzle_offsets) drive ONLY the pointer arm's operand tiles
    # (token replication at decode). Keeping scales affine avoids the non-affine gather
    # the wrap induces (~460us / 18pp at prefill).
    n_width: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
    offs_am_lin = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn_lin = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    as_ptrs = As + offs_am_lin * stride_as_m
    as_mask = offs_am_lin < M
    bs_mask = offs_bn_lin < N
    a_descriptor = operand_tile_descriptor(
        ADescriptor, A, M, K, stride_a_m, stride_a_k, BLOCK_SIZE_M, block_k, A_MEMORY_MODE
    )
    b_descriptor = operand_tile_descriptor(
        BDescriptor,
        B,
        N,
        K,
        stride_b_n,
        stride_b_k,
        BLOCK_SIZE_N,
        block_k,
        B_MEMORY_MODE,
    )
    # Explicit operand-tile pointers on the pointer arm; a scalar placeholder on a descriptor
    # arm (which reads its box via the descriptor — the [BM,BK]/[BK,BN] index tensor would stay
    # live across the K-loop for nothing and spill registers, ~460us / 20pp MFU at wide-N prefill).
    # Under GATE the weight is the stacked (2N, K) gate|up tile (matmul_weight_ptrs; pointer-only, the
    # descriptor box can't span the N-apart rows), and its per-weight-row block scale is a 2*BN gather
    # (gate rows [0,N), up rows offset by N//block_n scale blocks) — the block_dynamic_dot broadcast
    # then folds it per row exactly as in the dense case.
    a_ptrs = operand_tile_ptrs(A, offs_am, offs_k, stride_a_m, stride_a_k, A_MEMORY_MODE, not SWAP_AB)
    b_ptrs = matmul_weight_ptrs(B, offs_bn, offs_k, N, stride_b_n, stride_b_k, GATE, B_MEMORY_MODE, SWAP_AB)
    if GATE:
        bs_ptrs, bs_mask = gate_stacked_block_scale_ptrs(
            Bs, pid_n, N, block_n, stride_bs_n, BLOCK_SIZE_N, n_width
        )
    else:
        # the (BN,) scale-index gather decouples the tile from the scale grid: a wide tile
        # spans several scale columns, a narrow one shares a column
        bs_ptrs = Bs + (offs_bn_lin // block_n) * stride_bs_n
    accumulator = acc_init("dot", BLOCK_SIZE_M, n_width, SWAP_AB)

    for k in tl.range(0, tl.cdiv(K, block_k), warp_specialize=WARP_SPEC):
        a, a_s = load_act(
            "block_dynamic", a_ptrs, as_ptrs, None, None, as_mask, a_descriptor,
            pid_m * BLOCK_SIZE_M, k * block_k, A_MEMORY_MODE, SWAP_AB=SWAP_AB,
        )
        b, b_s = load_weight(
            "block_dynamic", b_ptrs, bs_ptrs, bs_mask, b_descriptor,
            pid_n * BLOCK_SIZE_N, k * block_k, B_MEMORY_MODE, SWAP_AB, GATE,
            BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=block_k,
        )
        accumulator = accumulate(
            accumulator, a, a_s, b, b_s, "block_dynamic", "dot", SWAP_AB, USE_DOT_SCALED,
            BLOCK_SIZE_M, n_width, block_k,
        )
        a_ptrs, as_ptrs, b_ptrs, bs_ptrs, _, _ = advance_ptrs(
            a_ptrs, as_ptrs, b_ptrs, bs_ptrs, b_ptrs, bs_ptrs,
            block_k * stride_a_k, 1, block_k * stride_b_k, stride_bs_k,
            A_MEMORY_MODE, B_MEMORY_MODE, True, True, False,
        )

    if GATE:
        gemm_epilogue(
            C, C, accumulator, offs_am_lin, pid_n, pid_m, as_mask,
            stride_c_m, stride_c_n, 0, 0,
            BLOCK_SIZE_M, BLOCK_SIZE_N, GATE, None, block_k,
            ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE,
            COMPUTE_MODE="dot", SWAP_AB=SWAP_AB, N_COLS=N,
        )
    else:
        store_masked_oriented(
            C,
            accumulator,
            pid_m,
            pid_n,
            M,
            N,
            stride_c_m,
            stride_c_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            SWAP_AB,
        )


@bayesian_autotune(
    # tune_block_m mirrors the block-dynamic kernel (adaptive BM=128 cost it 17% at prefill).
    get_accelerator_autotuning_configs(
        tune_block_nk=True, warp_spec=True, tune_block_m=True
    ),
    # GATE keys the gate|up arm separately (its stacked dot is 2*BN wide, a distinct config space).
    ["N", "K", "m_bit_length", "GATE"],
    n_trials=100,
    # block_k is a tuned axis and the loop below is maskless — veto non-dividing BKs;
    # WS is a pure perf axis here (non-WS is the validated state), compile-guarded.
    prune_configs_by={
        "early_config_prune": compose_pruners(
            block_within_dim_pruner("K"),
            warp_spec_compile_guard_pruner(),
        )
    },
)
@triton.jit
def w8a8_tensor_dynamic_fp8_matmul_kernel(
    A,  # (M, K) pre-quantized FP8 activations
    As,  # (M,) per-token activation scales
    B,  # (N, K) FP8 weights
    Bs,  # scalar/(1,) per-tensor weight scale
    C,  # (M, N) output
    # Shape
    M,
    N,
    K,
    m_bit_length,  # autotune key only (log2 M bucket); unused in body
    # Strides
    stride_a_m,
    stride_a_k,
    stride_as_m,
    stride_b_k,
    stride_b_n,
    stride_c_m,
    stride_c_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
    # Dense gate|up fusion: B is the (2N, K) gate|up stack, C the [M, N] GLU output. Every arm
    # folds out at compile time when GATE=False (the plain dense GEMM, unchanged).
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
):
    """Tensor-scale FP8 GEMM kernel.

    Computes ``C = A @ B.T`` with one activation scale per row and one
    weight scale for the full matrix.
    Uses a 2D grid with swizzle for L2 cache locality on B tiles.
    """
    n_width: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
    pid_m, pid_n, offs_am, offs_bn, offs_k = swizzle_offsets(
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    a_ptrs = operand_tile_ptrs(A, offs_am, offs_k, stride_a_m, stride_a_k, "pointer", True)
    # Under GATE the weight is the stacked (2N, K) gate|up tile; one per-tensor weight scale
    # covers both projections, applied (with the per-row act scale) before the GLU split.
    b_ptrs = matmul_weight_ptrs(B, offs_bn, offs_k, N, stride_b_n, stride_b_k, GATE, "pointer")

    a_s = tl.load(As + offs_am * stride_as_m)
    b_s = tl.load(Bs)

    # Accumulate raw dot products, apply scales once after the loop.
    accumulator = acc_init("dot", BLOCK_SIZE_M, n_width, False)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
        a, _as = load_act("tensor", a_ptrs, a_ptrs, None, None, None, a_ptrs, 0, 0, "pointer")
        b, _bs = load_weight(
            "tensor", b_ptrs, b_ptrs, None, b_ptrs, pid_n * BLOCK_SIZE_N, k * BLOCK_SIZE_K,
            "pointer", False, GATE, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        accumulator = accumulate(
            accumulator, a, _as, b, b_s, "tensor", "dot", False, False,
            BLOCK_SIZE_M, n_width, BLOCK_SIZE_K,
        )
        a_ptrs, _asp, b_ptrs, _bsp, _, _ = advance_ptrs(
            a_ptrs, a_ptrs, b_ptrs, b_ptrs, b_ptrs, b_ptrs,
            BLOCK_SIZE_K * stride_a_k, 0, BLOCK_SIZE_K * stride_b_k, 0,
            "pointer", "pointer", False, False, False,
        )

    accumulator = accumulator * a_s[:, None] * b_s

    if GATE:
        out_row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        gemm_epilogue(
            C, C, accumulator, out_row, pid_n, pid_m, out_row < M,
            stride_c_m, stride_c_n, 0, 0,
            BLOCK_SIZE_M, BLOCK_SIZE_N, GATE, None, BLOCK_SIZE_K,
            ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE,
            COMPUTE_MODE="dot", N_COLS=N,
        )
    else:
        store_masked(
            C,
            accumulator,
            pid_m,
            pid_n,
            M,
            N,
            stride_c_m,
            stride_c_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )


@bayesian_autotune(
    # tune_block_m + tune_block_n + WARP_SPEC + host-TMA memory modes mirror the block-dynamic
    # kernel above (same 2D swizzle loop physics). Both A/B memory axes are emitted and the tuner
    # routes descriptor vs pointer per (N, K, M) key; the A descriptor is read only on the offline
    # (pre-quantized fp8 A) arm — descriptor_needs_prequant_pruner keeps the M<threshold inline arm
    # pointer-only. The N tile is decoupled from the scale granularity.
    get_accelerator_autotuning_configs(
        warp_spec=True,
        tune_block_m=True,
        tune_block_n=True,
        a_memory_modes=("descriptor", "pointer"),
        b_memory_modes=("descriptor", "pointer"),
        pre_hook=_rebind_bd_descriptors,
    ),
    # GATE keys the gate|up arm separately (its stacked dot is 2*BN wide, a distinct config space).
    ["N", "K", "m_bit_length", "GATE"],
    n_trials=100,
    # The gate|up stack reads via the pointer arm (a contiguous descriptor box can't span the
    # N-apart gate/up rows), so prune the B-descriptor configs under GATE.
    prune_configs_by={
        "early_config_prune": compose_pruners(
            warp_spec_compile_guard_pruner(),
            descriptor_needs_prequant_pruner(),
            matched_memory_modes_pruner(),
            descriptor_box_pruner("block_k"),
            smem_pruner("block_k"),
            gate_pointer_only_pruner(),
        )
    },
)
@triton.jit
def w8a8_block_static_fp8_matmul_kernel(
    A,  # (M, K) E4M3 activations (pre-quantized against the static scale by the wrapper)
    ADescriptor,  # host TMA descriptor over A (M, K), box (BM, block_k); read iff A_MEMORY_MODE != "pointer"
    As,  # scalar — static per-tensor activation scale (calibration-time)
    B,  # (N, K) FP8 weights
    BDescriptor,  # host TMA descriptor over B (N, K), box (BN, block_k); read iff B_MEMORY_MODE != "pointer"
    Bs,  # (N // block_n, K // block_k) weight scales (fp32 or uint8/UE8M0)
    C,  # (M, N) output
    # Shape
    M,
    N,
    K,
    m_bit_length,  # autotune key only (log2 M bucket); unused in body
    # Strides
    stride_a_m,
    stride_a_k,
    stride_b_k,
    stride_b_n,
    stride_bs_k,
    stride_bs_n,
    stride_c_m,
    stride_c_n,
    # Weight-quantization blocks (see the block-dynamic kernel)
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
    A_MEMORY_MODE: tl.constexpr = "pointer",
    B_MEMORY_MODE: tl.constexpr = "pointer",
    # Dense gate|up fusion: B is the (2N, K) gate|up stack, C the [M, N] GLU output. Every arm
    # folds out at compile time when GATE=False (the plain dense GEMM, unchanged).
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
):
    """Block-scale FP8 GEMM with static (per-tensor) activation scale.

    ``A`` arrives pre-quantized (one elementwise ``(A / As).to(fp8)`` pass in the
    wrapper — an inline division would repeat per N-tile). Per-block weight scales apply
    per-K-tile during accumulation; the scalar activation scale is applied once at
    the end.
    """
    n_width: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
    pid_m, pid_n, offs_am, offs_bn, offs_k = swizzle_offsets(
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, block_k
    )
    a_descriptor = operand_tile_descriptor(
        ADescriptor, A, M, K, stride_a_m, stride_a_k, BLOCK_SIZE_M, block_k, A_MEMORY_MODE
    )
    b_descriptor = operand_tile_descriptor(
        BDescriptor, B, N, K, stride_b_n, stride_b_k, BLOCK_SIZE_N, block_k, B_MEMORY_MODE
    )
    a_ptrs = operand_tile_ptrs(A, offs_am, offs_k, stride_a_m, stride_a_k, A_MEMORY_MODE, True)
    # Weight-scale index off the AFFINE column offset + a bounds mask (the %-wrapped offs_bn
    # from swizzle_offsets drives only the pointer operand tile); an affine gather avoids the
    # non-affine scale read the wrap induces, matching the block-dynamic kernel above. Under GATE
    # the weight is the stacked (2N, K) gate|up tile (pointer-only) + a 2*BN per-weight-row scale
    # gather (gate rows [0,N), up rows offset by N//block_n scale blocks).
    b_ptrs = matmul_weight_ptrs(B, offs_bn, offs_k, N, stride_b_n, stride_b_k, GATE, B_MEMORY_MODE)
    if GATE:
        bs_ptrs, bs_mask = gate_stacked_block_scale_ptrs(
            Bs, pid_n, N, block_n, stride_bs_n, BLOCK_SIZE_N, n_width
        )
    else:
        offs_bn_lin = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        bs_ptrs = Bs + (offs_bn_lin // block_n) * stride_bs_n
        bs_mask = offs_bn_lin < N
    a_s_static = tl.load(As)

    accumulator = acc_init("dot", BLOCK_SIZE_M, n_width, False)
    for k in tl.range(0, tl.cdiv(K, block_k), warp_specialize=WARP_SPEC):
        a, _as = load_act(
            "static", a_ptrs, a_ptrs, None, None, None, a_descriptor, pid_m * BLOCK_SIZE_M, k * block_k,
            A_MEMORY_MODE, a_s_static=a_s_static,
        )
        b, b_s = load_weight(
            "static", b_ptrs, bs_ptrs, bs_mask, b_descriptor, pid_n * BLOCK_SIZE_N, k * block_k,
            B_MEMORY_MODE, False, GATE, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=block_k,
        )
        accumulator = accumulate(
            accumulator, a, _as, b, b_s, "static", "dot", False, False,
            BLOCK_SIZE_M, n_width, block_k,
        )
        a_ptrs, _asp, b_ptrs, bs_ptrs, _, _ = advance_ptrs(
            a_ptrs, a_ptrs, b_ptrs, bs_ptrs, b_ptrs, bs_ptrs,
            block_k * stride_a_k, 0, block_k * stride_b_k, stride_bs_k,
            A_MEMORY_MODE, B_MEMORY_MODE, False, True, False,
        )

    accumulator = accumulator * a_s_static
    if GATE:
        out_row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        gemm_epilogue(
            C, C, accumulator, out_row, pid_n, pid_m, out_row < M,
            stride_c_m, stride_c_n, 0, 0,
            BLOCK_SIZE_M, BLOCK_SIZE_N, GATE, None, block_k,
            ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE,
            COMPUTE_MODE="dot", N_COLS=N,
        )
    else:
        store_masked(
            C,
            accumulator,
            pid_m,
            pid_n,
            M,
            N,
            stride_c_m,
            stride_c_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )


@bayesian_autotune(
    # tune_block_m: BLOCK_SIZE_M becomes a config axis, so the tuner sizes the M tile
    # per workload — small at decode, large at prefill.
    # scalar is in the mode set for M=1 decode (attn projection): it avoids the MMA M->16 pad
    # that held MXFP8 attn decode 35% over block-dynamic at identical weight bytes; fp4-scalar
    # is dropped by the pruner (ALU-bound unpack). swap_ab intentionally OFF: an 18-cell forced-swap sweep (cudagraph,
    # M3+dsv4 decode) showed swap losing on the single matmul (adaptive BM>=16 fills the MMA atom;
    # M3 attn swap was −38%) while the tuner never picked it — emitting the configs only bloats
    # the search. Swap stays on the batched/fused experts kernels, where it wins ~30% on dsv4.
    get_accelerator_autotuning_configs(
        mx=True,
        tune_block_nk=True,
        compute_modes=("dot_scaled", "dot", "scalar"),
        tune_block_m=True,
        # host-TMA on both operands, tuner-picked vs pointers (measured +14-17pp: mxfp8
        # 54->71%, W4A8 50->67% via mixed A=ptr/B=desc, W4A4 +14%). Mixed A/B modes KEPT
        # (unlike dense bd): A=pointer + B=descriptor wins the mid-M band.
        a_memory_modes=("descriptor", "pointer"),
        b_memory_modes=("descriptor", "pointer"),
        pre_hook=_rebind_mx_descriptors,
        # WS tuner axis: dot_scaled + WS + TMA compiles + wins at prefill (1918 -> ~1995 @M=4k);
        # warp_spec_compile_guard_pruner keeps it to num_warps%4==0, BM>=64.
        warp_spec=True,
    ),
    # the MXFP4/MXFP8 split keys itself — the tuner appends every tensor arg's dtype to
    # its cache key (memory and disk);
    # m_bit_length (log2 M bucket) keys the M tile — the winner keeps shifting with M well past the
    # BM ceiling and it is NOT noise: cross-applying configs (N=K=4096) costs +62% at M=128 and +245%
    # at M=4096 (the thin M=128 tile can't saturate the wide GEMM), so don't collapse the buckets.
    # INPUT_RECIPE keys the inline act-quant grid: A stays raw bf16 under every
    # recipe below the pre-quant M threshold, so the dtype-appended key can't
    # split W4A8 from W4A4 itself.
    ["N", "K", "m_bit_length", "INPUT_RECIPE", "SWIZZLED_SCALES", "GATE"],
    n_trials=100,
    # BK-within-K veto (the loop loads are unmasked) + the sm_10x dot_scaled shape guards
    # + scalar restricted to decode-sized M (a BM=1 GEVM at prefill is TPE poison) + the
    # gate|up stack reads via the pointer arm (the descriptor's one contiguous box can't span
    # the N-apart gate/up rows — a (2,N,K) box is a follow-up), so prune descriptor under GATE.
    prune_configs_by={
        "early_config_prune": compose_pruners(
            mx_config_pruner("K"),
            scalar_max_m_pruner("M"),
            smem_pruner(),
            descriptor_box_pruner("BLOCK_SIZE_K"),
            gate_pointer_only_pruner(),
            warp_spec_compile_guard_pruner(),
        )
    },
)
@triton.jit
def mx_dynamic_matmul_kernel(
    A,  # (M, K) activations: E4M3 (pre-quantized) or raw bf16/fp16 (quantized inline)
    ADescriptor,  # host TMA descriptor over A (M, K_bytes), box (BM, BK_bytes); read iff A_MEMORY_MODE != "pointer"
    As,  # activation scales: SWIZZLE_32_4_4 (swizzled arm — bulk via ASDescriptor at BM=128, or this pointer for the BM<128 gather) or (M, K // 32) UE8M0 affine
    ASDescriptor,  # host TMA descriptor over the SWIZZLE_32_4_4 A scales (BM=128 bulk load); read iff SWIZZLED_SCALES
    B,  # (N, K) E4M3 (MXFP8) or (N, K // 2) packed E2M1 (MXFP4) weights
    BDescriptor,  # host TMA descriptor over B (N, K_bytes), box (BN, BK_bytes); read iff B_MEMORY_MODE != "pointer"
    Bs,  # weight scales: SWIZZLE_32_4_4 (swizzled arm — bulk via BSDescriptor at BN=128, or this pointer for the BN<128 gather) or (N, K // SCALE_GROUP_K) affine
    BSDescriptor,  # host TMA descriptor over the SWIZZLE_32_4_4 B scales (BN=128 bulk load); read iff SWIZZLED_SCALES
    C,  # (M, N) output (the GLU intermediate under GATE)
    Cs,  # (M, N // group) row-major requant output scale — written iff OUTPUT_RECIPE and not SWIZZLED_OUT
    CSDescriptor,  # SWIZZLE_32_4_4 requant-scale descriptor — written iff SWIZZLED_OUT (dummy else), like AS/BS
    AsGlobal,  # (1,) fp32 NVFP4 activation global g_a — SOLELY normalizes the inline raw-A quant (A/g_a); read iff not None
    AsBsGlobal,  # (1,) fp32 NVFP4 combined global g_a·g_b — recovers on the accumulator (one multiply); read iff not None
    CsGlobal,  # (1,) fp32 NVFP4 output global (next proj's provided input_scale); normalizes the requant; read iff not None
    # Shape
    M,
    N,
    K,
    m_bit_length,  # autotune key only (log2 M bucket); unused in body
    # Strides
    stride_a_m,
    stride_a_k,
    stride_as_m,
    stride_b_k,
    stride_b_n,
    stride_bs_k,
    stride_bs_n,
    stride_c_m,
    stride_c_n,
    stride_cs_m,  # requant output-scale strides (dead unless OUTPUT_RECIPE)
    stride_cs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
    A_MEMORY_MODE: tl.constexpr = "pointer",
    B_MEMORY_MODE: tl.constexpr = "pointer",
    INPUT_RECIPE: tl.constexpr = "mxfp8",
    SWIZZLED_SCALES: tl.constexpr = False,  # scales pre-swizzled (5D weight + matching acts); else affine/inline
    SWIZZLED_OUT: tl.constexpr = False,  # requant emits Cs in SWIZZLE_32_4_4 (CSDescriptor); single source: wrapper
    # Dense gate|up fusion: B is the (2N, K) gate|up stack, C the [M, N] GLU output. Every arm
    # folds out at compile time when GATE=False (the plain dense GEMM, unchanged).
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    OUTPUT_RECIPE: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
    WARP_SPEC: tl.constexpr = False,  # tuner axis; +4% at prefill (guard: num_warps%4, BM>=64)
):
    """Unified MXFP4/MXFP8 (W4A8/W8A8; a ``uint8`` ``A`` is packed E2M1 — W4A4) GEMM.

    ``C = A @ B.T``. ``A``'s dtype picks the activation form (compile-time folded):
    pre-quantized E4M3 + UE8M0 group-32 scales (one wrapper pass — the inline
    per-N-tile quant re-ran N//BN times per element, ~2x at prefill) vs raw bf16/fp16
    quantized inline (small M: the GEMM is weight-bandwidth-bound with idle ALU, so
    inline is free and a separate quant kernel only adds latency; both forms are
    bit-exact, same group-32 boundaries). Each operand's format is its dtype (``uint8``
    = packed E2M1, two values per byte; else E4M3). ``COMPUTE_MODE`` picks the MMA:
    ``tl.dot_scaled`` (native M=128 scaled MMA) vs fp8 ``tl.dot`` + per-group software
    rescale (wins at decode; FP4 unpacks E2M1->E4M3 first — lossless). 2D grid with
    swizzle for L2 reuse.
    """
    # uint8 A = packed-E2M1 activations (W4A4 — the dtype IS the activation format)
    ACT_VALUES_PER_BYTE: tl.constexpr = 2 if A.dtype.element_ty == tl.uint8 else 1
    WEIGHT_VALUES_PER_BYTE: tl.constexpr = 2 if B.dtype.element_ty == tl.uint8 else 1
    n_width: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
    SCALE_COLS: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
    offs_ka = tl.arange(0, BLOCK_SIZE_K // ACT_VALUES_PER_BYTE)
    offs_kb = tl.arange(0, BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE)
    # packed-fp4 weights (WEIGHT_VALUES_PER_BYTE==2) route swizzle_offsets to full L2 grouping
    pid_m, pid_n, offs_am, offs_bn, offs_k = swizzle_offsets(
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WEIGHT_VALUES_PER_BYTE
    )
    # Scales read affine off row/col offsets + bounds masks (the %-wrapped operand offsets would
    # make the scale load a non-affine gather); the SWIZZLED arm reads them via the SA/BS
    # descriptors in the loop instead, so its pointer tiles are dead (base scalars + null masks).
    as_ptrs, bs_ptrs, as_mask, bs_mask = mx_2d_scale_ptrs(
        As, Bs, pid_m, pid_n, M, N, stride_as_m, stride_bs_n, stride_bs_k,
        BLOCK_SIZE_M, BLOCK_SIZE_N, SCALE_COLS, SWIZZLED_SCALES,
    )
    # Operand tiles: activation + the (GATE-stacked-aware) weight, both single leaf calls (the
    # descriptor-vs-pointer and gate|up-stack branches fold inside the leaves).
    a_ptrs = operand_tile_ptrs(A, offs_am, offs_ka, stride_a_m, stride_a_k, A_MEMORY_MODE, True)
    b_ptrs = matmul_weight_ptrs(B, offs_bn, offs_kb, N, stride_b_n, stride_b_k, GATE, B_MEMORY_MODE)

    accumulator = acc_init("dot", BLOCK_SIZE_M, n_width, False)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
        a, a_s = load_act(
            "mx", a_ptrs, as_ptrs, AsGlobal, None, as_mask, ADescriptor, pid_m * BLOCK_SIZE_M,
            k * (BLOCK_SIZE_K // ACT_VALUES_PER_BYTE), A_MEMORY_MODE,
            as_descriptor=ASDescriptor, as_ptr=As, pid_m=pid_m, k=k, M=M, K=K,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_K=BLOCK_SIZE_K, SCALE_GROUP_K=SCALE_GROUP_K,
            SWIZZLED_SCALES=SWIZZLED_SCALES, INPUT_RECIPE=INPUT_RECIPE,
        )
        b, b_s = load_weight(
            "mx", b_ptrs, bs_ptrs, bs_mask, BDescriptor, pid_n * BLOCK_SIZE_N,
            k * (BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE), B_MEMORY_MODE, False,
            bs_descriptor=BSDescriptor, bs_ptr=Bs, pid_n=pid_n, k=k, N=N, K=K,
            BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, SCALE_GROUP_K=SCALE_GROUP_K,
            SWIZZLED_SCALES=SWIZZLED_SCALES, GATE=GATE,
            WEIGHT_VALUES_PER_BYTE=WEIGHT_VALUES_PER_BYTE,
            stride_bs_n=stride_bs_n, stride_bs_k=stride_bs_k,
        )
        accumulator = accumulate(
            accumulator, a, a_s, b, b_s, "mx", COMPUTE_MODE, False, False,
            BLOCK_SIZE_M, n_width, BLOCK_SIZE_K, SCALE_GROUP_K,
        )
        a_ptrs, as_ptrs, b_ptrs, bs_ptrs, _, _ = advance_ptrs(
            a_ptrs, as_ptrs, b_ptrs, bs_ptrs, b_ptrs, bs_ptrs,
            (BLOCK_SIZE_K // ACT_VALUES_PER_BYTE) * stride_a_k,
            BLOCK_SIZE_K // SCALE_GROUP_K,
            (BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE) * stride_b_k,
            (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_bs_k,
            A_MEMORY_MODE, B_MEMORY_MODE,
            not SWIZZLED_SCALES, (not SWIZZLED_SCALES) and not GATE, False,
        )

    # NVFP4 two-level: block e4m3 scales rode through dot_scaled; recover the combined per-tensor
    # global g_a·g_b on the accumulator — one multiply (g_a alone is used only by the inline-quant
    # arm). None folds out at trace time.
    if AsBsGlobal is not None:
        accumulator = accumulator * tl.load(AsBsGlobal).to(tl.float32)

    # one epilogue for both arms: GATE splits+SwiGLUs (+ optional requant), plain casts+stores.
    # affine output rows (swizzle's offs_am is %-wrapped — the store scatters to real rows);
    # N_COLS masks the 2D-dense column tail.
    out_row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    gemm_epilogue(
        C, Cs, accumulator, out_row, pid_n, pid_m, out_row < M,
        stride_c_m, stride_c_n, stride_cs_m, stride_cs_n,
        BLOCK_SIZE_M, BLOCK_SIZE_N, GATE, OUTPUT_RECIPE, SCALE_GROUP_K,
        ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE,
        COMPUTE_MODE=COMPUTE_MODE, N_COLS=N, SWIZZLED_OUT=SWIZZLED_OUT, CSDescriptor=CSDescriptor,
        CsGlobal=CsGlobal,
    )


@bayesian_autotune(
    # Same 2D-swizzle loop physics as the tensor kernel — tune the tile, WS a pure perf axis
    # (compile-guarded), no memory-mode/scale axes (unquantized, no scales anywhere).
    get_accelerator_autotuning_configs(tune_block_nk=True, warp_spec=True, tune_block_m=True),
    # GATE keys the gate|up arm separately (its stacked dot is 2*BN wide).
    ["N", "K", "m_bit_length", "GATE"],
    n_trials=100,
    prune_configs_by={
        "early_config_prune": compose_pruners(
            block_within_dim_pruner("K"),
            warp_spec_compile_guard_pruner(),
        )
    },
)
@triton.jit
def full_precision_matmul_2d_kernel(
    A,  # (M, K) BF16/FP16 activations
    B,  # (N, K) weights in A's dtype; under GATE the (2N, K) gate|up stack
    C,  # (M, N) output; under GATE the GLU intermediate
    # Shape
    M,
    N,
    K,
    m_bit_length,  # autotune key only (log2 M bucket); unused in body
    # Strides
    stride_a_m,
    stride_a_k,
    stride_b_k,
    stride_b_n,
    stride_c_m,
    stride_c_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
    # Dense gate|up fusion: B is the (2N, K) gate|up stack, C the [M, N] GLU output. Folds out at
    # compile time when GATE=False (the plain dense GEMM).
    GATE: tl.constexpr = False,
    ACT_FN: tl.constexpr = "silu",
    SWIGLU_ALPHA: tl.constexpr = None,
    SWIGLU_LIMIT: tl.constexpr = None,
    SIMULATE_UNFUSED: tl.constexpr = False,
    INTERMEDIATE_DTYPE: tl.constexpr = tl.bfloat16,
):
    """Full-precision (BF16/FP16) 2D matmul — the single-GEMM sibling of
    ``full_precision_matmul_grouped/batched``: plain ``tl.dot``, fp32 accumulation, no scales.
    ``GATE`` loads the stacked (2N, K) gate|up weight as one ``[BK, 2*BN]`` dot and applies the
    ``ACT_FN``/SwiGLU GLU; ``GATE=False`` is the plain dense GEMM. 2D grid with swizzle for L2 reuse."""
    n_width: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
    pid_m, pid_n, offs_am, offs_bn, offs_k = swizzle_offsets(
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    a_ptrs = operand_tile_ptrs(A, offs_am, offs_k, stride_a_m, stride_a_k, "pointer", True)
    b_ptrs = matmul_weight_ptrs(B, offs_bn, offs_k, N, stride_b_n, stride_b_k, GATE, "pointer")

    accumulator = acc_init("dot", BLOCK_SIZE_M, n_width, False)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
        a = tl.load(a_ptrs)
        w, _ = load_weight(
            "full_precision", b_ptrs, b_ptrs, None, b_ptrs, pid_n * BLOCK_SIZE_N, k * BLOCK_SIZE_K,
            "pointer", False, GATE, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        accumulator = accumulate(
            accumulator, a, a, w, w, "full_precision", "dot", False, False,
            BLOCK_SIZE_M, n_width, BLOCK_SIZE_K,
        )
        a_ptrs += BLOCK_SIZE_K * stride_a_k
        b_ptrs += BLOCK_SIZE_K * stride_b_k

    if GATE:
        out_row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        gemm_epilogue(
            C, C, accumulator, out_row, pid_n, pid_m, out_row < M,
            stride_c_m, stride_c_n, 0, 0,
            BLOCK_SIZE_M, BLOCK_SIZE_N, GATE, None, BLOCK_SIZE_K,
            ACT_FN, SWIGLU_ALPHA, SWIGLU_LIMIT, SIMULATE_UNFUSED, INTERMEDIATE_DTYPE,
            COMPUTE_MODE="dot", N_COLS=N,
        )
    else:
        store_masked(
            C, accumulator, pid_m, pid_n, M, N, stride_c_m, stride_c_n,
            BLOCK_SIZE_M, BLOCK_SIZE_N,
        )


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_block_dynamic_fp8_matmul"),
    mutates_args=(),
    opaque=True,
)
def w8a8_block_dynamic_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype | None = None,
    gate: bool = False,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
) -> list[torch.Tensor]:
    """Block-scale FP8 matmul: ``C = A @ B.T``; activations quantized offline in one pass.

    A:  (..., K) raw activations, bf16/fp16/fp32 (quantized to FP8 in one wrapper pass)
    B:  (N, K) FP8 weights — under ``gate`` the ``(2N, K)`` gate|up stack (gate rows [0,N), up [N,2N))
    Bs: (N // block_n, K // block_k) per-block weight scales (2N rows under ``gate``)

    ``gate`` fuses the gate|up projection into one stacked GEMM + SwiGLU, returning the
    ``[..., N]`` GLU intermediate (``output_dtype``); the fused-requant output recipes stay on
    the MX path. Returns a one-element list (mirrors the MX/grouped/batched op signature).
    """
    assert len(block_size) == 2, (
        f"block_size must be [block_n, block_k], got {block_size}"
    )
    block_n, block_k = block_size[0], block_size[1]

    validate_dense_2d_operands(A, B)

    rows, K = B.shape
    # Under gate|up fusion B is the (2N, K) gate|up stack; N is the per-projection output width.
    N = rows // 2 if gate else rows
    M = A.numel() // A.shape[-1]
    assert K % block_k == 0, f"K ({K}) must be divisible by block_k ({block_k})"
    assert not gate or N % block_n == 0, (
        f"gate|up fusion needs N ({N}) divisible by block_n ({block_n}) — the up-proj scale block"
    )

    assert Bs.ndim == 2, f"Bs must be 2D (rows//block_n, K//block_k), got ndim={Bs.ndim}"
    assert Bs.shape == (triton.cdiv(rows, block_n), K // block_k), (
        f"Bs shape {tuple(Bs.shape)} != expected ({triton.cdiv(rows, block_n)}, {K // block_k})"
    )

    bs_u8 = ue8m0_as_uint8(Bs)
    # UE8M0 weight scales are the DeepGEMM-Blackwell recipe — quantize activations to UE8M0
    # too so the kernel folds both group scales into the tcgen05 dot_scaled MMA (else fp32).
    A_q, A_s = fp8_act_quant_block_dynamic(
        A.view(M, K), block_k, use_ue8m0=bs_u8.dtype == torch.uint8
    )
    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)
    # Host TMA descriptors over the (M, K) / (N, K) matrices — the placeholder box is
    # re-bound per tuned config by _rebind_bd_descriptors. Read only by the descriptor
    # (host-TMA) configs the tuner picks for wide-N prefill; pointer configs never touch
    # them (the constexpr arm folds out in the load helpers).
    a_descriptor = TensorDescriptor.from_tensor(A_q, [1, block_k])
    b_descriptor = TensorDescriptor.from_tensor(B, [1, block_k])

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_block_dynamic_fp8_matmul_kernel)[grid](
            A_q,
            a_descriptor,
            A_s,
            B,
            b_descriptor,
            bs_u8,
            C,
            M,
            N,
            K,
            int(M).bit_length(),
            A_q.stride(0),
            A_q.stride(1),
            A_s.stride(0),
            B.stride(1),
            B.stride(0),
            bs_u8.stride(1),
            bs_u8.stride(0),
            C.stride(-2),
            C.stride(-1),
            block_n=block_n,
            block_k=block_k,
            GATE=gate,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(resolve_output_dtype(output_dtype, A, None)),
        )

    return [C]


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_block_static_fp8_matmul"),
    mutates_args=(),
    opaque=True,
)
def w8a8_block_static_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype | None = None,
    gate: bool = False,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
) -> list[torch.Tensor]:
    """Block-scale FP8 matmul with static (per-tensor) activation quantization.

    A:  (..., K) raw bf16/fp16 activations — pre-quantized against ``As`` in the wrapper
    B:  (N, K) FP8 weights — under ``gate`` the ``(2N, K)`` gate|up stack
    As: scalar / (1,) — per-tensor static activation scale
    Bs: (N // block_n, K // block_k) per-block weight scales (2N rows under ``gate``)

    ``gate`` fuses the gate|up projection into one stacked GEMM + SwiGLU, returning the
    ``[..., N]`` GLU intermediate. Returns a one-element list (mirrors the MX/grouped op).
    """
    assert len(block_size) == 2, (
        f"block_size must be [block_n, block_k], got {block_size}"
    )
    block_n, block_k = block_size[0], block_size[1]

    assert B.dtype != torch.int8, (
        "static activation quant is not supported on the FP4 path"
    )
    assert not (block_n == B.size(0) and block_k == B.size(1)), (
        "static activation quant requires block-wise weights, not tensor-mode"
    )
    validate_dense_2d_operands(A, B)
    assert As.numel() == 1, f"As must be scalar or (1,), got {tuple(As.shape)}"

    rows, K = B.shape
    # Under gate|up fusion B is the (2N, K) gate|up stack; N is the per-projection output width.
    N = rows // 2 if gate else rows
    M = A.numel() // A.shape[-1]

    assert Bs.ndim == 2, f"Bs must be 2D (rows//block_n, K//block_k), got ndim={Bs.ndim}"
    assert K % block_k == 0, f"K ({K}) must be divisible by block_k ({block_k})"
    assert not gate or N % block_n == 0, (
        f"gate|up fusion needs N ({N}) divisible by block_n ({block_n}) — the up-proj scale block"
    )
    assert Bs.shape == (triton.cdiv(rows, block_n), K // block_k), (
        f"Bs shape {tuple(Bs.shape)} != expected ({triton.cdiv(rows, block_n)}, {K // block_k})"
    )

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    bs_u8 = ue8m0_as_uint8(Bs)
    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)
    As = As.reshape(1).to(torch.float32)
    # M-gated static pre-quant (bit-exact with the inline arm: same scalar, same cast);
    # like MX, the inline form is cheap elementwise work — at M=1 a separate kernel is
    # pure added latency. The kernel picks its arm off A's dtype.
    A_q, _ = maybe_act_quant(
        A.view(M, K),
        lambda x: ((x.to(torch.float32) / As).to(FP8_DTYPE), As),
        STATIC_MATMUL_ACT_PREQUANT_MIN_M,
    )
    # Host-TMA descriptors over (M, K)/(N, K); placeholder boxes re-bound per config by
    # _rebind_bd_descriptors. Read only by the descriptor configs the tuner picks for wide-N
    # prefill (offline fp8 A); pointer/inline configs never touch them.
    a_descriptor = TensorDescriptor.from_tensor(A_q, [1, block_k])
    b_descriptor = TensorDescriptor.from_tensor(B, [1, block_k])

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_block_static_fp8_matmul_kernel)[grid](
            A_q,
            a_descriptor,
            As,
            B,
            b_descriptor,
            bs_u8,
            C,
            M,
            N,
            K,
            int(M).bit_length(),  # m_bit_length key bucket
            A_q.stride(0),
            A_q.stride(1),
            B.stride(1),
            B.stride(0),
            bs_u8.stride(1),
            bs_u8.stride(0),
            C.stride(-2),
            C.stride(-1),
            # Meta-parameters (BM and BN come from the config; BK is the caller's
            # block_k — see the block-dynamic kernel)
            block_n=block_n,
            block_k=block_k,
            GATE=gate,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(resolve_output_dtype(output_dtype, A, None)),
        )

    return [C]


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_tensor_dynamic_fp8_matmul"),
    mutates_args=(),
    opaque=True,
)
def w8a8_tensor_dynamic_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype | None = None,
    gate: bool = False,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
) -> list[torch.Tensor]:
    """Tensor-scale FP8 matmul: ``C = A @ B.T``; activations quantized offline per row.

    A:  (..., K) raw activations, bf16/fp16/fp32 (flattened to (M, K)
        internally) — per-row scales computed via ``fp8_act_quant_tensor_wide(A, K)``.
    B:  (N, K) FP8 weights — under ``gate`` the ``(2N, K)`` gate|up stack (one per-tensor scale).
    Bs: scalar, (1,), or (1, 1) — single tensor-scale weight scale.

    ``gate`` fuses the gate|up projection into one stacked GEMM + SwiGLU, returning the
    ``[..., N]`` GLU intermediate. Returns a one-element list (mirrors the MX/grouped op).
    """
    validate_dense_2d_operands(A, B)

    rows, K = B.shape
    # Under gate|up fusion B is the (2N, K) gate|up stack; N is the per-projection output width.
    N = rows // 2 if gate else rows
    M = A.numel() // A.shape[-1]

    assert Bs.numel() == 1, f"Bs must be scalar or (1,), got {tuple(Bs.shape)}"

    # Per-row scalar activation scale (one per token).
    qA, As = fp8_act_quant_tensor_wide(A, K)
    As = As.reshape(M)
    Bs = Bs.reshape(1)

    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_tensor_dynamic_fp8_matmul_kernel)[grid](
            qA,
            As,
            B,
            Bs,
            C,
            M,
            N,
            K,
            int(M).bit_length(),  # m_bit_length key bucket
            qA.stride(-2),
            qA.stride(-1),
            As.stride(0),
            B.stride(1),
            B.stride(0),
            C.stride(-2),
            C.stride(-1),
            GATE=gate,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(resolve_output_dtype(output_dtype, A, None)),
        )

    return [C]


@compile_time_only_triton_op(
    add_op_namespace_prefix("mx_dynamic_matmul"), mutates_args=(), opaque=True
)
def mx_dynamic_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor | None,
    Bs: torch.Tensor,
    output_dtype: torch.dtype | None = None,
    input_recipe: str | None = None,
    gate: bool = False,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    output_recipe: str | None = None,
    a_global_scale: torch.Tensor | None = None,
    b_global_scale: torch.Tensor | None = None,
    output_global_scale: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """MX/NVFP4 matmul ``C = A @ B.T``; activations quantized offline above the
    ``maybe_act_quant`` M threshold, inline below it. ``input_recipe`` sets the
    activation grid — E4M3 (``"mxfp8"``, the default) or packed E2M1 (``"mxfp4"``,
    W4A4); NVFP4 weights (E4M3 scales) pin ``"nvfp4"``. Weight format detected from
    ``B.dtype``: ``int8`` → packed E2M1 (``B`` is ``(N, K//2)``); ``float8_e4m3fn`` →
    unpacked E4M3. The scale dtype carries the family: UE8M0 = MX group-32, E4M3 =
    NVFP4 group-16 (``Bs`` is ``(N, K//group)``); tile + dot path are autotuned.

    A:  (..., K) raw activations, bf16/fp16/fp32;
        leading dims are flattened to (M, K) and restored on the output
    """
    assert B.ndim == 2 and Bs.ndim in (2, 5)  # 5D Bs = pre-swizzled SWIZZLE_32_4_4 weight scales
    assert B.dtype in (torch.int8, torch.float8_e4m3fn), (
        f"B must be int8 (packed E2M1) or float8_e4m3fn (E4M3), got {B.dtype}"
    )
    assert A.is_contiguous(), "A must be contiguous"
    assert B.is_contiguous(), "B must be contiguous"
    WEIGHT_VALUES_PER_BYTE = NIBBLES_PER_BYTE if B.dtype == torch.int8 else 1

    rows, K_b = B.shape
    K = A.shape[-1]
    M = A.numel() // K
    # Under gate|up fusion B is the (2N, K) gate|up stack; N is the per-projection output width.
    N = rows // 2 if gate else rows
    assert K == WEIGHT_VALUES_PER_BYTE * K_b, (
        f"K (={K}) must equal {WEIGHT_VALUES_PER_BYTE} * B.shape[1] (={K_b})"
    )
    # Weight scales arrive row-major (rows, K // scale_group) — read affine, no gain swizzling only
    # one operand — or already SWIZZLE_32_4_4 (5D), swizzled once at load: the deployment contract,
    # the op never swizzles in the hot path. The recipe is the scale dtype (E4M3 = NVFP4 group-16,
    # UE8M0 = MX group-32). SWIZZLED_SCALES governs both operands (a swizzled weight is paired with
    # swizzled acts). Callers wanting the tcgen05 fast path pre-swizzle the weight (5D).
    swizzled_scales = Bs.ndim == 5
    scale_group = mx_scale_family(Bs, K)
    if not swizzled_scales:
        assert Bs.shape == (rows, K // scale_group), (
            f"Bs shape {tuple(Bs.shape)} != ({rows}, {K // scale_group})"
        )
    b_u8 = e2m1_as_uint8(B)
    bs_u8 = ue8m0_as_uint8(Bs)  # caller's layout (5D swizzled / 2D affine); the op never swizzles
    input_recipe = resolve_input_recipe(input_recipe, None, Bs)
    # Activation quant is always maybe_act_quant — offline above the M threshold, inline in the
    # kernel below it (fp4 packs in-register). The offline arm writes swizzled scales directly when
    # the weight is swizzled (fused, no post-quant pass), else affine. Inline acts (small M) stay
    # affine in-register even under a swizzled weight — the dot_scaled reads the mixed layout fine.
    # NVFP4 two-level: a_global_scale (the calibrated activation global) normalizes the quant on both
    # arms — the offline kernel divides before the block quant, the inline arm via AsGlobal.
    if a_global_scale is not None:
        assert input_recipe == "nvfp4", "an activation global is NVFP4-only"
        act_quant = lambda a: MX_ACT_QUANT[input_recipe](  # noqa: E731
            a, swizzled=swizzled_scales, global_scale=a_global_scale
        )
    elif swizzled_scales:
        act_quant = lambda a: MX_ACT_QUANT[input_recipe](a, swizzled=True)  # noqa: E731
    else:
        act_quant = MX_ACT_QUANT[input_recipe]
    # NVFP4 accumulator correction: the g_a·g_b product folded onto the fp32 accumulator.
    input_global_scale = combine_global_scales(a_global_scale, b_global_scale, 1)
    # As given ⇒ A is already quantized (the routed-op parity: a pre-quantized activation + its
    # scales); else quantize raw A (offline above the M threshold, inline in the kernel below it).
    if As is not None:
        a_vals, as_scales = A.view(M, K), As
    else:
        a_vals, as_scales = maybe_act_quant(A.view(M, K), act_quant, MX_MATMUL_ACT_PREQUANT_MIN_M)
    A_q = e2m1_as_uint8(a_vals)
    as_u8 = ue8m0_as_uint8(as_scales)
    # acts are swizzled only when the offline arm ran (A quantized) under a swizzled weight
    act_swizzled = swizzled_scales and A_q.dtype != torch.bfloat16
    # gate|up output: the GLU intermediate [M, N] (output_dtype), or — under output_recipe —
    # requantized (mxfp8: fp8 + (M, N//32) uint8 Cs; fp4: packed (M, N//2) + (M, N//out_g) scale).
    out_recipe = output_recipe if gate else None
    requant = out_recipe is not None
    out_g = 16 if out_recipe == "nvfp4" else 32
    cs_dtype = torch.float8_e4m3fn if out_recipe == "nvfp4" else torch.uint8
    # Swizzled in -> swizzled out: a swizzled block requants Cs straight into the down's
    # SWIZZLE_32_4_4 layout (CSDescriptor), so the fused down reads it on the fast path. 2D dense
    # output rows are contiguous (no scatter), so it's always swizzle-able. Recipe-general — the
    # swizzle is a byte-tiling over the N // out_g scale grid (mxfp8/mxfp4 group-32, nvfp4 group-16).
    swizzled_out = requant and swizzled_scales
    if out_recipe in ("mxfp4", "nvfp4"):
        C = A.new_empty(A.shape[:-1] + (N // 2,), dtype=torch.int8)
    elif out_recipe == "mxfp8":
        C = A.new_empty(A.shape[:-1] + (N,), dtype=FP8_DTYPE)
    else:
        C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)
    if swizzled_out:
        cb_cs = triton.cdiv(N // out_g, 4)
        cs_ret = torch.empty(1, triton.cdiv(M, 128), cb_cs, 2, 256, device=A.device, dtype=cs_dtype)
        CSDescriptor = TensorDescriptor.from_tensor(cs_ret, [1, 1, 1, 2, 256])
        Cs = None  # row-major pointer unread here; CSDescriptor does the store
    elif requant:
        cs_ret = torch.empty(M, N // out_g, device=A.device, dtype=cs_dtype)
        Cs, CSDescriptor = cs_ret, None
    else:
        cs_ret, Cs, CSDescriptor = None, None, None  # unread (no OUTPUT_RECIPE)
    stride_cs_m = cs_ret.stride(0) if (requant and not swizzled_out) else 0
    stride_cs_n = cs_ret.stride(1) if (requant and not swizzled_out) else 0
    # Host-TMA descriptors over the packed (M, K_bytes) / (N, K_bytes) matrices — placeholder
    # box rebound per tuned config by _rebind_mx_descriptors. Read only by the descriptor
    # configs the tuner picks; pointer configs never touch them.
    a_descriptor = TensorDescriptor.from_tensor(A_q, [1, 32])
    b_descriptor = TensorDescriptor.from_tensor(b_u8, [1, 32])
    # One scale pointer per operand (as_u8/bs_u8) — the swizzled buffer on the fast path (also the
    # base for the BM<128 / BN<128 scalar gather), the affine scale otherwise — plus its descriptor,
    # built only on the swizzled path (dummy off the operand descriptor when un-swizzled, unread).
    box5 = [1, 1, 1, 2, 256]
    as_descriptor = TensorDescriptor.from_tensor(as_u8, box5) if act_swizzled else a_descriptor
    bs_descriptor = TensorDescriptor.from_tensor(bs_u8, box5) if swizzled_scales else b_descriptor

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    with device_context(A.device):
        compile_time_only_triton_wrap(mx_dynamic_matmul_kernel)[grid](
            A_q,
            a_descriptor,
            as_u8,
            as_descriptor,
            b_u8,
            b_descriptor,
            bs_u8,
            bs_descriptor,
            C,
            Cs,
            CSDescriptor,
            a_global_scale,  # AsGlobal: g_a for the inline-quant arm (A/g_a)
            input_global_scale,  # AsBsGlobal = g_a·g_b (acc)
            output_global_scale,  # CsGlobal: requant output normalization (next proj's provided input_scale); None folds out
            M,
            N,
            K,
            int(
                M
            ).bit_length(),  # m_bit_length key bucket; int() concretizes M (a SymInt under torch.compile has no .bit_length)
            A_q.stride(0),
            A_q.stride(1),
            as_u8.stride(0),  # As row stride (affine act-scale read; dead on the swizzled arm)
            b_u8.stride(1),
            b_u8.stride(0),
            bs_u8.stride(1),
            bs_u8.stride(0),
            C.stride(-2),
            C.stride(-1),
            stride_cs_m,
            stride_cs_n,
            SCALE_GROUP_K=scale_group,
            INPUT_RECIPE=input_recipe,
            SWIZZLED_SCALES=swizzled_scales,
            SWIZZLED_OUT=swizzled_out,
            GATE=gate,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            OUTPUT_RECIPE=out_recipe,
            SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(resolve_output_dtype(output_dtype, A, None)),
        )
    # NVFP4's block scales are already normalized by the provided output global (CsGlobal =
    # output_global_scale) at requant, so the caller pairs cs_ret with that global as the down's As.
    return [C, cs_ret] if cs_ret is not None else [C]


@compile_time_only_triton_op(
    add_op_namespace_prefix("full_precision_matmul_2d"), mutates_args=(), opaque=True
)
def full_precision_matmul_2d(
    A: torch.Tensor,
    B: torch.Tensor,
    output_dtype: torch.dtype | None = None,
    gate: bool = False,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
) -> list[torch.Tensor]:
    """Full-precision (BF16/FP16) 2D matmul ``C = A @ B.T``; no quantization anywhere. ``gate``
    fuses the ``(2N, K)`` gate|up projection into one stacked GEMM + SwiGLU, returning the
    ``[..., N]`` GLU intermediate. Returns a one-element list (mirrors the quantized 2D ops)."""
    validate_dense_2d_operands(A, B)
    assert A.dtype == B.dtype and A.dtype in (torch.bfloat16, torch.float16), (
        f"full-precision path needs matching BF16/FP16 A and B, got {A.dtype} / {B.dtype}"
    )
    rows, K = B.shape
    # Under gate|up fusion B is the (2N, K) gate|up stack; N is the per-projection output width.
    N = rows // 2 if gate else rows
    M = A.numel() // A.shape[-1]
    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    with device_context(A.device):
        compile_time_only_triton_wrap(full_precision_matmul_2d_kernel)[grid](
            A,
            B,
            C,
            M,
            N,
            K,
            int(M).bit_length(),  # m_bit_length key bucket
            A.stride(-2),
            A.stride(-1),
            B.stride(1),
            B.stride(0),
            C.stride(-2),
            C.stride(-1),
            GATE=gate,
            ACT_FN=act_fn,
            SWIGLU_ALPHA=swiglu_alpha,
            SWIGLU_LIMIT=swiglu_limit,
            SIMULATE_UNFUSED=simulate_unfused,
            INTERMEDIATE_DTYPE=tl_dtype(resolve_output_dtype(output_dtype, A, None)),
        )

    return [C]


def matmul_2d(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor | None = None,
    Bs: torch.Tensor | None = None,
    *,
    epilogue: Epilogue | None = None,
    quantization: Quantization | None = None,
    output_dtype: torch.dtype | None = None,
    a_global_scale: torch.Tensor | None = None,
    b_global_scale: torch.Tensor | None = None,
    output_global_scale: torch.Tensor | None = None,
) -> torch.Tensor | list[torch.Tensor]:
    """Dense (2D) quantized matmul dispatcher (W8A8 FP8, W4A8/W4A4 FP4) — the single-GEMM sibling of
    ``matmul_grouped``/``matmul_batched``, taking the same ``As``/``Bs`` scale spec,
    ``Epilogue``/``Quantization`` bundles, and inferred quant block (no ``block_size`` argument — it
    falls out of the weight-scale shape, so the data can't disagree with a parameter).

    ``As`` is the activation block-scale spec, same as the routed ops: ``None`` → the op quantizes raw
    ``A`` dynamically; a per-tensor scalar → static (calibrated) FP8 activation scale; block scales →
    pre-quantized ``A``. ``Bs`` is the weight block scale. ``a_global_scale``/``b_global_scale`` are the
    two-level NVFP4 per-tensor second-level scales (calibrated ``input_scale`` ``g_a`` and
    ``weight_scale_2`` ``g_b``) — provided, never computed; the op folds ``g_a·g_b`` onto the
    accumulator (``a_global_scale`` rides raw ``A`` or a pre-quantized ``As`` alike).
    ``output_global_scale`` (NVFP4 ``output_recipe`` only) is the NEXT proj's provided ``input_scale``:
    the fused requant normalizes the GLU intermediate by it, returning ``[C, Cs]`` the down-proj
    consumes as ``As=Cs, a_global_scale=output_global_scale``.

    ``epilogue`` is the fused output transform (``Epilogue(gate=True)`` → the ``(2N, K)`` gate|up
    stack + SwiGLU, returning the ``[..., N]`` GLU intermediate); ``quantization.input_recipe`` sets
    the MX activation grid (``"mxfp8"``/``"mxfp4"``/nvfp4), ``quantization.output_recipe`` (MX weights
    only) requantizes the GLU intermediate into ``[C, Cs]``. Returns the output tensor, or ``[C, Cs]``
    under ``output_recipe``.

    Routes by weight dtype + inferred block: ``Bs`` None → full-precision; MX/NVFP4 (int8/E4M3 weights
    with UE8M0 group-32 or E4M3 group-16 ``Bs``) → ``mx_dynamic_matmul``; tensor-wide FP8 (scalar
    ``Bs``) → ``w8a8_tensor_dynamic_fp8_matmul``; block FP8 → the static variant when ``As`` is a
    per-tensor scalar, else dynamic.
    """
    gate, act_fn, swiglu_alpha, swiglu_limit, simulate_unfused = (
        epilogue if epilogue is not None else Epilogue()
    ).as_args()
    input_recipe, output_recipe = (
        quantization if quantization is not None else Quantization()
    ).as_args()

    def _unwrap(ret: list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        return ret[0] if len(ret) == 1 else ret

    if Bs is None:  # unquantized BF16/FP16 weights — plain dot, no scales
        assert As is None and a_global_scale is None and input_recipe is None and output_recipe is None, (
            "the full-precision path (Bs=None) takes no activation scale or quantization recipe"
        )
        return _unwrap(
            full_precision_matmul_2d(
                A, B, output_dtype, gate, act_fn, swiglu_alpha, swiglu_limit, simulate_unfused
            )
        )

    if is_mx(B, Bs):
        return _unwrap(
            mx_dynamic_matmul(
                A, B, As, Bs, output_dtype, input_recipe,
                gate, act_fn, swiglu_alpha, swiglu_limit, simulate_unfused, output_recipe,
                a_global_scale, b_global_scale, output_global_scale,
            )
        )
    assert a_global_scale is None and b_global_scale is None and output_global_scale is None, (
        "two-level globals (a_global_scale / b_global_scale / output_global_scale) are NVFP4-only (MX weights)"
    )
    # FP8 activations are always E4M3 with the weight-implied scale granularity, so "fp8" is a
    # no-op recipe name (accepted for symmetry with the MoE ops); no other name applies here.
    assert input_recipe in (None, "fp8"), (
        f"FP8-weight activations are E4M3 ('fp8'), got {input_recipe!r}"
    )
    assert output_recipe is None, (
        f"output_recipe (fused-requant gate|up output) is MX-only, got {output_recipe!r}"
    )
    # Infer the FP8 quant block from the weight-scale shape (scalar Bs = tensor-wide). The contraction
    # dim K tiles evenly, so block_k is exact; N may be non-aligned (a partial last block, so
    # N % Bs.shape[0] != 0 and plain division would under-count) — recover block_n from the even dim,
    # the square-block convention for these FP8 weights.
    if Bs.numel() == 1:
        block_size = None
    else:
        block_k = B.shape[1] // Bs.shape[1]
        block_n = B.shape[0] // Bs.shape[0] if B.shape[0] % Bs.shape[0] == 0 else block_k
        block_size = [block_n, block_k]
    if block_size is None:  # tensor-wide (per-tensor) scale
        assert As is None, "tensor-wide FP8 quantizes A dynamically — no As"
        return _unwrap(
            w8a8_tensor_dynamic_fp8_matmul(
                A, B, Bs, output_dtype, gate, act_fn, swiglu_alpha, swiglu_limit, simulate_unfused
            )
        )
    # Block-wise FP8: a per-tensor scalar As is the static (calibrated) activation scale; else dynamic.
    if As is not None:
        assert As.numel() == 1, "block-wise FP8 As is a per-tensor static scale (scalar)"
        return _unwrap(
            w8a8_block_static_fp8_matmul(
                A, B, As, Bs, block_size, output_dtype,
                gate, act_fn, swiglu_alpha, swiglu_limit, simulate_unfused,
            )
        )
    return _unwrap(
        w8a8_block_dynamic_fp8_matmul(
            A, B, Bs, block_size, output_dtype,
            gate, act_fn, swiglu_alpha, swiglu_limit, simulate_unfused,
        )
    )
