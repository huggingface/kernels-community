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
    mx_compute,
    FP8_DTYPE,
    compile_time_only_triton_op,
    compile_time_only_triton_wrap,
    NIBBLES_PER_BYTE,
    acc_init,
    mx_config_pruner,
    smem_pruner,
    block_dynamic_dot,
    scalar_max_m_pruner,
    block_within_dim_pruner,
    compose_pruners,
    decode_group_scale,
    descriptor_box_pruner,
    matched_memory_modes_pruner,
    device_context,
    fp8_act_quant_tensor_wide,
    fp8_act_quant_block_dynamic,
    load_block_fp8_act_tile,
    load_weight_tile,
    load_mx_2d_act,
    load_mx_2d_weight,
    maybe_swizzle_mx_scales,
    oriented_tile_ptrs,
    weight_tile_descriptor,
    get_accelerator_autotuning_configs,
    warp_spec_compile_guard_pruner,
    block_dynamic_dot_scaled_ws_pruner,
    is_mx,
    is_tensor_wide,
    maybe_act_quant,
    MX_ACT_QUANT,
    mx_scale_family,
    resolve_input_recipe,
    store_masked,
    store_masked_oriented,
    swizzle_offsets,
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


def _rebind_bd_descriptors(nargs):
    """Per-config pre_hook: set the A and B host-TMA boxes to the tuned tile over the
    ``(rows, K)`` matrices — ``[BLOCK_SIZE_M, block_k]`` and ``[BLOCK_SIZE_N, block_k]``.
    MUST mutate ``block_shape`` in place (a rebind never reaches the launch); no-op for
    pointer configs, which never read the descriptor."""
    if nargs.get("A_MEMORY_MODE", "pointer") != "pointer" and not isinstance(
        nargs["ADescriptor"], int
    ):
        nargs["ADescriptor"].block_shape = [nargs["BLOCK_SIZE_M"], nargs["block_k"]]
    if nargs.get("B_MEMORY_MODE", "pointer") != "pointer" and not isinstance(
        nargs["BDescriptor"], int
    ):
        nargs["BDescriptor"].block_shape = [nargs["BLOCK_SIZE_N"], nargs["block_k"]]


def _rebind_mx_descriptors(nargs):
    """Per-config pre_hook for the MX kernel — set the A/B host-TMA boxes to the tuned tile
    in BYTES over the (rows, K_bytes) packed matrices: ``[BM, BK // ACT_VALUES_PER_BYTE]`` and
    ``[BN, BK // WEIGHT_VALUES_PER_BYTE]`` (values-per-byte read off the operand dtype; uint8 =
    packed E2M1). Mutates ``block_shape`` in place; no-op for pointer configs."""
    if nargs.get("A_MEMORY_MODE", "pointer") != "pointer" and not isinstance(
        nargs["ADescriptor"], int
    ):
        avpb = 2 if nargs["A"].dtype == torch.uint8 else 1
        nargs["ADescriptor"].block_shape = [
            nargs["BLOCK_SIZE_M"], nargs["BLOCK_SIZE_K"] // avpb
        ]
    if nargs.get("B_MEMORY_MODE", "pointer") != "pointer" and not isinstance(
        nargs["BDescriptor"], int
    ):
        wvpb = 2 if nargs["B"].dtype == torch.uint8 else 1
        nargs["BDescriptor"].block_shape = [
            nargs["BLOCK_SIZE_N"], nargs["BLOCK_SIZE_K"] // wvpb
        ]
    # SWIZZLE_32_4_4 scale boxes: [1, BLOCK//128, (BK // SCALE_GROUP_K) // 4, 2, 256]. Offline
    # (pre-quantized A: E4M3 or packed-E2M1) is the swizzled path — inferred from A's dtype,
    # matching the in-kernel SWIZZLED_SCALES constexpr.
    if nargs["A"].dtype in (torch.float8_e4m3fn, torch.uint8):
        rep_k = (nargs["BLOCK_SIZE_K"] // nargs["SCALE_GROUP_K"]) // 4
        nargs["ASDescriptor"].block_shape = [1, nargs["BLOCK_SIZE_M"] // 128, rep_k, 2, 256]
        nargs["BSDescriptor"].block_shape = [1, nargs["BLOCK_SIZE_N"] // 128, rep_k, 2, 256]


@bayesian_autotune(
    # tune_block_m: BLOCK_SIZE_M is a config axis. tune_block_n: the N tile is DECOUPLED
    # from the caller's scale granularity (block_n) — a BN=256 tile over 128-wide scale
    # columns halves activation re-reads; any BN is numerically fine (the offs_bn //
    # block_n gather spans or splits scale columns), while BK stays pinned to block_k
    # (activation scale groups are per block_k).
    #
    # Two memory regimes, tuner-routed per (N, K, M) key:
    #  - pointer: wins decode / short-K. BM=64 + WS is the winner at every model dim
    #    probed (H=5120/6144/7168), −17% e2e vs BM=128 at M=8192. Big tiles STARVE here —
    #    a full pointer sweep caps ~34% MFU at the wide-N prefill shape (load-bound).
    #  - host-TMA on BOTH operands (A_MEMORY_MODE / B_MEMORY_MODE = descriptor) in the
    #    NATURAL orientation (no SWAP_AB — the descriptor box transposes once, in
    #    load_weight_tile, to the pointer arm's K-major rhs): wins wide-N compute prefill.
    #    At M=8192 N=21504 K=7168 it takes the big tiles the pointer arm cannot feed —
    #    3502us/32% MFU -> 1681us/67% (2.08x, 90% of DeepGEMM), race-free, BM128 BN256 WS
    #    w8 (2026-07-16). WS is load-bearing: the per-iteration trans races without it.
    #    Identical pointer->natural-TMA transition the grouped kernel already took.
    # The earlier "TMA gives nothing" verdict was the SHORT-K gap shape (N=7168 K=2048):
    # there the store/load-heavy loop hides TMA behind the WS-pointer schedule and the
    # arms tie. Emitting both axes lets the tuner route per regime. B200 (sm_100) only —
    # re-chart on H100 or the target device.
    get_accelerator_autotuning_configs(
        warp_spec=True,
        tune_block_m=True,
        tune_block_n=True,
        a_memory_modes=("descriptor", "pointer"),
        b_memory_modes=("descriptor", "pointer"),
        pre_hook=_rebind_bd_descriptors,
    ),
    # m_bit_length (log2 M bucket) keys the M tile, mirroring mx_dynamic_matmul_kernel.
    ["N", "K", "m_bit_length"],
    n_trials=100,
    # WS compile guard + descriptor TMA-box limits (256/dim) + shared-memory fit (the big
    # TMA prefill tiles reach the smem ceiling; BM256 BN256 OOMs).
    prune_configs_by={
        "early_config_prune": compose_pruners(
            warp_spec_compile_guard_pruner(),
            block_dynamic_dot_scaled_ws_pruner(),
            matched_memory_modes_pruner(),
            descriptor_box_pruner("block_k"),
            smem_pruner("block_k"),
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
    offs_am_lin = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn_lin = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    as_ptrs = As + offs_am_lin * stride_as_m
    as_mask = offs_am_lin < M
    bs_mask = offs_bn_lin < N
    a_descriptor = weight_tile_descriptor(
        ADescriptor, A, M, K, stride_a_m, stride_a_k, BLOCK_SIZE_M, block_k, A_MEMORY_MODE
    )
    b_descriptor = weight_tile_descriptor(
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
    # Build (and, below, advance) the explicit operand-tile pointers ONLY on the pointer
    # arm. Under a descriptor arm they would stay live across the whole K-loop for
    # nothing — the [BM, BK] / [BK, BN] index tensors spill registers and starve
    # occupancy (~460us / 20pp MFU at the wide-N prefill shape), and the descriptor load
    # reads neither. A is a cheap scalar-pointer placeholder in that case.
    if A_MEMORY_MODE == "pointer":
        a_ptrs = oriented_tile_ptrs(A, offs_am, offs_k, stride_a_m, stride_a_k, not SWAP_AB)
    else:
        a_ptrs = A
    if B_MEMORY_MODE == "pointer":
        b_ptrs = oriented_tile_ptrs(B, offs_bn, offs_k, stride_b_n, stride_b_k, SWAP_AB)
    else:
        b_ptrs = B
    # the (BN,) scale-index gather decouples the tile from the scale grid: a wide tile
    # spans several scale columns, a narrow one shares a column
    bs_ptrs = Bs + (offs_bn_lin // block_n) * stride_bs_n
    accumulator = acc_init("dot", BLOCK_SIZE_M, BLOCK_SIZE_N, SWAP_AB)

    for k in tl.range(0, tl.cdiv(K, block_k), warp_specialize=WARP_SPEC):
        a, a_s = load_block_fp8_act_tile(
            a_ptrs,
            as_ptrs,
            a_descriptor,
            pid_m * BLOCK_SIZE_M,
            k * block_k,
            A_MEMORY_MODE,
            as_mask,
            TRANSPOSED=SWAP_AB,
        )
        b = load_weight_tile(
            b_ptrs, b_descriptor, pid_n * BLOCK_SIZE_N, k * block_k, B_MEMORY_MODE, SWAP_AB
        )
        b_s = tl.load(bs_ptrs, mask=bs_mask, other=0.0)
        accumulator = block_dynamic_dot(
            accumulator, a, a_s, b, b_s, block_k, SWAP_AB, USE_DOT_SCALED
        )
        as_ptrs += 1
        bs_ptrs += stride_bs_k
        if A_MEMORY_MODE == "pointer":
            a_ptrs += block_k * stride_a_k
        if B_MEMORY_MODE == "pointer":
            b_ptrs += block_k * stride_b_k

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
    ["N", "K", "m_bit_length"],
    n_trials=100,
    # block_k is a tuned axis and the loop below is maskless — veto non-dividing BKs;
    # WS is a pure perf axis here (non-WS is the validated state), compile-guarded.
    prune_configs_by={
        "early_config_prune": compose_pruners(
            block_within_dim_pruner("K"), warp_spec_compile_guard_pruner()
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
):
    """Tensor-scale FP8 GEMM kernel.

    Computes ``C = A @ B.T`` with one activation scale per row and one
    weight scale for the full matrix.
    Uses a 2D grid with swizzle for L2 cache locality on B tiles.
    """
    pid_m, pid_n, offs_am, offs_bn, offs_k = swizzle_offsets(
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    a_ptrs = A + offs_am[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
    b_ptrs = B + offs_k[:, None] * stride_b_k + offs_bn[None, :] * stride_b_n

    a_s = tl.load(As + offs_am * stride_as_m)
    b_s = tl.load(Bs)

    # Accumulate raw dot products, apply scales once after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_a_k
        b_ptrs += BLOCK_SIZE_K * stride_b_k

    accumulator = accumulator * a_s[:, None] * b_s

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
    # tune_block_m + tune_block_n + WARP_SPEC mirror the block-dynamic kernel above
    # (same 2D swizzle loop physics: the adaptive BM=128 heuristic + missing WS cost it
    # 2.3x at prefill, measured); the N tile is decoupled from the scale granularity.
    get_accelerator_autotuning_configs(
        warp_spec=True, tune_block_m=True, tune_block_n=True
    ),
    ["N", "K", "m_bit_length"],
    n_trials=100,
    prune_configs_by={"early_config_prune": warp_spec_compile_guard_pruner()},
)
@triton.jit
def w8a8_block_static_fp8_matmul_kernel(
    A,  # (M, K) E4M3 activations (pre-quantized against the static scale by the wrapper)
    As,  # scalar — static per-tensor activation scale (calibration-time)
    B,  # (N, K) FP8 weights
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
):
    """Block-scale FP8 GEMM with static (per-tensor) activation scale.

    ``A`` arrives pre-quantized (one elementwise ``(A / As).to(fp8)`` pass in the
    wrapper — an inline division would repeat per N-tile). Per-block weight scales apply
    per-K-tile during accumulation; the scalar activation scale is applied once at
    the end.
    """
    pid_m, pid_n, offs_am, offs_bn, offs_k = swizzle_offsets(
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, block_k
    )
    a_ptrs = A + (offs_am[:, None] * stride_a_m + offs_k[None, :] * stride_a_k)
    b_ptrs = B + (offs_k[:, None] * stride_b_k + offs_bn[None, :] * stride_b_n)

    # decoupled from the scale grid like the block-dynamic kernel above
    bs_ptrs = Bs + (offs_bn // block_n) * stride_bs_n
    a_s_static = tl.load(As)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, block_k), warp_specialize=WARP_SPEC):
        if A.dtype.element_ty == tl.float8e4nv:  # pre-quantized offline
            a = tl.load(a_ptrs)
        else:  # raw bf16/fp16 — quantize inline against the static scale
            a = (tl.load(a_ptrs).to(tl.float32) / a_s_static).to(tl.float8e4nv)
        b = tl.load(b_ptrs)
        b_s = decode_group_scale(tl.load(bs_ptrs))
        accumulator += tl.dot(a, b) * b_s[None, :]
        a_ptrs += block_k * stride_a_k
        b_ptrs += block_k * stride_b_k
        bs_ptrs += stride_bs_k

    accumulator = accumulator * a_s_static
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
    ),
    # the MXFP4/MXFP8 split keys itself — the tuner appends every tensor arg's dtype to
    # its cache key (memory and disk);
    # m_bit_length (log2 M bucket) keys the M tile — the winner keeps shifting with M well past the
    # BM ceiling and it is NOT noise: cross-applying configs (N=K=4096) costs +62% at M=128 and +245%
    # at M=4096 (the thin M=128 tile can't saturate the wide GEMM), so don't collapse the buckets.
    # INPUT_RECIPE keys the inline act-quant grid: A stays raw bf16 under every
    # recipe below the pre-quant M threshold, so the dtype-appended key can't
    # split W4A8 from W4A4 itself.
    ["N", "K", "m_bit_length", "INPUT_RECIPE"],
    n_trials=100,
    # BK-within-K veto (the loop loads are unmasked) + the sm_10x dot_scaled shape guards
    # + scalar restricted to decode-sized M (a BM=1 GEVM at prefill is TPE poison).
    prune_configs_by={
        "early_config_prune": compose_pruners(
            mx_config_pruner("K"),
            scalar_max_m_pruner("M"),
            smem_pruner(),
            # veto descriptor boxes past the 256/dim TMA limit (packed: BK//values-per-byte,
            # so fp4 BK512 -> 256-byte box is legal and keeps TMA)
            descriptor_box_pruner("BLOCK_SIZE_K"),
        )
    },
)
@triton.jit
def mx_dynamic_matmul_kernel(
    A,  # (M, K) activations: E4M3 (pre-quantized) or raw bf16/fp16 (quantized inline)
    ADescriptor,  # host TMA descriptor over A (M, K_bytes), box (BM, BK_bytes); read iff A_MEMORY_MODE != "pointer"
    As,  # (M, K // 32) UE8M0 group-32 activation scales (pre-quantized arm only)
    ASDescriptor,  # host TMA descriptor over the SWIZZLE_32_4_4 A scales; read iff SWIZZLED_SCALES
    B,  # (N, K) E4M3 (MXFP8) or (N, K // 2) packed E2M1 (MXFP4) weights
    BDescriptor,  # host TMA descriptor over B (N, K_bytes), box (BN, BK_bytes); read iff B_MEMORY_MODE != "pointer"
    Bs,  # (N, K // SCALE_GROUP_K) UE8M0 weight scales
    BSDescriptor,  # host TMA descriptor over the SWIZZLE_32_4_4 B scales; read iff SWIZZLED_SCALES
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
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
    A_MEMORY_MODE: tl.constexpr = "pointer",
    B_MEMORY_MODE: tl.constexpr = "pointer",
    INPUT_RECIPE: tl.constexpr = "mxfp8",
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
    # Offline (pre-quantized A: E4M3 or packed-E2M1) => scales are swizzled by the wrapper;
    # inline (raw bf16/fp16 A) computes scales in-register (nothing to swizzle). Inferred from
    # A's dtype — the wrapper swizzles + builds the SA/SB descriptors on exactly this condition.
    SWIZZLED_SCALES: tl.constexpr = (
        A.dtype.element_ty == tl.float8e4nv or A.dtype.element_ty == tl.uint8
    )
    # packed-fp4 weights (WEIGHT_VALUES_PER_BYTE==2) route swizzle_offsets to full grouping
    pid_m, pid_n, offs_am, offs_bn, offs_k = swizzle_offsets(
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WEIGHT_VALUES_PER_BYTE
    )
    offs_ka = tl.arange(0, BLOCK_SIZE_K // ACT_VALUES_PER_BYTE)
    offs_kb = tl.arange(0, BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)
    # Scales: SWIZZLED loads from the SA/BSDescriptors (SWIZZLE_32_4_4) + un-swizzles, so its
    # as/bs pointer tiles are dead — skip building them. Else affine-offset + bounds-mask
    # pointer loads (the %-wrapped offs drive only the operand tiles; the wrap would make the
    # scale load a non-affine gather — mirrors the bd 2D affine-scale fix).
    if SWIZZLED_SCALES:
        as_ptrs, bs_ptrs, as_mask, bs_mask = As, Bs, None, None
    else:
        offs_am_lin = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn_lin = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        as_mask = offs_am_lin < M
        bs_mask = offs_bn_lin < N
        as_ptrs = As + offs_am_lin[:, None] * stride_as_m + offs_sf[None, :]
        bs_ptrs = Bs + (offs_bn_lin[:, None] * stride_bs_n + offs_sf[None, :] * stride_bs_k)
    # Operand tiles are built ONLY on the pointer arm; the descriptor arm reads neither (the
    # [BM,BK]/[BK,BN] index tensors would stay live for nothing and spill registers).
    if A_MEMORY_MODE == "pointer":
        a_ptrs = A + (offs_am[:, None] * stride_a_m + offs_ka[None, :] * stride_a_k)
    else:
        a_ptrs = A
    if B_MEMORY_MODE == "pointer":
        b_ptrs = B + (offs_kb[:, None] * stride_b_k + offs_bn[None, :] * stride_b_n)
    else:
        b_ptrs = B

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a, a_scale = load_mx_2d_act(
            a_ptrs,
            as_ptrs,
            as_mask,
            ADescriptor,
            ASDescriptor,
            pid_m * BLOCK_SIZE_M,
            k * (BLOCK_SIZE_K // ACT_VALUES_PER_BYTE),
            pid_m,
            k,
            BLOCK_SIZE_M,
            BLOCK_SIZE_K,
            SCALE_GROUP_K,
            A_MEMORY_MODE,
            SWIZZLED_SCALES,
            INPUT_RECIPE,
        )
        b, b_s = load_mx_2d_weight(
            b_ptrs,
            bs_ptrs,
            bs_mask,
            BDescriptor,
            BSDescriptor,
            pid_n * BLOCK_SIZE_N,
            k * (BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE),
            pid_n,
            k,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            SCALE_GROUP_K,
            B_MEMORY_MODE,
            SWIZZLED_SCALES,
        )
        accumulator = mx_compute(
            accumulator,
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
        if not SWIZZLED_SCALES:  # swizzled arm re-derives the scale box offset from k
            as_ptrs += BLOCK_SIZE_K // SCALE_GROUP_K
            bs_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_bs_k
        if A_MEMORY_MODE == "pointer":  # descriptor arm re-derives the box K offset from k
            a_ptrs += (BLOCK_SIZE_K // ACT_VALUES_PER_BYTE) * stride_a_k
        if B_MEMORY_MODE == "pointer":
            b_ptrs += (BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE) * stride_b_k

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
) -> torch.Tensor:
    """Block-scale FP8 matmul: ``C = A @ B.T``; activations quantized offline in one pass.

    A:  (..., K) raw activations, bf16/fp16/fp32 (quantized to FP8 in one wrapper pass)
    B:  (N, K) FP8 weights
    Bs: (N // block_n, K // block_k) per-block weight scales
    """
    assert len(block_size) == 2, (
        f"block_size must be [block_n, block_k], got {block_size}"
    )
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1], (
        f"K mismatch: A has K={A.shape[-1]}, B has K={B.shape[-1]}"
    )
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 2, f"B must be 2D (N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"

    N, K = B.shape
    M = A.numel() // A.shape[-1]
    assert K % block_k == 0, f"K ({K}) must be divisible by block_k ({block_k})"

    assert Bs.ndim == 2, f"Bs must be 2D (N//block_n, K//block_k), got ndim={Bs.ndim}"
    assert Bs.shape == (triton.cdiv(N, block_n), K // block_k), (
        f"Bs shape {tuple(Bs.shape)} != expected ({triton.cdiv(N, block_n)}, {K // block_k})"
    )

    bs_u8 = ue8m0_as_uint8(Bs)
    # UE8M0 weight scales are the DeepGEMM-Blackwell recipe — quantize activations to UE8M0
    # too so the kernel folds both group scales into the tcgen05 dot_scaled MMA (else fp32).
    A_q, A_s = fp8_act_quant_block_dynamic(
        A.view(M, K), block_k, use_ue8m0=bs_u8.dtype == torch.uint8
    )
    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)
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
        )

    return C


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_block_static_fp8_matmul"),
    mutates_args=(),
    opaque=True,
)
def w8a8_block_static_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    As: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Block-scale FP8 matmul with static (per-tensor) activation quantization.

    A:  (..., K) raw bf16/fp16 activations — pre-quantized against ``As`` in the wrapper
    B:  (N, K) FP8 weights
    Bs: (N // block_n, K // block_k) per-block weight scales
    As: scalar / (1,) — per-tensor static activation scale
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
    assert A.shape[-1] == B.shape[-1], (
        f"K mismatch: A has K={A.shape[-1]}, B has K={B.shape[-1]}"
    )
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 2, f"B must be 2D (N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"
    assert As.numel() == 1, f"As must be scalar or (1,), got {tuple(As.shape)}"

    N, K = B.shape
    M = A.numel() // A.shape[-1]

    assert Bs.ndim == 2, f"Bs must be 2D (N//block_n, K//block_k), got ndim={Bs.ndim}"
    assert K % block_k == 0, f"K ({K}) must be divisible by block_k ({block_k})"
    assert Bs.shape == (triton.cdiv(N, block_n), K // block_k), (
        f"Bs shape {tuple(Bs.shape)} != expected ({triton.cdiv(N, block_n)}, {K // block_k})"
    )

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    bs_u8 = ue8m0_as_uint8(Bs)
    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)
    As = As.reshape(1).to(torch.float32)
    # M-gated static pre-quant (bit-exact with the inline arm: same scalar, same cast);
    # like MX, the inline form is cheap elementwise work — at M=1 a separate kernel is
    # pure added latency. The kernel picks its arm off A's dtype.
    A_q, _ = maybe_act_quant(
        A.view(M, K),
        lambda x: ((x.to(torch.float32) / As).to(FP8_DTYPE), As),
        STATIC_MATMUL_ACT_PREQUANT_MIN_M,
    )

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_block_static_fp8_matmul_kernel)[grid](
            A_q,
            As,
            B,
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
        )

    return C


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
) -> torch.Tensor:
    """Tensor-scale FP8 matmul: ``C = A @ B.T``; activations quantized offline per row.

    A:  (..., K) raw activations, bf16/fp16/fp32 (flattened to (M, K)
        internally) — per-row scales computed via ``fp8_act_quant_tensor_wide(A, K)``.
    B:  (N, K) FP8 weights.
    Bs: scalar, (1,), or (1, 1) — single tensor-scale weight scale.
    """
    assert A.shape[-1] == B.shape[-1], (
        f"K mismatch: A has K={A.shape[-1]}, B has K={B.shape[-1]}"
    )
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 2, f"B must be 2D (N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"

    N, K = B.shape
    M = A.numel() // A.shape[-1]

    assert Bs.numel() == 1, f"Bs must be scalar or (1,), got {tuple(Bs.shape)}"

    # Per-row scalar activation scale (one per token).
    qA, As = fp8_act_quant_tensor_wide(A, K)
    As = As.reshape(M)
    Bs = Bs.reshape(1)

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

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
        )

    return C


@compile_time_only_triton_op(
    add_op_namespace_prefix("mx_dynamic_matmul"), mutates_args=(), opaque=True
)
def mx_dynamic_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype | None = None,
    input_recipe: str | None = None,
) -> torch.Tensor:
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
    assert B.ndim == 2 and Bs.ndim == 2
    assert B.dtype in (torch.int8, torch.float8_e4m3fn), (
        f"B must be int8 (packed E2M1) or float8_e4m3fn (E4M3), got {B.dtype}"
    )
    assert A.is_contiguous(), "A must be contiguous"
    assert B.is_contiguous(), "B must be contiguous"
    WEIGHT_VALUES_PER_BYTE = NIBBLES_PER_BYTE if B.dtype == torch.int8 else 1

    N, K_b = B.shape
    K = A.shape[-1]
    M = A.numel() // K
    assert K == WEIGHT_VALUES_PER_BYTE * K_b, (
        f"K (={K}) must equal {WEIGHT_VALUES_PER_BYTE} * B.shape[1] (={K_b})"
    )
    nvfp4, scale_group = mx_scale_family(Bs, K)
    assert Bs.shape == (N, K // scale_group), (
        f"Bs shape {tuple(Bs.shape)} != ({N}, {K // scale_group})"
    )

    b_u8 = e2m1_as_uint8(B)
    bs_u8 = ue8m0_as_uint8(Bs)
    # the kernel quantizes raw A inline on this grid (fp4 recipes pack in-register)
    input_recipe = resolve_input_recipe(input_recipe, None, Bs)
    A_q, A_s = maybe_act_quant(
        A.view(M, K), MX_ACT_QUANT[input_recipe], MX_MATMUL_ACT_PREQUANT_MIN_M
    )
    A_q = e2m1_as_uint8(A_q)
    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)
    # Host-TMA descriptors over the packed (M, K_bytes) / (N, K_bytes) matrices — placeholder
    # box rebound per tuned config by _rebind_mx_descriptors. Read only by the descriptor
    # configs the tuner picks; pointer configs never touch them.
    a_descriptor = TensorDescriptor.from_tensor(A_q, [1, 32])
    b_descriptor = TensorDescriptor.from_tensor(b_u8, [1, 32])
    # Swizzled (SWIZZLE_32_4_4) scale descriptors for the offline/dot_scaled path — the kernel
    # bulk-loads + un-swizzles them (tcgen05 fast path). Only the offline arm (pre-quantized A)
    # takes it; inline (raw-bf16 A, decode) keeps the in-register 2D scales. Placeholder boxes
    # are rebound per config by _rebind_mx_descriptors.
    SWIZZLED_SCALES = A_q.dtype != torch.bfloat16
    if SWIZZLED_SCALES:
        cb = triton.cdiv(K // scale_group, 4)
        as_sw = maybe_swizzle_mx_scales(ue8m0_as_uint8(A_s)).reshape(
            1, triton.cdiv(M, 128), cb, 2, 256
        )
        bs_sw = maybe_swizzle_mx_scales(bs_u8).reshape(1, triton.cdiv(N, 128), cb, 2, 256)
        as_descriptor = TensorDescriptor.from_tensor(as_sw, [1, 1, 1, 2, 256])
        bs_descriptor = TensorDescriptor.from_tensor(bs_sw, [1, 1, 1, 2, 256])
    else:
        as_descriptor, bs_descriptor = a_descriptor, b_descriptor  # dummy, unread

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    with device_context(A.device):
        compile_time_only_triton_wrap(mx_dynamic_matmul_kernel)[grid](
            A_q,
            a_descriptor,
            A_s,
            as_descriptor,
            b_u8,
            b_descriptor,
            bs_u8,
            bs_descriptor,
            C,
            M,
            N,
            K,
            int(
                M
            ).bit_length(),  # m_bit_length key bucket; int() concretizes M (a SymInt under torch.compile has no .bit_length)
            A_q.stride(0),
            A_q.stride(1),
            A_s.stride(0),
            b_u8.stride(1),
            b_u8.stride(0),
            bs_u8.stride(1),
            bs_u8.stride(0),
            C.stride(-2),
            C.stride(-1),
            SCALE_GROUP_K=scale_group,
            INPUT_RECIPE=input_recipe,
        )
    return C


def matmul_2d(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int] | None,
    output_dtype: torch.dtype | None = None,
    activation_scale: torch.Tensor | None = None,
    input_recipe: str | None = None,
) -> torch.Tensor:
    """Quantized matmul dispatcher (W8A8 FP8, W4A8 or W4A4 FP4).

    ``A`` is always raw bf16/fp16/fp32; quantization is fused into every path.
    With ``activation_scale`` set, the kernel uses that per-tensor scalar
    (static quant); otherwise it computes its own scale from ``A`` (dynamic).

    ``output_dtype`` defaults to ``A.dtype``.

    Routes by weight dtype and ``block_size``:
    - MX/NVFP4 weights — ``int8`` (packed E2M1) or ``float8_e4m3fn`` (E4M3) with UE8M0
      group-32 or E4M3 group-16 ``Bs`` (shape ``[N, K//group]``) → ``mx_dynamic_matmul``
      (``block_size`` ignored, ``activation_scale`` unsupported; the group is autotuned-
      around, fixed by the scale dtype). ``input_recipe`` sets the activation grid
      (``"mxfp8"`` default, ``"mxfp4"`` = W4A4; NVFP4 weights pin ``"nvfp4"``).
    - ``block_size`` None or full ``[N, K]`` → ``w8a8_tensor_dynamic_fp8_matmul``.
    - otherwise → ``w8a8_block_dynamic_fp8_matmul`` (or its static variant when
      ``activation_scale`` is given).
    """
    if is_mx(B, Bs):
        if activation_scale is not None:
            raise NotImplementedError(
                "activation_scale (static activation quant) is not supported for MX weights — "
                "the MX path quantizes activations dynamically per group. Omit activation_scale."
            )
        return mx_dynamic_matmul(A, B, Bs, output_dtype, input_recipe)
    assert input_recipe is None, (
        f"input_recipe applies to MX/NVFP4 weights only, got {input_recipe!r}"
    )

    if is_tensor_wide(block_size, B):
        return w8a8_tensor_dynamic_fp8_matmul(A, B, Bs, output_dtype)

    # Block-wise FP8: static when a per-tensor activation scale is supplied, else dynamic.
    if activation_scale is not None:
        return w8a8_block_static_fp8_matmul(
            A, B, Bs, activation_scale, block_size, output_dtype
        )
    return w8a8_block_dynamic_fp8_matmul(A, B, Bs, block_size, output_dtype)
