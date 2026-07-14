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

from ._ops import add_op_namespace_prefix
from .bayesian_autotuner import bayesian_autotune
from .utils import (
    mx_compute,
    FP8_DTYPE,
    compile_time_only_triton_op,
    compile_time_only_triton_wrap,
    NIBBLES_PER_BYTE,
    MX_SCALE_GROUP_K,
    UE8M0_SCALE_DTYPES,
    acc_init,
    mx_config_pruner,
    smem_pruner,
    block_scaled_fp8_dot,
    scalar_max_m_pruner,
    block_k_within_k_pruner,
    compose_pruners,
    decode_group_scale,
    descriptor_config_pruner,
    device_context,
    fp8_act_quant_tensor_wide,
    fp8_act_quant_block_dynamic,
    load_block_fp8_act_tile,
    load_weight_tile,
    oriented_tile_ptrs,
    weight_tile_descriptor,
    get_accelerator_autotuning_configs,
    warp_spec_compile_guard_pruner,
    is_mxfp,
    is_tensor_wide,
    maybe_act_quant,
    mxfp8_act_quant,
    store_masked,
    store_masked_oriented,
    swizzle_offsets,
    load_mx_act_tile,
    e2m1_as_uint8,
    ue8m0_as_uint8,
)

# Swizzle group size for the 2D-grid kernels' L2-locality tiling (``swizzle_offsets``) —
# a perf knob passed as the ``GROUP_SIZE_M`` constexpr, not a correctness parameter.
GROUP_SIZE_M = 8

# maybe_act_quant crossovers (min rows for offline pre-quant). MXFP: MEASURED — B200
# 2D-matmul sweep, graph-timed, H=6144 MXFP8 / H=4096 MXFP4: inline wins only at M=1
# (33 vs 44us / 20 vs 30us), offline from M=16 for MXFP8 (22 vs 33us), 2-3x by M>=64
# (MXFP4's M=16 cell marginally favors inline — outweighed). STATIC: inherited estimate,
# not swept — its inline arm is cheaper elementwise work, so the true crossover is at or
# above the MXFP one; the M=1 decode case (the one that matters) is inline either way.
MXFP8_MATMUL_ACT_PREQUANT_MIN_M = 16
STATIC_MATMUL_ACT_PREQUANT_MIN_M = 16
# Block-dynamic has NO gate: offline wins at EVERY M incl. M=1 (24.7 vs 31.0us, isolated
# arm A/B, graph-timed, H=6144 b128) — the inline arm pays a per-tile fp32 amax+div. The
# always-offline paths (bd 2D/batched/grouped, fused gate_ups, MX grouped) quantize
# unconditionally; only the two gates above have a real M=1 inline win.


@bayesian_autotune(
    # tune_block_m: BLOCK_SIZE_M is a config axis (not a launch-time heuristic):
    # pointer+WS at BM=64 is the winner at every model dim probed (H=5120/
    # 6144/7168), −17% e2e vs the heuristic's BM=128 at M=8192. tune_block_n: the N tile
    # is DECOUPLED from the caller's scale granularity (block_n) — a BN=256 tile
    # over 128-wide scale columns halves activation re-reads; any BN is numerically fine
    # (the offs_bn // block_n gather spans or splits scale columns), while BK stays
    # pinned to block_k (activation scale groups are per block_k).
    # The SWAP_AB/MEMORY_MODE arms below are implemented and validated but NOT emitted:
    # logged tunes rank them measurable seconds everywhere (812us WS-pointer vs 910us
    # pointer+swap vs 926us descriptor+swap at H=6144) and axes must earn an e2e win to
    # grow the grid. To re-evaluate (e.g. on a Triton bump), add
    # memory_modes=("descriptor", "pointer") and swap_ab=True — descriptor_config_pruner
    # already fences the invalid regions.
    get_accelerator_autotuning_configs(
        warp_spec=True, tune_block_m=True, tune_block_n=True
    ),
    # m_bit_length (log2 M bucket) keys the M tile, mirroring mx_dynamic_matmul_kernel.
    ["N", "K", "m_bit_length"],
    n_trials=100,
    # WS compile guard (non-WS is race-free at every (BM, warps) here) + the descriptor
    # modes' measured can't-win fences.
    prune_configs_by={
        "early_config_prune": compose_pruners(
            warp_spec_compile_guard_pruner(), descriptor_config_pruner()
        )
    },
)
@triton.jit
def w8a8_block_dynamic_fp8_matmul_kernel(
    A,  # (M, K) E4M3 activations (pre-quantized once by the wrapper)
    As,  # (M, K // block_k) fp32 per-row, per-K-block activation scales
    B,  # (N, K) FP8 weights
    BDescriptor,  # TMA descriptor over B, box (BLOCK_SIZE_N, block_k); read only under MEMORY_MODE == "host_descriptor"
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
    GROUP_SIZE_M: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
    MEMORY_MODE: tl.constexpr = "pointer",
    SWAP_AB: tl.constexpr = False,
):
    """Block-scale FP8 GEMM kernel.

    Computes ``C = A @ B.T``. Activations arrive pre-quantized (one pass in the
    wrapper — the inline per-N-tile quant would repeat N//BN times per element; see the
    grouped kernels). 2D grid with swizzle for L2 cache locality on B.

    ``SWAP_AB`` is an independent orientation knob: the weight tile sits in the MMA
    M dim and the activation tile (loaded transposed for free via strides) drops to
    the N side. The descriptor MEMORY_MODEs REQUIRE it (enforced by
    ``descriptor_config_pruner``): the natural orientation would need a per-iteration
    ``tl.trans`` on the descriptor tile, which races without WS (Triton 3.7.1
    pipeliner) and loses 2.3x with it; the swapped descriptor form is race-sound
    (12-process flake) and −11% vs same-config pointers at prefill on B200.
    """
    pid_m, pid_n, offs_am, offs_bn, offs_k = swizzle_offsets(
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, block_k, GROUP_SIZE_M
    )
    a_ptrs = oriented_tile_ptrs(A, offs_am, offs_k, stride_a_m, stride_a_k, not SWAP_AB)
    as_ptrs = As + offs_am * stride_as_m
    b_ptrs = oriented_tile_ptrs(B, offs_bn, offs_k, stride_b_n, stride_b_k, SWAP_AB)
    b_descriptor = weight_tile_descriptor(
        BDescriptor,
        B,
        N,
        K,
        stride_b_n,
        stride_b_k,
        BLOCK_SIZE_N,
        block_k,
        MEMORY_MODE,
    )
    # the (BN,) scale-index gather decouples the tile from the scale grid: a wide tile
    # spans several scale columns, a narrow one shares a column
    bs_ptrs = Bs + (offs_bn // block_n) * stride_bs_n
    accumulator = acc_init("dot", BLOCK_SIZE_M, BLOCK_SIZE_N, SWAP_AB)

    for k in tl.range(0, tl.cdiv(K, block_k), warp_specialize=WARP_SPEC):
        a, a_s = load_block_fp8_act_tile(a_ptrs, as_ptrs, TRANSPOSED=SWAP_AB)
        b = load_weight_tile(
            b_ptrs, b_descriptor, pid_n * BLOCK_SIZE_N, k * block_k, MEMORY_MODE
        )
        b_s = decode_group_scale(tl.load(bs_ptrs))
        accumulator += block_scaled_fp8_dot(a, a_s, b, b_s, SWAP_AB)
        a_ptrs += block_k * stride_a_k
        as_ptrs += 1
        b_ptrs += block_k * stride_b_k
        bs_ptrs += stride_bs_k

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
            block_k_within_k_pruner("K"), warp_spec_compile_guard_pruner()
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
    GROUP_SIZE_M: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
):
    """Tensor-scale FP8 GEMM kernel.

    Computes ``C = A @ B.T`` with one activation scale per row and one
    weight scale for the full matrix.
    Uses a 2D grid with swizzle for L2 cache locality on B tiles.
    """
    pid_m, pid_n, offs_am, offs_bn, offs_k = swizzle_offsets(
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
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
    GROUP_SIZE_M: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
):
    """Block-scale FP8 GEMM with static (per-tensor) activation scale.

    ``A`` arrives pre-quantized (one elementwise ``(A / As).to(fp8)`` pass in the
    wrapper — an inline division would repeat per N-tile). Per-block weight scales apply
    per-K-tile during accumulation; the scalar activation scale is applied once at
    the end.
    """
    pid_m, pid_n, offs_am, offs_bn, offs_k = swizzle_offsets(
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, block_k, GROUP_SIZE_M
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
    ),
    # the MXFP4/MXFP8 split keys itself — the tuner appends every tensor arg's dtype to
    # its cache key (memory and disk);
    # m_bit_length (log2 M bucket) keys the M tile — the winner keeps shifting with M well past the
    # BM ceiling and it is NOT noise: cross-applying configs (N=K=4096) costs +62% at M=128 and +245%
    # at M=4096 (the thin M=128 tile can't saturate the wide GEMM), so don't collapse the buckets.
    ["N", "K", "m_bit_length"],
    n_trials=100,
    # BK-within-K veto (the loop loads are unmasked) + the sm_10x dot_scaled shape guards
    # + scalar restricted to decode-sized M (a BM=1 GEVM at prefill is TPE poison).
    prune_configs_by={
        "early_config_prune": compose_pruners(
            mx_config_pruner("K"),
            scalar_max_m_pruner("M"),
            smem_pruner(),
        )
    },
)
@triton.jit
def mx_dynamic_matmul_kernel(
    A,  # (M, K) activations: E4M3 (pre-quantized) or raw bf16/fp16 (quantized inline)
    As,  # (M, K // 32) UE8M0 group-32 activation scales (pre-quantized arm only)
    B,  # (N, K) E4M3 (MXFP8) or (N, K // 2) packed E2M1 (MXFP4) weights
    Bs,  # (N, K // SCALE_GROUP_K) UE8M0 weight scales
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
    GROUP_SIZE_M: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
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
    pid_m, pid_n, offs_am, offs_bn, offs_k = swizzle_offsets(
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )
    # uint8 A = packed-E2M1 activations (W4A4 — the dtype IS the activation format);
    # the 2D op currently always passes raw/E4M3, so this folds to 1 there.
    ACT_VALUES_PER_BYTE: tl.constexpr = 2 if A.dtype.element_ty == tl.uint8 else 1
    WEIGHT_VALUES_PER_BYTE: tl.constexpr = 2 if B.dtype.element_ty == tl.uint8 else 1
    offs_ka = tl.arange(0, BLOCK_SIZE_K // ACT_VALUES_PER_BYTE)
    offs_kb = tl.arange(0, BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)
    a_ptrs = A + (offs_am[:, None] * stride_a_m + offs_ka[None, :] * stride_a_k)
    as_ptrs = As + offs_am[:, None] * stride_as_m + offs_sf[None, :]
    b_ptrs = B + (offs_kb[:, None] * stride_b_k + offs_bn[None, :] * stride_b_n)
    bs_ptrs = Bs + (offs_bn[:, None] * stride_bs_n + offs_sf[None, :] * stride_bs_k)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a, a_scale = load_mx_act_tile(
            a_ptrs, as_ptrs, None, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K
        )
        b = tl.load(b_ptrs)
        b_s = tl.load(bs_ptrs).to(tl.uint8)
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
        a_ptrs += (BLOCK_SIZE_K // ACT_VALUES_PER_BYTE) * stride_a_k
        as_ptrs += BLOCK_SIZE_K // SCALE_GROUP_K
        b_ptrs += (BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE) * stride_b_k
        bs_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_bs_k

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
    add_op_namespace_prefix("w8a8_block_dynamic_fp8_matmul"), mutates_args=()
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

    Bs = ue8m0_as_uint8(Bs)
    A_q, A_s = fp8_act_quant_block_dynamic(A.view(M, K), block_k)
    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)
    # The descriptor MEMORY_MODEs are not emitted (see the tuner note), so no live
    # TensorDescriptor is passed: the launcher encodes a tensormap for descriptor args
    # on EVERY launch even when the kernel never reads them, and with the (BM, BN) tile
    # now tuned the fixed box also no longer matches every config. Re-enabling the
    # descriptor rows means passing TensorDescriptor.from_tensor(B, box) again with a
    # per-config pre_hook re-binding the box to the autotuned (BM, BN) tile.
    b_descriptor = 0

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_block_dynamic_fp8_matmul_kernel)[grid](
            A_q,
            A_s,
            B,
            b_descriptor,
            Bs,
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
            Bs.stride(1),
            Bs.stride(0),
            C.stride(-2),
            C.stride(-1),
            # Meta-parameters (BM and BN come from the config; BK is the caller's
            # block_k — the activation scale groups are per block_k)
            block_n=block_n,
            block_k=block_k,
            GROUP_SIZE_M=GROUP_SIZE_M,
        )

    return C


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_block_static_fp8_matmul"), mutates_args=()
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

    Bs = ue8m0_as_uint8(Bs)
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
            Bs,
            C,
            M,
            N,
            K,
            int(M).bit_length(),  # m_bit_length key bucket
            A_q.stride(0),
            A_q.stride(1),
            B.stride(1),
            B.stride(0),
            Bs.stride(1),
            Bs.stride(0),
            C.stride(-2),
            C.stride(-1),
            # Meta-parameters (BM and BN come from the config; BK is the caller's
            # block_k — see the block-dynamic kernel)
            block_n=block_n,
            block_k=block_k,
            GROUP_SIZE_M=GROUP_SIZE_M,
        )

    return C


@compile_time_only_triton_op(
    add_op_namespace_prefix("w8a8_tensor_dynamic_fp8_matmul"), mutates_args=()
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
            GROUP_SIZE_M=GROUP_SIZE_M,
        )

    return C


@compile_time_only_triton_op(
    add_op_namespace_prefix("mx_dynamic_matmul"), mutates_args=()
)
def mx_dynamic_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """MX matmul ``C = A @ B.T``; activations quantized offline above the
    ``maybe_act_quant`` M threshold, inline below it. Weight format detected
    from ``B.dtype``: ``int8`` → packed E2M1 (MXFP4, ``B`` is ``(N, K//2)``);
    ``float8_e4m3fn`` → unpacked E4M3 (MXFP8, ``(N, K)``). Both use UE8M0 group-32 scales
    ``(N, K//32)``; tile + dot path are autotuned (scale granularity fixed at 32).

    A:  (..., K) raw activations, bf16/fp16/fp32 (quantized inline to E4M3); leading dims are
        flattened to (M, K) and restored on the output
    """
    assert B.ndim == 2 and Bs.ndim == 2
    assert B.dtype in (torch.int8, torch.float8_e4m3fn), (
        f"B must be int8 (packed E2M1) or float8_e4m3fn (E4M3), got {B.dtype}"
    )
    assert Bs.dtype in UE8M0_SCALE_DTYPES, (
        f"Bs must be float8_e8m0fnu or uint8 (UE8M0), got {Bs.dtype}"
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
    assert K % MX_SCALE_GROUP_K == 0, (
        f"K (={K}) must be a multiple of {MX_SCALE_GROUP_K}"
    )
    assert Bs.shape == (N, K // MX_SCALE_GROUP_K), (
        f"Bs shape {tuple(Bs.shape)} != ({N}, {K // MX_SCALE_GROUP_K})"
    )

    B = e2m1_as_uint8(B)
    bs_u8 = ue8m0_as_uint8(Bs)
    A_q, A_s = maybe_act_quant(
        A.view(M, K), mxfp8_act_quant, MXFP8_MATMUL_ACT_PREQUANT_MIN_M
    )
    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    with device_context(A.device):
        compile_time_only_triton_wrap(mx_dynamic_matmul_kernel)[grid](
            A_q,
            A_s,
            B,
            bs_u8,
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
            B.stride(1),
            B.stride(0),
            bs_u8.stride(1),
            bs_u8.stride(0),
            C.stride(-2),
            C.stride(-1),
            GROUP_SIZE_M=GROUP_SIZE_M,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
        )
    return C


def matmul_2d(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int] | None,
    output_dtype: torch.dtype | None = None,
    activation_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantized matmul dispatcher (W8A8 FP8 or W4A8 FP4).

    ``A`` is always raw bf16/fp16/fp32; quantization is fused into every path.
    With ``activation_scale`` set, the kernel uses that per-tensor scalar
    (static quant); otherwise it computes its own scale from ``A`` (dynamic).

    ``output_dtype`` defaults to ``A.dtype``.

    Routes by weight dtype and ``block_size``:
    - MX weights — ``int8`` (packed E2M1) or ``float8_e4m3fn`` (E4M3) with UE8M0
      group-32 ``Bs`` (shape ``[N, K//32]``) → ``mx_dynamic_matmul`` (``block_size``
      ignored, ``activation_scale`` unsupported; scale granularity fixed at 32, autotuned).
    - ``block_size`` None or full ``[N, K]`` → ``w8a8_tensor_dynamic_fp8_matmul``.
    - otherwise → ``w8a8_block_dynamic_fp8_matmul`` (or its static variant when
      ``activation_scale`` is given).
    """
    if is_mxfp(B, Bs):
        if activation_scale is not None:
            raise NotImplementedError(
                "activation_scale (static activation quant) is not supported for MX weights — "
                "the MX path quantizes activations dynamically per group. Omit activation_scale."
            )
        return mx_dynamic_matmul(A, B, Bs, output_dtype)

    if is_tensor_wide(block_size, B):
        return w8a8_tensor_dynamic_fp8_matmul(A, B, Bs, output_dtype)

    # Block-wise FP8: static when a per-tensor activation scale is supplied, else dynamic.
    if activation_scale is not None:
        return w8a8_block_static_fp8_matmul(
            A, B, Bs, activation_scale, block_size, output_dtype
        )
    return w8a8_block_dynamic_fp8_matmul(A, B, Bs, block_size, output_dtype)
