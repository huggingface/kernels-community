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
    adaptive_block_size_m,
    batched_mx_pruner,
    scalar_max_m_pruner,
    block_k_within_k_pruner,
    compose_pruners,
    decode_ue8m0_scale,
    device_context,
    fp8_act_quant,
    fp8_act_quant_2d,
    load_block_fp8_act_tile,
    get_accelerator_autotuning_configs,
    warp_spec_compile_guard_pruner,
    get_mxfp_autotuning_configs,
    is_mxfp,
    is_tensor_wide,
    maybe_act_quant,
    mxfp_act_quant,
    load_mx_act_tile,
    e2m1_as_uint8,
    ue8m0_as_uint8,
)

# Swizzle group size for the 2D-grid kernels' L2-locality tiling (``_swizzle_offsets``) —
# a perf knob passed as the ``GROUP_SIZE_M`` constexpr, not a correctness parameter.
GROUP_SIZE_M = 8

# maybe_act_quant crossovers (min rows for offline pre-quant). MXFP: MEASURED — B200
# 2D-matmul sweep, graph-timed, H=6144 MXFP8 / H=4096 MXFP4: inline wins only at M=1
# (33 vs 44us / 20 vs 30us), offline from M=16 for MXFP8 (22 vs 33us), 2-3x by M>=64
# (MXFP4's M=16 cell marginally favors inline — outweighed). STATIC: inherited estimate,
# not swept — its inline arm is cheaper elementwise work, so the true crossover is at or
# above the MXFP one; the M=1 decode case (the one that matters) is inline either way.
MXFP_MATMUL_ACT_PREQUANT_MIN_M = 16
STATIC_MATMUL_ACT_PREQUANT_MIN_M = 16
# MEASURED (isolated arm A/B, graph-timed, H=6144 b128): offline wins at EVERY M incl.
# M=1 (24.7 vs 31.0us) — the inline arm pays a per-tile fp32 amax+div.
BLOCK_DYNAMIC_MATMUL_ACT_PREQUANT_MIN_M = 0


@triton.jit
def _swizzle_offsets(
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """2D-grid tile scheduling shared by the kernels below: swizzle the
    ``(pid_m, pid_n)`` program ids for L2 locality on B, then build the operand
    offset vectors. Returns ``(pid_m, pid_n, offs_am, offs_bn, offs_k)`` — the
    swizzled ids (reused by the output store) and the ``%``-wrapped row/col offsets
    plus the K range."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    return pid_m, pid_n, offs_am, offs_bn, offs_k


@triton.jit
def _store_masked(
    C,
    accumulator,
    pid_m,
    pid_n,
    M,
    N,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Shared output epilogue of the kernels below: cast the fp32 accumulator to
    ``C``'s dtype and store the ``(BLOCK_SIZE_M, BLOCK_SIZE_N)`` tile at the swizzled
    ``(pid_m, pid_n)``, masked to the ``(M, N)`` bounds."""
    c = accumulator.to(C.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@bayesian_autotune(
    get_accelerator_autotuning_configs(warp_spec=True),
    ["N", "K", "BLOCK_SIZE_M"],
    n_trials=100,
    # WS compile guard only — non-WS is race-free at every (BM, warps) here (see the pruner).
    prune_configs_by={"early_config_prune": warp_spec_compile_guard_pruner()},
)
@triton.jit
def w8a8_block_dynamic_fp8_matmul_kernel(
    A,  # (M, K) E4M3 activations (pre-quantized once by the wrapper)
    AScale,  # (M, K // BLOCK_SIZE_K) fp32 per-row, per-K-block activation scales
    B,  # (N, K) FP8 weights
    C,  # (M, N) output
    Bs,  # (N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales (fp32 or uint8/UE8M0)
    # Shape
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_as_m,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bs_k,
    stride_bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    WARP_SPEC: tl.constexpr = False,
):
    """Block-scale FP8 GEMM kernel.

    Computes ``C = A @ B.T``. Activations arrive pre-quantized (one pass in the
    wrapper — the inline per-N-tile quant re-ran N//BN times per element; see the
    grouped kernels). 2D grid with swizzle for L2 cache locality on B.
    """
    pid_m, pid_n, offs_am, offs_bn, offs_k = _swizzle_offsets(
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    as_ptrs = AScale + offs_am * stride_as_m
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_bsn = offs_bn // BLOCK_SIZE_N
    bs_ptrs = Bs + offs_bsn * stride_bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
        a, a_s = load_block_fp8_act_tile(a_ptrs, as_ptrs)
        b = tl.load(b_ptrs)
        b_s = decode_ue8m0_scale(tl.load(bs_ptrs))
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        as_ptrs += 1
        b_ptrs += BLOCK_SIZE_K * stride_bk
        bs_ptrs += stride_bs_k

    _store_masked(
        C,
        accumulator,
        pid_m,
        pid_n,
        M,
        N,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )


@bayesian_autotune(
    get_accelerator_autotuning_configs(tune_block_nk=True, warp_spec=True),
    ["N", "K", "BLOCK_SIZE_M"],
    n_trials=100,
    # BLOCK_SIZE_K is a tuned axis and the loop below is maskless — veto non-dividing BKs;
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
    B,  # (N, K) FP8 weights
    C,  # (M, N) output
    As,  # (M,) per-token activation scales
    Bs,  # scalar/(1,) per-tensor weight scale
    # Shape
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_as_m,
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
    pid_m, pid_n, offs_am, offs_bn, offs_k = _swizzle_offsets(
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )
    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    a_s = tl.load(As + offs_am * stride_as_m)
    b_s = tl.load(Bs)

    # Accumulate raw dot products, apply scales once after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPEC):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator * a_s[:, None] * b_s

    _store_masked(
        C,
        accumulator,
        pid_m,
        pid_n,
        M,
        N,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )


@bayesian_autotune(
    get_accelerator_autotuning_configs(),
    ["N", "K", "BLOCK_SIZE_M"],
    n_trials=100,
)
@triton.jit
def w8a8_block_static_fp8_matmul_kernel(
    A,  # (M, K) E4M3 activations (pre-quantized against the static scale by the wrapper)
    B,  # (N, K) FP8 weights
    C,  # (M, N) output
    As,  # scalar — static per-tensor activation scale (calibration-time)
    Bs,  # (N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales (fp32 or uint8/UE8M0)
    # Shape
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bs_k,
    stride_bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Block-scale FP8 GEMM with static (per-tensor) activation scale.

    ``A`` arrives pre-quantized (one elementwise ``(A / As).to(fp8)`` pass in the
    wrapper — the inline division re-ran per N-tile). Per-block weight scales apply
    per-K-tile during accumulation; the scalar activation scale is applied once at
    the end.
    """
    pid_m, pid_n, offs_am, offs_bn, offs_k = _swizzle_offsets(
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_bsn = offs_bn // BLOCK_SIZE_N
    bs_ptrs = Bs + offs_bsn * stride_bs_n
    a_s_static = tl.load(As)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if A.dtype.element_ty == tl.float8e4nv:  # pre-quantized offline
            a = tl.load(a_ptrs)
        else:  # raw bf16/fp16 — quantize inline against the static scale
            a = (tl.load(a_ptrs).to(tl.float32) / a_s_static).to(tl.float8e4nv)
        b = tl.load(b_ptrs)
        b_s = decode_ue8m0_scale(tl.load(bs_ptrs))
        accumulator += tl.dot(a, b) * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        bs_ptrs += stride_bs_k

    accumulator = accumulator * a_s_static
    _store_masked(
        C,
        accumulator,
        pid_m,
        pid_n,
        M,
        N,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )


@bayesian_autotune(
    # tune_block_m: BLOCK_SIZE_M becomes a config axis (not the adaptive_block_size_m launch
    # heuristic), so the tuner sizes the M tile per workload — small at decode, large at prefill.
    # scalar is in the mode set for M=1 decode (attn projection): it avoids the MMA M->16 pad
    # that held MXFP8 attn decode 35% over block-dynamic at identical weight bytes; fp4-scalar
    # is dropped by the pruner (ALU-bound unpack). swap_ab intentionally OFF: an 18-cell forced-swap sweep (cudagraph,
    # M3+dsv4 decode) showed swap losing on the single matmul (adaptive BM>=16 fills the MMA atom;
    # M3 attn swap was −38%) while the tuner never picked it — emitting the configs only bloats
    # the search. Swap stays on the batched/fused experts kernels, where it wins ~30% on dsv4.
    get_mxfp_autotuning_configs(
        compute_modes=("dot_scaled", "dot", "scalar"), tune_block_m=True
    ),
    # VALUES_PER_BYTE keys the MXFP4/MXFP8 split so a cached winner is only reused for its packing;
    # m_bit_length (log2 M bucket) keys the M tile — the winner keeps shifting with M well past the
    # BM ceiling and it is NOT noise: cross-applying configs (N=K=4096) costs +62% at M=128 and +245%
    # at M=4096 (the thin M=128 tile can't saturate the wide GEMM), so don't collapse the buckets.
    ["N", "K", "m_bit_length", "VALUES_PER_BYTE"],
    n_trials=100,
    # BK-within-K veto (the loop loads are unmasked) + the sm_10x dot_scaled shape guards
    # + scalar restricted to decode-sized M (a BM=1 GEVM at prefill is TPE poison).
    prune_configs_by={
        "early_config_prune": compose_pruners(
            batched_mx_pruner("K"), scalar_max_m_pruner("M")
        )
    },
)
@triton.jit
def mxfp_dynamic_matmul_kernel(
    A,  # (M, K) activations: E4M3 (pre-quantized) or raw bf16/fp16 (quantized inline)
    AScale,  # (M, K // 32) UE8M0 group-32 activation scales (pre-quantized arm only)
    B,  # (N, K) E4M3 (MXFP8) or (N, K // 2) packed E2M1 (MXFP4) weights
    C,  # (M, N) output
    Bs,  # (N, K // SCALE_GROUP_K) UE8M0 weight scales
    # Shape
    M,
    N,
    K,
    m_bit_length,  # autotune key only (log2 M bucket); unused in body
    # Strides
    stride_am,
    stride_ak,
    stride_as_m,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bs_k,
    stride_bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
):
    """Unified MXFP4/MXFP8 (W4A8/W8A8) GEMM.

    ``C = A @ B.T``. ``A``'s dtype picks the activation form (compile-time folded):
    pre-quantized E4M3 + UE8M0 group-32 scales (one wrapper pass — the inline
    per-N-tile quant re-ran N//BN times per element, ~2x at prefill) vs raw bf16/fp16
    quantized inline (small M: the GEMM is weight-bandwidth-bound with idle ALU, so
    inline is free and a separate quant kernel only adds latency; both forms are
    bit-exact, same group-32 boundaries). ``VALUES_PER_BYTE`` picks the weight format:
    2 = packed E2M1 (MXFP4), 1 = unpacked E4M3 (MXFP8). ``COMPUTE_MODE`` picks the MMA:
    ``tl.dot_scaled`` (native M=128 scaled MMA) vs fp8 ``tl.dot`` + per-group software
    rescale (wins at decode; FP4 unpacks E2M1->E4M3 first — lossless). 2D grid with
    swizzle for L2 reuse.
    """
    pid_m, pid_n, offs_am, offs_bn, offs_k = _swizzle_offsets(
        M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )
    offs_kb = tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    as_ptrs = AScale + offs_am[:, None] * stride_as_m + offs_sf[None, :]
    b_ptrs = B + (offs_kb[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
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
            VALUES_PER_BYTE,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            SCALE_GROUP_K,
        )
        a_ptrs += BLOCK_SIZE_K * stride_ak
        as_ptrs += BLOCK_SIZE_K // SCALE_GROUP_K
        b_ptrs += (BLOCK_SIZE_K // VALUES_PER_BYTE) * stride_bk
        bs_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_bs_k

    _store_masked(
        C,
        accumulator,
        pid_m,
        pid_n,
        M,
        N,
        stride_cm,
        stride_cn,
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

    BLOCK_SIZE_K = block_k
    BLOCK_SIZE_N = block_n
    Bs = ue8m0_as_uint8(Bs)
    C_shape = A.shape[:-1] + (N,)
    BLOCK_SIZE_M = adaptive_block_size_m(M)
    C = A.new_empty(C_shape, dtype=output_dtype)
    A_q, A_s = maybe_act_quant(
        A.view(M, K),
        lambda x: fp8_act_quant_2d(x, block_k),
        BLOCK_DYNAMIC_MATMUL_ACT_PREQUANT_MIN_M,
    )
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_block_dynamic_fp8_matmul_kernel)[grid](
            A_q,
            A_s,
            B,
            C,
            Bs,
            M,
            N,
            K,
            A_q.stride(0),
            A_q.stride(1),
            A_s.stride(0),
            B.stride(1),
            B.stride(0),
            C.stride(-2),
            C.stride(-1),
            Bs.stride(1),
            Bs.stride(0),
            # Meta-parameters
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
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

    BLOCK_SIZE_K = block_k
    BLOCK_SIZE_N = block_n
    BLOCK_SIZE_M = adaptive_block_size_m(M)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

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
            B,
            C,
            As,
            Bs,
            M,
            N,
            K,
            A_q.stride(0),
            A_q.stride(1),
            B.stride(1),
            B.stride(0),
            C.stride(-2),
            C.stride(-1),
            Bs.stride(1),
            Bs.stride(0),
            # Meta-parameters
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
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
        internally) — per-row scales computed via ``fp8_act_quant(A, K)``.
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
    qA, As = fp8_act_quant(A, K)
    As = As.reshape(M)
    Bs = Bs.reshape(1)

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)
    BLOCK_SIZE_M = adaptive_block_size_m(M)

    def grid(META):
        return (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, META["BLOCK_SIZE_N"]))

    with device_context(A.device):
        compile_time_only_triton_wrap(w8a8_tensor_dynamic_fp8_matmul_kernel)[grid](
            qA,
            B,
            C,
            As,
            Bs,
            M,
            N,
            K,
            qA.stride(-2),
            qA.stride(-1),
            B.stride(1),
            B.stride(0),
            C.stride(-2),
            C.stride(-1),
            As.stride(0),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            GROUP_SIZE_M=GROUP_SIZE_M,
        )

    return C


@compile_time_only_triton_op(
    add_op_namespace_prefix("mxfp_dynamic_matmul"), mutates_args=()
)
def mxfp_dynamic_matmul(
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
    VALUES_PER_BYTE = NIBBLES_PER_BYTE if B.dtype == torch.int8 else 1

    N, K_b = B.shape
    K = A.shape[-1]
    M = A.numel() // K
    assert K == VALUES_PER_BYTE * K_b, (
        f"K (={K}) must equal {VALUES_PER_BYTE} * B.shape[1] (={K_b})"
    )
    assert K % MX_SCALE_GROUP_K == 0, (
        f"K (={K}) must be a multiple of {MX_SCALE_GROUP_K}"
    )
    assert Bs.shape == (N, K // MX_SCALE_GROUP_K), (
        f"Bs shape {tuple(Bs.shape)} != ({N}, {K // MX_SCALE_GROUP_K})"
    )

    B = e2m1_as_uint8(B)
    bs_u8 = ue8m0_as_uint8(Bs)
    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)
    A_q, A_s = maybe_act_quant(
        A.view(M, K), mxfp_act_quant, MXFP_MATMUL_ACT_PREQUANT_MIN_M
    )

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    with device_context(A.device):
        compile_time_only_triton_wrap(mxfp_dynamic_matmul_kernel)[grid](
            A_q,
            A_s,
            B,
            C,
            bs_u8,
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
            C.stride(-2),
            C.stride(-1),
            bs_u8.stride(1),
            bs_u8.stride(0),
            GROUP_SIZE_M=GROUP_SIZE_M,
            VALUES_PER_BYTE=VALUES_PER_BYTE,
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
      group-32 ``Bs`` (shape ``[N, K//32]``) → ``mxfp_dynamic_matmul`` (``block_size``
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
        return mxfp_dynamic_matmul(A, B, Bs, output_dtype)

    if is_tensor_wide(block_size, B):
        return w8a8_tensor_dynamic_fp8_matmul(A, B, Bs, output_dtype)

    # Block-wise FP8: static when a per-tensor activation scale is supplied, else dynamic.
    if activation_scale is not None:
        return w8a8_block_static_fp8_matmul(
            A, B, Bs, activation_scale, block_size, output_dtype
        )
    return w8a8_block_dynamic_fp8_matmul(A, B, Bs, block_size, output_dtype)
