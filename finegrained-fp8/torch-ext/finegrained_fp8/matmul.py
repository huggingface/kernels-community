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
from torch.library import triton_op, wrap_triton

from ._ops import add_op_namespace_prefix, ops
from .utils import (
    NIBBLES_PER_BYTE,
    MX_SCALE_GROUP_K,
    adaptive_block_size_m,
    device_context,
    fp8_act_quant,
    fp8_act_quant_inline,
    get_accelerator_autotuning_configs,
    mx_act_quant_inline,
)


@triton.autotune(
    configs=get_accelerator_autotuning_configs(),
    key=["N", "K", "BLOCK_SIZE_M"],
)
@triton.jit
def w8a8_block_dynamic_fp8_matmul_kernel(
    A,  # (M, K) raw BF16/FP16 activations
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
    """Block-scale FP8 GEMM kernel with fused activation quantization.

    Computes ``C = A @ B.T`` with bf16/fp16 ``A`` quantized to FP8 per-K-tile
    inline (one scale per M-row per BLOCK_SIZE_K) and pre-quantized FP8 weights
    with per-block scales. 2D grid with swizzle for L2 cache locality on B.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_bsn = offs_bn // BLOCK_SIZE_N
    bs_ptrs = Bs + offs_bsn * stride_bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a_raw = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0).to(
            tl.float32
        )
        a, a_s = fp8_act_quant_inline(a_raw)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        b_s = tl.load(bs_ptrs)
        if b_s.dtype == tl.uint8:
            # UE8M0 decode: value = 2^(exp - 127); build the fp32 bit pattern.
            b_s = (b_s.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        bs_ptrs += stride_bs_k

    c = accumulator.to(C.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=get_accelerator_autotuning_configs(),
    key=["N", "K", "BLOCK_SIZE_M"],
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
):
    """Tensor-scale FP8 GEMM kernel.

    Computes ``C = A @ B.T`` with one activation scale per row and one
    weight scale for the full matrix.
    Uses a 2D grid with swizzle for L2 cache locality on B tiles.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    a_s = tl.load(As + offs_am * stride_as_m)
    b_s = tl.load(Bs)

    # Accumulate raw dot products, apply scales once after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator * a_s[:, None] * b_s

    c = accumulator.to(C.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=get_accelerator_autotuning_configs(),
    key=["N", "K", "BLOCK_SIZE_M"],
)
@triton.jit
def w8a8_block_static_fp8_matmul_kernel(
    A,  # (M, K) raw BF16/FP16 activations
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

    ``A`` is raw bf16/fp16; the kernel divides by the scalar ``As`` and casts
    to FP8 inline. Per-block weight scales apply per-K-tile during
    accumulation; the scalar activation scale factors out of the loop and
    is applied once at the end.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_bsn = offs_bn // BLOCK_SIZE_N
    bs_ptrs = Bs + offs_bsn * stride_bs_n
    a_s_static = tl.load(As)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a_raw = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0).to(
            tl.float32
        )
        a = (a_raw / a_s_static).to(tl.float8e4nv)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        b_s = tl.load(bs_ptrs)
        if b_s.dtype == tl.uint8:
            # UE8M0 decode: value = 2^(exp - 127); build the fp32 bit pattern.
            b_s = (b_s.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        accumulator += tl.dot(a, b) * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        bs_ptrs += stride_bs_k

    accumulator = accumulator * a_s_static
    c = accumulator.to(C.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=get_accelerator_autotuning_configs(for_mx=True),
    key=["N", "K", "BLOCK_SIZE_M"],
)
@triton.jit
def w4a8_mx_dynamic_fp4_matmul_kernel(
    A,  # (M, K) raw BF16/FP16 activations
    B,  # (N, K // 2) packed FP4 (E2M1) weights as int8
    C,  # (M, N) output
    Bs,  # (N, K // SCALE_GROUP_K) UE8M0 weight scales
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
    NIBBLES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    """MXFP4 (W4A8) GEMM kernel with fused activation quantization.

    Computes ``C = A @ B.T`` with bf16/fp16 ``A`` quantized to FP8 (E4M3) per
    K-group of ``SCALE_GROUP_K`` elements inline (UE8M0 scale), and packed FP4
    (E2M1) weights with their own UE8M0 scales. 2D grid with swizzle for L2
    cache locality on B tiles, ``tl.dot_scaled`` for the scaled MMA.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_k_byte = tl.arange(0, BLOCK_SIZE_K // NIBBLES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k_byte[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    bs_ptrs = Bs + (offs_bn[:, None] * stride_bs_n + offs_sf[None, :] * stride_bs_k)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a_raw = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0).to(
            tl.float32
        )
        a, a_scale = mx_act_quant_inline(
            a_raw, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K
        )
        b = tl.load(
            b_ptrs, mask=offs_k_byte[:, None] < k_remaining // NIBBLES_PER_BYTE, other=0
        ).to(tl.uint8)
        b_s = tl.load(
            bs_ptrs, mask=offs_sf[None, :] < k_remaining // SCALE_GROUP_K, other=0
        ).to(tl.uint8)
        accumulator = tl.dot_scaled(a, a_scale, "e4m3", b, b_s, "e2m1", acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // NIBBLES_PER_BYTE) * stride_bk
        bs_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_bs_k

    c = accumulator.to(C.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=get_accelerator_autotuning_configs(for_mx=True),
    key=["N", "K", "BLOCK_SIZE_M"],
)
@triton.jit
def w8a8_mx_dynamic_fp8_matmul_kernel(
    A,  # (M, K) raw BF16/FP16 activations
    B,  # (N, K) E4M3 weights (unpacked, one byte per value)
    C,  # (M, N) output
    Bs,  # (N, K // SCALE_GROUP_K) UE8M0 weight scales
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
    SCALE_GROUP_K: tl.constexpr,
):
    """MXFP8 block-scale GEMM kernel with fused activation quantization.

    Computes ``C = A @ B.T`` with bf16/fp16 ``A`` quantized to FP8 (E4M3) per
    K-group of ``SCALE_GROUP_K`` elements inline (UE8M0 scale), against E4M3
    weights with their own UE8M0 K-group scales. Mirrors the W4A8 FP4 kernel but
    keeps weights unpacked (one E4M3 byte per value); ``tl.dot_scaled`` drives
    the MX scaled MMA with ``"e4m3"`` on both operands.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    bs_ptrs = Bs + (offs_bn[:, None] * stride_bs_n + offs_sf[None, :] * stride_bs_k)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a_raw = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0).to(
            tl.float32
        )
        a, a_scale = mx_act_quant_inline(
            a_raw, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        b_s = tl.load(
            bs_ptrs, mask=offs_sf[None, :] < k_remaining // SCALE_GROUP_K, other=0
        ).to(tl.uint8)
        accumulator = tl.dot_scaled(a, a_scale, "e4m3", b, b_s, "e4m3", acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        bs_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_bs_k

    c = accumulator.to(C.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton_op(add_op_namespace_prefix("w8a8_block_dynamic_fp8_matmul"), mutates_args=())
def _w8a8_block_dynamic_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Block-scale FP8 matmul: ``C = A @ B.T`` with fused activation quantization.

    A:  (..., K) raw activations, bf16/fp16/fp32 (quantized inline to FP8)
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

    assert Bs.ndim == 2, f"Bs must be 2D (N//block_n, K//block_k), got ndim={Bs.ndim}"
    assert Bs.shape == (triton.cdiv(N, block_n), triton.cdiv(K, block_k)), (
        f"Bs shape {tuple(Bs.shape)} != expected ({triton.cdiv(N, block_n)}, {triton.cdiv(K, block_k)})"
    )

    BLOCK_SIZE_K = block_k
    BLOCK_SIZE_N = block_n
    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)
    BLOCK_SIZE_M = adaptive_block_size_m(M)
    # UE8M0 scales: pass as uint8 (Triton binder doesn't recognize
    # float8_e8m0fnu); kernel decodes 2^(exp-127) inline.
    if Bs.dtype == torch.float8_e8m0fnu:
        Bs = Bs.view(torch.uint8)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    with device_context(A.device):
        wrap_triton(w8a8_block_dynamic_fp8_matmul_kernel)[grid](
            A,
            B,
            C,
            Bs,
            M,
            N,
            K,
            A.stride(-2),
            A.stride(-1),
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
            GROUP_SIZE_M=8,
        )

    return C


@triton_op(add_op_namespace_prefix("w8a8_block_static_fp8_matmul"), mutates_args=())
def _w8a8_block_static_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    As: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Block-scale FP8 matmul with static (per-tensor) activation quantization.

    A:  (..., K) raw bf16/fp16 activations — quantized to FP8 inline against ``As``
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
    assert Bs.shape == (triton.cdiv(N, block_n), triton.cdiv(K, block_k)), (
        f"Bs shape {tuple(Bs.shape)} != expected ({triton.cdiv(N, block_n)}, {triton.cdiv(K, block_k)})"
    )

    As = As.reshape(1).to(torch.float32)
    BLOCK_SIZE_K = block_k
    BLOCK_SIZE_N = block_n
    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)
    BLOCK_SIZE_M = adaptive_block_size_m(M)
    # UE8M0 scales: pass as uint8 (Triton binder doesn't recognize
    # float8_e8m0fnu); kernel decodes 2^(exp-127) inline.
    if Bs.dtype == torch.float8_e8m0fnu:
        Bs = Bs.view(torch.uint8)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    with device_context(A.device):
        wrap_triton(w8a8_block_static_fp8_matmul_kernel)[grid](
            A,
            B,
            C,
            As,
            Bs,
            M,
            N,
            K,
            A.stride(-2),
            A.stride(-1),
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
            GROUP_SIZE_M=8,
        )

    return C


@triton_op(add_op_namespace_prefix("w8a8_tensor_dynamic_fp8_matmul"), mutates_args=())
def _w8a8_tensor_dynamic_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Tensor-scale FP8 matmul: ``C = A @ B.T`` with fused activation quantization.

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

    # Normalize Bs to (1,)
    assert Bs.numel() == 1, f"Bs must be scalar or (1,), got {tuple(Bs.shape)}"
    Bs = Bs.reshape(1)

    # Per-row scalar activation scale (one per token).
    qA, As = fp8_act_quant(A, K)
    As = As.reshape(M)

    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128
    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)
    BLOCK_SIZE_M = adaptive_block_size_m(M)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    with device_context(A.device):
        wrap_triton(w8a8_tensor_dynamic_fp8_matmul_kernel)[grid](
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
            # Meta-parameters
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=8,
        )

    return C


@triton_op(add_op_namespace_prefix("w4a8_mx_dynamic_fp4_matmul"), mutates_args=())
def _w4a8_mx_dynamic_fp4_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """MXFP4 (W4A8) matmul: ``C = A @ B.T`` with fused activation quant.

    A:  (M, K) raw activations, bf16/fp16/fp32 (quantized inline to FP8)
    B:  (N, K // 2) packed FP4 (E2M1) weights, two codes per int8
    Bs: (N, K // 32) UE8M0 weight scales

    On CUDA and other backends, BLOCK_SIZE_N and BLOCK_SIZE_K are autotuned.
    On XPU they are fixed to 128x128 and only num_warps/num_stages are autotuned.
    FP4 scale granularity is fixed at 32, so tile shape is purely a perf knob.
    """
    assert A.ndim == 2 and B.ndim == 2 and Bs.ndim == 2
    assert B.dtype == torch.int8, f"B must be int8 (packed FP4), got {B.dtype}"
    assert Bs.dtype == torch.float8_e8m0fnu, (
        f"Bs must be float8_e8m0fnu, got {Bs.dtype}"
    )
    assert A.is_contiguous(), "A must be contiguous"
    assert B.is_contiguous(), "B must be contiguous"

    M, K = A.shape
    N, K_half = B.shape
    assert K == NIBBLES_PER_BYTE * K_half, (
        f"K (={K}) must equal {NIBBLES_PER_BYTE} * B.shape[1] (={K_half})"
    )
    assert K % MX_SCALE_GROUP_K == 0, (
        f"K (={K}) must be a multiple of {MX_SCALE_GROUP_K}"
    )
    assert Bs.shape == (N, K // MX_SCALE_GROUP_K), (
        f"Bs shape {tuple(Bs.shape)} != ({N}, {K // MX_SCALE_GROUP_K})"
    )

    bs_u8 = Bs.view(torch.uint8)
    C = A.new_empty((M, N), dtype=output_dtype)
    BLOCK_SIZE_M = adaptive_block_size_m(M)

    def grid(META):
        return (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, META["BLOCK_SIZE_N"]))

    with device_context(A.device):
        wrap_triton(w4a8_mx_dynamic_fp4_matmul_kernel)[grid](
            A,
            B,
            C,
            bs_u8,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(1),
            B.stride(0),
            C.stride(0),
            C.stride(1),
            bs_u8.stride(1),
            bs_u8.stride(0),
            # Meta-parameters (BLOCK_SIZE_N, BLOCK_SIZE_K come from autotune Config)
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            GROUP_SIZE_M=8,
            NIBBLES_PER_BYTE=NIBBLES_PER_BYTE,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
        )
    return C


@triton_op(add_op_namespace_prefix("w8a8_mx_dynamic_fp8_matmul"), mutates_args=())
def _w8a8_mx_dynamic_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """MXFP8 W8A8 matmul: ``C = A @ B.T`` (E4M3 × E4M3, UE8M0 group-32 scales).

    A:  (M, K) raw activations, bf16/fp16/fp32 (quantized inline to E4M3,
        MX group-32 UE8M0 scales)
    B:  (N, K) E4M3 weights (unpacked)
    Bs: (N, K // 32) UE8M0 weight scales

    Tile shape (BLOCK_SIZE_N, BLOCK_SIZE_K) is autotuned; MX scale granularity is
    fixed at 32 (the MX-format spec), so tile shape is purely a perf knob.
    """
    assert A.ndim == 2 and B.ndim == 2 and Bs.ndim == 2
    assert B.dtype == torch.float8_e4m3fn, (
        f"B must be float8_e4m3fn (E4M3 weights), got {B.dtype}"
    )
    assert Bs.dtype == torch.float8_e8m0fnu, (
        f"Bs must be float8_e8m0fnu, got {Bs.dtype}"
    )
    assert A.is_contiguous(), "A must be contiguous"
    assert B.is_contiguous(), "B must be contiguous"

    M, K = A.shape
    N, K_b = B.shape
    assert K == K_b, f"K mismatch: A has K={K}, B has K={K_b}"
    assert K % MX_SCALE_GROUP_K == 0, (
        f"K (={K}) must be a multiple of {MX_SCALE_GROUP_K}"
    )
    assert Bs.shape == (N, K // MX_SCALE_GROUP_K), (
        f"Bs shape {tuple(Bs.shape)} != ({N}, {K // MX_SCALE_GROUP_K})"
    )

    bs_u8 = Bs.view(torch.uint8)
    C = A.new_empty((M, N), dtype=output_dtype)
    BLOCK_SIZE_M = adaptive_block_size_m(M)

    def grid(META):
        return (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, META["BLOCK_SIZE_N"]))

    with device_context(A.device):
        wrap_triton(w8a8_mx_dynamic_fp8_matmul_kernel)[grid](
            A,
            B,
            C,
            bs_u8,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(1),
            B.stride(0),
            C.stride(0),
            C.stride(1),
            bs_u8.stride(1),
            bs_u8.stride(0),
            # Meta-parameters (BLOCK_SIZE_N, BLOCK_SIZE_K come from autotune Config)
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            GROUP_SIZE_M=8,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
        )
    return C


def w8a8_block_dynamic_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Block-wise W8A8 FP8 matrix multiplication with fused activation quantization.

    Args:
        A: Raw activation tensor ``[..., M, K]`` in bf16/fp16/fp32 — quantized
            inline to ``float8_e4m3fn`` against per-K-tile per-row UE8 scales.
        B: FP8 weight tensor ``[N, K]`` in ``float8_e4m3fn``.
        Bs: Per-block weight scales ``[N // block_size[0], K // block_size[1]]``.
        block_size: ``[block_n, block_k]`` weight quantization block dimensions.
        output_dtype: dtype of the returned tensor (default: ``torch.float32``).

    Returns:
        Output tensor ``[..., M, N]`` in ``output_dtype``.
    """
    return ops.w8a8_block_dynamic_fp8_matmul(A, B, Bs, block_size, output_dtype)


def w8a8_block_static_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    As: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Block-wise W8A8 FP8 matmul with static (per-tensor) activation quantization.

    Args:
        A: Raw activation tensor ``[..., M, K]`` in bf16/fp16/fp32 — quantized
            inline against the per-tensor scalar ``As``.
        B: FP8 weight tensor ``[N, K]`` in ``float8_e4m3fn``.
        Bs: Per-block weight scales ``[N // block_size[0], K // block_size[1]]``.
        As: Static per-tensor activation scale (scalar or ``[1]``).
        block_size: ``[block_n, block_k]`` weight quantization block dimensions.
        output_dtype: dtype of the returned tensor (default: ``torch.float32``).

    Returns:
        Output tensor ``[..., M, N]`` in ``output_dtype``.
    """
    return ops.w8a8_block_static_fp8_matmul(A, B, Bs, As, block_size, output_dtype)


def w8a8_tensor_dynamic_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Tensor-scale W8A8 FP8 matmul with fused activation quantization.

    Computes ``C = A @ B.T`` with raw bf16/fp16/fp32 ``A`` quantized to FP8
    per-row (one scale per token) before the dot.

    Args:
        A: Raw activation tensor ``[M, K]`` in bf16/fp16/fp32.
        B: FP8 weight tensor ``[N, K]`` in ``float8_e4m3fn``.
        Bs: Single weight scale, scalar or ``[1]``.
        output_dtype: dtype of the returned tensor.
    """
    return ops.w8a8_tensor_dynamic_fp8_matmul(A, B, Bs, output_dtype)


def w4a8_mx_dynamic_fp4_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """MXFP4 (W4A8) matmul with fused activation quantization.

    Computes ``C = A @ B.T`` with bf16/fp16/fp32 ``A`` quantized to FP8 (E4M3)
    inline and packed FP4 (E2M1) weights; both scales are UE8M0 at K-group
    granularity 32 (the MX-format spec). Tile shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
    is autotuned — FP4 has no caller-controlled quantization block.

    Args:
        A: Raw activations ``[M, K]`` in bf16/fp16/fp32.
        B: Packed FP4 weights ``[N, K // 2]`` (``int8``, two codes per byte).
        Bs: UE8M0 weight scales ``[N, K // 32]``.
        output_dtype: dtype of the returned tensor (default ``bfloat16``).
    """
    return ops.w4a8_mx_dynamic_fp4_matmul(A, B, Bs, output_dtype)


def w8a8_mx_dynamic_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """MXFP8 W8A8 matmul (E4M3 × E4M3, UE8M0 group-32 scales) with fused act quant.

    Computes ``C = A @ B.T`` with bf16/fp16/fp32 ``A`` quantized inline to E4M3
    against per-row, per-32-K-group UE8M0 scales, and E4M3 weights with their own
    UE8M0 K-group scales. Both scales feed ``tl.dot_scaled`` (the MX scaled MMA),
    unlike ``w8a8_block_dynamic_fp8_matmul`` which applies 128×128 block scales in
    software. Tile shape is autotuned; MX scale granularity is fixed at 32.

    Args:
        A: Raw activations ``[M, K]`` in bf16/fp16/fp32.
        B: E4M3 weights ``[N, K]`` (``float8_e4m3fn``, unpacked).
        Bs: UE8M0 weight scales ``[N, K // 32]`` (``float8_e8m0fnu``).
        output_dtype: dtype of the returned tensor (default ``bfloat16``).
    """
    return ops.w8a8_mx_dynamic_fp8_matmul(A, B, Bs, output_dtype)


def matmul(
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

    Routes by weight dtype and ``block_size``:
    - ``B.dtype == int8`` (packed FP4) → ``w4a8_mx_dynamic_fp4_matmul``
      (``block_size`` is ignored; FP4 scale granularity is fixed at 32 and
      tile sizes are autotuned).
    - ``B.dtype == float8_e4m3fn`` with UE8M0 group-32 ``Bs`` (shape ``[N, K//32]``)
      → ``w8a8_mx_dynamic_fp8_matmul`` (MX scaled MMA; ``block_size`` ignored).
    - ``block_size`` None or full ``[N, K]`` → ``w8a8_tensor_dynamic_fp8_matmul``.
    - otherwise → ``w8a8_block_dynamic_fp8_matmul`` (or its static variant when
      ``activation_scale`` is given).
    """
    if activation_scale is not None:
        if B.dtype == torch.int8:
            raise NotImplementedError(
                "static activation_scale is not supported on the FP4 path"
            )
        if block_size is None or (
            block_size[0] == B.size(0) and block_size[1] == B.size(1)
        ):
            raise NotImplementedError(
                "static activation_scale requires block-wise weights, "
                "not tensor-mode (block_size None or full [N, K])"
            )
        return w8a8_block_static_fp8_matmul(
            A, B, Bs, activation_scale, block_size, output_dtype
        )

    if B.dtype == torch.int8:
        return w4a8_mx_dynamic_fp4_matmul(A, B, Bs, output_dtype)

    # MXFP8: E4M3 weights with UE8M0 group-32 scales (Bs shape [N, K//32]).
    # Distinguished from 128×128 block-scaled FP8 (Bs shape [cdiv(N,128), cdiv(K,128)]).
    if B.dtype == torch.float8_e4m3fn and Bs.dtype == torch.float8_e8m0fnu:
        N, K = B.shape
        if Bs.ndim == 2 and tuple(Bs.shape) == (N, K // MX_SCALE_GROUP_K):
            return w8a8_mx_dynamic_fp8_matmul(A, B, Bs, output_dtype)

    if block_size is None or (
        block_size[0] == B.size(0) and block_size[1] == B.size(1)
    ):
        return w8a8_tensor_dynamic_fp8_matmul(A, B, Bs, output_dtype)

    return w8a8_block_dynamic_fp8_matmul(A, B, Bs, block_size, output_dtype)
