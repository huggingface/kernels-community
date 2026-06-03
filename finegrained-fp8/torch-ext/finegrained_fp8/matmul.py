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

from .utils import (
    FP4_SCALE_GROUP_K,
    FP4_VALUES_PER_BYTE,
    device_context,
    fp4_expand_activation_scales,
    fp4_resolve_block_k,
)


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8_kernel.py
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [2, 4, 8, 16]
        for s in [2, 3, 4]
    ],
    key=["N", "K", "BLOCK_SIZE_M"],
)
@triton.jit
def w8a8_block_fp8_matmul_kernel(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_as_m,
    stride_as_k,
    stride_bs_k,
    stride_bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Block-scale FP8 GEMM kernel.

    Computes ``C = A @ B.T`` with block-wise activation/weight scales.
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
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    as_ptrs = As + offs_am * stride_as_m
    offs_bsn = offs_bn // BLOCK_SIZE_N
    bs_ptrs = Bs + offs_bsn * stride_bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        a_s = tl.load(as_ptrs + k * stride_as_k)
        b_s = tl.load(bs_ptrs + k * stride_bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [2, 4, 8, 16]
        for s in [2, 3, 4]
    ],
    key=["N", "K", "BLOCK_SIZE_M"],
)
@triton.jit
def w8a8_tensor_fp8_matmul_kernel(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_as_m,
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

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [8, 16]
        for s in [2, 3, 4]
    ],
    key=["M", "N", "K", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"],
)
@triton.jit
def w4a8_block_fp4_matmul_kernel(
    A_ptr,
    AS_ptr,
    B_ptr,
    SF_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
    stride_bn,
    stride_bk,
    stride_sfn,
    stride_sfk,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k_byte = tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    a_base = A_ptr + offs_m[:, None] * stride_am
    as_base = AS_ptr + offs_m[:, None] * stride_asm
    b_base = B_ptr + offs_n[:, None] * stride_bn
    sf_base = SF_ptr + offs_n[:, None] * stride_sfn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_base = k0 * BLOCK_SIZE_K
        a_k = k_base + tl.arange(0, BLOCK_SIZE_K)
        b_k = k_base // VALUES_PER_BYTE + offs_k_byte
        s_k = k_base // SCALE_GROUP_K + offs_sf
        a = tl.load(
            a_base + a_k[None, :] * stride_ak,
            mask=mask_m[:, None] & (a_k[None, :] < K),
            other=0.0,
        )
        a_scale = tl.load(
            as_base + s_k[None, :] * stride_ask,
            mask=mask_m[:, None] & (s_k[None, :] < K // SCALE_GROUP_K),
            other=0,
        ).to(tl.uint8)
        b_nk = tl.load(
            b_base + b_k[None, :] * stride_bk,
            mask=mask_n[:, None] & (b_k[None, :] < (K // VALUES_PER_BYTE)),
            other=0,
        ).to(tl.uint8)
        b = tl.trans(b_nk)
        sf = tl.load(
            sf_base + s_k[None, :] * stride_sfk,
            mask=mask_n[:, None] & (s_k[None, :] < (K // SCALE_GROUP_K)),
            other=0,
        ).to(tl.uint8)
        acc = tl.dot_scaled(
            a,
            a_scale,
            "e4m3",
            b,
            sf,
            "e2m1",
            acc=acc,
        )

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)


@triton_op("finegrained_fp8::w8a8_block_fp8_matmul", mutates_args=())
def _w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Block-scale FP8 matmul: C = A @ B.T with per-block scales.

    As: (M, K // block_k) — per-token-group activation scales
    Bs: (N // block_n, K // block_k) — per-block weight scales
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

    assert As.ndim >= 2, f"As must be at least 2D, got ndim={As.ndim}"
    assert As.shape[-1] == triton.cdiv(K, block_k), (
        f"As last dim {As.shape[-1]} != expected {triton.cdiv(K, block_k)} (cdiv(K={K}, block_k={block_k}))"
    )
    assert Bs.ndim == 2, f"Bs must be 2D (N//block_n, K//block_k), got ndim={Bs.ndim}"
    assert Bs.shape == (triton.cdiv(N, block_n), triton.cdiv(K, block_k)), (
        f"Bs shape {tuple(Bs.shape)} != expected ({triton.cdiv(N, block_n)}, {triton.cdiv(K, block_k)})"
    )

    BLOCK_SIZE_K = block_k
    BLOCK_SIZE_N = block_n
    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)
    # Adaptive BLOCK_SIZE_M: smallest power-of-2 >= M, floored at 16, capped at 128.
    # Matches the WGMMA tile to the actual row count — smaller tiles use less
    # register pressure and a better-matched FP8 WGMMA instruction, improving
    # both accuracy and performance for small M (decode).
    BLOCK_SIZE_M = min(max(triton.next_power_of_2(M), 16), 128)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    with device_context(A.device):
        wrap_triton(w8a8_block_fp8_matmul_kernel)[grid](
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
            As.stride(-2),
            As.stride(-1),
            Bs.stride(1),
            Bs.stride(0),
            # Meta-parameters
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=8,
        )

    return C


@triton_op("finegrained_fp8::w8a8_tensor_fp8_matmul", mutates_args=())
def _w8a8_tensor_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Tensor-scale FP8 matmul: C = A @ B.T with per-row / per-tensor scales.

    As: scalar, (M,), or (M, 1) — per-row activation scales
    Bs: scalar, (1,), or (1, 1) — single weight scale
    """
    assert A.shape[-1] == B.shape[-1], (
        f"K mismatch: A has K={A.shape[-1]}, B has K={B.shape[-1]}"
    )
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 2, f"B must be 2D (N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"

    N, K = B.shape
    M = A.numel() // A.shape[-1]

    # Normalize As to (M,)
    if As.numel() == 1:
        As = As.reshape(1).expand(M).contiguous()
    elif As.ndim == 2:
        As = As.reshape(M)
    assert As.ndim == 1 and As.shape[0] == M, (
        f"As must be scalar, (M,), or (M,1) with M={M}, got {tuple(As.shape)}"
    )

    # Normalize Bs to (1,)
    assert Bs.numel() == 1, f"Bs must be scalar or (1,), got {tuple(Bs.shape)}"
    Bs = Bs.reshape(1)

    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128
    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)
    BLOCK_SIZE_M = min(max(triton.next_power_of_2(M), 16), 128)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    with device_context(A.device):
        wrap_triton(w8a8_tensor_fp8_matmul_kernel)[grid](
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
            As.stride(0),
            # Meta-parameters
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=8,
        )

    return C


@triton_op("finegrained_fp8::w4a8_block_fp4_matmul", mutates_args=())
def _w4a8_block_fp4_matmul(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert A.ndim == 2 and As.ndim == 2 and B.ndim == 2 and Bs.ndim == 2
    assert A.dtype == torch.float8_e4m3fn
    assert B.dtype == torch.int8
    assert Bs.dtype == torch.float8_e8m0fnu
    assert As.dtype in (torch.float32, torch.float8_e8m0fnu)
    assert A.is_contiguous() and B.is_contiguous()
    assert len(block_size) == 2, (
        f"block_size must be [block_n, block_k], got {block_size}"
    )

    M, K = A.shape
    N, K_half = B.shape
    assert K == FP4_VALUES_PER_BYTE * K_half, (
        f"K (={K}) must equal {FP4_VALUES_PER_BYTE} * B.shape[1] (={K_half})"
    )
    assert Bs.shape == (N, K // FP4_SCALE_GROUP_K), (
        f"Bs shape {tuple(Bs.shape)} != ({N}, {K // FP4_SCALE_GROUP_K})"
    )
    assert K % FP4_SCALE_GROUP_K == 0, f"K (={K}) must be a multiple of {FP4_SCALE_GROUP_K}"

    is_tensor_scale_input = As.shape[-1] == 1
    block_n = block_size[0]
    block_k = fp4_resolve_block_k(block_size, K)
    As = fp4_expand_activation_scales(As, K, block_k)
    assert As.shape == (M, K // FP4_SCALE_GROUP_K), (
        f"As shape {tuple(As.shape)} != ({M}, {K // FP4_SCALE_GROUP_K})"
    )

    as_u8 = As.to(torch.float8_e8m0fnu).view(torch.uint8)
    sf_u8 = Bs.view(torch.uint8)
    launch_block_n = 128 if is_tensor_scale_input else block_n
    launch_block_k = 128 if is_tensor_scale_input and K % 128 == 0 else block_k
    C = A.new_empty((M, N), dtype=output_dtype)
    BLOCK_SIZE_M = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, launch_block_n),)
    with device_context(A.device):
        wrap_triton(w4a8_block_fp4_matmul_kernel)[grid](
            A,
            as_u8,
            B,
            sf_u8,
            C,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            as_u8.stride(0),
            as_u8.stride(1),
            B.stride(0),
            B.stride(1),
            sf_u8.stride(0),
            sf_u8.stride(1),
            C.stride(0),
            C.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=launch_block_n,
            BLOCK_SIZE_K=launch_block_k,
            GROUP_SIZE_M=8,
            VALUES_PER_BYTE=FP4_VALUES_PER_BYTE,
            SCALE_GROUP_K=FP4_SCALE_GROUP_K,
        )
    return C


def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Block-wise W8A8 FP8 matrix multiplication.

    Computes ``C = A @ B.T`` where both operands are pre-quantized to
    ``float8_e4m3fn`` with per-block scales, and accumulates in float32
    before casting to ``output_dtype``.

    Args:
        A: Quantized activation tensor ``[M, K]`` in ``float8_e4m3fn``.
        B: Quantized weight tensor ``[N, K]`` in ``float8_e4m3fn``.
        As: Per-token-group activation scales ``[M, K // block_size[1]]``.
        Bs: Per-block weight scales ``[N // block_size[0], K // block_size[1]]``.
        block_size: ``[block_n, block_k]`` quantization block dimensions, e.g. ``[128, 128]``.
        output_dtype: dtype of the returned tensor (default: ``torch.float32``).

    Returns:
        Output tensor ``[M, N]`` in ``output_dtype``.
    """
    return torch.ops.finegrained_fp8.w8a8_block_fp8_matmul(
        A, B, As, Bs, block_size, output_dtype
    )


def w8a8_tensor_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Tensor-scale W8A8 FP8 matrix multiplication.

    Computes ``C = A @ B.T`` in tensor-scale mode using pre-quantized FP8
    activations/weights and tensor scales.

    Args:
        A: Quantized activation tensor ``[M, K]`` in ``float8_e4m3fn``.
        B: Quantized weight tensor ``[N, K]`` in ``float8_e4m3fn``.
        As: Per-row activation scales ``[M]``.
        Bs: Single weight scale, scalar or ``[1]``.
        output_dtype: dtype of the returned tensor.

    Returns:
        Output tensor ``[M, N]`` in ``output_dtype``.
    """
    return torch.ops.finegrained_fp8.w8a8_tensor_fp8_matmul(A, B, As, Bs, output_dtype)


def w4a8_block_fp4_matmul(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Block-scale W4A8 FP4 matrix multiplication.

    Computes ``C = A @ B.T`` with FP8 (E4M3) activations and packed FP4 (E2M1)
    weights, both with UE8M0 block scales.

    Args:
        A: FP8 activations ``[M, K]`` (``float8_e4m3fn``).
        As: UE8M0 activation scales ``[M, K // block_size[1]]``.
        B: Packed FP4 weights ``[N, K // 2]`` (``int8``, two codes per byte).
        Bs: UE8M0 weight scales ``[N, K // 32]``.
        block_size: ``[block_n, block_k]``; ``block_k`` must divide ``K``.
        output_dtype: dtype of the returned tensor (default ``bfloat16``).
    """
    return torch.ops.finegrained_fp8.w4a8_block_fp4_matmul(
        A, As, B, Bs, block_size, output_dtype
    )


def fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int] | None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Neutral FP8 matmul dispatcher.

    Routes by weight dtype and scale shape:
    - ``int8`` weights (packed FP4) → ``w4a8_block_fp4_matmul`` (block mode only;
      tensor-mode FP4 — ``block_size`` None or ``[N, K]`` — is not yet supported).
    - ``block_size`` None or ``[N, K]`` → ``w8a8_tensor_fp8_matmul``
    - otherwise → ``w8a8_block_fp8_matmul``
    """
    is_fp4 = B.dtype == torch.int8
    is_tensor_mode = block_size is None or (
        block_size[0] == B.size(0) and block_size[1] == B.size(1)
    )

    if is_fp4:
        if is_tensor_mode:
            raise NotImplementedError(
                "W4A8 FP4 path only supports block mode; tensor-mode "
                "(block_size=None or [N, K]) is not yet implemented."
            )
        return w4a8_block_fp4_matmul(A, As, B, Bs, block_size, output_dtype)

    if is_tensor_mode:
        return w8a8_tensor_fp8_matmul(A, B, As, Bs, output_dtype)

    return w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype)
