# Copyright 2026 Google LLC and The HuggingFace Inc. team. All rights reserved.
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

from .gemv import (
    _triton_gemv_int2_kernel,
    _triton_gemv_int4_kernel,
    _triton_gemv_int8_kernel,
)


@triton.jit
def _triton_gemm_int4_kernel(
    a_ptr,
    b_ptr,
    scale_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    K_PACKED,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K_PACKED, BLOCK_K_PACKED):
        offs_kp = k + tl.arange(0, BLOCK_K_PACKED)
        mask_k = offs_kp < K_PACKED

        offs_k_low = offs_kp * 2
        offs_k_high = offs_kp * 2 + 1

        mask_m = offs_m[:, None] < M
        mask_low = mask_m & (offs_k_low[None, :] < K)
        mask_high = mask_m & (offs_k_high[None, :] < K)

        a_low = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k_low[None, :] * stride_ak, mask=mask_low, other=0.0
        )
        a_high = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k_high[None, :] * stride_ak, mask=mask_high, other=0.0
        )

        mask_n = offs_n[:, None] < N
        mask_b = mask_n & mask_k[None, :]
        b_packed = tl.load(
            b_ptr + offs_n[:, None] * stride_bn + offs_kp[None, :] * stride_bk, mask=mask_b, other=0
        )

        low = ((b_packed & 0x0F).to(tl.int8) - 8).to(a_ptr.dtype.element_ty)
        high = (((b_packed >> 4) & 0x0F).to(tl.int8) - 8).to(a_ptr.dtype.element_ty)

        low = tl.where(mask_b, low, 0.0)
        high = tl.where(mask_b, high, 0.0)

        acc = tl.dot(a_low, tl.trans(low), acc)
        acc = tl.dot(a_high, tl.trans(high), acc)

    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0).to(tl.float32)
    acc = acc * scale[None, :]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(
        c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=mask_c,
    )


@triton.jit
def _triton_gemm_int2_kernel(
    a_ptr,
    b_ptr,
    scale_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    K_PACKED,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K_PACKED, BLOCK_K_PACKED):
        offs_kp = k + tl.arange(0, BLOCK_K_PACKED)
        mask_k = offs_kp < K_PACKED

        offs_k0 = offs_kp * 4
        offs_k1 = offs_kp * 4 + 1
        offs_k2 = offs_kp * 4 + 2
        offs_k3 = offs_kp * 4 + 3

        mask_m = offs_m[:, None] < M
        mask0 = mask_m & (offs_k0[None, :] < K)
        mask1 = mask_m & (offs_k1[None, :] < K)
        mask2 = mask_m & (offs_k2[None, :] < K)
        mask3 = mask_m & (offs_k3[None, :] < K)

        a0 = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k0[None, :] * stride_ak, mask=mask0, other=0.0)
        a1 = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k1[None, :] * stride_ak, mask=mask1, other=0.0)
        a2 = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k2[None, :] * stride_ak, mask=mask2, other=0.0)
        a3 = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k3[None, :] * stride_ak, mask=mask3, other=0.0)

        mask_n = offs_n[:, None] < N
        mask_b = mask_n & mask_k[None, :]
        b_packed = tl.load(
            b_ptr + offs_n[:, None] * stride_bn + offs_kp[None, :] * stride_bk, mask=mask_b, other=0
        )

        v0 = ((b_packed & 0x03).to(tl.int8) - 2).to(a_ptr.dtype.element_ty)
        v1 = (((b_packed >> 2) & 0x03).to(tl.int8) - 2).to(a_ptr.dtype.element_ty)
        v2 = (((b_packed >> 4) & 0x03).to(tl.int8) - 2).to(a_ptr.dtype.element_ty)
        v3 = (((b_packed >> 6) & 0x03).to(tl.int8) - 2).to(a_ptr.dtype.element_ty)

        v0 = tl.where(mask_b, v0, 0.0)
        v1 = tl.where(mask_b, v1, 0.0)
        v2 = tl.where(mask_b, v2, 0.0)
        v3 = tl.where(mask_b, v3, 0.0)

        acc = tl.dot(a0, tl.trans(v0), acc)
        acc = tl.dot(a1, tl.trans(v1), acc)
        acc = tl.dot(a2, tl.trans(v2), acc)
        acc = tl.dot(a3, tl.trans(v3), acc)

    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0).to(tl.float32)
    acc = acc * scale[None, :]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(
        c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=mask_c,
    )


@triton.jit
def _triton_gemm_int8_kernel(
    a_ptr,
    b_ptr,
    scale_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        mask_m = offs_m[:, None] < M
        mask_a = mask_m & mask_k[None, :]
        a_tile = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=mask_a, other=0.0)

        mask_n = offs_n[:, None] < N
        mask_b = mask_n & mask_k[None, :]
        b_tile = tl.load(b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk, mask=mask_b, other=0)

        b_cast = b_tile.to(a_ptr.dtype.element_ty)
        b_cast = tl.where(mask_b, b_cast, 0.0)

        acc = tl.dot(a_tile, tl.trans(b_cast), acc)

    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0).to(tl.float32)
    acc = acc * scale[None, :]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(
        c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=mask_c,
    )


@torch.library.custom_op(
    add_op_namespace_prefix("lowbit_gemm"), mutates_args=(), device_types="cuda"
)
def lowbit_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    num_bits: int = 4,
) -> torch.Tensor:
    """Fused low-bit GEMV (M==1) and tiled GEMM (M>1) for quantized weights (int2, int4, int8).

    Streams packed integer weights directly from VRAM into registers without
    allocating intermediate float weight tensors.
    """
    if num_bits not in (2, 4, 8):
        raise ValueError(f"Unsupported num_bits: {num_bits}")

    orig_shape = x.shape
    in_features = orig_shape[-1]
    out_features = weight_scale.shape[0]

    x_2d = x.contiguous().view(-1, in_features)
    M = x_2d.shape[0]
    N = out_features
    K = in_features

    pack_factor = 8 // num_bits
    expected_k_packed = (K + pack_factor - 1) // pack_factor
    if weight.dim() != 2 or weight.shape[0] != N or weight.shape[1] != expected_k_packed:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match expected ({N}, {expected_k_packed}) "
            f"for K={K}, num_bits={num_bits}"
        )

    weight_scale = weight_scale.squeeze(-1) if weight_scale.dim() > 1 else weight_scale
    scale_flat = weight_scale.contiguous()
    bias_flat = bias.contiguous() if bias is not None else None

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    with torch.cuda.device(x.device):
        if M == 1:
            x_vec = x_2d.squeeze(0)
            out_vec = out.squeeze(0)
            BLOCK_N = 64
            grid = (triton.cdiv(N, BLOCK_N),)
            if num_bits == 4:
                BLOCK_K_PACKED = 64
                K_PACKED = (K + 1) // 2
                _triton_gemv_int4_kernel[grid](
                    x_vec,
                    weight,
                    scale_flat,
                    bias_flat if bias_flat is not None else x_vec,
                    out_vec,
                    N,
                    K,
                    K_PACKED,
                    weight.stride(0),
                    weight.stride(1),
                    HAS_BIAS=bias is not None,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K_PACKED=BLOCK_K_PACKED,
                )
            elif num_bits == 2:
                BLOCK_K_PACKED = 64
                K_PACKED = (K + 3) // 4
                _triton_gemv_int2_kernel[grid](
                    x_vec,
                    weight,
                    scale_flat,
                    bias_flat if bias_flat is not None else x_vec,
                    out_vec,
                    N,
                    K,
                    K_PACKED,
                    weight.stride(0),
                    weight.stride(1),
                    HAS_BIAS=bias is not None,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K_PACKED=BLOCK_K_PACKED,
                )
            elif num_bits == 8:
                BLOCK_K = 64
                _triton_gemv_int8_kernel[grid](
                    x_vec,
                    weight,
                    scale_flat,
                    bias_flat if bias_flat is not None else x_vec,
                    out_vec,
                    N,
                    K,
                    weight.stride(0),
                    weight.stride(1),
                    HAS_BIAS=bias is not None,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                )
            else:
                raise ValueError(f"Unsupported num_bits: {num_bits}")
        else:
            BLOCK_M = 16 if M <= 16 else 32
            BLOCK_N = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            if num_bits == 4:
                BLOCK_K_PACKED = 32
                K_PACKED = (K + 1) // 2
                _triton_gemm_int4_kernel[grid](
                    x_2d,
                    weight,
                    scale_flat,
                    bias_flat if bias_flat is not None else x_2d,
                    out,
                    M,
                    N,
                    K,
                    K_PACKED,
                    x_2d.stride(0),
                    x_2d.stride(1),
                    weight.stride(0),
                    weight.stride(1),
                    out.stride(0),
                    out.stride(1),
                    HAS_BIAS=bias is not None,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K_PACKED=BLOCK_K_PACKED,
                )
            elif num_bits == 2:
                BLOCK_K_PACKED = 32
                K_PACKED = (K + 3) // 4
                _triton_gemm_int2_kernel[grid](
                    x_2d,
                    weight,
                    scale_flat,
                    bias_flat if bias_flat is not None else x_2d,
                    out,
                    M,
                    N,
                    K,
                    K_PACKED,
                    x_2d.stride(0),
                    x_2d.stride(1),
                    weight.stride(0),
                    weight.stride(1),
                    out.stride(0),
                    out.stride(1),
                    HAS_BIAS=bias is not None,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K_PACKED=BLOCK_K_PACKED,
                )
            elif num_bits == 8:
                BLOCK_K = 32
                _triton_gemm_int8_kernel[grid](
                    x_2d,
                    weight,
                    scale_flat,
                    bias_flat if bias_flat is not None else x_2d,
                    out,
                    M,
                    N,
                    K,
                    x_2d.stride(0),
                    x_2d.stride(1),
                    weight.stride(0),
                    weight.stride(1),
                    out.stride(0),
                    out.stride(1),
                    HAS_BIAS=bias is not None,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                )
            else:
                raise ValueError(f"Unsupported num_bits: {num_bits}")

    return out.view(*orig_shape[:-1], out_features)


@lowbit_gemm.register_fake
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    num_bits: int = 4,
) -> torch.Tensor:
    out_features = weight_scale.shape[0]
    return x.new_empty((*x.shape[:-1], out_features))
