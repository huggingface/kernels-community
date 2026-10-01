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

import triton
import triton.language as tl


@triton.jit
def _triton_gemv_int4_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    N,
    K,
    K_PACKED,
    stride_wn,
    stride_wk,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k in range(0, K_PACKED, BLOCK_K_PACKED):
        offs_kp = k + tl.arange(0, BLOCK_K_PACKED)
        mask_k = offs_kp < K_PACKED

        offs_k_low = offs_kp * 2
        offs_k_high = offs_kp * 2 + 1

        mask_low = offs_k_low < K
        mask_high = offs_k_high < K

        x_low = tl.load(x_ptr + offs_k_low, mask=mask_low, other=0.0).to(tl.float32)
        x_high = tl.load(x_ptr + offs_k_high, mask=mask_high, other=0.0).to(tl.float32)

        mask_b = mask_n[:, None] & mask_k[None, :]
        w_packed = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_kp[None, :] * stride_wk, mask=mask_b, other=0
        )

        low = ((w_packed & 0x0F).to(tl.int8) - 8).to(tl.float32)
        high = (((w_packed >> 4) & 0x0F).to(tl.int8) - 8).to(tl.float32)

        low = tl.where(mask_low[None, :], low, 0.0)
        high = tl.where(mask_high[None, :], high, 0.0)

        acc += tl.sum(low * x_low[None, :] + high * x_high[None, :], axis=1)

    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    acc = acc * scale

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += bias

    tl.store(out_ptr + offs_n, acc.to(out_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def _triton_gemv_int2_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    N,
    K,
    K_PACKED,
    stride_wn,
    stride_wk,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k in range(0, K_PACKED, BLOCK_K_PACKED):
        offs_kp = k + tl.arange(0, BLOCK_K_PACKED)
        mask_k = offs_kp < K_PACKED

        offs_k0 = offs_kp * 4
        offs_k1 = offs_kp * 4 + 1
        offs_k2 = offs_kp * 4 + 2
        offs_k3 = offs_kp * 4 + 3

        mask0 = offs_k0 < K
        mask1 = offs_k1 < K
        mask2 = offs_k2 < K
        mask3 = offs_k3 < K

        x0 = tl.load(x_ptr + offs_k0, mask=mask0, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + offs_k1, mask=mask1, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + offs_k2, mask=mask2, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + offs_k3, mask=mask3, other=0.0).to(tl.float32)

        mask_b = mask_n[:, None] & mask_k[None, :]
        w_packed = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_kp[None, :] * stride_wk, mask=mask_b, other=0
        )

        v0 = ((w_packed & 0x03).to(tl.int8) - 2).to(tl.float32)
        v1 = (((w_packed >> 2) & 0x03).to(tl.int8) - 2).to(tl.float32)
        v2 = (((w_packed >> 4) & 0x03).to(tl.int8) - 2).to(tl.float32)
        v3 = (((w_packed >> 6) & 0x03).to(tl.int8) - 2).to(tl.float32)

        v0 = tl.where(mask0[None, :], v0, 0.0)
        v1 = tl.where(mask1[None, :], v1, 0.0)
        v2 = tl.where(mask2[None, :], v2, 0.0)
        v3 = tl.where(mask3[None, :], v3, 0.0)

        acc += tl.sum(v0 * x0[None, :] + v1 * x1[None, :] + v2 * x2[None, :] + v3 * x3[None, :], axis=1)

    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    acc = acc * scale

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += bias

    tl.store(out_ptr + offs_n, acc.to(out_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def _triton_gemv_int8_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    N,
    K,
    stride_wn,
    stride_wk,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x_val = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        mask_b = mask_n[:, None] & mask_k[None, :]
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk, mask=mask_b, other=0).to(
            tl.float32
        )

        acc += tl.sum(w * x_val[None, :], axis=1)

    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    acc = acc * scale

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += bias

    tl.store(out_ptr + offs_n, acc.to(out_ptr.dtype.element_ty), mask=mask_n)
