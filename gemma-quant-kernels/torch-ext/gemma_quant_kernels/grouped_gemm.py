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


@triton.jit
def _triton_grouped_gemm_int4_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    offsets_ptr,
    out_ptr,
    N,
    K,
    K_PACKED,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_se,
    stride_sn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    expert_id = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_idx = tl.load(offsets_ptr + expert_id)
    end_idx = tl.load(offsets_ptr + expert_id + 1)
    M_e = end_idx - start_idx
    if M_e <= 0:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    for m_start in range(0, M_e, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M_e

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K_PACKED, BLOCK_K_PACKED):
            offs_kp = k + tl.arange(0, BLOCK_K_PACKED)
            mask_k = offs_kp < K_PACKED

            offs_k0 = offs_kp * 2
            offs_k1 = offs_kp * 2 + 1

            mask_a0 = mask_m[:, None] & (offs_k0[None, :] < K)
            mask_a1 = mask_m[:, None] & (offs_k1[None, :] < K)

            a0 = tl.load(
                x_ptr + (start_idx + offs_m)[:, None] * stride_xm + offs_k0[None, :] * stride_xk,
                mask=mask_a0,
                other=0.0,
            )
            a1 = tl.load(
                x_ptr + (start_idx + offs_m)[:, None] * stride_xm + offs_k1[None, :] * stride_xk,
                mask=mask_a1,
                other=0.0,
            )

            mask_b = mask_n[:, None] & mask_k[None, :]
            b_packed = tl.load(
                w_ptr + expert_id * stride_we + offs_n[:, None] * stride_wn + offs_kp[None, :] * stride_wk,
                mask=mask_b,
                other=0,
            )

            low = ((b_packed & 0x0F).to(tl.int8) - 8).to(x_ptr.dtype.element_ty)
            high = (((b_packed >> 4) & 0x0F).to(tl.int8) - 8).to(x_ptr.dtype.element_ty)

            low = tl.where(mask_b, low, 0.0)
            high = tl.where(mask_b, high, 0.0)

            acc = tl.dot(a0, tl.trans(low), acc)
            acc = tl.dot(a1, tl.trans(high), acc)

        scale = tl.load(scale_ptr + expert_id * stride_se + offs_n * stride_sn, mask=mask_n, other=1.0).to(
            tl.float32
        )
        acc = acc * scale[None, :]

        tl.store(
            out_ptr + (start_idx + offs_m)[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


@triton.jit
def _triton_grouped_gemm_int2_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    offsets_ptr,
    out_ptr,
    N,
    K,
    K_PACKED,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_se,
    stride_sn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    expert_id = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_idx = tl.load(offsets_ptr + expert_id)
    end_idx = tl.load(offsets_ptr + expert_id + 1)
    M_e = end_idx - start_idx
    if M_e <= 0:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    for m_start in range(0, M_e, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M_e

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K_PACKED, BLOCK_K_PACKED):
            offs_kp = k + tl.arange(0, BLOCK_K_PACKED)
            mask_k = offs_kp < K_PACKED

            offs_k0 = offs_kp * 4
            offs_k1 = offs_kp * 4 + 1
            offs_k2 = offs_kp * 4 + 2
            offs_k3 = offs_kp * 4 + 3

            mask_a0 = mask_m[:, None] & (offs_k0[None, :] < K)
            mask_a1 = mask_m[:, None] & (offs_k1[None, :] < K)
            mask_a2 = mask_m[:, None] & (offs_k2[None, :] < K)
            mask_a3 = mask_m[:, None] & (offs_k3[None, :] < K)

            a0 = tl.load(
                x_ptr + (start_idx + offs_m)[:, None] * stride_xm + offs_k0[None, :] * stride_xk,
                mask=mask_a0,
                other=0.0,
            )
            a1 = tl.load(
                x_ptr + (start_idx + offs_m)[:, None] * stride_xm + offs_k1[None, :] * stride_xk,
                mask=mask_a1,
                other=0.0,
            )
            a2 = tl.load(
                x_ptr + (start_idx + offs_m)[:, None] * stride_xm + offs_k2[None, :] * stride_xk,
                mask=mask_a2,
                other=0.0,
            )
            a3 = tl.load(
                x_ptr + (start_idx + offs_m)[:, None] * stride_xm + offs_k3[None, :] * stride_xk,
                mask=mask_a3,
                other=0.0,
            )

            mask_b = mask_n[:, None] & mask_k[None, :]
            b_packed = tl.load(
                w_ptr + expert_id * stride_we + offs_n[:, None] * stride_wn + offs_kp[None, :] * stride_wk,
                mask=mask_b,
                other=0,
            )

            v0 = ((b_packed & 0x03).to(tl.int8) - 2).to(x_ptr.dtype.element_ty)
            v1 = (((b_packed >> 2) & 0x03).to(tl.int8) - 2).to(x_ptr.dtype.element_ty)
            v2 = (((b_packed >> 4) & 0x03).to(tl.int8) - 2).to(x_ptr.dtype.element_ty)
            v3 = (((b_packed >> 6) & 0x03).to(tl.int8) - 2).to(x_ptr.dtype.element_ty)

            v0 = tl.where(mask_b, v0, 0.0)
            v1 = tl.where(mask_b, v1, 0.0)
            v2 = tl.where(mask_b, v2, 0.0)
            v3 = tl.where(mask_b, v3, 0.0)

            acc = tl.dot(a0, tl.trans(v0), acc)
            acc = tl.dot(a1, tl.trans(v1), acc)
            acc = tl.dot(a2, tl.trans(v2), acc)
            acc = tl.dot(a3, tl.trans(v3), acc)

        scale = tl.load(scale_ptr + expert_id * stride_se + offs_n * stride_sn, mask=mask_n, other=1.0).to(
            tl.float32
        )
        acc = acc * scale[None, :]

        tl.store(
            out_ptr + (start_idx + offs_m)[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


@triton.jit
def _triton_grouped_gemm_int8_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    offsets_ptr,
    out_ptr,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_se,
    stride_sn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_idx = tl.load(offsets_ptr + expert_id)
    end_idx = tl.load(offsets_ptr + expert_id + 1)
    M_e = end_idx - start_idx
    if M_e <= 0:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    for m_start in range(0, M_e, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M_e

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            mask_a = mask_m[:, None] & mask_k[None, :]
            a_tile = tl.load(
                x_ptr + (start_idx + offs_m)[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=mask_a,
                other=0.0,
            )

            mask_b = mask_n[:, None] & mask_k[None, :]
            b_tile = tl.load(
                w_ptr + expert_id * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=mask_b,
                other=0,
            )

            b_cast = b_tile.to(x_ptr.dtype.element_ty)
            b_cast = tl.where(mask_b, b_cast, 0.0)

            acc = tl.dot(a_tile, tl.trans(b_cast), acc)

        scale = tl.load(scale_ptr + expert_id * stride_se + offs_n * stride_sn, mask=mask_n, other=1.0).to(
            tl.float32
        )
        acc = acc * scale[None, :]

        tl.store(
            out_ptr + (start_idx + offs_m)[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


@torch.library.custom_op(
    add_op_namespace_prefix("grouped_lowbit_gemm"), mutates_args=(), device_types="cuda"
)
def grouped_lowbit_gemm(
    permuted_x: torch.Tensor,
    packed_w: torch.Tensor,
    scales: torch.Tensor,
    offsets: torch.Tensor,
    num_bits: int = 4,
) -> torch.Tensor:
    """Fused Grouped GEMM for low-bit MoE experts (int2, int4, int8).

    Dispatches all routed tokens and experts in a single kernel launch.
    """
    if num_bits not in (2, 4, 8):
        raise ValueError(f"Unsupported num_bits: {num_bits}")

    permuted_x = permuted_x.contiguous()
    P = permuted_x.shape[0]
    num_experts = packed_w.shape[0]
    N = packed_w.shape[1]
    K = permuted_x.shape[1]
    K_PACKED = packed_w.shape[2]

    pack_factor = 8 // num_bits
    expected_k_packed = (K + pack_factor - 1) // pack_factor
    if K_PACKED != expected_k_packed:
        raise ValueError(
            f"packed_w shape {tuple(packed_w.shape)} does not match expected K_PACKED={expected_k_packed} "
            f"for K={K}, num_bits={num_bits}"
        )

    if scales.dim() == 3:
        scales = scales.squeeze(-1)
    scales = scales.contiguous()
    offsets = offsets.contiguous()

    out = torch.zeros((P, N), dtype=permuted_x.dtype, device=permuted_x.device)

    BLOCK_M = 16
    BLOCK_N = 64
    grid = (num_experts, triton.cdiv(N, BLOCK_N))

    with torch.cuda.device(permuted_x.device):
        if num_bits == 4:
            BLOCK_K_PACKED = 32
            _triton_grouped_gemm_int4_kernel[grid](
                permuted_x,
                packed_w,
                scales,
                offsets,
                out,
                N,
                K,
                K_PACKED,
                permuted_x.stride(0),
                permuted_x.stride(1),
                packed_w.stride(0),
                packed_w.stride(1),
                packed_w.stride(2),
                scales.stride(0),
                scales.stride(1),
                out.stride(0),
                out.stride(1),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K_PACKED=BLOCK_K_PACKED,
            )
        elif num_bits == 2:
            BLOCK_K_PACKED = 32
            _triton_grouped_gemm_int2_kernel[grid](
                permuted_x,
                packed_w,
                scales,
                offsets,
                out,
                N,
                K,
                K_PACKED,
                permuted_x.stride(0),
                permuted_x.stride(1),
                packed_w.stride(0),
                packed_w.stride(1),
                packed_w.stride(2),
                scales.stride(0),
                scales.stride(1),
                out.stride(0),
                out.stride(1),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K_PACKED=BLOCK_K_PACKED,
            )
        elif num_bits == 8:
            BLOCK_K = 32
            _triton_grouped_gemm_int8_kernel[grid](
                permuted_x,
                packed_w,
                scales,
                offsets,
                out,
                N,
                K,
                permuted_x.stride(0),
                permuted_x.stride(1),
                packed_w.stride(0),
                packed_w.stride(1),
                packed_w.stride(2),
                scales.stride(0),
                scales.stride(1),
                out.stride(0),
                out.stride(1),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
            )
        else:
            raise ValueError(f"Unsupported num_bits: {num_bits}")

    return out


@grouped_lowbit_gemm.register_fake
def _(
    permuted_x: torch.Tensor,
    packed_w: torch.Tensor,
    scales: torch.Tensor,
    offsets: torch.Tensor,
    num_bits: int = 4,
) -> torch.Tensor:
    return permuted_x.new_empty((permuted_x.shape[0], packed_w.shape[1]))
