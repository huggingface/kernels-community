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

import functools

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton

from .act_quant import fp8_act_quant
from .utils import device_context


FP4_VALUES_PER_BYTE = 2
FP4_SCALE_GROUP_K = 32
DEFAULT_BLOCK_M = 32
DEFAULT_BLOCK_N = 128
DEFAULT_BLOCK_K = 128
DEFAULT_GROUP_M = 8
FP8_MAX = 448.0


def _autotune_configs():
    return [
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [2, 4, 8, 16]
        for s in [2, 3, 4]
    ]


@triton.autotune(configs=_autotune_configs(), key=["M", "N", "K", "BLOCK_M", "BLOCK_N", "BLOCK_K"])
@triton.jit
def w4a8_tensor_fp8_matmul_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_byte = tl.arange(0, BLOCK_K // VALUES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_K // SCALE_GROUP_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    a_base = A_ptr + offs_m[:, None] * stride_am
    as_base = AS_ptr + offs_m[:, None] * stride_asm
    b_base = B_ptr + offs_n[:, None] * stride_bn
    sf_base = SF_ptr + offs_n[:, None] * stride_sfn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_base = k0 * BLOCK_K
        a_k = k_base + tl.arange(0, BLOCK_K)
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


@triton.autotune(configs=_autotune_configs(), key=["N", "K", "BLOCK_M", "BLOCK_N", "BLOCK_K"])
@triton.jit
def w4a8_tensor_fp8_matmul_batched_kernel(
    A_ptr,
    AS_ptr,
    B_ptr,
    SF_ptr,
    C_ptr,
    ExpertIds_ptr,
    S,
    N,
    K,
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
    stride_be,
    stride_bn,
    stride_bk,
    stride_sfe,
    stride_sfn,
    stride_sfk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    expert_id = tl.load(ExpertIds_ptr + batch_id).to(tl.int64)

    A_ptr = A_ptr + batch_id * stride_am
    AS_ptr = AS_ptr + batch_id * stride_asm
    B_ptr = B_ptr + expert_id * stride_be
    SF_ptr = SF_ptr + expert_id * stride_sfe
    C_ptr = C_ptr + batch_id * stride_cm

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_byte = tl.arange(0, BLOCK_K // VALUES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_K // SCALE_GROUP_K)

    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_base = k0 * BLOCK_K
        a_k = k_base + tl.arange(0, BLOCK_K)
        b_k = k_base // VALUES_PER_BYTE + offs_k_byte
        s_k = k_base // SCALE_GROUP_K + offs_sf
        a = tl.load(
            A_ptr + offs_m[:, None] * 0 + a_k[None, :] * stride_ak,
            mask=a_k[None, :] < K,
            other=0.0,
        )
        a_scale = tl.load(
            AS_ptr + offs_m[:, None] * 0 + s_k[None, :] * stride_ask,
            mask=s_k[None, :] < K // SCALE_GROUP_K,
            other=0,
        ).to(tl.uint8)
        b_nk = tl.load(
            B_ptr + offs_n[:, None] * stride_bn + b_k[None, :] * stride_bk,
            mask=mask_n[:, None] & (b_k[None, :] < (K // VALUES_PER_BYTE)),
            other=0,
        ).to(tl.uint8)
        b = tl.trans(b_nk)
        sf = tl.load(
            SF_ptr + offs_n[:, None] * stride_sfn + s_k[None, :] * stride_sfk,
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

    c_ptrs = C_ptr + offs_m[:, None] * 0 + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=(offs_m == 0)[:, None] & (offs_n[None, :] < N))


@triton.autotune(configs=_autotune_configs(), key=["S", "N", "K", "BLOCK_M", "BLOCK_N", "BLOCK_K"])
@triton.jit
def w4a8_tensor_fp8_matmul_grouped_kernel(
    A_ptr,
    AS_ptr,
    B_ptr,
    SF_ptr,
    C_ptr,
    Offsets_ptr,
    TileOffsets_ptr,
    S,
    N,
    K,
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
    stride_be,
    stride_bn,
    stride_bk,
    stride_sfe,
    stride_sfn,
    stride_sfk,
    stride_cm,
    stride_cn,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_EXPERTS_BIT_LENGTH: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    total_tiles = tl.load(TileOffsets_ptr + NUM_EXPERTS - 1)
    if pid_m >= total_tiles:
        return

    lo = 0
    hi = NUM_EXPERTS - 1
    for _ in tl.static_range(NUM_EXPERTS_BIT_LENGTH):
        mid = (lo + hi) >> 1
        mid_val = tl.load(TileOffsets_ptr + mid)
        is_left = mid_val <= pid_m
        lo = tl.where(is_left, mid + 1, lo)
        hi = tl.where(is_left, hi, mid)
    expert_id = tl.minimum(lo, NUM_EXPERTS - 1).to(tl.int64)

    prev_eid = tl.maximum(expert_id - 1, 0)
    expert_start = tl.where(expert_id == 0, 0, tl.load(Offsets_ptr + prev_eid))
    expert_end = tl.load(Offsets_ptr + expert_id)
    m_expert = expert_end - expert_start

    expert_tile_start = tl.where(expert_id == 0, 0, tl.load(TileOffsets_ptr + prev_eid))
    local_tile = pid_m - expert_tile_start
    m_off = local_tile * BLOCK_M

    offs_m = m_off + tl.arange(0, BLOCK_M)
    row_mask = offs_m < m_expert
    offs_global_m = expert_start + offs_m
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_byte = tl.arange(0, BLOCK_K // VALUES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_K // SCALE_GROUP_K)

    a_base = A_ptr + offs_global_m[:, None] * stride_am
    as_base = AS_ptr + offs_global_m[:, None] * stride_asm
    b_base = B_ptr + expert_id * stride_be + offs_n[:, None] * stride_bn
    sf_base = SF_ptr + expert_id * stride_sfe + offs_n[:, None] * stride_sfn

    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_base = k0 * BLOCK_K
        a_k = k_base + tl.arange(0, BLOCK_K)
        b_k = k_base // VALUES_PER_BYTE + offs_k_byte
        s_k = k_base // SCALE_GROUP_K + offs_sf
        a = tl.load(
            a_base + a_k[None, :] * stride_ak,
            mask=row_mask[:, None] & (a_k[None, :] < K),
            other=0.0,
        )
        a_scale = tl.load(
            as_base + s_k[None, :] * stride_ask,
            mask=row_mask[:, None] & (s_k[None, :] < K // SCALE_GROUP_K),
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

    c_ptrs = C_ptr + offs_global_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = row_mask[:, None] & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)


@functools.cache
def _get_tensor_grouped_kernel():
    return w4a8_tensor_fp8_matmul_grouped_kernel


@functools.cache
def _get_tensor_batched_kernel():
    return w4a8_tensor_fp8_matmul_batched_kernel


@functools.cache
def _get_tensor_kernel():
    return w4a8_tensor_fp8_matmul_kernel


def _resolve_block_k(block_size: list[int] | None, K: int) -> int:
    block_k = DEFAULT_BLOCK_K if block_size is None else block_size[1]
    if K % block_k == 0:
        return block_k
    if block_size is None:
        return FP4_SCALE_GROUP_K
    raise AssertionError(f"K (={K}) must be divisible by block_k (={block_k}) for fp8_act_quant")


def _expand_activation_scales_for_fp4(As: torch.Tensor, K: int, block_k: int) -> torch.Tensor:
    expected_groups = K // FP4_SCALE_GROUP_K
    if As.shape[-1] == expected_groups:
        return As

    expected_blocks = K // block_k
    assert As.shape[-1] == expected_blocks, (
        f"As shape {tuple(As.shape)} incompatible with K={K}, block_k={block_k}; "
        f"expected last dim {expected_blocks} or {expected_groups}"
    )
    assert block_k % FP4_SCALE_GROUP_K == 0, (
        f"block_k (={block_k}) must be divisible by {FP4_SCALE_GROUP_K} for FP4 scale expansion"
    )
    return As.repeat_interleave(block_k // FP4_SCALE_GROUP_K, dim=-1)


def w4a8_tensor_fp8_matmul(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int] | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert A.ndim == 2 and As.ndim == 2 and B.ndim == 2 and Bs.ndim == 2
    assert A.dtype == torch.float8_e4m3fn
    assert B.dtype == torch.int8
    assert Bs.dtype == torch.float8_e8m0fnu
    assert As.dtype in (torch.float32, torch.float8_e8m0fnu)
    assert A.is_contiguous() and As.is_contiguous() and B.is_contiguous() and Bs.is_contiguous()

    M, K = A.shape
    N, K_half = B.shape
    assert K == FP4_VALUES_PER_BYTE * K_half, (
        f"K (={K}) must equal {FP4_VALUES_PER_BYTE} * B.shape[1] (={K_half})"
    )
    assert Bs.shape == (N, K // FP4_SCALE_GROUP_K), (
        f"Bs shape {tuple(Bs.shape)} != ({N}, {K // FP4_SCALE_GROUP_K})"
    )
    assert K % FP4_SCALE_GROUP_K == 0, f"K (={K}) must be a multiple of {FP4_SCALE_GROUP_K}"

    block_k = _resolve_block_k(block_size, K)
    As = _expand_activation_scales_for_fp4(As, K, block_k)
    assert As.shape == (M, K // FP4_SCALE_GROUP_K), f"As shape {tuple(As.shape)} != ({M}, {K // FP4_SCALE_GROUP_K})"

    as_u8 = As.to(torch.float8_e8m0fnu).contiguous().view(torch.uint8)
    sf_u8 = Bs.contiguous().view(torch.uint8)
    block_n = DEFAULT_BLOCK_N if block_size is None else block_size[0]
    C = A.new_empty((M, N), dtype=output_dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    with device_context(A.device):
        _get_tensor_kernel()[grid](
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
            BLOCK_M=DEFAULT_BLOCK_M,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=DEFAULT_GROUP_M,
            VALUES_PER_BYTE=FP4_VALUES_PER_BYTE,
            SCALE_GROUP_K=FP4_SCALE_GROUP_K,
        )
    return C


def w4a8_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int] | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert A.ndim == 2
    assert A.dtype in (torch.bfloat16, torch.float16)
    block_k = _resolve_block_k(block_size, A.shape[1])
    qA, As = fp8_act_quant(A, block_k)
    return w4a8_tensor_fp8_matmul(qA, As, B, Bs, [DEFAULT_BLOCK_N if block_size is None else block_size[0], block_k], output_dtype)


def w4a8_tensor_fp8_matmul_batched(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int] | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert A.ndim == 2 and As.ndim == 2 and B.ndim == 3 and Bs.ndim == 3
    assert expert_ids.ndim == 1
    assert A.dtype == torch.float8_e4m3fn
    assert As.dtype in (torch.float32, torch.float8_e8m0fnu)
    assert B.dtype == torch.int8
    assert Bs.dtype == torch.float8_e8m0fnu

    S, K = A.shape
    E, N, K_half = B.shape
    assert expert_ids.shape[0] == S
    assert K == FP4_VALUES_PER_BYTE * K_half, (
        f"K (={K}) must equal {FP4_VALUES_PER_BYTE} * B.shape[2] (={K_half})"
    )
    assert Bs.shape == (E, N, K // FP4_SCALE_GROUP_K), (
        f"Bs shape {tuple(Bs.shape)} != ({E}, {N}, {K // FP4_SCALE_GROUP_K})"
    )
    assert K % FP4_SCALE_GROUP_K == 0, f"K (={K}) must be a multiple of {FP4_SCALE_GROUP_K}"

    block_k = _resolve_block_k(block_size, K)
    As = _expand_activation_scales_for_fp4(As, K, block_k)
    assert As.shape == (S, K // FP4_SCALE_GROUP_K), f"As shape {tuple(As.shape)} != ({S}, {K // FP4_SCALE_GROUP_K})"

    as_u8 = As.to(torch.float8_e8m0fnu).contiguous().view(torch.uint8)
    sf_u8 = Bs.contiguous().view(torch.uint8)
    expert_ids = expert_ids.to(device=A.device, dtype=torch.int32).contiguous()
    block_n = DEFAULT_BLOCK_N if block_size is None else block_size[0]
    C = A.new_empty((S, N), dtype=output_dtype)
    grid = (S, triton.cdiv(N, block_n))
    with device_context(A.device):
        _get_tensor_batched_kernel()[grid](
            A,
            as_u8,
            B,
            sf_u8,
            C,
            expert_ids,
            S,
            N,
            K,
            A.stride(0),
            A.stride(1),
            as_u8.stride(0),
            as_u8.stride(1),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            sf_u8.stride(0),
            sf_u8.stride(1),
            sf_u8.stride(2),
            C.stride(0),
            C.stride(1),
            BLOCK_M=DEFAULT_BLOCK_M,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            VALUES_PER_BYTE=FP4_VALUES_PER_BYTE,
            SCALE_GROUP_K=FP4_SCALE_GROUP_K,
        )
    return C


def w4a8_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int] | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert A.ndim == 2
    assert A.dtype in (torch.bfloat16, torch.float16)
    block_k = _resolve_block_k(block_size, A.shape[1])
    qA, As = fp8_act_quant(A, block_k)
    return w4a8_tensor_fp8_matmul_batched(
        qA,
        As,
        B,
        Bs,
        expert_ids,
        [DEFAULT_BLOCK_N if block_size is None else block_size[0], block_k],
        output_dtype,
    )


def w4a8_tensor_fp8_matmul_grouped(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_size: list[int] | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert A.ndim == 2 and As.ndim == 2 and B.ndim == 3 and Bs.ndim == 3
    assert offsets.ndim == 1 and tokens_per_expert.ndim == 1
    assert A.dtype == torch.float8_e4m3fn
    assert As.dtype in (torch.float32, torch.float8_e8m0fnu)
    assert B.dtype == torch.int8
    assert Bs.dtype == torch.float8_e8m0fnu

    S, K = A.shape
    E, N, K_half = B.shape
    assert offsets.shape[0] == E and tokens_per_expert.shape[0] == E
    assert K == FP4_VALUES_PER_BYTE * K_half, (
        f"K (={K}) must equal {FP4_VALUES_PER_BYTE} * B.shape[2] (={K_half})"
    )
    assert Bs.shape == (E, N, K // FP4_SCALE_GROUP_K), (
        f"Bs shape {tuple(Bs.shape)} != ({E}, {N}, {K // FP4_SCALE_GROUP_K})"
    )
    assert K % FP4_SCALE_GROUP_K == 0, f"K (={K}) must be a multiple of {FP4_SCALE_GROUP_K}"

    block_k = _resolve_block_k(block_size, K)
    As = _expand_activation_scales_for_fp4(As, K, block_k)
    assert As.shape == (S, K // FP4_SCALE_GROUP_K), f"As shape {tuple(As.shape)} != ({S}, {K // FP4_SCALE_GROUP_K})"

    as_u8 = As.to(torch.float8_e8m0fnu).contiguous().view(torch.uint8)
    sf_u8 = Bs.contiguous().view(torch.uint8)
    offsets = offsets.to(device=A.device, dtype=torch.int32).contiguous()
    tokens_per_expert = tokens_per_expert.to(device=A.device, dtype=torch.int32).contiguous()
    block_n = DEFAULT_BLOCK_N if block_size is None else block_size[0]
    block_m = DEFAULT_BLOCK_M
    C = A.new_empty((S, N), dtype=output_dtype)
    tiles_per_expert = (tokens_per_expert + block_m - 1) // block_m
    tile_offsets = torch.cumsum(tiles_per_expert, dim=0).to(torch.int32)
    max_m_tiles = triton.cdiv(S, block_m) + E
    grid = (max_m_tiles, triton.cdiv(N, block_n))
    with device_context(A.device):
        _get_tensor_grouped_kernel()[grid](
            A,
            as_u8,
            B,
            sf_u8,
            C,
            offsets,
            tile_offsets,
            S,
            N,
            K,
            A.stride(0),
            A.stride(1),
            as_u8.stride(0),
            as_u8.stride(1),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            sf_u8.stride(0),
            sf_u8.stride(1),
            sf_u8.stride(2),
            C.stride(0),
            C.stride(1),
            NUM_EXPERTS=E,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            NUM_EXPERTS_BIT_LENGTH=E.bit_length(),
            VALUES_PER_BYTE=FP4_VALUES_PER_BYTE,
            SCALE_GROUP_K=FP4_SCALE_GROUP_K,
        )
    return C


def w4a8_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_size: list[int] | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert A.ndim == 2
    assert A.dtype in (torch.bfloat16, torch.float16)
    block_k = _resolve_block_k(block_size, A.shape[1])
    qA, As = fp8_act_quant(A, block_k)
    return w4a8_tensor_fp8_matmul_grouped(
        qA,
        As,
        B,
        Bs,
        offsets,
        tokens_per_expert,
        [DEFAULT_BLOCK_N if block_size is None else block_size[0], block_k],
        output_dtype,
    )