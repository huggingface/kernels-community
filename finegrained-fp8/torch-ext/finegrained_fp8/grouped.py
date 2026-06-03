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

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from .act_quant import fp8_act_quant
from .utils import (
    FP4_SCALE_GROUP_K,
    FP4_VALUES_PER_BYTE,
    device_context,
    fp4_expand_activation_scales,
    fp4_resolve_block_k,
)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [2, 4, 8, 16]
        for s in [2, 3, 4, 5]
    ],
    key=["N", "K", "BLOCK_SIZE_M"],
)
@triton.jit
def w8a8_block_fp8_matmul_grouped_kernel(
    A,  # (S, K)  raw BF16/FP16 activations, sorted/grouped by expert id
    B,  # (E, N, K) FP8 weight matrices
    C,  # (S, N)  output
    Bs,  # (E, N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales
    Offsets,  # (E,) int32 — cumulative row-end per expert
    TileOffsets,  # (E,) int32 — cumulative tile-end per expert
    # Shape
    S,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bs_e,
    stride_bs_k,
    stride_bs_n,
    # Meta-parameters
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    NUM_EXPERTS_BIT_LENGTH: tl.constexpr,
):
    """Block-scale grouped FP8 expert matmul kernel.

    Tokens are assumed sorted by expert. The kernel maps each M-tile to its
    owning expert via ``TileOffsets`` and applies fused activation quantization.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Exit early for programs beyond the actual tile count.
    total_tiles = tl.load(TileOffsets + NUM_EXPERTS - 1)
    if pid_m >= total_tiles:
        return

    # Binary search in TileOffsets to find the owning expert.
    # Finds the smallest e such that TileOffsets[e] > pid_m (upper_bound semantics),
    # which is the expert whose tile range contains pid_m.
    # O(log2(NUM_EXPERTS)) loads instead of the O(NUM_EXPERTS) linear scan.
    # NUM_EXPERTS_BIT_LENGTH is ceil(log2(E))+1 for powers-of-two, giving one
    # harmless extra iteration when lo==hi; it's a compile-time constant so the
    # loop is fully unrolled by the compiler.
    lo = 0
    hi = NUM_EXPERTS
    for _ in tl.static_range(NUM_EXPERTS_BIT_LENGTH):
        mid = (lo + hi) >> 1
        mid_val = tl.load(TileOffsets + mid)
        is_left = mid_val <= pid_m
        lo = tl.where(is_left, mid + 1, lo)
        hi = tl.where(is_left, hi, mid)

    # Cast expert_id to int64 to prevent int32 overflow when computing
    # expert_id * stride_be (e.g. 255 * 9_437_184 > 2^31 for 256 experts of
    # 3072×3072 FP8 weights).
    expert_id = lo.to(tl.int64)

    prev_eid = tl.maximum(expert_id - 1, 0)
    expert_start = tl.where(expert_id == 0, 0, tl.load(Offsets + prev_eid))
    expert_end = tl.load(Offsets + expert_id)
    M_expert = expert_end - expert_start

    expert_tile_start = tl.where(expert_id == 0, 0, tl.load(TileOffsets + prev_eid))
    local_tile = pid_m - expert_tile_start
    m_off = local_tile * BLOCK_SIZE_M

    offs_am = m_off + tl.arange(0, BLOCK_SIZE_M)
    row_mask = offs_am < M_expert
    offs_global_m = expert_start + offs_am

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + offs_global_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        B
        + expert_id * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )
    bs_ptrs = Bs + expert_id * stride_bs_e + pid_n * stride_bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # ---- fused fp8_act_quant ----
        a_raw = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        a_s = tl.max(tl.abs(a_raw), axis=1) / 448.0
        a = (a_raw / tl.maximum(a_s[:, None], 1e-12)).to(tl.float8e4nv)
        # ---- matmul ----
        b = tl.load(b_ptrs)
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

    c_ptrs = C + stride_cm * offs_global_m[:, None] + stride_cn * offs_bn[None, :]
    c_mask = row_mask[:, None]
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [2, 4, 8, 16]
        for s in [2, 3, 4, 5]
    ],
    key=["N", "K", "BLOCK_SIZE_M"],
)
@triton.jit
def w8a8_tensor_fp8_matmul_grouped_kernel(
    A,  # (S, K) raw BF16/FP16 activations, sorted/grouped by expert idc
    B,  # (E, N, K) FP8 weight matrices
    C,  # (S, N) output
    As,  # (S, 1) activation scales
    Bs,  # (E, 1, 1) per-tensor weight scales
    Offsets,
    TileOffsets,
    S,
    N,
    K,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_as_m,
    stride_bs_e,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    NUM_EXPERTS_BIT_LENGTH: tl.constexpr,
):
    """Tensor-scale grouped FP8 expert matmul kernel.

    Uses grouped expert scheduling with pre-quantized activations plus
    per-token activation scales and per-expert tensor weight scales.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    total_tiles = tl.load(TileOffsets + NUM_EXPERTS - 1)
    if pid_m >= total_tiles:
        return

    lo = 0
    hi = NUM_EXPERTS
    for _ in tl.static_range(NUM_EXPERTS_BIT_LENGTH):
        mid = (lo + hi) >> 1
        mid_val = tl.load(TileOffsets + mid)
        is_left = mid_val <= pid_m
        lo = tl.where(is_left, mid + 1, lo)
        hi = tl.where(is_left, hi, mid)
    expert_id = lo.to(tl.int64)

    prev_eid = tl.maximum(expert_id - 1, 0)
    expert_start = tl.where(expert_id == 0, 0, tl.load(Offsets + prev_eid))
    expert_end = tl.load(Offsets + expert_id)
    M_expert = expert_end - expert_start

    expert_tile_start = tl.where(expert_id == 0, 0, tl.load(TileOffsets + prev_eid))
    local_tile = pid_m - expert_tile_start
    m_off = local_tile * BLOCK_SIZE_M

    offs_am = m_off + tl.arange(0, BLOCK_SIZE_M)
    row_mask = offs_am < M_expert
    offs_global_m = expert_start + offs_am

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + offs_global_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        B
        + expert_id * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    a_s = tl.load(As + offs_global_m * stride_as_m, mask=row_mask, other=0.0)
    b_s = tl.load(Bs + expert_id * stride_bs_e)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)

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

    c_ptrs = C + stride_cm * offs_global_m[:, None] + stride_cn * offs_bn[None, :]
    c_mask = row_mask[:, None]
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [4, 8]
        for s in [2, 3, 4]
    ],
    key=["S", "N", "K", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"],
)
@triton.jit
def w4a8_block_fp4_matmul_grouped_kernel(
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
    stride_offsets,
    stride_tile_offsets,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS_BIT_LENGTH: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    total_tiles = tl.load(TileOffsets_ptr + (NUM_EXPERTS - 1) * stride_tile_offsets)
    if pid_m >= total_tiles:
        return

    lo = 0
    hi = NUM_EXPERTS - 1
    for _ in tl.static_range(NUM_EXPERTS_BIT_LENGTH):
        mid = (lo + hi) >> 1
        mid_val = tl.load(TileOffsets_ptr + mid * stride_tile_offsets)
        is_left = mid_val <= pid_m
        lo = tl.where(is_left, mid + 1, lo)
        hi = tl.where(is_left, hi, mid)
    expert_id = tl.minimum(lo, NUM_EXPERTS - 1).to(tl.int64)

    prev_eid = tl.maximum(expert_id - 1, 0)
    expert_start = tl.where(expert_id == 0, 0, tl.load(Offsets_ptr + prev_eid * stride_offsets))
    expert_end = tl.load(Offsets_ptr + expert_id * stride_offsets)
    m_expert = expert_end - expert_start

    expert_tile_start = tl.where(expert_id == 0, 0, tl.load(TileOffsets_ptr + prev_eid * stride_tile_offsets))
    local_tile = pid_m - expert_tile_start
    m_off = local_tile * BLOCK_SIZE_M

    offs_m = m_off + tl.arange(0, BLOCK_SIZE_M)
    row_mask = offs_m < m_expert
    offs_global_m = expert_start + offs_m
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k_byte = tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)

    a_base = A_ptr + offs_global_m[:, None] * stride_am
    as_base = AS_ptr + offs_global_m[:, None] * stride_asm
    b_base = B_ptr + expert_id * stride_be + offs_n[:, None] * stride_bn
    sf_base = SF_ptr + expert_id * stride_sfe + offs_n[:, None] * stride_sfn

    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_base = k0 * BLOCK_SIZE_K
        a_k = k_base + tl.arange(0, BLOCK_SIZE_K)
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


@triton_op("finegrained_fp8::w8a8_block_fp8_matmul_grouped", mutates_args=())
def _w8a8_block_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """Block-scale grouped FP8 matmul: C = A @ B.T per expert, with fused act quant.

    A:  (S, K) raw bf16/fp16 activations, sorted by expert
    B:  (E, N, K) FP8 expert weights
    Bs: (E, N // block_n, K // block_k) per-block weight scales
    """
    assert A.ndim == 2, f"A must be 2D (S, K), got ndim={A.ndim}"
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 3, f"B must be 3D (E, N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"
    assert A.shape[1] == B.shape[2], (
        f"K mismatch: A has K={A.shape[1]}, B has K={B.shape[2]}"
    )

    S, K = A.shape
    E, N, _ = B.shape

    assert len(block_size) == 2, (
        f"block_size must be [block_n, block_k], got {block_size}"
    )
    block_n, block_k = block_size[0], block_size[1]
    # MoE expert dimensions must be block-aligned; non-aligned N/K is not supported.
    assert N % block_n == 0, f"N ({N}) must be divisible by block_n ({block_n})"
    assert K % block_k == 0, f"K ({K}) must be divisible by block_k ({block_k})"
    assert Bs.ndim == 3, (
        f"Bs must be 3D (E, N//block_n, K//block_k), got ndim={Bs.ndim}"
    )
    assert Bs.shape == (E, N // block_n, K // block_k), (
        f"Bs shape {tuple(Bs.shape)} != expected ({E}, {N // block_n}, {K // block_k})"
    )

    C = A.new_empty(S, N)
    # Adaptive BLOCK_SIZE_M: match tile to average tokens per expert.
    BLOCK_SIZE_M = min(max(triton.next_power_of_2((S + E - 1) // E), 16), 128)
    tiles_per_expert = (tokens_per_expert + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    tile_offsets = torch.cumsum(tiles_per_expert, dim=0).to(torch.int32)
    # Upper bound on M-tiles: sum_e ceil(M_e / BLOCK_M) <= ceil(S / BLOCK_M) + E.
    # Programs beyond the real tile count exit immediately via the early-return
    # guard inside the kernel. This is faster than syncing for the exact count
    # and keeps the grid size data-independent (cuda-graph / torch.compile safe).
    max_M_tiles = triton.cdiv(S, BLOCK_SIZE_M) + E
    grid = (max_M_tiles, triton.cdiv(N, block_n))
    with device_context(A.device):
        wrap_triton(w8a8_block_fp8_matmul_grouped_kernel)[grid](
            A,
            B,
            C,
            Bs,
            offsets,
            tile_offsets,
            S,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            Bs.stride(0),
            Bs.stride(2),
            Bs.stride(1),
            # Meta-parameters
            NUM_EXPERTS=E,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            NUM_EXPERTS_BIT_LENGTH=E.bit_length(),
        )

    return C


@triton_op("finegrained_fp8::w8a8_tensor_fp8_matmul_grouped", mutates_args=())
def _w8a8_tensor_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Tensor-scale grouped FP8 matmul: C = A @ B.T per expert, with fused act quant.

    A:  (S, K) raw bf16/fp16 activations, sorted by expert
    B:  (E, N, K) FP8 expert weights
    Bs: (E,) or (E, 1, 1) per-expert weight scales
    """
    assert A.ndim == 2, f"A must be 2D (S, K), got ndim={A.ndim}"
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 3, f"B must be 3D (E, N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"
    assert A.shape[1] == B.shape[2], (
        f"K mismatch: A has K={A.shape[1]}, B has K={B.shape[2]}"
    )

    S, K = A.shape
    E, N, _ = B.shape

    # Normalize Bs to (E, 1, 1)
    if Bs.ndim == 1:
        assert Bs.shape[0] == E, f"Bs shape {tuple(Bs.shape)} != expected ({E},)"
        Bs = Bs.reshape(E, 1, 1)
    else:
        assert Bs.shape == (E, 1, 1), (
            f"Bs shape {tuple(Bs.shape)} != expected ({E}, 1, 1)"
        )

    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128
    C = A.new_empty(S, N)
    qA, As = fp8_act_quant(A, K)
    BLOCK_SIZE_M = min(max(triton.next_power_of_2((S + E - 1) // E), 16), 128)
    tiles_per_expert = (tokens_per_expert + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    tile_offsets = torch.cumsum(tiles_per_expert, dim=0).to(torch.int32)
    # Upper bound on M-tiles: sum_e ceil(M_e / BLOCK_M) <= ceil(S / BLOCK_M) + E.
    # Programs beyond the real tile count exit immediately via the early-return
    # guard inside the kernel. This is faster than syncing for the exact count
    # and keeps the grid size data-independent (cuda-graph / torch.compile safe).
    max_M_tiles = triton.cdiv(S, BLOCK_SIZE_M) + E
    grid = (max_M_tiles, triton.cdiv(N, BLOCK_SIZE_N))
    with device_context(A.device):
        wrap_triton(w8a8_tensor_fp8_matmul_grouped_kernel)[grid](
            qA,
            B,
            C,
            As,
            Bs,
            offsets,
            tile_offsets,
            S,
            N,
            K,
            qA.stride(0),
            qA.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            As.stride(0),
            Bs.stride(0),
            # Meta-parameters
            NUM_EXPERTS=E,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            NUM_EXPERTS_BIT_LENGTH=E.bit_length(),
        )

    return C


def w8a8_block_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """Block-scale grouped FP8 matmul with fused activation quantization.

    A:  (S, K) raw activations sorted by expert, bf16/fp16/fp32
    B:  (E, N, K) FP8 expert weights
    Bs: (E, N // block_n, K // block_k) per-block weight scales
    """
    return torch.ops.finegrained_fp8.w8a8_block_fp8_matmul_grouped(
        A, B, Bs, offsets, tokens_per_expert, block_size
    )


def w8a8_tensor_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Tensor-scale grouped FP8 matmul with fused activation quantization.

    A:  (S, K) raw activations sorted by expert, bf16/fp16/fp32
    B:  (E, N, K) FP8 expert weights
    Bs: (E,) or (E, 1, 1) per-expert weight scales
    """
    return torch.ops.finegrained_fp8.w8a8_tensor_fp8_matmul_grouped(
        A, B, Bs, offsets, tokens_per_expert
    )


@triton_op("finegrained_fp8::w4a8_block_fp4_matmul_grouped", mutates_args=())
def _w4a8_block_fp4_matmul_grouped(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert len(block_size) == 2, (
        f"block_size must be [block_n, block_k], got {block_size}"
    )
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

    is_tensor_scale_input = As.shape[-1] == 1
    quant_block_k = fp4_resolve_block_k(block_size, K)
    As = fp4_expand_activation_scales(As, K, quant_block_k)
    assert As.shape == (S, K // FP4_SCALE_GROUP_K), (
        f"As shape {tuple(As.shape)} != ({S}, {K // FP4_SCALE_GROUP_K})"
    )

    as_u8 = As.to(torch.float8_e8m0fnu).view(torch.uint8)
    sf_u8 = Bs.view(torch.uint8)
    offsets = offsets.to(device=A.device, dtype=torch.int32)
    tokens_per_expert = tokens_per_expert.to(device=A.device, dtype=torch.int32)
    BLOCK_SIZE_N = 128 if is_tensor_scale_input else block_size[0]
    launch_block_k = 128 if is_tensor_scale_input and K % 128 == 0 else quant_block_k
    BLOCK_SIZE_M = min(max(triton.next_power_of_2((S + E - 1) // E), 16), 128)
    C = A.new_empty((S, N), dtype=output_dtype)
    tiles_per_expert = (tokens_per_expert + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    tile_offsets = torch.cumsum(tiles_per_expert, dim=0).to(torch.int32)
    max_m_tiles = triton.cdiv(S, BLOCK_SIZE_M) + E
    grid = (max_m_tiles, triton.cdiv(N, BLOCK_SIZE_N))
    with device_context(A.device):
        wrap_triton(w4a8_block_fp4_matmul_grouped_kernel)[grid](
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
            offsets.stride(0),
            tile_offsets.stride(0),
            NUM_EXPERTS=E,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=launch_block_k,
            NUM_EXPERTS_BIT_LENGTH=E.bit_length(),
            VALUES_PER_BYTE=FP4_VALUES_PER_BYTE,
            SCALE_GROUP_K=FP4_SCALE_GROUP_K,
        )
    return C


def w4a8_block_fp4_matmul_grouped(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Block-scale grouped W4A8 FP4 matmul: per-expert ``C[s] = A[s] @ B[e].T``
    over contiguous, expert-sorted rows of A.

    A:  (S, K) FP8 activations, expert-sorted
    As: (S, K // block_k) UE8M0 activation scales
    B:  (E, N, K // 2) packed FP4 expert weights (int8, two codes per byte)
    Bs: (E, N, K // 32) UE8M0 weight scales
    offsets: (E,) — exclusive prefix of expert token counts (cumsum)
    tokens_per_expert: (E,) — per-expert row count
    """
    return torch.ops.finegrained_fp8.w4a8_block_fp4_matmul_grouped(
        A, As, B, Bs, offsets, tokens_per_expert, block_size, output_dtype
    )


def fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_size: list[int] | None,
) -> torch.Tensor:
    """Neutral grouped FP8 matmul dispatcher.

    Routes by weight dtype and scale shape:
    - ``int8`` weights (packed FP4) → ``w4a8_block_fp4_matmul_grouped`` (block
      mode only; tensor-mode FP4 — ``block_size`` None or ``[N, K]`` — is not
      yet supported). The W8A8 kernels fuse activation quantization but the W4A8
      block kernel takes pre-quantized FP8 activations, so this wrapper runs
      ``fp8_act_quant`` first.
    - ``block_size`` None or ``[N, K]`` → ``w8a8_tensor_fp8_matmul_grouped``
    - otherwise → ``w8a8_block_fp8_matmul_grouped``
    """
    is_fp4 = B.dtype == torch.int8
    is_tensor_mode = block_size is None or (
        block_size[0] == B.size(1) and block_size[1] == B.size(2)
    )

    if is_fp4:
        if is_tensor_mode:
            raise NotImplementedError(
                "W4A8 FP4 grouped path only supports block mode; tensor-mode "
                "(block_size=None or [N, K]) is not yet implemented."
            )
        qA, As = fp8_act_quant(A, block_size[1])
        return w4a8_block_fp4_matmul_grouped(
            qA, As, B, Bs, offsets, tokens_per_expert, block_size, A.dtype
        )

    if is_tensor_mode:
        return w8a8_tensor_fp8_matmul_grouped(A, B, Bs, offsets, tokens_per_expert)

    return w8a8_block_fp8_matmul_grouped(
        A, B, Bs, offsets, tokens_per_expert, block_size
    )
