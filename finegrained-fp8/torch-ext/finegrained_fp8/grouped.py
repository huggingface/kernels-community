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

from .utils import (
    FP4_SCALE_GROUP_K,
    FP4_VALUES_PER_BYTE,
    adaptive_block_size_m,
    device_context,
    fp4_act_quant_inline,
    fp8_act_quant,
    fp8_act_quant_inline,
    grouped_expert_lookup,
    grouped_tile_layout,
)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [2, 4, 8, 16]
        for s in [2, 3, 4]
    ],
    key=["N", "K", "BLOCK_SIZE_M"],
)
@triton.jit
def w8a8_block_fp8_matmul_grouped_kernel(
    A,  # (S, K) raw BF16/FP16 activations, sorted/grouped by expert id
    B,  # (E, N, K) FP8 weight matrices
    C,  # (S, N) output
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
    stride_offs,
    stride_tile,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_BIT_LENGTH: tl.constexpr,
):
    """Block-scale grouped FP8 expert matmul kernel.

    Tokens are assumed sorted by expert. The kernel maps each M-tile to its
    owning expert via ``TileOffsets`` and applies fused activation quantization.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    total_tiles = tl.load(TileOffsets + (NUM_EXPERTS - 1) * stride_tile)
    if pid_m >= total_tiles:
        return

    expert_id, offs_global_m, row_mask = grouped_expert_lookup(
        pid_m,
        Offsets,
        TileOffsets,
        stride_offs,
        stride_tile,
        NUM_EXPERTS,
        NUM_EXPERTS_BIT_LENGTH,
        BLOCK_SIZE_M,
    )
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
        a_raw = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        a, a_s = fp8_act_quant_inline(a_raw)
        b = tl.load(b_ptrs)
        b_s = tl.load(bs_ptrs + k * stride_bs_k)
        if b_s.dtype == tl.uint8:
            # UE8M0 decode: value = 2^(exp - 127); build the fp32 bit pattern.
            b_s = (b_s.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(C.dtype.element_ty)
    c_ptrs = C + stride_cm * offs_global_m[:, None] + stride_cn * offs_bn[None, :]
    c_mask = row_mask[:, None]
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
def w8a8_tensor_fp8_matmul_grouped_kernel(
    A,  # (S, K) pre-quantized FP8 activations, sorted/grouped by expert id
    B,  # (E, N, K) FP8 weight matrices
    C,  # (S, N) output
    As,  # (S,) per-token activation scales
    Bs,  # (E, 1, 1) per-tensor weight scales
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
    stride_as_m,
    stride_bs_e,
    stride_offs,
    stride_tile,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_BIT_LENGTH: tl.constexpr,
):
    """Tensor-scale grouped FP8 expert matmul kernel.

    Uses grouped expert scheduling with pre-quantized activations plus
    per-token activation scales and per-expert tensor weight scales.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    total_tiles = tl.load(TileOffsets + (NUM_EXPERTS - 1) * stride_tile)
    if pid_m >= total_tiles:
        return

    expert_id, offs_global_m, row_mask = grouped_expert_lookup(
        pid_m,
        Offsets,
        TileOffsets,
        stride_offs,
        stride_tile,
        NUM_EXPERTS,
        NUM_EXPERTS_BIT_LENGTH,
        BLOCK_SIZE_M,
    )
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

    c = accumulator.to(C.dtype.element_ty)
    c_ptrs = C + stride_cm * offs_global_m[:, None] + stride_cn * offs_bn[None, :]
    c_mask = row_mask[:, None]
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk},
            num_warps=w,
            num_stages=s,
        )
        for bn in [64, 128, 256]
        for bk in [64, 128, 256]
        for w in [2, 4, 8, 16]
        for s in [2, 3, 4]
    ],
    key=["N", "K", "BLOCK_SIZE_M"],
)
@triton.jit
def w4a8_fp4_matmul_grouped_kernel(
    A,  # (S, K) raw BF16/FP16 activations, sorted by expert id
    B,  # (E, N, K // 2) packed FP4 (E2M1) expert weights as int8
    C,  # (S, N) output
    Bs,  # (E, N, K // SCALE_GROUP_K) UE8M0 weight scales
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
    stride_offs,
    stride_tile,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_BIT_LENGTH: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    """Block-scale grouped W4A8 FP4 expert matmul with fused activation quant.

    Tokens are assumed sorted by expert. The kernel maps each M-tile to its
    owning expert via ``TileOffsets``, quantizes ``A`` to FP8 per K-group inline
    (UE8M0 scale), then ``tl.dot_scaled`` against packed FP4 weights.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    total_tiles = tl.load(TileOffsets + (NUM_EXPERTS - 1) * stride_tile)
    if pid_m >= total_tiles:
        return

    expert_id, offs_global_m, row_mask = grouped_expert_lookup(
        pid_m,
        Offsets,
        TileOffsets,
        stride_offs,
        stride_tile,
        NUM_EXPERTS,
        NUM_EXPERTS_BIT_LENGTH,
        BLOCK_SIZE_M,
    )
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_k_byte = tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)

    a_ptrs = A + offs_global_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        B
        + expert_id * stride_be
        + offs_k_byte[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )
    bs_ptrs = (
        Bs
        + expert_id * stride_bs_e
        + offs_bn[:, None] * stride_bs_n
        + offs_sf[None, :] * stride_bs_k
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_raw = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        a, a_scale = fp4_act_quant_inline(
            a_raw, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K
        )
        b = tl.load(b_ptrs).to(tl.uint8)
        b_s = tl.load(bs_ptrs).to(tl.uint8)
        accumulator = tl.dot_scaled(a, a_scale, "e4m3", b, b_s, "e2m1", acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // VALUES_PER_BYTE) * stride_bk
        bs_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_bs_k

    c_ptrs = C + stride_cm * offs_global_m[:, None] + stride_cn * offs_bn[None, :]
    c_mask = row_mask[:, None]
    tl.store(c_ptrs, accumulator.to(C.dtype.element_ty), mask=c_mask)


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
    BLOCK_SIZE_M = adaptive_block_size_m((S + E - 1) // E)
    tile_offsets, max_m_tiles = grouped_tile_layout(
        tokens_per_expert, BLOCK_SIZE_M, S, E
    )
    # UE8M0 scales: pass as uint8 (Triton binder doesn't recognize
    # float8_e8m0fnu); kernel decodes 2^(exp-127) inline.
    if Bs.dtype == torch.float8_e8m0fnu:
        Bs = Bs.view(torch.uint8)
    grid = (max_m_tiles, triton.cdiv(N, block_n))
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
            offsets.stride(0),
            tile_offsets.stride(0),
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
    BLOCK_SIZE_M = adaptive_block_size_m((S + E - 1) // E)
    tile_offsets, max_m_tiles = grouped_tile_layout(
        tokens_per_expert, BLOCK_SIZE_M, S, E
    )
    grid = (max_m_tiles, triton.cdiv(N, BLOCK_SIZE_N))
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
            offsets.stride(0),
            tile_offsets.stride(0),
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
    offsets = offsets.to(device=A.device, dtype=torch.int32)
    tokens_per_expert = tokens_per_expert.to(device=A.device, dtype=torch.int32)
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
    offsets = offsets.to(device=A.device, dtype=torch.int32)
    tokens_per_expert = tokens_per_expert.to(device=A.device, dtype=torch.int32)
    return torch.ops.finegrained_fp8.w8a8_tensor_fp8_matmul_grouped(
        A, B, Bs, offsets, tokens_per_expert
    )


@triton_op("finegrained_fp8::w4a8_fp4_matmul_grouped", mutates_args=())
def _w4a8_fp4_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Block-scale grouped W4A8 FP4 matmul with fused activation quant.

    A:  (S, K) raw activations, bf16/fp16/fp32, expert-sorted (quantized inline)
    B:  (E, N, K // 2) packed FP4 (E2M1) expert weights, two codes per int8
    Bs: (E, N, K // 32) UE8M0 weight scales
    offsets: (E,) — exclusive prefix of expert token counts (cumsum)
    tokens_per_expert: (E,) — per-expert row count

    BLOCK_SIZE_N and BLOCK_SIZE_K are autotuned.
    """
    assert A.ndim == 2 and B.ndim == 3 and Bs.ndim == 3
    assert offsets.ndim == 1 and tokens_per_expert.ndim == 1
    assert B.dtype == torch.int8, f"B must be int8 (packed FP4), got {B.dtype}"
    assert Bs.dtype == torch.float8_e8m0fnu, (
        f"Bs must be float8_e8m0fnu, got {Bs.dtype}"
    )

    S, K = A.shape
    E, N, K_half = B.shape
    assert offsets.shape[0] == E and tokens_per_expert.shape[0] == E
    assert K == FP4_VALUES_PER_BYTE * K_half, (
        f"K (={K}) must equal {FP4_VALUES_PER_BYTE} * B.shape[2] (={K_half})"
    )
    assert K % FP4_SCALE_GROUP_K == 0, (
        f"K (={K}) must be a multiple of {FP4_SCALE_GROUP_K}"
    )
    assert Bs.shape == (E, N, K // FP4_SCALE_GROUP_K), (
        f"Bs shape {tuple(Bs.shape)} != ({E}, {N}, {K // FP4_SCALE_GROUP_K})"
    )

    bs_u8 = Bs.view(torch.uint8)
    BLOCK_SIZE_M = adaptive_block_size_m((S + E - 1) // E)
    C = A.new_empty((S, N), dtype=output_dtype)
    tile_offsets, max_m_tiles = grouped_tile_layout(
        tokens_per_expert, BLOCK_SIZE_M, S, E
    )

    def grid(META):
        return (max_m_tiles, triton.cdiv(N, META["BLOCK_SIZE_N"]))

    with device_context(A.device):
        wrap_triton(w4a8_fp4_matmul_grouped_kernel)[grid](
            A,
            B,
            C,
            bs_u8,
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
            bs_u8.stride(0),
            bs_u8.stride(2),
            bs_u8.stride(1),
            offsets.stride(0),
            tile_offsets.stride(0),
            # Meta-parameters (BLOCK_SIZE_N, BLOCK_SIZE_K come from autotune Config)
            NUM_EXPERTS=E,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            NUM_EXPERTS_BIT_LENGTH=E.bit_length(),
            VALUES_PER_BYTE=FP4_VALUES_PER_BYTE,
            SCALE_GROUP_K=FP4_SCALE_GROUP_K,
        )
    return C


def w4a8_fp4_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Block-scale grouped W4A8 FP4 matmul with fused activation quant. Per-expert
    ``C[s] = A[s] @ B[e].T`` over contiguous, expert-sorted rows. Tile shape
    autotuned; FP4 scale granularity is fixed at 32."""
    offsets = offsets.to(device=A.device, dtype=torch.int32)
    tokens_per_expert = tokens_per_expert.to(device=A.device, dtype=torch.int32)
    return torch.ops.finegrained_fp8.w4a8_fp4_matmul_grouped(
        A, B, Bs, offsets, tokens_per_expert, output_dtype
    )


def matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_size: list[int] | None,
) -> torch.Tensor:
    """Grouped quantized matmul dispatcher (W8A8 FP8 or W4A8 FP4). Tokens must
    be sorted by expert; M-tiles are mapped to experts via ``offsets``.

    Routes by weight dtype and ``block_size``:
    - ``B.dtype == int8`` (packed FP4) → ``w4a8_fp4_matmul_grouped``
      (``block_size`` is ignored; FP4 tile shape is autotuned).
    - ``block_size`` None or full ``[N, K]`` → ``w8a8_tensor_fp8_matmul_grouped``.
    - otherwise → ``w8a8_block_fp8_matmul_grouped``.
    """
    if B.dtype == torch.int8:
        return w4a8_fp4_matmul_grouped(A, B, Bs, offsets, tokens_per_expert, A.dtype)

    if block_size is None or (
        block_size[0] == B.size(1) and block_size[1] == B.size(2)
    ):
        return w8a8_tensor_fp8_matmul_grouped(A, B, Bs, offsets, tokens_per_expert)

    return w8a8_block_fp8_matmul_grouped(
        A, B, Bs, offsets, tokens_per_expert, block_size
    )
