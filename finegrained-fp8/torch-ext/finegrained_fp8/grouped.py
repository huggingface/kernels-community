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
from .act_quant import fp8_act_quant
from torch.library import triton_op, wrap_triton


@triton.jit
def w8a8_block_fp8_matmul_grouped_kernel(
    A,  # (S, K)  raw BF16/FP16 activations, sorted/grouped by expert id
    B,  # (E, N, K) FP8 weight matrices
    C,  # (S, N)  output
    Bs,  # (E, N // block_n, K // block_k) weight scales
    Offsets,  # (E,) int32 — cumulative row-end per expert
    TileOffsets,  # (E,) int32 — cumulative tile-end per expert
    # Shape
    S,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_Eb,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_Esb,
    stride_Bsk,
    stride_Bsn,
    # Meta-parameters
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
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
    # expert_id * stride_Eb (e.g. 255 * 9_437_184 > 2^31 for 256 experts of
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

    offs_bn = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)

    a_ptrs = A + offs_global_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        B
        + expert_id * stride_Eb
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )
    offs_bsn = offs_bn // block_n
    Bs_ptrs = Bs + expert_id * stride_Esb + offs_bsn * stride_Bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, block_n), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, block_k)):
        # ---- fused fp8_act_quant ----
        a_raw = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        a_s = tl.max(tl.abs(a_raw), axis=1) / 448.0  # per-row scale  (BLOCK_SIZE_M,)
        a = (a_raw / tl.maximum(a_s[:, None], 1e-12)).to(tl.float8e4nv)
        # ---- same as baseline from here ----
        b = tl.load(b_ptrs)
        k_start = k * block_k
        offs_ks = k_start // block_k
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bsk)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    c_ptrs = C + stride_cm * offs_global_m[:, None] + stride_cn * offs_bn[None, :]
    c_mask = row_mask[:, None]
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def w8a8_tensor_fp8_matmul_grouped_kernel(
    A,  # (S, K) pre-quantized FP8 activations
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
    stride_Eb,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_Esb,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
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
    for _ in tl.static_range(NUM_EXPERTS.bit_length()):
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

    offs_bn = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)

    a_ptrs = A + offs_global_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        B
        + expert_id * stride_Eb
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    a_s = tl.load(As + offs_global_m * stride_As_m, mask=row_mask, other=0.0)
    b_s = tl.load(Bs + expert_id * stride_Esb)

    accumulator = tl.zeros((BLOCK_SIZE_M, block_n), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, block_k)):
        a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    c_ptrs = C + stride_cm * offs_global_m[:, None] + stride_cn * offs_bn[None, :]
    c_mask = row_mask[:, None]
    tl.store(c_ptrs, c, mask=c_mask)


@triton_op("finegrained_fp8::w8a8_block_fp8_matmul_grouped", mutates_args=())
def _w8a8_block_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_size: list[int] | None,
) -> torch.Tensor:
    """Internal block-scale grouped FP8 matmul op.

    ``A`` must be sorted by expert, ``offsets``/``tokens_per_expert`` define
    grouped scheduling, and activation quantization is fused into the matmul.
    """
    assert A.ndim == 2, "A must be (S, K)"
    assert A.is_contiguous(), "A must be contiguous"

    assert B.ndim == 3, "B must be (E, N, K)"
    assert B.is_contiguous(), "B must be contiguous"

    assert A.shape[1] == B.shape[2], "K dimension mismatch between A and B"
    assert tokens_per_expert.is_contiguous(), "tokens_per_expert must be contiguous"
    assert offsets.is_contiguous(), "offsets must be contiguous"
    assert Bs.is_contiguous(), "Bs must be contiguous"

    S, K = A.shape
    E, N, K = B.shape

    if block_size is None:
        block_n, block_k = 128, 128
    else:
        block_n, block_k = block_size[0], block_size[1]

    if block_n == N and block_k == K:
        block_n, block_k = 128, 128

    # we specifically require block-aligned shapes to keep the kernel simpler
    assert N % block_n == 0, f"N ({N}) must be divisible by block_n ({block_n})"
    assert K % block_k == 0, f"K ({K}) must be divisible by block_k ({block_k})"

    # For per-tensor scales, expand to block-scale shape with strides (0, 0).
    # This is a zero-copy view; all loads for a given expert hit the same value.
    if Bs.ndim == 1:
        Bs = Bs.reshape(E, 1, 1).expand(E, N // block_n, K // block_k)
    elif Bs.ndim == 3 and Bs.shape[0] == E and Bs.shape[1] == 1 and Bs.shape[2] == 1:
        Bs = Bs.expand(E, N // block_n, K // block_k)
    else:
        assert Bs.ndim == 3, (
            "Bs must be either (E,) / (E,1,1) for per-tensor scales or (E,N//block_n,K//block_k)"
        )
        assert Bs.shape[0] == E, (
            f"Bs expert dim mismatch: expected {E}, got {Bs.shape[0]}"
        )
        assert Bs.shape[1] == N // block_n, (
            f"Bs N-block dim mismatch: expected {N // block_n}, got {Bs.shape[1]}"
        )
        assert Bs.shape[2] == K // block_k, (
            f"Bs K-block dim mismatch: expected {K // block_k}, got {Bs.shape[2]}"
        )

    C = A.new_empty(S, N)

    # Adaptive BLOCK_SIZE_M: match tile to average tokens per expert.
    BLOCK_SIZE_M = min(max(triton.next_power_of_2((S + E - 1) // E), 16), 128)
    tiles_per_expert = (tokens_per_expert + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    tile_offsets = torch.cumsum(tiles_per_expert, dim=0).to(torch.int32)
    # Upper bound on M-tiles: sum_e ceil(M_e / BLOCK_M) <= ceil(S / BLOCK_M) + E.
    # Using a static upper bound keeps the grid size data-independent, which is
    # required for cuda-graph compatibility.  Programs beyond the real tile count
    # exit immediately via the early-return guard inside the kernel.
    max_M_tiles = triton.cdiv(S, BLOCK_SIZE_M) + E

    grid = (max_M_tiles, triton.cdiv(N, block_n))
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
        A.stride(0),  # stride_am
        A.stride(1),  # stride_ak
        B.stride(0),  # stride_Eb
        B.stride(2),  # stride_bk
        B.stride(1),  # stride_bn
        C.stride(0),  # stride_cm
        C.stride(1),  # stride_cn
        Bs.stride(0),  # stride_Esb
        Bs.stride(2),  # stride_Bsk
        Bs.stride(1),  # stride_Bsn
        # Meta-parameters
        block_n=block_n,
        block_k=block_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        NUM_EXPERTS=E,
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
    block_size: list[int] | None,
) -> torch.Tensor:
    """Tensor-scale grouped FP8 matmul for sorted routed experts.

    Uses the same grouped scheduling as block mode, but with per-token tensor
    activation scales and per-expert tensor weight scales.

    Accepted ``Bs`` layouts: ``[E]`` or ``[E,1,1]``.
    """
    assert A.ndim == 2, "A must be (S, K)"
    assert A.is_contiguous(), "A must be contiguous"

    assert B.ndim == 3, "B must be (E, N, K)"
    assert B.is_contiguous(), "B must be contiguous"

    assert A.shape[1] == B.shape[2], "K dimension mismatch between A and B"
    assert tokens_per_expert.is_contiguous(), "tokens_per_expert must be contiguous"
    assert offsets.is_contiguous(), "offsets must be contiguous"
    assert Bs.is_contiguous(), "Bs must be contiguous"

    S, K = A.shape
    E, N, K = B.shape

    if Bs.ndim == 1:
        Bs = Bs.reshape(E, 1, 1)
    elif Bs.ndim == 3 and Bs.shape[0] == E and Bs.shape[1] == 1 and Bs.shape[2] == 1:
        pass
    else:
        assert Bs.ndim == 3, "Tensor mode expects Bs in (E,) or (E,1,1)"
        assert Bs.shape[0] == E and Bs.shape[1] == 1 and Bs.shape[2] == 1, (
            f"Tensor mode expects Bs shape (E,1,1), got {tuple(Bs.shape)}"
        )

    block_n = 128 if N % 128 == 0 else N
    block_k = 128 if K % 128 == 0 else K
    assert N % block_n == 0, f"N ({N}) must be divisible by block_n ({block_n})"
    assert K % block_k == 0, f"K ({K}) must be divisible by block_k ({block_k})"

    C = A.new_empty(S, N)

    BLOCK_SIZE_M = min(max(triton.next_power_of_2((S + E - 1) // E), 16), 128)
    tiles_per_expert = (tokens_per_expert + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    tile_offsets = torch.cumsum(tiles_per_expert, dim=0).to(torch.int32)
    max_M_tiles = triton.cdiv(S, BLOCK_SIZE_M) + E

    qA, As = fp8_act_quant(A, K)
    grid = (max_M_tiles, triton.cdiv(N, block_n))
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
        block_n=block_n,
        block_k=block_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        NUM_EXPERTS=E,
    )

    return C


def w8a8_block_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_size: list[int] | None,
) -> torch.Tensor:
    """Grouped W8A8 FP8 matmul for MoE expert dispatch with fused activation quantization.

    Tokens in ``A`` must be pre-sorted by expert id. The kernel quantizes ``A``
    to FP8 on-the-fly (fused ``act_quant``), uses a static over-provisioned grid
    for CUDA-graph compatibility, and resolves each tile's expert via an O(log E)
    binary search over ``offsets``.

    Args:
        A: Raw activation matrix ``[S, K]`` sorted by expert, in bf16/fp16/fp32.
        B: Stacked expert weight tensor ``[E, N, K]`` in ``float8_e4m3fn``.
        Bs: Expert weight scales, accepted as ``[E, nb, kb]`` (block)
            or ``[E]`` / ``[E,1,1]`` (per-tensor; expanded internally).
        offsets: Cumulative token counts per expert ``[E]`` (i.e. ``cumsum(tokens_per_expert)``).
        tokens_per_expert: Number of tokens routed to each expert ``[E]``.
        block_size: ``[block_n, block_k]`` quantization block dimensions, e.g. ``[128, 128]``.

    Returns:
        Output tensor ``[S, N]`` in the same dtype as ``A``, in expert-sorted order.
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
    block_size: list[int] | None,
) -> torch.Tensor:
    """Tensor-scale grouped W8A8 FP8 matmul for MoE expert dispatch.

    Args:
        A: Raw activation matrix ``[S, K]`` sorted by expert, in bf16/fp16/fp32.
        B: Stacked expert weight tensor ``[E, N, K]`` in ``float8_e4m3fn``.
        Bs: Per-expert tensor scales ``[E]`` or ``[E,1,1]``.
        offsets: Cumulative token counts per expert ``[E]``.
        tokens_per_expert: Number of tokens routed to each expert ``[E]``.
        block_size: Kept for API consistency; tensor path derives tile sizes from ``N`` and ``K``.

    Returns:
        Output tensor ``[S, N]`` in the same dtype as ``A``, in expert-sorted order.
    """
    return torch.ops.finegrained_fp8.w8a8_tensor_fp8_matmul_grouped(
        A, B, Bs, offsets, tokens_per_expert, block_size
    )


def w8a8_fp8_matmul_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    block_size: list[int] | None,
) -> torch.Tensor:
    """Unified grouped W8A8 FP8 matmul dispatcher.

    Dispatch rules:
    - tensor mode when ``block_size is None``
    - tensor mode when ``block_size == [N, K]``
    - otherwise block mode

    Returns:
        Output tensor ``[S, N]`` in the same dtype as ``A``, in expert-sorted order.
    """
    if block_size is None or (
        block_size[0] == B.size(1) and block_size[1] == B.size(2)
    ):
        return w8a8_tensor_fp8_matmul_grouped(
            A, B, Bs, offsets, tokens_per_expert, block_size
        )

    return w8a8_block_fp8_matmul_grouped(
        A, B, Bs, offsets, tokens_per_expert, block_size
    )
