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


@triton.jit
def w8a8_block_fp8_grouped_mm_kernel(
    A,  # (S, K)  raw BF16/FP16 activations, sorted/grouped by expert id
    B,  # (E, N, K) FP8 weight matrices
    C,  # (S, N)  output
    Bs,  # (E, N // group_n, K // group_k) weight scales
    Offsets,  # (E,) int32 — cumulative row-end per expert
    TileOffsets,  # (E,) int32 — cumulative tile-end per expert
    # Shape
    S,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
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
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Exit early for programs beyond the actual tile count.
    total_tiles = tl.load(TileOffsets + NUM_EXPERTS - 1)
    if pid_m >= total_tiles:
        return

    # Binary search in TileOffsets to find the owning expert.
    # Finds the smallest e such that TileOffsets[e] > pid_m (upper_bound semantics),
    # which is the expert whose tile range contains pid_m.
    # O(log2(NUM_EXPERTS)) loads instead of the O(NUM_EXPERTS) linear scan.
    # NUM_EXPERTS.bit_length() is ceil(log2(E))+1 for powers-of-two, giving one
    # harmless extra iteration when lo==hi; it's a compile-time constant so the
    # loop is fully unrolled by the compiler.
    lo = 0
    hi = NUM_EXPERTS
    for _ in tl.static_range(NUM_EXPERTS.bit_length()):
        mid = (lo + hi) >> 1
        mid_val = tl.load(TileOffsets + mid)
        is_left = mid_val <= pid_m
        lo = tl.where(is_left, mid + 1, lo)
        hi = tl.where(is_left, hi, mid)
    expert_id = lo

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
    offs_bn_safe = offs_bn % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    offs_am_safe = offs_global_m % S

    a_ptrs = A + offs_am_safe[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        B
        + expert_id * stride_Eb
        + offs_k[:, None] * stride_bk
        + offs_bn_safe[None, :] * stride_bn
    )
    offs_bsn_safe = offs_bn_safe // group_n
    Bs_ptrs = Bs + expert_id * stride_Esb + offs_bsn_safe * stride_Bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # ---- fused act_quant (replaces: a = tl.load(a_ptrs); a_s = tl.load(As_ptrs)) ----
        a_raw = tl.load(
            a_ptrs,
            mask=row_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        ).to(tl.float32)
        a_s = tl.max(tl.abs(a_raw), axis=1) / 448.0  # per-row scale  (BLOCK_SIZE_M,)
        # clamp denominator so masked all-zero rows don't produce NaN
        # (their a_s multiplier is 0 anyway, so the output row is correct)
        a = (a_raw / tl.maximum(a_s[:, None], 1e-12)).to(tl.float8e4nv)
        # ---- same as baseline from here ----
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bsk)

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
    c_mask = row_mask[:, None] & (offs_bn[None, :] < N)
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
    """Grouped FP8 block-wise matmul with fused activation quantization.

    Mirrors ``_grouped_linear`` / ``_grouped_mm`` for FP8 weights: A is the
    raw (BF16/FP16) activation matrix sorted by expert, B / Bs are the stacked
    expert weights / scales.  Activation quantization (``act_quant``) is fused
    into the matmul loop.  ``tokens_per_expert`` is needed (in addition to
    ``offsets``) to build the per-expert tile schedule inside the kernel.
    """
    assert A.ndim == 2, "A must be (S, K)"
    assert A.is_contiguous()

    assert B.ndim == 3, "B must be (E, N, K)"
    assert B.is_contiguous()

    assert tokens_per_expert.is_contiguous()
    assert offsets.is_contiguous()
    assert Bs.is_contiguous()

    if block_size is None:
        block_n, block_k = 128, 128
    else:
        assert len(block_size) == 2
        block_n, block_k = block_size[0], block_size[1]

    S, K = A.shape
    E, N, _ = B.shape
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

    grid = (max_M_tiles * triton.cdiv(N, block_n),)
    wrap_triton(w8a8_block_fp8_grouped_mm_kernel)[grid](
        A,
        B,
        C,
        Bs,
        offsets,
        tile_offsets,
        S,
        N,
        K,
        block_n,
        block_k,
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
        NUM_EXPERTS=E,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
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
        Bs: Stacked expert weight scales ``[E, N // block_size[0], K // block_size[1]]``.
        offsets: Cumulative token counts per expert ``[E]`` (i.e. ``cumsum(tokens_per_expert)``).
        tokens_per_expert: Number of tokens routed to each expert ``[E]``.
        block_size: ``[block_n, block_k]`` quantization block dimensions, e.g. ``[128, 128]``.

    Returns:
        Output tensor ``[S, N]`` in the same dtype as ``A``, in expert-sorted order.
    """
    return torch.ops.finegrained_fp8.w8a8_block_fp8_matmul_grouped(
        A, B, Bs, offsets, tokens_per_expert, block_size
    )
