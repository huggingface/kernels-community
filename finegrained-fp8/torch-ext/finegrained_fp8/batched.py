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
def w8a8_block_fp8_matmul_batched_kernel(
    A,  # (S, K)  raw BF16/FP16 activations
    B,  # (E, N, K) FP8 weight matrices
    C,  # (S, N)  output
    Bs,  # (E, N // group_n, K // group_k) weight scales
    ExpertIds,  # (S,) — which expert each batch element routes to
    # Shape
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Per-row strides
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cn,
    stride_Bs_k,
    stride_Bs_n,
    # Batch / expert strides
    stride_Ab,  # stride between rows in A (one token per program)
    stride_Eb,  # stride between experts in B
    stride_Cb,
    stride_Esb,  # stride between experts in Bs
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)

    # Advance base pointers to this token's activation row and its expert's
    # weight / scale slice. No pre-gather of weights needed (like in non-fp8 impls)
    expert_id = tl.load(ExpertIds + batch_id)
    A = A + batch_id * stride_Ab
    B = B + expert_id * stride_Eb
    C = C + batch_id * stride_Cb
    Bs = Bs + expert_id * stride_Esb

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # M=1: broadcast the single activation row to BLOCK_SIZE_M identical rows
    # so tl.dot gets the required (BLOCK_SIZE_M, BLOCK_SIZE_K) shape.
    a_ptrs = A + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 + offs_k[None, :] * stride_ak
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # ---- fused act_quant (replaces: a = tl.load(a_ptrs); a_s = tl.load(As_ptrs)) ----
        a_raw = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0).to(tl.float32)
        a_s = tl.max(tl.abs(a_raw)) / 448.0  # per-block scale (scalar for M=1)
        a = (a_raw / tl.maximum(a_s, 1e-12)).to(tl.float8e4nv)
        # ---- same as baseline from here ----
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    # Only write row 0 (M=1); the broadcast rows are discarded.
    offs_cm = tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offs_cm[:, None] * 0 + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < 1) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton_op("finegrained_fp8::w8a8_block_fp8_matmul_batched", mutates_args=())
def _w8a8_block_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int] | None,
) -> torch.Tensor:
    """Batched FP8 block-wise matmul with fused activation quantization.

    Mirrors ``_batched_linear`` for FP8 weights: A is the raw (BF16/FP16)
    activation matrix, B / Bs are the stacked expert weights / scales.
    The kernel looks up ``expert_ids[batch_id]`` to address the correct expert
    slice of B directly — no (S, N, K) weight gather is needed.
    Activation quantization (``act_quant``) is fused into the matmul loop.
    """
    assert A.ndim == 2, "A must be (S, K)"
    assert A.is_contiguous()

    assert B.ndim == 3, "B must be (E, N, K)"
    assert B.is_contiguous()

    assert A.shape[1] == B.shape[2], "K dimension mismatch between A and B"
    assert expert_ids.is_contiguous()
    assert Bs.is_contiguous()

    if block_size is None:
        block_n, block_k = 128, 128
    else:
        assert len(block_size) == 2
        block_n, block_k = block_size[0], block_size[1]

    S, K = A.shape
    E, N, _ = B.shape
    C = A.new_empty(S, N)

    # Adaptive BLOCK_SIZE_M: match the tile to the average tokens per expert
    BLOCK_SIZE_M = min(max(triton.next_power_of_2((S + E - 1) // E), 16), 128)

    grid = (triton.cdiv(N, block_n), S)
    wrap_triton(w8a8_block_fp8_matmul_batched_kernel)[grid](
        A,
        B,
        C,
        Bs,
        expert_ids,
        N,
        K,
        block_n,
        block_k,
        A.stride(1),   # stride_ak
        B.stride(2),   # stride_bk
        B.stride(1),   # stride_bn
        C.stride(1),   # stride_cn
        Bs.stride(2),  # stride_Bs_k
        Bs.stride(1),  # stride_Bs_n
        A.stride(0),   # stride_Ab
        B.stride(0),   # stride_Eb
        C.stride(0),   # stride_Cb
        Bs.stride(0),  # stride_Esb
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
    )

    return C


def w8a8_block_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int] | None,
) -> torch.Tensor:
    """Batched W8A8 FP8 matmul for MoE expert dispatch with fused activation quantization.

    Each token in ``A`` is routed to one expert via ``expert_ids``. The kernel
    quantizes ``A`` to FP8 on-the-fly (fused ``act_quant``), reads the correct
    expert weight slice of ``B`` directly using ``expert_ids[batch_id]`` — no
    ``(S, N, K)`` weight gather needed — and accumulates in float32.

    Args:
        A: Raw activation matrix ``[S, K]`` in bf16/fp16/fp32.
        B: Stacked expert weight tensor ``[E, N, K]`` in ``float8_e4m3fn``.
        Bs: Stacked expert weight scales ``[E, N // block_size[0], K // block_size[1]]``.
        expert_ids: Expert index per token ``[S]``, values in ``[0, E)``.
        block_size: ``[block_n, block_k]`` quantization block dimensions, e.g. ``[128, 128]``.

    Returns:
        Output tensor ``[S, N]`` in the same dtype as ``A``.
    """
    return torch.ops.finegrained_fp8.w8a8_block_fp8_matmul_batched(A, B, Bs, expert_ids, block_size)
