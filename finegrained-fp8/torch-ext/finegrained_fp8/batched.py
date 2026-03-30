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

from .utils import device_context


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [2, 4, 8, 16]
        for s in [2, 3, 4, 5]
    ],
    key=["N", "K"],
)
@triton.jit
def w8a8_block_fp8_matmul_batched_kernel(
    A,  # (S, K)  raw BF16/FP16 activations
    B,  # (E, N, K) FP8 weight matrices
    C,  # (S, N)  output
    Bs,  # (E, N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales
    ExpertIds,  # (S,) — which expert each batch element routes to
    # Shape
    S,
    N,
    K,
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
    stride_Cb,  # stride between rows in C (one token per program)
    stride_Esb,  # stride between experts in Bs
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Block-scale batched FP8 expert matmul kernel.

    Each program handles one routed token row and one N-tile, looks up the
    owning expert from ``ExpertIds``, and applies fused activation quantization.
    """
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Cast expert_id to int64 to prevent int32 overflow when computing
    # expert_id * stride_Eb (e.g. 255 * 9_437_184 > 2^31 for 256 experts of
    # 3072×3072 FP8 weights).
    expert_id = tl.load(ExpertIds + batch_id).to(tl.int64)

    A = A + batch_id * stride_Ab
    B = B + expert_id * stride_Eb
    C = C + batch_id * stride_Cb
    Bs = Bs + expert_id * stride_Esb

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 + offs_k[None, :] * stride_ak
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    Bs_ptrs = Bs + pid_n * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # ---- fused fp8_act_quant ----
        a_raw = tl.load(a_ptrs).to(tl.float32)
        a_s = tl.max(tl.abs(a_raw)) / 448.0
        a = (a_raw / tl.maximum(a_s, 1e-12)).to(tl.float8e4nv)
        # ---- matmul ----
        b = tl.load(b_ptrs)
        b_s = tl.load(Bs_ptrs + k * stride_Bs_k)
        accumulator += tl.dot(a, b) * a_s * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offs_cm[:, None] * 0 + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [2, 4, 8, 16]
        for s in [2, 3, 4, 5]
    ],
    key=["N", "K"],
)
@triton.jit
def w8a8_tensor_fp8_matmul_batched_kernel(
    A,  # (S, K) pre-quantized FP8 activations
    B,  # (E, N, K) FP8 weight matrices
    C,  # (S, N) output
    As,  # (S, 1) per-tensor activation scales
    Bs,  # (E, 1, 1) per-tensor weight scales
    ExpertIds,
    S,
    N,
    K,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cn,
    stride_As_b,
    stride_Ab,
    stride_Eb,
    stride_Cb,
    stride_Esb,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Tensor-scale batched FP8 expert matmul kernel.

    Activations are already quantized; the kernel applies per-token activation
    scales and per-expert tensor weight scales.
    """
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    expert_id = tl.load(ExpertIds + batch_id).to(tl.int64)

    A = A + batch_id * stride_Ab
    B = B + expert_id * stride_Eb
    C = C + batch_id * stride_Cb
    Bs = Bs + expert_id * stride_Esb

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 + offs_k[None, :] * stride_ak
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    b_s = tl.load(Bs)
    a_s = tl.load(As + batch_id * stride_As_b)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator * a_s * b_s

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offs_cm[:, None] * 0 + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)


@triton_op("finegrained_fp8::w8a8_block_fp8_matmul_batched", mutates_args=())
def _w8a8_block_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """Block-scale batched FP8 matmul: C[s] = A[s] @ B[expert_ids[s]].T, with fused act quant.

    A:  (S, K) raw bf16/fp16 activations
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
    BLOCK_SIZE_N = block_n
    BLOCK_SIZE_K = block_k
    grid = (S, triton.cdiv(N, block_n))
    # Adaptive BLOCK_SIZE_M: smallest power-of-2 >= M, floored at 16, capped at 128.
    # Matches the WGMMA tile to the actual row count — smaller tiles use less
    # register pressure and a better-matched FP8 WGMMA instruction, improving
    # both accuracy and performance for small M (decode).
    BLOCK_SIZE_M = min(max(triton.next_power_of_2((S + E - 1) // E), 16), 128)
    with device_context(A.device):
        wrap_triton(w8a8_block_fp8_matmul_batched_kernel)[grid](
            A,
            B,
            C,
            Bs,
            expert_ids,
            S,
            N,
            K,
            A.stride(1),
            B.stride(2),
            B.stride(1),
            C.stride(1),
            Bs.stride(2),
            Bs.stride(1),
            A.stride(0),
            B.stride(0),
            C.stride(0),
            Bs.stride(0),
            # Meta-parameters
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
        )

    return C


@triton_op("finegrained_fp8::w8a8_tensor_fp8_matmul_batched", mutates_args=())
def _w8a8_tensor_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
) -> torch.Tensor:
    """Tensor-scale batched FP8 matmul: C[s] = A[s] @ B[expert_ids[s]].T, with fused act quant.

    A:  (S, K) raw bf16/fp16 activations
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
    grid = (S, triton.cdiv(N, BLOCK_SIZE_N))
    # Adaptive BLOCK_SIZE_M: smallest power-of-2 >= M, floored at 16, capped at 128.
    # Matches the WGMMA tile to the actual row count — smaller tiles use less
    # register pressure and a better-matched FP8 WGMMA instruction, improving
    # both accuracy and performance for small M (decode).
    BLOCK_SIZE_M = min(max(triton.next_power_of_2((S + E - 1) // E), 16), 128)
    with device_context(A.device):
        wrap_triton(w8a8_tensor_fp8_matmul_batched_kernel)[grid](
            qA,
            B,
            C,
            As,
            Bs,
            expert_ids,
            S,
            N,
            K,
            qA.stride(1),
            B.stride(2),
            B.stride(1),
            C.stride(1),
            As.stride(0),
            qA.stride(0),
            B.stride(0),
            C.stride(0),
            Bs.stride(0),
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
        )

    return C


def w8a8_block_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """Block-scale batched FP8 matmul with fused activation quantization.

    A:  (S, K) raw activations, bf16/fp16/fp32
    B:  (E, N, K) FP8 expert weights
    Bs: (E, N // block_n, K // block_k) per-block weight scales
    """
    return torch.ops.finegrained_fp8.w8a8_block_fp8_matmul_batched(
        A, B, Bs, expert_ids, block_size
    )


def w8a8_tensor_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
) -> torch.Tensor:
    """Tensor-scale batched FP8 matmul with fused activation quantization.

    A:  (S, K) raw activations, bf16/fp16/fp32
    B:  (E, N, K) FP8 expert weights
    Bs: (E,) or (E, 1, 1) per-expert weight scales
    """
    return torch.ops.finegrained_fp8.w8a8_tensor_fp8_matmul_batched(
        A, B, Bs, expert_ids
    )


def w8a8_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int] | None,
) -> torch.Tensor:
    """Unified batched W8A8 FP8 matmul dispatcher.

    Dispatch rules:
    - tensor mode when ``block_size is None``
    - tensor mode when ``block_size == [N, K]``
    - otherwise block mode

    Returns:
        Output tensor ``[S, N]`` in the same dtype as ``A``.
    """
    if block_size is None or (
        block_size[0] == B.size(1) and block_size[1] == B.size(2)
    ):
        return w8a8_tensor_fp8_matmul_batched(A, B, Bs, expert_ids)

    return w8a8_block_fp8_matmul_batched(A, B, Bs, expert_ids, block_size)
