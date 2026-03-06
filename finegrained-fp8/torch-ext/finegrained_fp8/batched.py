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
def w8a8_block_fp8_matmul_batched_kernel(
    A,  # (S, K)  raw BF16/FP16 activations
    B,  # (E, N, K) FP8 weight matrices
    C,  # (S, N)  output
    Bs,  # (E, N // block_n, K // block_k) weight scales
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
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Block-scale batched FP8 expert matmul kernel.

    Each program handles one routed token row and one N-tile, looks up the
    owning expert from ``ExpertIds``, and applies fused activation quantization.
    """
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Advance base pointers to this token's activation row and its expert's
    # weight / scale slice. No pre-gather of weights needed (like in non-fp8 impls)
    # Cast expert_id to int64 to prevent int32 overflow when computing
    # expert_id * stride_Eb (e.g. 255 * 9_437_184 > 2^31 for 256 experts of
    # 3072×3072 FP8 weights).
    expert_id = tl.load(ExpertIds + batch_id).to(tl.int64)

    A = A + batch_id * stride_Ab
    B = B + expert_id * stride_Eb
    C = C + batch_id * stride_Cb
    Bs = Bs + expert_id * stride_Esb

    offs_bn = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)
    a_ptrs = A + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 + offs_k[None, :] * stride_ak
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_bsn = offs_bn // block_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, block_n), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, block_k)):
        # ---- fused fp8_act_quant ----
        a_raw = tl.load(a_ptrs).to(tl.float32)
        a_s = tl.max(tl.abs(a_raw)) / 448.0  # per-block scale (scalar for M=1)
        a = (a_raw / tl.maximum(a_s, 1e-12)).to(tl.float8e4nv)
        # ---- same as baseline from here ----
        b = tl.load(b_ptrs)
        k_start = k * block_k
        offs_ks = k_start // block_k
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)
        accumulator += tl.dot(a, b) * a_s * b_s[None, :]
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * block_n + tl.arange(0, block_n)
    c_ptrs = C + offs_cm[:, None] * 0 + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)


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
    block_n: tl.constexpr,
    block_k: tl.constexpr,
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

    offs_bn = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)
    a_ptrs = A + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 + offs_k[None, :] * stride_ak
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    b_s = tl.load(Bs)
    a_s = tl.load(As + batch_id * stride_As_b)

    accumulator = tl.zeros((BLOCK_SIZE_M, block_n), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, block_k)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        accumulator += tl.dot(a, b) * a_s * b_s[None, :]
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * block_n + tl.arange(0, block_n)
    c_ptrs = C + offs_cm[:, None] * 0 + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)


@triton_op("finegrained_fp8::w8a8_block_fp8_matmul_batched", mutates_args=())
def _w8a8_block_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int] | None,
) -> torch.Tensor:
    """Internal block-scale batched FP8 matmul op.

    ``A`` is raw activations, ``B``/``Bs`` are stacked expert weights/scales,
    and ``expert_ids`` routes each token row to one expert. Per-token
    activation quantization is fused into the matmul loop.
    """
    assert A.ndim == 2, "A must be (S, K)"
    assert A.is_contiguous(), "A must be contiguous"

    assert B.ndim == 3, "B must be (E, N, K)"
    assert B.is_contiguous(), "B must be contiguous"

    assert A.shape[1] == B.shape[2], "K dimension mismatch between A and B"
    assert expert_ids.is_contiguous(), "expert_ids must be contiguous"
    assert Bs.is_contiguous(), "Bs must be contiguous"

    S, K = A.shape
    E, N, K = B.shape

    if block_size is None:
        block_n, block_k = 128, 128
    else:
        block_n, block_k = block_size[0], block_size[1]

    # if we have per-tensor quantization, we use 128x128 block size for tiled matmul multiplication
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
    grid = (S, triton.cdiv(N, block_n))
    # Adaptive BLOCK_SIZE_M: smallest power-of-2 >= M, floored at 16, capped at 128.
    # Matches the WGMMA tile to the actual row count — smaller tiles use less
    # register pressure and a better-matched FP8 WGMMA instruction, improving
    # both accuracy and performance for small M (decode).
    BLOCK_SIZE_M = min(max(triton.next_power_of_2((S + E - 1) // E), 16), 128)
    wrap_triton(w8a8_block_fp8_matmul_batched_kernel)[grid](
        A,
        B,
        C,
        Bs,
        expert_ids,
        S,
        N,
        K,
        A.stride(1),  # stride_ak
        B.stride(2),  # stride_bk
        B.stride(1),  # stride_bn
        C.stride(1),  # stride_cn
        Bs.stride(2),  # stride_Bs_k
        Bs.stride(1),  # stride_Bs_n
        A.stride(0),  # stride_Ab
        B.stride(0),  # stride_Eb
        C.stride(0),  # stride_Cb
        Bs.stride(0),  # stride_Esb
        # Meta-parameters
        block_n=block_n,
        block_k=block_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )

    return C


@triton_op("finegrained_fp8::w8a8_tensor_fp8_matmul_batched", mutates_args=())
def _w8a8_tensor_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int] | None,
) -> torch.Tensor:
    """Tensor-scale batched FP8 matmul for routed experts.

    Activations are quantized once with per-token tensor scales and multiplied
    with expert-selected FP8 weights using per-expert tensor scales.

    Accepted ``Bs`` layouts: ``[E]`` or ``[E,1,1]``.
    """
    assert A.ndim == 2, "A must be (S, K)"
    assert A.is_contiguous(), "A must be contiguous"

    assert B.ndim == 3, "B must be (E, N, K)"
    assert B.is_contiguous(), "B must be contiguous"

    assert A.shape[1] == B.shape[2], "K dimension mismatch between A and B"
    assert expert_ids.is_contiguous(), "expert_ids must be contiguous"
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
    qA, As = fp8_act_quant(A, K)

    grid = (S, triton.cdiv(N, block_n))
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
        block_n=block_n,
        block_k=block_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
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
        Bs: Expert weight scales, accepted as ``[E, nb, kb]`` (block)
            or ``[E]`` / ``[E,1,1]`` (per-tensor; expanded internally).
        expert_ids: Expert index per token ``[S]``, values in ``[0, E)``.
        block_size: ``[block_n, block_k]`` quantization block dimensions, e.g. ``[128, 128]``.

    Returns:
        Output tensor ``[S, N]`` in the same dtype as ``A``.
    """
    return torch.ops.finegrained_fp8.w8a8_block_fp8_matmul_batched(
        A, B, Bs, expert_ids, block_size
    )


def w8a8_tensor_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int] | None,
) -> torch.Tensor:
    """Tensor-scale batched W8A8 FP8 matmul for MoE expert dispatch.

    Args:
        A: Raw activation matrix ``[S, K]`` in bf16/fp16/fp32.
        B: Stacked expert weight tensor ``[E, N, K]`` in ``float8_e4m3fn``.
        Bs: Per-expert tensor scales ``[E]`` or ``[E,1,1]``.
        expert_ids: Expert index per token ``[S]``, values in ``[0, E)``.
        block_size: Kept for API consistency; tensor path derives tile sizes from ``N`` and ``K``.

    Returns:
        Output tensor ``[S, N]`` in the same dtype as ``A``.
    """
    return torch.ops.finegrained_fp8.w8a8_tensor_fp8_matmul_batched(
        A, B, Bs, expert_ids, block_size
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
        return w8a8_tensor_fp8_matmul_batched(A, B, Bs, expert_ids, block_size)

    return w8a8_block_fp8_matmul_batched(A, B, Bs, expert_ids, block_size)
