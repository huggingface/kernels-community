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

    A = A + batch_id * stride_am
    B = B + expert_id * stride_be
    C = C + batch_id * stride_cm
    Bs = Bs + expert_id * stride_bs_e

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 + offs_k[None, :] * stride_ak
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    bs_ptrs = Bs + pid_n * stride_bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # ---- fused fp8_act_quant ----
        a_raw = tl.load(a_ptrs).to(tl.float32)
        a_s = tl.max(tl.abs(a_raw)) / 448.0
        a = (a_raw / tl.maximum(a_s, 1e-12)).to(tl.float8e4nv)
        # ---- matmul ----
        b = tl.load(b_ptrs)
        b_s = tl.load(bs_ptrs + k * stride_bs_k)
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
    # Fake-batch trick aliases all BLOCK_SIZE_M lanes to the same C row, so emitting
    # `tl.store(c_ptrs, c)` issues BLOCK_SIZE_M duplicate-address stores. On NVIDIA
    # WGMMA this is usually benign (last-write-wins of identical bytes), but on Intel
    # XPU the duplicate-address store has hardware-undefined behavior and corrupts the
    # output. Mask so only lane 0 stores — the (M, N) accumulator rows are
    # mathematically identical (same A row × same B), so lane 0 holds the right value.
    tl.store(c_ptrs, c, mask=(offs_cm == 0)[:, None])


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
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_as_m,
    stride_bs_e,
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

    A = A + batch_id * stride_am
    B = B + expert_id * stride_be
    C = C + batch_id * stride_cm
    Bs = Bs + expert_id * stride_bs_e

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 + offs_k[None, :] * stride_ak
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    b_s = tl.load(Bs)
    a_s = tl.load(As + batch_id * stride_as_m)

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
    # See block-FP8 kernel above: BLOCK_SIZE_M lanes alias the same C row;
    # mask so only lane 0 stores to avoid hardware-undefined duplicate writes on XPU.
    tl.store(c_ptrs, c, mask=(offs_cm == 0)[:, None])


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [8, 16]
        for s in [2, 3, 4]
    ],
    key=["N", "K", "BLOCK_SIZE_N", "BLOCK_SIZE_K"],
)
@triton.jit
def w4a8_block_fp4_matmul_batched_kernel(
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
    stride_expert_ids,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    expert_id = tl.load(ExpertIds_ptr + batch_id * stride_expert_ids).to(tl.int64)

    A_ptr = A_ptr + batch_id * stride_am
    AS_ptr = AS_ptr + batch_id * stride_asm
    B_ptr = B_ptr + expert_id * stride_be
    SF_ptr = SF_ptr + expert_id * stride_sfe
    C_ptr = C_ptr + batch_id * stride_cm

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k_byte = tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)

    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_base = k0 * BLOCK_SIZE_K
        a_k = k_base + tl.arange(0, BLOCK_SIZE_K)
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
    # Adaptive BLOCK_SIZE_M: smallest power-of-2 >= M, floored at 16, capped at 128.
    # Matches the WGMMA tile to the actual row count — smaller tiles use less
    # register pressure and a better-matched FP8 WGMMA instruction, improving
    # both accuracy and performance for small M (decode).
    BLOCK_SIZE_M = min(max(triton.next_power_of_2((S + E - 1) // E), 16), 128)
    grid = (S, triton.cdiv(N, block_n))
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
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
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
    grid = (S, triton.cdiv(N, BLOCK_SIZE_N))
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
            qA.stride(0),
            qA.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            As.stride(0),
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


@triton_op("finegrained_fp8::w4a8_block_fp4_matmul_batched", mutates_args=())
def _w4a8_block_fp4_matmul_batched(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert A.ndim == 2 and As.ndim == 2 and B.ndim == 3 and Bs.ndim == 3
    assert expert_ids.ndim == 1
    assert A.dtype == torch.float8_e4m3fn
    assert As.dtype in (torch.float32, torch.float8_e8m0fnu)
    assert B.dtype == torch.int8
    assert Bs.dtype == torch.float8_e8m0fnu
    assert len(block_size) == 2, (
        f"block_size must be [block_n, block_k], got {block_size}"
    )

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

    is_tensor_scale_input = As.shape[-1] == 1
    block_n = block_size[0]
    block_k = fp4_resolve_block_k(block_size, K)
    As = fp4_expand_activation_scales(As, K, block_k)
    assert As.shape == (S, K // FP4_SCALE_GROUP_K), (
        f"As shape {tuple(As.shape)} != ({S}, {K // FP4_SCALE_GROUP_K})"
    )

    as_u8 = As.to(torch.float8_e8m0fnu).view(torch.uint8)
    sf_u8 = Bs.view(torch.uint8)
    expert_ids = expert_ids.to(device=A.device, dtype=torch.int32)
    launch_block_n = 128 if is_tensor_scale_input else block_n
    launch_block_k = 128 if is_tensor_scale_input and K % 128 == 0 else block_k
    C = A.new_empty((S, N), dtype=output_dtype)
    # Decode handles one routed row per program; BLOCK_SIZE_M > 1 would just
    # duplicate the same row computation and keep one row on store.
    BLOCK_SIZE_M = 1
    grid = (S, triton.cdiv(N, launch_block_n))
    with device_context(A.device):
        wrap_triton(w4a8_block_fp4_matmul_batched_kernel)[grid](
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
            expert_ids.stride(0),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=launch_block_n,
            BLOCK_SIZE_K=launch_block_k,
            VALUES_PER_BYTE=FP4_VALUES_PER_BYTE,
            SCALE_GROUP_K=FP4_SCALE_GROUP_K,
        )
    return C


def w4a8_block_fp4_matmul_batched(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Block-scale batched W4A8 FP4 matmul: ``C[s] = A[s] @ B[expert_ids[s]].T``.

    A:  (S, K) FP8 activations
    As: (S, K // block_k) UE8M0 activation scales
    B:  (E, N, K // 2) packed FP4 expert weights (int8, two codes per byte)
    Bs: (E, N, K // 32) UE8M0 weight scales
    """
    return torch.ops.finegrained_fp8.w4a8_block_fp4_matmul_batched(
        A, As, B, Bs, expert_ids, block_size, output_dtype
    )


def fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int] | None,
) -> torch.Tensor:
    """Neutral batched FP8 matmul dispatcher.

    Routes by weight dtype and scale shape:
    - ``int8`` weights (packed FP4) → ``w4a8_block_fp4_matmul_batched`` (block
      mode only; tensor-mode FP4 — ``block_size`` None or ``[N, K]`` — is not
      yet supported). The W8A8 kernels fuse activation quantization but the W4A8
      block kernel takes pre-quantized FP8 activations, so this wrapper runs
      ``fp8_act_quant`` first.
    - ``block_size`` None or ``[N, K]`` → ``w8a8_tensor_fp8_matmul_batched``
    - otherwise → ``w8a8_block_fp8_matmul_batched``
    """
    is_fp4 = B.dtype == torch.int8
    is_tensor_mode = block_size is None or (
        block_size[0] == B.size(1) and block_size[1] == B.size(2)
    )

    if is_fp4:
        if is_tensor_mode:
            raise NotImplementedError(
                "W4A8 FP4 batched path only supports block mode; tensor-mode "
                "(block_size=None or [N, K]) is not yet implemented."
            )
        qA, As = fp8_act_quant(A, block_size[1])
        return w4a8_block_fp4_matmul_batched(
            qA, As, B, Bs, expert_ids, block_size, A.dtype
        )

    if is_tensor_mode:
        return w8a8_tensor_fp8_matmul_batched(A, B, Bs, expert_ids)

    return w8a8_block_fp8_matmul_batched(A, B, Bs, expert_ids, block_size)
