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

from ._ops import add_op_namespace_prefix, ops
from .utils import (
    MX_SCALE_GROUP_K,
    NIBBLES_PER_BYTE,
    device_context,
    mx_act_quant_inline,
    fp8_act_quant,
    fp8_act_quant_inline,
    get_accelerator_autotuning_configs,
    is_mxfp4,
    is_mxfp8,
    ue8m0_as_uint8,
)


@triton.autotune(
    configs=get_accelerator_autotuning_configs(),
    key=["N", "K"],
)
@triton.jit
def w8a8_block_dynamic_fp8_matmul_batched_kernel(
    A,  # (S, K) raw BF16/FP16 activations
    B,  # (E, N, K) FP8 weight matrices
    C,  # (S, N) output
    Bs,  # (E, N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales
    ExpertIds,  # (S,) — which expert each batch element routes to
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
    stride_eid,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
):
    """Block-scale batched FP8 expert matmul kernel.

    Each program handles one routed token row and one N-tile, looks up the
    owning expert from ``ExpertIds``, and applies fused activation quantization.
    """
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Cast to int64 to prevent overflow on expert_id * stride_be.
    expert_id = tl.load(ExpertIds + batch_id * stride_eid).to(tl.int64)
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= NUM_EXPERTS:
        return

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
        a_raw = tl.load(a_ptrs).to(tl.float32)
        a, a_s = fp8_act_quant_inline(a_raw)
        b = tl.load(b_ptrs)
        b_s = tl.load(bs_ptrs)
        if b_s.dtype == tl.uint8:
            # UE8M0 decode: value = 2^(exp - 127); build the fp32 bit pattern.
            b_s = (b_s.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        bs_ptrs += stride_bs_k

    c = accumulator.to(C.dtype.element_ty)

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
    configs=get_accelerator_autotuning_configs(with_block_sizes=True),
    key=["N", "K"],
)
@triton.jit
def w8a8_tensor_dynamic_fp8_matmul_batched_kernel(
    A,  # (S, K) pre-quantized FP8 activations
    B,  # (E, N, K) FP8 weight matrices
    C,  # (S, N) output
    As,  # (S,) per-token activation scales
    Bs,  # (E, 1, 1) per-tensor weight scales
    ExpertIds,  # (S,) — which expert each batch element routes to
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
    stride_eid,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
):
    """Tensor-scale batched FP8 expert matmul kernel.

    Activations are already quantized; the kernel applies per-token activation
    scales and per-expert tensor weight scales.
    """
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    expert_id = tl.load(ExpertIds + batch_id * stride_eid).to(tl.int64)
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= NUM_EXPERTS:
        return

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

    c = accumulator.to(C.dtype.element_ty)

    offs_cm = tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offs_cm[:, None] * 0 + stride_cn * offs_cn[None, :]
    # See block-FP8 kernel above: BLOCK_SIZE_M lanes alias the same C row;
    # mask so only lane 0 stores to avoid hardware-undefined duplicate writes on XPU.
    tl.store(c_ptrs, c, mask=(offs_cm == 0)[:, None])


@triton.autotune(
    configs=get_accelerator_autotuning_configs(with_block_sizes=True),
    key=["N", "K"],
)
@triton.jit
def w4a8_mx_dynamic_fp4_matmul_batched_kernel(
    A,  # (S, K) raw BF16/FP16 activations
    B,  # (E, N, K // 2) packed FP4 (E2M1) expert weights as int8
    C,  # (S, N) output
    Bs,  # (E, N, K // SCALE_GROUP_K) UE8M0 weight scales
    ExpertIds,  # (S,) — which expert each routed row uses
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
    stride_eid,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NIBBLES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
):
    """Batched MXFP4 (W4A8) expert matmul with fused activation quant.

    Each program handles one routed token row and one N-tile, looks up the
    owning expert from ``ExpertIds``, quantizes ``A`` to FP8 per K-group inline
    (UE8M0 scale), then ``tl.dot_scaled`` against packed FP4 weights.
    """
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Cast to int64 to prevent overflow on expert_id * stride_be.
    expert_id = tl.load(ExpertIds + batch_id * stride_eid).to(tl.int64)
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= NUM_EXPERTS:
        return

    A = A + batch_id * stride_am
    B = B + expert_id * stride_be
    C = C + batch_id * stride_cm
    Bs = Bs + expert_id * stride_bs_e

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_k_byte = tl.arange(0, BLOCK_SIZE_K // NIBBLES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)
    a_ptrs = A + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 + offs_k[None, :] * stride_ak
    b_ptrs = B + (offs_k_byte[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    bs_ptrs = Bs + offs_bn[:, None] * stride_bs_n + offs_sf[None, :] * stride_bs_k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_raw = tl.load(a_ptrs).to(tl.float32)
        a, a_scale = mx_act_quant_inline(
            a_raw, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K
        )
        b = tl.load(b_ptrs).to(tl.uint8)
        b_s = tl.load(bs_ptrs).to(tl.uint8)
        accumulator = tl.dot_scaled(a, a_scale, "e4m3", b, b_s, "e2m1", acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // NIBBLES_PER_BYTE) * stride_bk
        bs_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_bs_k

    offs_cm = tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offs_cm[:, None] * 0 + stride_cn * offs_cn[None, :]
    # See block-FP8 batched kernel above: BLOCK_SIZE_M lanes alias the same C row;
    # mask so only lane 0 stores to avoid hardware-undefined duplicate writes on XPU.
    tl.store(c_ptrs, accumulator.to(C.dtype.element_ty), mask=(offs_cm == 0)[:, None])


@triton.autotune(
    configs=get_accelerator_autotuning_configs(with_block_sizes=True),
    key=["N", "K"],
)
@triton.jit
def w8a8_mx_dynamic_fp8_matmul_batched_kernel(
    A,  # (S, K) raw BF16/FP16 activations
    B,  # (E, N, K) E4M3 expert weights (unpacked)
    C,  # (S, N) output
    Bs,  # (E, N, K // SCALE_GROUP_K) UE8M0 weight scales
    ExpertIds,  # (S,) — which expert each routed row uses
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
    stride_eid,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
):
    """Batched MXFP8 (W8A8) expert matmul with fused activation quant.

    Each program handles one routed token row and one N-tile, looks up the
    owning expert from ``ExpertIds``, quantizes ``A`` to E4M3 per K-group inline
    (UE8M0 scale), then ``tl.dot_scaled`` against E4M3 weights. Mirrors the
    batched MXFP4 kernel but with unpacked weights and ``"e4m3"`` on both operands.
    """
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Cast to int64 to prevent overflow on expert_id * stride_be.
    expert_id = tl.load(ExpertIds + batch_id * stride_eid).to(tl.int64)
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= NUM_EXPERTS:
        return

    A = A + batch_id * stride_am
    B = B + expert_id * stride_be
    C = C + batch_id * stride_cm
    Bs = Bs + expert_id * stride_bs_e

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)
    a_ptrs = A + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 + offs_k[None, :] * stride_ak
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    bs_ptrs = Bs + offs_bn[:, None] * stride_bs_n + offs_sf[None, :] * stride_bs_k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_raw = tl.load(a_ptrs).to(tl.float32)
        a, a_scale = mx_act_quant_inline(
            a_raw, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K
        )
        b = tl.load(b_ptrs)
        b_s = tl.load(bs_ptrs).to(tl.uint8)
        accumulator = tl.dot_scaled(a, a_scale, "e4m3", b, b_s, "e4m3", acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        bs_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_bs_k

    offs_cm = tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offs_cm[:, None] * 0 + stride_cn * offs_cn[None, :]
    # See block-FP8 batched kernel above: BLOCK_SIZE_M lanes alias the same C row;
    # mask so only lane 0 stores to avoid hardware-undefined duplicate writes on XPU.
    tl.store(c_ptrs, accumulator.to(C.dtype.element_ty), mask=(offs_cm == 0)[:, None])


@triton_op(
    add_op_namespace_prefix("w8a8_block_dynamic_fp8_matmul_batched"), mutates_args=()
)
def _w8a8_block_dynamic_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype | None = None,
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

    # Decode handles one routed row per program; BLOCK_SIZE_M > 1 would just
    # duplicate the same row computation and keep one row on store.
    BLOCK_SIZE_M = 1
    Bs = ue8m0_as_uint8(Bs)
    grid = (S, triton.cdiv(N, block_n))
    C = A.new_empty(S, N, dtype=output_dtype)

    with device_context(A.device):
        wrap_triton(w8a8_block_dynamic_fp8_matmul_batched_kernel)[grid](
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
            expert_ids.stride(0),
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            NUM_EXPERTS=E,
        )

    return C


@triton_op(
    add_op_namespace_prefix("w8a8_tensor_dynamic_fp8_matmul_batched"), mutates_args=()
)
def _w8a8_tensor_dynamic_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    output_dtype: torch.dtype | None = None,
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

    # Decode handles one routed row per program; BLOCK_SIZE_M > 1 would just
    # duplicate the same row computation and keep one row on store.
    BLOCK_SIZE_M = 1
    Bs = ue8m0_as_uint8(Bs)
    qA, As = fp8_act_quant(A, K)
    C = A.new_empty(S, N, dtype=output_dtype)

    def grid(META):
        return (S, triton.cdiv(N, META["BLOCK_SIZE_N"]))

    with device_context(A.device):
        wrap_triton(w8a8_tensor_dynamic_fp8_matmul_batched_kernel)[grid](
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
            expert_ids.stride(0),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            NUM_EXPERTS=E,
        )

    return C


def w8a8_block_dynamic_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Block-scale batched FP8 matmul with fused activation quantization.

    A:  (S, K) raw activations, bf16/fp16/fp32
    B:  (E, N, K) FP8 expert weights
    Bs: (E, N // block_n, K // block_k) per-block weight scales
    expert_ids: (S,) — kernel loads stride-aware, any int dtype works
    output_dtype: defaults to ``A.dtype``
    """
    return ops.w8a8_block_dynamic_fp8_matmul_batched(
        A, B, Bs, expert_ids, block_size, output_dtype
    )


def w8a8_tensor_dynamic_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Tensor-scale batched FP8 matmul with fused activation quantization.

    A:  (S, K) raw activations, bf16/fp16/fp32
    B:  (E, N, K) FP8 expert weights
    Bs: (E,) or (E, 1, 1) per-expert weight scales
    output_dtype: defaults to ``A.dtype``
    """
    return ops.w8a8_tensor_dynamic_fp8_matmul_batched(
        A, B, Bs, expert_ids, output_dtype
    )


@triton_op(
    add_op_namespace_prefix("w4a8_mx_dynamic_fp4_matmul_batched"), mutates_args=()
)
def _w4a8_mx_dynamic_fp4_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Batched MXFP4 (W4A8) matmul with fused activation quant.

    A:  (S, K) raw activations, bf16/fp16/fp32 (quantized inline to FP8)
    B:  (E, N, K // 2) packed FP4 (E2M1) expert weights, two codes per int8
    Bs: (E, N, K // 32) UE8M0 weight scales
    expert_ids: (S,) which expert each routed row uses

    BLOCK_SIZE_N and BLOCK_SIZE_K are autotuned.
    """
    assert A.ndim == 2 and B.ndim == 3 and Bs.ndim == 3
    assert expert_ids.ndim == 1
    assert B.dtype == torch.int8, f"B must be int8 (packed FP4), got {B.dtype}"
    assert Bs.dtype == torch.float8_e8m0fnu, (
        f"Bs must be float8_e8m0fnu, got {Bs.dtype}"
    )

    S, K = A.shape
    E, N, K_half = B.shape
    assert expert_ids.shape[0] == S
    assert K == NIBBLES_PER_BYTE * K_half, (
        f"K (={K}) must equal {NIBBLES_PER_BYTE} * B.shape[2] (={K_half})"
    )
    assert K % MX_SCALE_GROUP_K == 0, (
        f"K (={K}) must be a multiple of {MX_SCALE_GROUP_K}"
    )
    assert Bs.shape == (E, N, K // MX_SCALE_GROUP_K), (
        f"Bs shape {tuple(Bs.shape)} != ({E}, {N}, {K // MX_SCALE_GROUP_K})"
    )

    # Decode handles one routed row per program; BLOCK_SIZE_M > 1 would just
    # duplicate the same row computation and keep one row on store.
    BLOCK_SIZE_M = 1
    bs_u8 = ue8m0_as_uint8(Bs)
    C = A.new_empty((S, N), dtype=output_dtype)

    def grid(META):
        return (S, triton.cdiv(N, META["BLOCK_SIZE_N"]))

    with device_context(A.device):
        wrap_triton(w4a8_mx_dynamic_fp4_matmul_batched_kernel)[grid](
            A,
            B,
            C,
            bs_u8,
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
            bs_u8.stride(0),
            bs_u8.stride(2),
            bs_u8.stride(1),
            expert_ids.stride(0),
            # Meta-parameters (BLOCK_SIZE_N, BLOCK_SIZE_K come from autotune Config)
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            NIBBLES_PER_BYTE=NIBBLES_PER_BYTE,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
            NUM_EXPERTS=E,
        )
    return C


def w4a8_mx_dynamic_fp4_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Batched MXFP4 (W4A8) matmul with fused activation quant. Tile
    shape autotuned; FP4 scale granularity is fixed at 32."""
    return ops.w4a8_mx_dynamic_fp4_matmul_batched(A, B, Bs, expert_ids, output_dtype)


@triton_op(
    add_op_namespace_prefix("w8a8_mx_dynamic_fp8_matmul_batched"), mutates_args=()
)
def _w8a8_mx_dynamic_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Batched MXFP8 (W8A8) matmul: ``C[s] = A[s] @ B[expert_ids[s]].T`` (E4M3 ×
    E4M3, UE8M0 group-32 scales) with fused activation quant.

    A:  (S, K) raw activations, bf16/fp16/fp32 (quantized inline to E4M3)
    B:  (E, N, K) E4M3 expert weights (unpacked)
    Bs: (E, N, K // 32) UE8M0 weight scales
    expert_ids: (S,) which expert each routed row uses

    BLOCK_SIZE_N and BLOCK_SIZE_K are autotuned.
    """
    assert A.ndim == 2 and B.ndim == 3 and Bs.ndim == 3
    assert expert_ids.ndim == 1
    assert B.dtype == torch.float8_e4m3fn, (
        f"B must be float8_e4m3fn (E4M3 weights), got {B.dtype}"
    )
    assert Bs.dtype == torch.float8_e8m0fnu, (
        f"Bs must be float8_e8m0fnu, got {Bs.dtype}"
    )

    S, K = A.shape
    E, N, K_b = B.shape
    assert expert_ids.shape[0] == S
    assert K == K_b, f"K mismatch: A has K={K}, B has K={K_b}"
    assert K % MX_SCALE_GROUP_K == 0, (
        f"K (={K}) must be a multiple of {MX_SCALE_GROUP_K}"
    )
    assert Bs.shape == (E, N, K // MX_SCALE_GROUP_K), (
        f"Bs shape {tuple(Bs.shape)} != ({E}, {N}, {K // MX_SCALE_GROUP_K})"
    )

    # Decode handles one routed row per program; BLOCK_SIZE_M > 1 would just
    # duplicate the same row computation and keep one row on store.
    BLOCK_SIZE_M = 1
    bs_u8 = ue8m0_as_uint8(Bs)
    C = A.new_empty((S, N), dtype=output_dtype)

    def grid(META):
        return (S, triton.cdiv(N, META["BLOCK_SIZE_N"]))

    with device_context(A.device):
        wrap_triton(w8a8_mx_dynamic_fp8_matmul_batched_kernel)[grid](
            A,
            B,
            C,
            bs_u8,
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
            bs_u8.stride(0),
            bs_u8.stride(2),
            bs_u8.stride(1),
            expert_ids.stride(0),
            # Meta-parameters (BLOCK_SIZE_N, BLOCK_SIZE_K come from autotune Config)
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
            NUM_EXPERTS=E,
        )
    return C


def w8a8_mx_dynamic_fp8_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Batched MXFP8 (W8A8) matmul with fused activation quant (E4M3 × E4M3,
    UE8M0 group-32) via the MX scaled MMA. Tile shape autotuned."""
    return ops.w8a8_mx_dynamic_fp8_matmul_batched(A, B, Bs, expert_ids, output_dtype)


def matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int] | None,
) -> torch.Tensor:
    """Batched quantized matmul dispatcher (W8A8 FP8 or W4A8 FP4). Routes one
    routed row per program to ``B[expert_ids[s]]``.

    Routes by weight dtype and ``block_size``:
    - ``B.dtype == int8`` (packed FP4) → ``w4a8_mx_dynamic_fp4_matmul_batched``
      (``block_size`` is ignored; FP4 tile shape is autotuned).
    - ``B.dtype == float8_e4m3fn`` with UE8M0 group-32 ``Bs`` (shape ``[E, N, K//32]``)
      → ``w8a8_mx_dynamic_fp8_matmul_batched`` (MX scaled MMA; ``block_size`` ignored).
    - ``block_size`` None or full ``[N, K]`` → ``w8a8_tensor_dynamic_fp8_matmul_batched``.
    - otherwise → ``w8a8_block_dynamic_fp8_matmul_batched``.
    """
    if is_mxfp4(B, Bs):
        return w4a8_mx_dynamic_fp4_matmul_batched(A, B, Bs, expert_ids, A.dtype)

    if is_mxfp8(B, Bs):
        return w8a8_mx_dynamic_fp8_matmul_batched(A, B, Bs, expert_ids, A.dtype)

    if block_size is None or (
        block_size[0] == B.size(1) and block_size[1] == B.size(2)
    ):
        return w8a8_tensor_dynamic_fp8_matmul_batched(A, B, Bs, expert_ids)

    return w8a8_block_dynamic_fp8_matmul_batched(A, B, Bs, expert_ids, block_size)
