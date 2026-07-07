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
from .bayesian_autotuner import bayesian_autotune
from .utils import (
    MX_SCALE_GROUP_K,
    NIBBLES_PER_BYTE,
    UE8M0_SCALE_DTYPES,
    DECODE_BLOCK_SIZE_M,
    decode_ue8m0_scale,
    device_context,
    mx_dot_rescale,
    mx_dot_scaled,
    mx_scalar_reduce,
    mxfp_act_quant_inline,
    fp8_act_quant,
    fp8_act_quant_inline,
    get_accelerator_autotuning_configs,
    get_mxfp_autotuning_configs,
    is_mxfp,
    is_tensor_wide,
    e2m1_as_uint8,
    ue8m0_as_uint8,
)


@triton.jit
def expert_setup(
    A,
    B,
    C,
    Bs,
    ExpertIds,
    stride_am,
    stride_be,
    stride_cm,
    stride_bs_e,
    stride_eid,
):
    """Per-(row, expert) prologue shared by the batched kernels: read the program
    ids, look up the routed expert, and advance the A/B/C/Bs base pointers to this
    row's slice. Returns ``(batch_id, pid_n, expert_id, A, B, C, Bs)``.

    The caller must early-return on the EP sentinel (``expert_id >= num_experts``)
    before any load — the pointer arithmetic itself is harmless, only the loads on a
    non-local expert would be out of bounds."""
    batch_id = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    # Cast to int64 to prevent overflow on expert_id * stride_be.
    expert_id = tl.load(ExpertIds + batch_id * stride_eid).to(tl.int64)
    A = A + batch_id * stride_am
    B = B + expert_id * stride_be
    C = C + batch_id * stride_cm
    Bs = Bs + expert_id * stride_bs_e
    return batch_id, pid_n, expert_id, A, B, C, Bs


@triton.jit
def store_row(
    C,
    accumulator,
    pid_n,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Output epilogue shared by the batched kernels (``C`` already advanced to the
    row). The fake-batch trick aliases all ``BLOCK_SIZE_M`` lanes to the same C row,
    so a plain store would issue ``BLOCK_SIZE_M`` duplicate-address writes — benign on
    NVIDIA WGMMA (last-write-wins of identical bytes) but hardware-undefined on Intel
    XPU, where it corrupts the output. Mask so only lane 0 stores; the accumulator
    rows are mathematically identical (same A row × same B), so lane 0 is correct."""
    c = accumulator.to(C.dtype.element_ty)
    offs_cm = tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offs_cm[:, None] * 0 + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm == 0)[:, None])


@triton.autotune(
    configs=get_accelerator_autotuning_configs(),
    key=["N", "K", "S"],
)
@triton.jit
def w8a8_block_dynamic_fp8_matmul_batched_kernel(
    A,  # (S, K) raw BF16/FP16 activations
    B,  # (num_experts, N, K) FP8 weight matrices
    C,  # (S, N) output
    Bs,  # (num_experts, N // BLOCK_SIZE_N, K // BLOCK_SIZE_K) weight scales
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
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Block-scale batched FP8 expert matmul kernel.

    Each program handles one routed token row and one N-tile, looks up the
    owning expert from ``ExpertIds``, and applies fused activation quantization.
    """
    _, pid_n, expert_id, A, B, C, Bs = expert_setup(
        A, B, C, Bs, ExpertIds, stride_am, stride_be, stride_cm, stride_bs_e, stride_eid
    )
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= num_experts:
        return

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
        b_s = decode_ue8m0_scale(tl.load(bs_ptrs))
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        bs_ptrs += stride_bs_k

    store_row(C, accumulator, pid_n, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N)


@triton.autotune(
    configs=get_accelerator_autotuning_configs(tune_block_nk=True),
    key=["N", "K"],
)
@triton.jit
def w8a8_tensor_dynamic_fp8_matmul_batched_kernel(
    A,  # (S, K) pre-quantized FP8 activations
    B,  # (num_experts, N, K) FP8 weight matrices
    C,  # (S, N) output
    As,  # (S,) per-token activation scales
    Bs,  # (num_experts, 1, 1) per-tensor weight scales
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
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Tensor-scale batched FP8 expert matmul kernel.

    Activations are already quantized; the kernel applies per-token activation
    scales and per-expert tensor weight scales.
    """
    batch_id, pid_n, expert_id, A, B, C, Bs = expert_setup(
        A, B, C, Bs, ExpertIds, stride_am, stride_be, stride_cm, stride_bs_e, stride_eid
    )
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= num_experts:
        return

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

    store_row(C, accumulator, pid_n, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N)


@bayesian_autotune(get_mxfp_autotuning_configs(), ["N", "K", "S"], n_trials=60)
@triton.jit
def mxfp_dynamic_matmul_batched_kernel(
    A,  # (S, K) raw BF16/FP16 activations
    B,  # (num_experts, N, K) E4M3 (MXFP8) or (num_experts, N, K // 2) packed E2M1 (MXFP4) expert weights
    C,  # (S, N) output
    Bs,  # (num_experts, N, K // SCALE_GROUP_K) UE8M0 weight scales
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
    num_experts,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    VALUES_PER_BYTE: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    COMPUTE_MODE: tl.constexpr,
):
    """Unified batched MXFP4/MXFP8 (W4A8/W8A8) expert matmul with fused act quant.

    One routed row + one N-tile per program; expert looked up from ``ExpertIds``. ``A`` is
    quantized to E4M3 per K-group inline (UE8M0 scale). ``VALUES_PER_BYTE`` picks the
    weight format (2 = packed E2M1 / MXFP4, 1 = unpacked E4M3 / MXFP8); ``COMPUTE_MODE``
    picks ``tl.dot_scaled`` (native M=128) vs fp8 ``tl.dot`` + per-group rescale (wins at
    decode where the scaled MMA's M→128 pad is waste; FP4 unpacks E2M1->E4M3, lossless).
    """
    _, pid_n, expert_id, A, B, C, Bs = expert_setup(
        A, B, C, Bs, ExpertIds, stride_am, stride_be, stride_cm, stride_bs_e, stride_eid
    )
    # EP sentinel: row routed to a non-local expert; output is left uninit.
    if expert_id >= num_experts:
        return

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_kb = tl.arange(0, BLOCK_SIZE_K // VALUES_PER_BYTE)
    offs_sf = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_K)
    a_ptrs = A + tl.arange(0, BLOCK_SIZE_M)[:, None] * 0 + offs_k[None, :] * stride_ak
    b_ptrs = B + (offs_kb[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    bs_ptrs = Bs + offs_bn[:, None] * stride_bs_n + offs_sf[None, :] * stride_bs_k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_raw = tl.load(a_ptrs).to(tl.float32)
        a, a_scale = mxfp_act_quant_inline(
            a_raw, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K
        )
        b = tl.load(b_ptrs)
        b_s = tl.load(bs_ptrs).to(tl.uint8)
        if COMPUTE_MODE == "dot_scaled":
            accumulator = mx_dot_scaled(
                accumulator, a, a_scale, b, b_s, VALUES_PER_BYTE
            )
        elif COMPUTE_MODE == "dot":
            accumulator = mx_dot_rescale(accumulator, a, b, a_scale, b_s, VALUES_PER_BYTE)
        else:  # scalar
            accumulator = mx_scalar_reduce(
                accumulator,
                a,
                a_scale,
                b,
                b_s,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                SCALE_GROUP_K,
                VALUES_PER_BYTE,
            )
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // VALUES_PER_BYTE) * stride_bk
        bs_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_K) * stride_bs_k

    store_row(C, accumulator, pid_n, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N)


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
    B:  (num_experts, N, K) FP8 expert weights
    Bs: (num_experts, N // block_n, K // block_k) per-block weight scales
    """
    assert A.ndim == 2, f"A must be 2D (S, K), got ndim={A.ndim}"
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 3, f"B must be 3D (num_experts, N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"
    assert A.shape[1] == B.shape[2], (
        f"K mismatch: A has K={A.shape[1]}, B has K={B.shape[2]}"
    )

    S, K = A.shape
    num_experts, N, _ = B.shape

    assert len(block_size) == 2, (
        f"block_size must be [block_n, block_k], got {block_size}"
    )
    block_n, block_k = block_size[0], block_size[1]
    # MoE expert dimensions must be block-aligned; non-aligned N/K is not supported.
    assert N % block_n == 0, f"N ({N}) must be divisible by block_n ({block_n})"
    assert K % block_k == 0, f"K ({K}) must be divisible by block_k ({block_k})"
    assert Bs.ndim == 3, (
        f"Bs must be 3D (num_experts, N//block_n, K//block_k), got ndim={Bs.ndim}"
    )
    assert Bs.shape == (num_experts, N // block_n, K // block_k), (
        f"Bs shape {tuple(Bs.shape)} != expected ({num_experts}, {N // block_n}, {K // block_k})"
    )

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
            BLOCK_SIZE_M=DECODE_BLOCK_SIZE_M,
            num_experts=num_experts,
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
    B:  (num_experts, N, K) FP8 expert weights
    Bs: (num_experts,) or (num_experts, 1, 1) per-expert weight scales
    """
    assert A.ndim == 2, f"A must be 2D (S, K), got ndim={A.ndim}"
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 3, f"B must be 3D (num_experts, N, K), got ndim={B.ndim}"
    assert B.is_contiguous(), "B must be contiguous"
    assert A.shape[1] == B.shape[2], (
        f"K mismatch: A has K={A.shape[1]}, B has K={B.shape[2]}"
    )

    S, K = A.shape
    num_experts, N, _ = B.shape

    # Normalize Bs to (num_experts, 1, 1)
    if Bs.ndim == 1:
        assert Bs.shape[0] == num_experts, (
            f"Bs shape {tuple(Bs.shape)} != expected ({num_experts},)"
        )
        Bs = Bs.reshape(num_experts, 1, 1)
    else:
        assert Bs.shape == (num_experts, 1, 1), (
            f"Bs shape {tuple(Bs.shape)} != expected ({num_experts}, 1, 1)"
        )

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
            BLOCK_SIZE_M=DECODE_BLOCK_SIZE_M,
            num_experts=num_experts,
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
    B:  (num_experts, N, K) FP8 expert weights
    Bs: (num_experts, N // block_n, K // block_k) per-block weight scales
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
    B:  (num_experts, N, K) FP8 expert weights
    Bs: (num_experts,) or (num_experts, 1, 1) per-expert weight scales
    output_dtype: defaults to ``A.dtype``
    """
    return ops.w8a8_tensor_dynamic_fp8_matmul_batched(
        A, B, Bs, expert_ids, output_dtype
    )


@triton_op(add_op_namespace_prefix("mxfp_dynamic_matmul_batched"), mutates_args=())
def _mxfp_dynamic_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Batched MX matmul ``C[s] = A[s] @ B[expert_ids[s]].T`` with fused act quant.
    Weight format is detected from ``B.dtype``: ``int8`` → packed E2M1 (MXFP4, ``B`` is
    ``(num_experts, N, K//2)``); ``float8_e4m3fn`` → unpacked E4M3 (MXFP8, ``(num_experts, N, K)``). Both use
    UE8M0 group-32 scales ``(num_experts, N, K//32)``; tile + dot path are autotuned.

    A:  (S, K) raw activations, bf16/fp16/fp32 (quantized inline to E4M3)
    expert_ids: (S,) which expert each routed row uses
    """
    assert A.ndim == 2 and B.ndim == 3 and Bs.ndim == 3
    assert expert_ids.ndim == 1
    assert B.dtype in (torch.int8, torch.float8_e4m3fn), (
        f"B must be int8 (packed E2M1) or float8_e4m3fn (E4M3), got {B.dtype}"
    )
    assert Bs.dtype in UE8M0_SCALE_DTYPES, (
        f"Bs must be float8_e8m0fnu or uint8 (UE8M0), got {Bs.dtype}"
    )
    VALUES_PER_BYTE = NIBBLES_PER_BYTE if B.dtype == torch.int8 else 1

    S, K = A.shape
    num_experts, N, K_b = B.shape
    assert expert_ids.shape[0] == S
    assert K == VALUES_PER_BYTE * K_b, (
        f"K (={K}) must equal {VALUES_PER_BYTE} * B.shape[2] (={K_b})"
    )
    assert K % MX_SCALE_GROUP_K == 0, (
        f"K (={K}) must be a multiple of {MX_SCALE_GROUP_K}"
    )
    assert Bs.shape == (num_experts, N, K // MX_SCALE_GROUP_K), (
        f"Bs shape {tuple(Bs.shape)} != ({num_experts}, {N}, {K // MX_SCALE_GROUP_K})"
    )

    B = e2m1_as_uint8(B)
    bs_u8 = ue8m0_as_uint8(Bs)
    C = A.new_empty((S, N), dtype=output_dtype)

    def grid(META):
        return (S, triton.cdiv(N, META["BLOCK_SIZE_N"]))

    with device_context(A.device):
        wrap_triton(mxfp_dynamic_matmul_batched_kernel)[grid](
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
            BLOCK_SIZE_M=DECODE_BLOCK_SIZE_M,
            VALUES_PER_BYTE=VALUES_PER_BYTE,
            SCALE_GROUP_K=MX_SCALE_GROUP_K,
            num_experts=num_experts,
        )
    return C


def mxfp_dynamic_matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Batched MX (MXFP4/MXFP8) matmul with fused act quant; the weight format
    (packed E2M1 vs E4M3) is detected from ``B.dtype``. Tile + dot path autotuned."""
    return ops.mxfp_dynamic_matmul_batched(A, B, Bs, expert_ids, output_dtype)


def matmul_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: list[int] | None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Batched quantized matmul dispatcher (W8A8 FP8 or W4A8 FP4). Routes one
    routed row per program to ``B[expert_ids[s]]``.

    ``output_dtype`` defaults to ``A.dtype``.

    Routes by weight dtype and ``block_size``:
    - MX weights — ``int8`` (packed E2M1) or ``float8_e4m3fn`` (E4M3) with UE8M0
      group-32 ``Bs`` (shape ``[num_experts, N, K//32]``) → ``mxfp_dynamic_matmul_batched``
      (``block_size`` ignored; tile + dot path autotuned).
    - ``block_size`` None or full ``[N, K]`` → ``w8a8_tensor_dynamic_fp8_matmul_batched``.
    - otherwise → ``w8a8_block_dynamic_fp8_matmul_batched``.
    """
    if is_mxfp(B, Bs):
        return mxfp_dynamic_matmul_batched(A, B, Bs, expert_ids, output_dtype)

    if is_tensor_wide(block_size, B):
        return w8a8_tensor_dynamic_fp8_matmul_batched(
            A, B, Bs, expert_ids, output_dtype
        )

    return w8a8_block_dynamic_fp8_matmul_batched(
        A, B, Bs, expert_ids, block_size, output_dtype
    )
