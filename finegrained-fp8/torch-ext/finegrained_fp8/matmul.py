# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/fp8_kernel.py
@triton.jit
def w8a8_block_fp8_matmul_kernel(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and
    store the result in output tensor `C`.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton_op("finegrained_fp8::w8a8_block_fp8_matmul", mutates_args=())
def _w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int] | None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """This function performs matrix multiplication with block-wise
    quantization.
    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.
    Args:
        A: The input tensor, e.g., activation.
        B: The input tensor, e.g., weight.
        As: The per-token-group quantization scale for `A`.
        Bs: The per-block quantization scale for `B`.
        block_size: The block size for per-block quantization. It should
        be 2-dim, e.g., [128, 128].
        output_dtype: The dtype of the returned tensor.
    Returns:
        torch.Tensor: The result of matmul.
    """
    if block_size is None:
        block_n, block_k = 128, 128
    else:
        assert len(block_size) == 2
        block_n, block_k = block_size[0], block_size[1]

    # if we have per-tensor quantization, we use 128x128 block size for tiled matmul multiplication
    if block_n == B.shape[-2] and block_k == B.shape[-1]:
        block_n = 128
        block_k = 128

    assert A.shape[-1] == B.shape[-1]
    assert A.is_contiguous()

    assert B.ndim == 2
    assert B.is_contiguous()

    N, K = B.shape
    M = A.numel() // A.shape[-1]

    # For per-tensor scales (scalar), expand to block-scale shape with strides (0, 0).
    # This is a zero-copy view; all loads inside the kernel hit the same cached value.
    if As.numel() == 1:
        As = As.reshape(1, 1).expand(M, triton.cdiv(K, block_k))
    else:
        assert A.shape[:-1] == As.shape[:-1]
        assert triton.cdiv(K, block_k) == As.shape[-1]
    if Bs.numel() == 1:
        Bs = Bs.reshape(1, 1).expand(triton.cdiv(N, block_n), triton.cdiv(K, block_k))
    else:
        assert Bs.ndim == 2
        assert triton.cdiv(N, block_n) == Bs.shape[0], f"{N}, {block_n}, {Bs.shape}"
        assert triton.cdiv(K, block_k) == Bs.shape[1], f"{K}, {block_k}, {Bs.shape}"

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    # Adaptive BLOCK_SIZE_M: smallest power-of-2 >= M, floored at 16, capped at 128.
    # Matches the WGMMA tile to the actual row count — smaller tiles use less
    # register pressure and a better-matched FP8 WGMMA instruction, improving
    # both accuracy and performance for small M (decode).
    BLOCK_SIZE_M = min(max(triton.next_power_of_2(M), 16), 128)
    BLOCK_SIZE_K = block_k
    BLOCK_SIZE_N = block_n

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    wrap_triton(w8a8_block_fp8_matmul_kernel)[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
    )

    return C


def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int] | None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Block-wise W8A8 FP8 matrix multiplication.

    Computes ``C = A @ B.T`` where both operands are pre-quantized to
    ``float8_e4m3fn`` with per-block scales, and accumulates in float32
    before casting to ``output_dtype``.

    Args:
        A: Quantized activation tensor ``[M, K]`` in ``float8_e4m3fn``.
        B: Quantized weight tensor ``[N, K]`` in ``float8_e4m3fn``.
        As: Per-token-group activation scales ``[M, K // block_size[1]]``.
        Bs: Per-block weight scales ``[N // block_size[0], K // block_size[1]]``.
        block_size: ``[block_n, block_k]`` quantization block dimensions, e.g. ``[128, 128]``.
        output_dtype: dtype of the returned tensor (default: ``torch.float32``).

    Returns:
        Output tensor ``[M, N]`` in ``output_dtype``.
    """
    return torch.ops.finegrained_fp8.w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype)
