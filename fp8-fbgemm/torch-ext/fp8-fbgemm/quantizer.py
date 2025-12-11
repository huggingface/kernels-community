# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license
# copied from https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/experimental/gemm/triton_gemm/fp8_gemm.py


import torch
import triton
import triton.language as tl
from torch import nn
from triton import Config
from typing import Any, Optional

def get_fp8_constants() -> tuple[torch.dtype, tl.dtype, float, float]:
    """
    Helper function to get constant values for the current platform.

    Returns:
        pt_dtype (torch.dtype): The correct torch fp8 datatype.
        tl_dtype (tl.dtype): The correct triton fp8 datatype.
        max_fp8 (float): The maximum reprsentable value for the fp8 datatype.
        eps (float): Minimum clip value to prevent divide by zero.
    """
    pt_fp8_dtype = torch.float8_e4m3fn
    tl_fp8_dtype = tl.float8e4nv
    return pt_fp8_dtype, tl_fp8_dtype, torch.finfo(pt_fp8_dtype).max, 1e-12


@triton.autotune(
    configs=[
        Config({"BLOCK_SIZE": 512}),
        Config({"BLOCK_SIZE": 1024}),
        Config({"BLOCK_SIZE": 2048}),
        Config({"BLOCK_SIZE": 4096}),
        Config({"BLOCK_SIZE": 8192}),
    ],
    key=["K"],
)
@triton.jit
def _kernel_quantize_fp8_row(
    A,
    A_scale,
    A_fp8,
    scale_ub,
    zero_start_index_M,
    B,
    M,
    N,
    K,
    K_fp8,  # used when padding
    stride_ab,
    stride_am,
    stride_an,
    stride_ak,
    stride_ob,
    stride_om,
    stride_on,
    stride_ok,
    stride_zb,
    stride_zm,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    JAGGED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_INT64: tl.constexpr,
) -> None:
    """Quantize and scale each row.

    Scale per row i is computed as MAX_FP8 / max(abs(A[i, :]))

    Kernel naively iterates through  matrix with [1, BLOCK_SIZE] tiles
    in a max pass then scale/quantize pass.

    Todo:
        * Better tiling schemes.

    Args:
        A (Tensor): higher precision input tensor of 4 dimension.
        A_scale (Tensor): [B * M * N] reciprocal scale tensor per row.
        A_fp8 (Tensor): fp8 scaled tensor. A_fp8 = A / a_scale
        scale_ub (Tensor): [1] Maximum value allowed for scale.
        B (int): Size of dimenion 0
        M (int): Size of dimenion 1
        N (int): Size of dimenion 2
        K (int): Size of dimenion 3 (input row size)
        K_fp8 (int): Size of dimenion 3 for A_fp8 (output row size, can be >= K)
        stride_ab (int): Stride of b dimension of A.
        stride_am (int): Stride of m dimension of A.
        stride_an (int): Stride of n dimension of A.
        stride_ak (int): Stride of k dimension of A.
        stride_ob (int): Stride of b dimension of output.
        stride_om (int): Stride of m dimension of output.
        stride_on (int): Stride of n dimension of output.
        stride_ok (int): Stride of k dimension of output.
        stride_zb (int): Stride of b dimension of jagged index.
        stride_zm (int): Stride of m dimension of jagged index.
        TL_FP8_DTYPE (tl.dtype): Target fp8 datatype.
        MAX_FP8 (float): Maxmimum expressible value for FP8.
        EPS (float): Epsilon value for numerical stability.
        CLAMP_MAX (bool): Whethar to apply scale_ub.
        JAGGED (bool): Whether to use jagged indexing.
        BLOCK_SIZE (int): Block size for reduction.
        USE_INT64 (bool): Whether to use int64 indexing for large inputs.
    """
    pid = tl.program_id(0)
    # Use int64 indexing for large inputs. This is slower, but
    # needed to avoid index overflows.
    if USE_INT64:
        pid = pid.to(tl.int64)
    n_offset = tl.arange(0, BLOCK_SIZE)
    a_offset_base = pid // (M * N) * stride_ab + (pid % (M * N)) // N * stride_am + (pid % (M * N)) % N * stride_an
    a_fp8_offset_base = pid // (M * N) * stride_ob + (pid % (M * N)) // N * stride_om + (pid % (M * N)) % N * stride_on

    K_in = K
    if JAGGED:
        z_offset_base = pid // (M * N) * stride_zb + (pid % (M * N)) // N * stride_zm
        group_rows = tl.load(zero_start_index_M + z_offset_base)
        current_row = pid % N
        # If this row is empty, dont process any of it.
        if current_row >= group_rows:
            K_in = 0

    # Calculate max.
    cur_max = 0.0
    for _k in range(0, tl.cdiv(K_in, BLOCK_SIZE)):
        a = tl.load(
            A + a_offset_base + n_offset * stride_ak,
            mask=n_offset < K_in,
            other=0.0,
        )
        tile_max = tl.max(tl.abs(a))
        cur_max = tl.maximum(tile_max, cur_max)
        n_offset += BLOCK_SIZE
    # Clamp max value appropriately.
    if CLAMP_MAX:
        ub = tl.load(scale_ub)
        cur_max = tl.clamp(cur_max, EPS, ub)
    else:
        cur_max = tl.maximum(cur_max, EPS)
    # Scale and quantize.
    a_scale = MAX_FP8 / cur_max
    tl.store(A_scale + pid, 1.0 / a_scale)
    n_offset = tl.arange(0, BLOCK_SIZE)

    # Write quantized values for the first K elements (from A), and pad the rest with zeros up to K_fp8
    for _k in range(0, tl.cdiv(K_fp8, BLOCK_SIZE)):
        # Load from A if in range, else 0 (we're going all the way to K_fp8)
        a = tl.load(
            A + a_offset_base + n_offset * stride_ak,
            mask=n_offset < K_in,
            other=0.0,
        )
        # For elements >= K, a will be 0
        a_fp8 = a * a_scale
        # Clamp A to fp8 range to make sure there's no overflow.
        # This is required for AMD. Nvidia's default saturation
        # handles it, but it's nice to have anyway.
        a_fp8 = tl.clamp(a_fp8, -MAX_FP8, MAX_FP8).to(TL_FP8_DTYPE)

        # Store the full new row in its place (for elements >= K, a_fp8 is already 0)
        tl.store(
            A_fp8 + a_fp8_offset_base + n_offset * stride_ok,
            a_fp8,
            mask=n_offset < K_fp8,
        )
        n_offset += BLOCK_SIZE


def quantize_fp8_per_row(
    a: torch.Tensor,
    scale_ub: Optional[torch.Tensor] = None,
    zero_start_index_M: Optional[torch.Tensor] = None,
    align_rows_to: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Call the triton quantize fp8 row kernel to quantize a tensor to fp8 with row-wise scalings.

    Args:
        a (Tensor): higher precision input tensor of 4 dimension.
        scale_ub (Tensor): Maximum allowed value for scale.
        zero_start_index_M (Tensor): Indicates number of nonzero elements in each row.
        align_rows_to: Pad rows to align to this value. Useful for downstream kernels accepting specific sizes (e.g., multiple of 16)
    Returns:
        torch.Tensor: fp8 scaled tensor.
        torch.Tensor: reciprocal scale tensor per row.
    """
    # Handle meta tensors (skip kernel execution)
    if a.device.type == "meta":
        pt_dtype, _, _, _ = get_fp8_constants()
        a_shape = list(a.shape)
        if align_rows_to is not None:
            last_dim = a_shape[-1]
            padded_last_dim = ((last_dim + align_rows_to - 1) // align_rows_to) * align_rows_to
            a_shape[-1] = padded_last_dim

        # Return empty meta tensors with correct shapes
        return (
            torch.empty(a_shape, device="meta", dtype=pt_dtype),
            torch.empty(a_shape[:-1], device="meta", dtype=torch.float32)
        )

    if scale_ub is not None and scale_ub.device != a.device:
        raise Exception("'scale_ub' must be on the same device as 'a'")
    if zero_start_index_M is not None and zero_start_index_M.device != a.device:
        raise Exception("'zero_start_index_M' must be on the same device as 'a'")

    assert a.dim() <= 4, "Triton only supports up to 4 dimension input tensor."
    a_shape = a.shape
    while a.dim() < 4:
        a = a.unsqueeze(0)
    if zero_start_index_M is not None:
        # There should be one value of zero_start_index_M per NxK matrix.
        zero_start_index_M = zero_start_index_M.view(a.shape[0], a.shape[1])
    # Get constant values.
    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()
    num_rows = a.numel() // a.shape[-1]
    a_scale = torch.empty((num_rows), dtype=torch.float32, device=a.device)
    # If align_rows_to is provided, pad the last dimension to be a multiple of it
    if align_rows_to is not None:
        last_dim = a.shape[-1]
        padded_last_dim = ((last_dim + align_rows_to - 1) // align_rows_to) * align_rows_to
        a_fp8 = torch.empty((*a.shape[:-1], padded_last_dim), device=a.device, dtype=pt_dtype)
        a_shape = torch.Size((*a_shape[:-1], padded_last_dim))
    else:
        a_fp8 = torch.empty(a.shape, device=a.device, dtype=pt_dtype)

    # If input tensor is sufficiently large, we need to use int64 indexing.
    use_int64 = a.numel() > (2**31 - 1)
    grid = (num_rows,)
    _kernel_quantize_fp8_row[grid](
        a,
        a_scale,
        a_fp8,
        scale_ub,
        zero_start_index_M,
        a.shape[0],
        a.shape[1],
        a.shape[2],
        a.shape[3],
        a_fp8.shape[3],
        a.stride(0),
        a.stride(1),
        a.stride(2),
        a.stride(3),
        a_fp8.stride(0),
        a_fp8.stride(1),
        a_fp8.stride(2),
        a_fp8.stride(3),
        (zero_start_index_M.stride(0) if zero_start_index_M is not None else None),
        (zero_start_index_M.stride(1) if zero_start_index_M is not None else None),
        TL_FP8_DTYPE=tl_dtype,
        MAX_FP8=max_fp8,
        EPS=eps,
        CLAMP_MAX=scale_ub is not None,
        JAGGED=zero_start_index_M is not None,
        USE_INT64=use_int64,
    )

    return a_fp8.view(a_shape), a_scale.view(a_shape[:-1])
