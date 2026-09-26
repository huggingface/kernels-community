# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from .. import gptq_marlin_gemm, gptq_marlin_repack

from .marlin_utils import USE_FP32_REDUCE_DEFAULT, marlin_make_workspace, marlin_permute_scales


def is_fp8_marlin_supported():
    capability = torch.cuda.get_device_capability()
    capability = capability[0] * 10 + capability[1]
    return capability >= 80


def fp8_fused_exponent_bias_into_scales(scales):
    fp8_exponent = 4
    if scales.dtype == torch.half:
        target_exponent = 5
    elif scales.dtype == torch.bfloat16:
        target_exponent = 8
    # exponent_bias_fp16 = 2 ** 4 - 2 ** 3 = 8
    # exponent_bias_bf16 = 2 ** 7 - 2 ** 3 = 120
    exponent_bias = 2**(target_exponent - 1) - 2**(fp8_exponent - 1)
    s = torch.ones_like(scales) * 2
    s = s**exponent_bias
    return scales * s


def apply_fp8_marlin_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        workspace: torch.Tensor,
        size_n: int,
        size_k: int,
        bias: Optional[torch.Tensor],
        use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT) -> torch.Tensor:
    # For GPUs that lack FP8 hardware support, we can leverage the
    # Marlin kernel for fast weight-only FP8 quantization

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n, )

    use_atomic_add = should_use_atomic_add_reduce(m=reshaped_x.size(0),
                                                  n=size_n,
                                                  k=size_k,
                                                  device=input.device,
                                                  dtype=input.dtype)

    output = gptq_marlin_gemm(a=reshaped_x,
                                  c=None,
                                  b_q_weight=weight,
                                  b_scales=weight_scale,
                                  global_scale=None,
                                  b_zeros=None,
                                  g_idx=None,
                                  perm=None,
                                  workspace=workspace,
                                  b_q_type=scalar_types.float8_e4m3fn,
                                  size_m=reshaped_x.size(0),
                                  size_n=size_n,
                                  size_k=size_k,
                                  use_atomic_add=use_atomic_add,
                                  use_fp32_reduce=use_fp32_reduce)

    if bias is not None:
        output.add_(bias)  # In-place add

    return output.reshape(out_shape)

def pack_fp8_to_int32(fp8_tensor: torch.Tensor,
                      size_k_first: bool = True) -> torch.Tensor:
    """
    Repack FP8 weights to gptq format (packed int32 elements)
    """
    assert fp8_tensor.dtype == torch.float8_e4m3fn
    assert fp8_tensor.ndim == 2

    fp8_tensor = fp8_tensor.T if size_k_first else fp8_tensor
    fp8_tensor = fp8_tensor.contiguous()
    # fp8_tensor is contiguous and have shape (N, K) now
    # with `.view(torch.int32)`, it become (N, K // 4)
    int32_tensor = fp8_tensor.view(torch.int32)
    return int32_tensor.T.contiguous() if size_k_first else int32_tensor


def marlin_quant_fp8_torch(weight, group_size):
    size_n, size_k = weight.shape
    device = weight.device

    if group_size != -1:
        scales = weight.view(size_n, -1, group_size).abs().max(-1)[0] / 448
        repeated_scales = scales.repeat_interleave(group_size, 1)
        fp8_weight = (weight / repeated_scales).to(torch.float8_e4m3fn)
        weight_ref = fp8_weight.to(weight.dtype) * repeated_scales
    else:
        scales = weight.view(size_n, 1, group_size).abs().max(-1)[0] / 448
        repeated_scales = scales.repeat_interleave(size_k, 1)
        fp8_weight = (weight / repeated_scales).to(torch.float8_e4m3fn)
        weight_ref = fp8_weight.to(weight.dtype) * repeated_scales

    packed_weight = pack_fp8_to_int32(fp8_weight, False).T.contiguous()
    marlin_qweight = gptq_marlin_repack(
        b_q_weight=packed_weight,
        perm=torch.empty(0, dtype=torch.int, device=device),
        size_k=size_k,
        size_n=size_n,
        num_bits=8,
    )

    marlin_scales = marlin_permute_scales(s=scales.T,
                                          size_k=size_k,
                                          size_n=size_n,
                                          group_size=group_size)

    marlin_scales = fp8_fused_exponent_bias_into_scales(marlin_scales)

    return weight_ref.T, marlin_qweight, marlin_scales
