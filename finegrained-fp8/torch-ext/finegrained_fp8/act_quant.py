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


_FP8_DTYPE = torch.float8_e4m3fn


# Copied from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py
@triton.jit
def _fp8_act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0  # float8_e4m3fn max
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


@triton_op("finegrained_fp8::fp8_act_quant", mutates_args=())
def _fp8_act_quant(
    x: torch.Tensor, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.shape[-1] % block_size == 0
    y = torch.empty_like(x, dtype=_FP8_DTYPE)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)

    grid = (triton.cdiv(x.numel(), block_size),)
    wrap_triton(_fp8_act_quant_kernel)[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


def fp8_act_quant(
    x: torch.Tensor, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to FP8 with per-block dynamic scaling.

    Splits the last dimension of ``x`` into blocks of ``block_size`` elements,
    computes ``scale = max(|x_block|) / 448`` per block, and quantizes to
    ``float8_e4m3fn``.

    Args:
        x: Input tensor in bf16/fp16/fp32. Last dimension must be divisible by
            ``block_size`` and the tensor must be contiguous.
        block_size: Number of elements per quantization block (default: 128).

    Returns:
        A tuple ``(quantized, scales)`` where ``quantized`` has dtype
        ``float8_e4m3fn`` with the same shape as ``x``, and ``scales`` has
        shape ``(*x.shape[:-1], x.shape[-1] // block_size)`` in float32.
    """
    return torch.ops.finegrained_fp8.fp8_act_quant(x, block_size)
