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

"""Selected-row lookup for scalar-scaled FP8 and group-16 NVFP4 embedding tables."""

import torch
import triton
import triton.language as tl

from .compat import (
    add_op_namespace_prefix,
    compile_time_only_triton_op,
    compile_time_only_triton_wrap,
    device_context,
)
from .quant import e2m1_cols_to_f32


@triton.jit
def _embedding_kernel(
    W,
    Ids,
    S,
    G,
    Out,
    N: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    W_ROW: tl.constexpr,
    W_COL: tl.constexpr,
    S_ROW: tl.constexpr,
    S_COL: tl.constexpr,
    NVFP4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Read each packed byte once and let the shared decoder interleave its two values.
    packing: tl.constexpr = 2 if NVFP4 else 1
    offsets = tl.program_id(0) * (BLOCK // packing) + tl.arange(0, BLOCK // packing)
    valid = offsets < N * (D // packing)
    row = tl.load(Ids + offsets // (D // packing), valid, other=0).to(tl.int64)
    in_bounds = (row >= 0) & (row < V)
    tl.device_assert(in_bounds | ~valid, "embedding index out of range")
    valid = valid & in_bounds
    col = offsets % (D // packing)
    if NVFP4:
        packed = tl.load(W + row * W_ROW + col * W_COL, valid, other=0).to(tl.uint8)
        value = e2m1_cols_to_f32(packed)
        scale = tl.load(S + row * S_ROW + (col // 8) * S_COL, valid, other=0.0).to(
            tl.float32
        )
        value = (value * tl.interleave(scale, scale)) * tl.load(G).to(tl.float32)
    else:
        value = tl.load(W + row * W_ROW + col * W_COL, valid, other=0.0).to(tl.float32)
        value = value * tl.load(S).to(tl.float32)
    output_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Out + output_offsets, value, output_offsets < N * D)


@compile_time_only_triton_op(add_op_namespace_prefix("embedding_lookup"))
def embedding_lookup(
    weight: torch.Tensor,
    indices: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Gather and dequantize only selected rows, without expanding the embedding table.

    FP8 takes an E4M3 table and one scale (output defaults to the scale dtype). NVFP4 takes
    low-nibble-first packed int8/uint8 weights, an affine E4M3 scale grid per 16 values, and
    one FP32 multiplier (output defaults to bf16). Table/grid strides may describe TP slices.
    Indices follow ``torch.nn.functional.embedding``: negative and out-of-range values fail;
    padding indices look up their stored row. Empty, repeated, and noncontiguous indices work.
    """
    packed = global_scale is not None
    if weight.ndim != 2 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expected a 2D embedding table and int32/int64 indices.")
    if weight.device != indices.device or weight.device != scale.device:
        raise ValueError("Embedding table, indices, and scales must share a device.")
    width = weight.shape[1] * (2 if packed else 1)
    if packed:
        if weight.dtype not in (torch.int8, torch.uint8) or width % 16:
            raise ValueError(
                "NVFP4 tables must be packed bytes with width divisible by 16."
            )
        if scale.dtype != torch.float8_e4m3fn or scale.shape != (
            weight.shape[0],
            width // 16,
        ):
            raise ValueError(
                "NVFP4 scales must be an E4M3 grid with one scale per 16 values."
            )
        if (
            global_scale.dtype != torch.float32
            or global_scale.numel() != 1
            or global_scale.device != weight.device
        ):
            raise ValueError(
                "NVFP4 requires one FP32 global multiplier on the table's device."
            )
        dtype = output_dtype or torch.bfloat16
    else:
        if (
            weight.dtype != torch.float8_e4m3fn
            or scale.numel() != 1
            or scale.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise ValueError("FP8 tables require E4M3 weights and a scalar scale.")
        dtype = output_dtype or scale.dtype
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Embedding output must be float16, bfloat16, or float32.")
    if width == 0:
        raise ValueError("Embedding width must be positive.")
    if weight.device.type not in ("cuda", "xpu"):
        raise ValueError("Fused embedding lookup requires a CUDA, ROCm, or XPU device.")
    out = torch.empty((*indices.shape, width), dtype=dtype, device=weight.device)
    if indices.numel() == 0:
        return out
    with device_context(weight.device):
        compile_time_only_triton_wrap(_embedding_kernel)[
            (triton.cdiv(indices.numel() * width, 256),)
        ](
            weight,
            indices.reshape(-1),
            scale,
            global_scale,
            out,
            indices.numel(),
            width,
            weight.shape[0],
            weight.stride(0),
            weight.stride(1),
            scale.stride(0) if packed else 0,
            scale.stride(1) if packed else 0,
            packed,
            256,
            num_warps=4,
            enable_fp_fusion=False,
            debug=True,
        )
    return out
