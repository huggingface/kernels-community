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

"""Selected-row lookup for floating-point and integer quantized embedding tables."""

import torch
import triton
import triton.language as tl

from .compat import (
    FP8_DTYPE,
    MX_SCALE_GROUP_K,
    NVFP4_SCALE_GROUP_K,
    add_op_namespace_prefix,
    compile_time_only_triton_op,
    compile_time_only_triton_wrap,
    device_context,
)
from .formats import ue8m0_as_uint8
from .loading.scales import decode_group_scale
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
    SCALE_BLOCK_ROW: tl.constexpr,
    SCALE_BLOCK_COL: tl.constexpr,
    FORMAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # FP4 packs adjacent columns; INT4 packs adjacent vocabulary rows.
    packing: tl.constexpr = 2 if (FORMAT == "nvfp4" or FORMAT == "mxfp4") else 1
    program = tl.program_id(0).to(tl.int64)
    offsets = program * (BLOCK // packing) + tl.arange(0, BLOCK // packing)
    valid = offsets < N * (D // packing)
    # Tiles contained in one row share its index and per-row scale across all lanes.
    if D % BLOCK == 0:
        indices = program // (D // BLOCK)
    else:
        indices = offsets // (D // packing)
    valid_rows = indices < N
    row = tl.load(Ids + indices, valid_rows, other=0).to(tl.int64)
    in_bounds = (row >= 0) & (row < V)
    tl.device_assert(in_bounds | ~valid_rows, "embedding index out of range")
    valid = valid & in_bounds
    col = offsets % (D // packing)
    weight_row = row // 2 if FORMAT == "int4" else row
    value = tl.load(W + weight_row * W_ROW + col * W_COL, valid, other=0.0)
    if FORMAT == "nvfp4" or FORMAT == "mxfp4":
        value = e2m1_cols_to_f32(value.to(tl.uint8))
    elif FORMAT == "int4":
        nibble = (value.to(tl.uint8).to(tl.int32) >> ((row % 2) * 4)) & 15
        value = ((nibble ^ 8) - 8).to(tl.float32)
    else:
        value = value.to(tl.float32)
    scale_offsets = (row // SCALE_BLOCK_ROW) * S_ROW
    if S_COL != 0:
        scale_offsets += (col * packing // SCALE_BLOCK_COL) * S_COL
    scale = tl.load(
        S + scale_offsets,
        (valid_rows & in_bounds) if S_COL == 0 else valid,
        other=0.0,
    )
    scale = decode_group_scale(scale).to(tl.float32)
    if packing == 2:
        scale = tl.broadcast_to(scale, (BLOCK // packing,))
        scale = tl.interleave(scale, scale)
    value = value * scale
    if FORMAT == "nvfp4":
        value = value * tl.load(G).to(tl.float32)
    output_offsets = program * BLOCK + tl.arange(0, BLOCK)
    tl.store(Out + output_offsets, value, output_offsets < N * D)


@compile_time_only_triton_op(add_op_namespace_prefix("embedding_lookup"))
def embedding_lookup(
    weight: torch.Tensor,
    indices: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    *,
    weight_format: str | None = None,
    block_size: int = 128,
) -> torch.Tensor:
    """Gather and dequantize selected rows without expanding the embedding table.

    ``weight_format`` selects ``fp8``, ``nvfp4``, ``mxfp8``, ``mxfp4``, ``int8``, or ``int4``.
    By default, a global scale selects NVFP4, signed int8 weights select INT8, and otherwise
    FP8 is used. FP8 accepts scalar, per-row, or square block scales (``block_size``).
    NVFP4 uses E4M3 group-16 scales and one FP32 multiplier; MX formats use UE8M0 group-32
    scales. FP4 packs adjacent columns, low nibble first. INT8 uses per-row scales; INT4
    packs adjacent vocabulary rows and uses per-row column groups inferred from its scale
    grid. Integer values are signed, symmetric, and have no zero point.

    Outputs default to the floating scale dtype, or bf16 for NVFP4/MX. Table/grid strides
    may describe TP slices. Indices follow ``torch.nn.functional.embedding``: negative and
    out-of-range values fail; padding indices look up their stored row. Empty, repeated,
    and noncontiguous indices work.
    """
    if weight_format is None:
        if global_scale is not None:
            weight_format = "nvfp4"
        else:
            weight_format = "int8" if weight.dtype == torch.int8 else "fp8"
    if weight_format not in ("fp8", "nvfp4", "mxfp8", "mxfp4", "int8", "int4"):
        raise ValueError(f"Unsupported embedding weight format: {weight_format}.")
    if weight.ndim != 2 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expected a 2D embedding table and int32/int64 indices.")
    if weight.device != indices.device or weight.device != scale.device:
        raise ValueError("Embedding table, indices, and scales must share a device.")
    if weight.device.type not in ("cuda", "xpu"):
        raise ValueError("Fused embedding lookup requires a CUDA, ROCm, or XPU device.")
    packed_columns = weight_format in ("nvfp4", "mxfp4")
    width = weight.shape[1] * (2 if packed_columns else 1)
    vocab = weight.shape[0] * (2 if weight_format == "int4" else 1)
    if width == 0:
        raise ValueError("Embedding width must be positive.")
    float_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    scale_row = scale_col = 0
    row_block = col_block = 1
    if weight_format in ("fp8", "int8"):
        expected_dtype = FP8_DTYPE if weight_format == "fp8" else torch.int8
        if weight.dtype != expected_dtype or scale.dtype not in float_dtypes:
            raise ValueError(
                f"{weight_format.upper()} requires {expected_dtype} weights and floating scales."
            )
        if scale.numel() == 1 and weight_format == "fp8":
            pass
        elif scale.shape in ((vocab,), (vocab, 1)):
            scale_row = scale.stride(0)
        elif (
            weight_format == "fp8"
            and block_size > 0
            and scale.shape
            == (triton.cdiv(vocab, block_size), triton.cdiv(width, block_size))
        ):
            row_block = col_block = block_size
            scale_row, scale_col = scale.stride()
        else:
            raise ValueError(
                "FP8 scales must be scalar, per-row, or a block grid; INT8 scales must be per-row."
            )
    else:
        if weight_format == "mxfp8":
            if weight.dtype != FP8_DTYPE:
                raise ValueError("MXFP8 requires E4M3 weights.")
        elif weight.dtype not in (torch.int8, torch.uint8):
            raise ValueError("FP4 and INT4 weights must be packed int8/uint8 bytes.")
        if weight_format == "nvfp4":
            col_block = NVFP4_SCALE_GROUP_K
            scale_dtypes = (FP8_DTYPE,)
        elif weight_format in ("mxfp8", "mxfp4"):
            col_block = MX_SCALE_GROUP_K
            scale = ue8m0_as_uint8(scale)
            scale_dtypes = (torch.uint8,)
        else:
            if scale.ndim != 2 or scale.shape[1] == 0 or width % scale.shape[1]:
                raise ValueError(
                    "INT4 scales must be a per-row grid of equal column groups."
                )
            col_block = width // scale.shape[1]
            scale_dtypes = float_dtypes
        if (
            width % col_block
            or scale.shape != (vocab, width // col_block)
            or scale.dtype not in scale_dtypes
        ):
            raise ValueError(
                f"{weight_format.upper()} requires a ({vocab}, {width // col_block}) scale grid with dtype in {scale_dtypes}."
            )
        scale_row, scale_col = scale.stride()
    if weight_format == "nvfp4":
        if (
            global_scale is None
            or global_scale.dtype != torch.float32
            or global_scale.numel() != 1
            or global_scale.device != weight.device
        ):
            raise ValueError(
                "NVFP4 requires one FP32 global multiplier on the table's device."
            )
    elif global_scale is not None:
        raise ValueError("Only NVFP4 embeddings use a global multiplier.")
    dtype = output_dtype or (
        torch.bfloat16 if weight_format in ("nvfp4", "mxfp8", "mxfp4") else scale.dtype
    )
    if dtype not in float_dtypes:
        raise ValueError("Embedding output must be float16, bfloat16, or float32.")
    out = torch.empty((*indices.shape, width), dtype=dtype, device=weight.device)
    if indices.numel() == 0:
        return out
    # Keep small lookups parallel; larger tiles amortize scheduling for bulk lookups.
    block = 256
    if out.numel() >= 256 * 1024:
        block = 1024
    if out.numel() >= 4 * 1024 * 1024:
        block = 4096
    with device_context(weight.device):
        compile_time_only_triton_wrap(_embedding_kernel)[
            (triton.cdiv(out.numel(), block),)
        ](
            weight,
            indices.reshape(-1),
            scale,
            global_scale,
            out,
            indices.numel(),
            width,
            vocab,
            weight.stride(0),
            weight.stride(1),
            scale_row,
            scale_col,
            row_block,
            col_block,
            weight_format,
            block,
            num_warps=4,
            enable_fp_fusion=False,
            debug=True,
        )
    return out
