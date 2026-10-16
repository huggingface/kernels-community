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

"""Lossless NVFP4 CSF scales: original streams plus a compact row directory.

Codec: byte-window4-fixed-stream-u24-exceptions/1. Preparation runs on CPU
at load time; GEMMs read these compressed buffers directly on the device.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
import triton
import triton.language as tl

from .compat import (
    add_op_namespace_prefix,
    compile_time_only_triton_op,
    compile_time_only_triton_wrap,
    device_context,
)
from .loading.csf import load_nvfp4_csf_scale


@dataclass(frozen=True)
class NVFP4CSFScales:
    fixed: torch.Tensor
    exceptions: torch.Tensor
    row_offsets: torch.Tensor
    search_steps: int

    @property
    def shape(self):
        experts, slabs, slab_bytes = self.fixed.shape
        return experts, slabs * 16, (slab_bytes // 16 - 1) * 2

    @property
    def storage_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (self.fixed, self.exceptions, self.row_offsets)
        )

    def to(self, device, *, non_blocking=False):
        return NVFP4CSFScales(
            self.fixed.to(device, non_blocking=non_blocking),
            self.exceptions.to(device, non_blocking=non_blocking),
            self.row_offsets.to(device, non_blocking=non_blocking),
            self.search_steps,
        )


def prepare_nvfp4_csf_scales(
    fixed: torch.Tensor, exceptions: torch.Tensor | Sequence[torch.Tensor]
) -> NVFP4CSFScales:
    """Prepare one matrix or an expert stack on CPU, without expanding its scales.

    `fixed` is uint8 `[rows/16, 16*(1+cols/2)]`, or the same shape with an
    expert dimension. Exceptions are uint32 `[count]` for one matrix, or one
    tensor per expert; low 24 bits locate a scale and high 8 bits replace it.
    The returned buffers retain both original streams and add uint32 row
    offsets for bounded per-row lookup. Move the result with `.to(device)`.
    """
    if fixed.dtype != torch.uint8 or fixed.ndim not in (2, 3):
        raise ValueError("CSF fixed stream must be a 2D or 3D uint8 tensor")
    if fixed.device.type != "cpu":
        raise ValueError("Prepare CSF scales on CPU before moving them to the device")
    fixed = fixed.unsqueeze(0) if fixed.ndim == 2 else fixed
    experts, slabs, slab_bytes = fixed.shape
    if experts == 0 or slabs == 0 or slab_bytes <= 16 or slab_bytes % 16:
        raise ValueError("Invalid CSF slab shape")
    rows, cols = slabs * 16, (slab_bytes // 16 - 1) * 2
    if rows * cols > 1 << 24:
        raise ValueError(
            "CSF exception positions support at most 2**24 scales per matrix"
        )
    if bool((fixed[:, :, :16] > 240).any()):
        raise ValueError("CSF row bases must be in 0..240")
    streams = [exceptions] if isinstance(exceptions, torch.Tensor) else list(exceptions)
    if len(streams) != experts:
        raise ValueError("CSF needs one exception stream per expert")
    counts, words = [], []
    max_count = 0
    for stream in streams:
        if (
            stream.dtype != torch.uint32
            or stream.ndim != 1
            or stream.device.type != "cpu"
        ):
            raise ValueError("CSF exception streams must be 1D CPU uint32 tensors")
        values = stream.to(torch.int64)
        positions = values & 0xFFFFFF
        if positions.numel() and (
            int(positions[-1]) >= rows * cols
            or bool((positions[1:] <= positions[:-1]).any())
        ):
            raise ValueError(
                "CSF exception positions must be sorted, unique and in range"
            )
        per_row = torch.bincount(positions // cols, minlength=rows)
        max_count = max(max_count, int(per_row.max()))
        counts.append(per_row)
        words.append(stream)
    count = sum(stream.numel() for stream in streams)
    if count >= 1 << 32:
        raise ValueError("CSF row directory exceeds uint32 offsets")
    row_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int64), torch.cat(counts).cumsum(0))
    ).to(torch.uint32)
    return NVFP4CSFScales(
        fixed.contiguous(), torch.cat(words), row_offsets, max_count.bit_length()
    )


@triton.jit
def _decode_csf_kernel(
    Fixed,
    Exceptions,
    RowOffsets,
    Out,
    TOTAL: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = index // (ROWS * COLS)
    row = (index // COLS) % ROWS
    col = index % COLS
    scale = load_nvfp4_csf_scale(
        Fixed,
        Exceptions,
        RowOffsets,
        expert,
        row,
        col,
        ROWS,
        COLS,
        STEPS,
        mask=index < TOTAL,
    )
    tl.store(Out + index, scale, mask=index < TOTAL)


@compile_time_only_triton_op(
    add_op_namespace_prefix("decode_nvfp4_csf_scales"), mutates_args=()
)
def _decode_csf(
    fixed: torch.Tensor,
    exceptions: torch.Tensor,
    row_offsets: torch.Tensor,
    rows: int,
    cols: int,
    search_steps: int,
) -> torch.Tensor:
    output = torch.empty(
        (fixed.shape[0], rows, cols), device=fixed.device, dtype=torch.float8_e4m3fn
    )
    with device_context(fixed.device):
        compile_time_only_triton_wrap(_decode_csf_kernel)[
            (triton.cdiv(output.numel(), 256),)
        ](
            fixed,
            exceptions,
            row_offsets,
            output,
            output.numel(),
            rows,
            cols,
            search_steps,
            256,
        )
    return output


def decompress_nvfp4_csf_scales(scales: NVFP4CSFScales) -> torch.Tensor:
    """Explicitly allocate dense E4M3 scales, for conversion or diagnostics.

    CSF GEMM entrypoints do not call this function or allocate dense scales.
    """
    experts, rows, cols = scales.shape
    if scales.fixed.device.type != "cpu":
        return _decode_csf(
            scales.fixed,
            scales.exceptions,
            scales.row_offsets,
            rows,
            cols,
            scales.search_steps,
        )
    bases = scales.fixed[:, :, :16].reshape(experts, rows, 1).to(torch.int16)
    packed = scales.fixed[:, :, 16:].reshape(experts, rows, cols // 2)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(experts, rows, cols)
    output = (bases + codes).to(torch.uint8)
    for expert in range(experts):
        start = int(scales.row_offsets[expert * rows])
        end = int(scales.row_offsets[(expert + 1) * rows])
        words = scales.exceptions[start:end].to(torch.int64)
        output[expert].view(-1)[words & 0xFFFFFF] = (words >> 24).to(torch.uint8)
    return output.view(torch.float8_e4m3fn)


def _validate_csf_matmul(A, B, scales):
    if B.dtype != torch.int8 or B.ndim != 3:
        raise ValueError("CSF requires packed int8 NVFP4 weights [experts, rows, K/2]")
    expected = (B.shape[0], B.shape[1], B.shape[2] // 8)
    if B.shape[2] % 16 or scales.shape != expected:
        raise ValueError(
            f"CSF scales {scales.shape} do not match NVFP4 weights {tuple(B.shape)}"
        )
    if any(
        t.device != A.device
        for t in (B, scales.fixed, scales.exceptions, scales.row_offsets)
    ):
        raise ValueError(
            "Activations, weights and all CSF buffers must be on the same device"
        )


def _matmul_csf(
    A,
    B,
    scales,
    routing,
    *,
    grouped,
    As=None,
    activation_format=None,
    gate=False,
    act_fn="silu",
    swiglu_alpha=None,
    swiglu_limit=None,
    quantize_output=False,
    simulate_unfused=False,
    output_dtype=None,
    gather_idx=None,
    scatter_idx=None,
    a_global_scale=None,
    b_global_scale=None,
    output_global_scale=None,
    bias=None,
):
    _validate_csf_matmul(A, B, scales)
    if quantize_output and not gate:
        raise ValueError("CSF output requantization requires gate=True")
    if torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad
        for tensor in (A, As, a_global_scale, b_global_scale, output_global_scale, bias)
    ):
        raise ValueError("CSF matmul currently supports inference only")
    common = dict(
        gate=gate,
        act_fn=act_fn,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        simulate_unfused=simulate_unfused,
        output_dtype=output_dtype,
        gather_idx=gather_idx,
        scatter_idx=scatter_idx,
        b_global_scale=b_global_scale,
        bias=bias,
        csf_exceptions=scales.exceptions,
        csf_row_offsets=scales.row_offsets,
        csf_search_steps=scales.search_steps,
    )
    if activation_format == "bf16":
        if (
            As is not None
            or quantize_output
            or a_global_scale is not None
            or output_global_scale is not None
        ):
            raise ValueError(
                "CSF weight-only matmul takes raw activations and produces an unquantized output"
            )
        if grouped:
            from .grouped import mx_weight_only_matmul_grouped

            out = mx_weight_only_matmul_grouped(A, B, scales.fixed, routing, **common)
        else:
            from .batched import mx_weight_only_matmul_batched

            out = mx_weight_only_matmul_batched(A, B, scales.fixed, routing, **common)
    else:
        if grouped:
            from .grouped import mx_dynamic_matmul_grouped

            op = mx_dynamic_matmul_grouped
        else:
            from .batched import mx_dynamic_matmul_batched

            op = mx_dynamic_matmul_batched
        out = op(
            A,
            B,
            As,
            scales.fixed,
            routing,
            activation_format=activation_format,
            quantize_output=quantize_output,
            a_global_scale=a_global_scale,
            output_global_scale=output_global_scale,
            **common,
        )
    return (out[0], out[1]) if quantize_output else out[0]


def matmul_csf_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    scales: NVFP4CSFScales,
    *,
    expert_ids: torch.Tensor,
    As: torch.Tensor | None = None,
    activation_format: str | None = None,
    gate: bool = False,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    quantize_output: bool = False,
    simulate_unfused: bool = False,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    a_global_scale: torch.Tensor | None = None,
    b_global_scale: torch.Tensor | None = None,
    output_global_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
):
    """Experimental NVFP4 expert decode with compressed scales read directly inside the GEMM.

    Matches ``matmul_batched`` routing and epilogue arguments. ``activation_format``
    is ``nvfp4`` (default) or ``bf16`` (weight-only). Inference only; no scale plane
    is materialized. CSF row order must match B, including its gate/up layout.
    """
    return _matmul_csf(
        A,
        B,
        scales,
        expert_ids,
        grouped=False,
        As=As,
        activation_format=activation_format,
        gate=gate,
        act_fn=act_fn,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        quantize_output=quantize_output,
        simulate_unfused=simulate_unfused,
        output_dtype=output_dtype,
        gather_idx=gather_idx,
        scatter_idx=scatter_idx,
        a_global_scale=a_global_scale,
        b_global_scale=b_global_scale,
        output_global_scale=output_global_scale,
        bias=bias,
    )


def matmul_csf_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    scales: NVFP4CSFScales,
    *,
    expert_start: torch.Tensor,
    As: torch.Tensor | None = None,
    activation_format: str | None = None,
    gate: bool = False,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    quantize_output: bool = False,
    simulate_unfused: bool = False,
    output_dtype: torch.dtype | None = None,
    gather_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    a_global_scale: torch.Tensor | None = None,
    b_global_scale: torch.Tensor | None = None,
    output_global_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
):
    """Experimental NVFP4 expert prefill with compressed scales decoded inside each GEMM tile.

    Matches ``matmul_grouped`` routing and epilogue arguments, with the same
    inference-only and row-order contract as ``matmul_csf_batched``.
    """
    return _matmul_csf(
        A,
        B,
        scales,
        expert_start,
        grouped=True,
        As=As,
        activation_format=activation_format,
        gate=gate,
        act_fn=act_fn,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        quantize_output=quantize_output,
        simulate_unfused=simulate_unfused,
        output_dtype=output_dtype,
        gather_idx=gather_idx,
        scatter_idx=scatter_idx,
        a_global_scale=a_global_scale,
        b_global_scale=b_global_scale,
        output_global_scale=output_global_scale,
        bias=bias,
    )
