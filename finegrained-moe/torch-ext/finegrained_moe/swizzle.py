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


import contextvars
import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal


import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ._ops import add_op_namespace_prefix
from .bayesian_autotuner import bayesian_autotune

from .compat import *  # noqa: F401,F403
from .recipes import *  # noqa: F401,F403



@triton.jit
def swizzle_store_block(DST, s, blk, cb, NCB):
    """Pack one row-major ``(128, 4)`` scale block ``s`` into its SWIZZLE_32_4_4 ``(32, 16)`` block
    and store it at flat offset ``(blk * NCB + cb) * 512`` — the inverse of the un-swizzle in
    ``load_swizzled_scale``. Shared by every scale-swizzle kernel below."""
    sw = s.reshape(4, 32, 4).trans(1, 0, 2).reshape(32, 16)
    r = tl.arange(0, 32)
    c = tl.arange(0, 16)
    tl.store(DST + (blk * NCB + cb) * 512 + r[:, None] * 16 + c[None, :], sw)



@triton.jit
def _swizzle_scales_kernel(
    SRC,  # (rows, cols) row-major block scales (uint8 / e8m0 / e4m3 — 1 byte)
    DST,  # flat SWIZZLE_32_4_4 output, (n_row_blocks * n_col_blocks * 512,)
    GatherIdx,  # (padded_rows,) output-row -> source row (-1 = pad); read only when not None
    ROWS,
    COLS,
    NCB,  # number of 4-wide column blocks
    stride_src_m,
):
    """One 128x4 SWIZZLE_32_4_4 block per (row-block, col-block) program: gather+pad+swizzle in
    a single launch (replaces the torch view/permute/transpose chain — several kernels + two
    full-tensor copies). Per block: ``(128, 4) -> (4, 32, 4) -> trans(1,0,2) -> (32, 16)`` — the
    exact reorder the torch packer did, so bit-identical."""
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    ri = rb * 128 + tl.arange(0, 128)
    if GatherIdx is not None:
        src = tl.load(GatherIdx + ri, mask=ri < ROWS, other=-1)
        valid = src >= 0
        src = tl.where(valid, src, 0)
    else:
        src = ri
        valid = ri < ROWS
    cj = cb * 4 + tl.arange(0, 4)
    s = tl.load(
        SRC + src[:, None] * stride_src_m + cj[None, :],
        mask=valid[:, None] & (cj[None, :] < COLS),
        other=0,
    )
    swizzle_store_block(DST, s, rb, cb, NCB)



def swizzle_mx_scales(
    scale: torch.Tensor, gather_idx: torch.Tensor | None = None
) -> torch.Tensor:
    """Reorder a block-scale tensor ``(rows, K // group)`` into the ``SWIZZLE_32_4_4`` layout
    the Blackwell tcgen05 scaled-MMA consumes, in a SINGLE triton launch (``_swizzle_scales_kernel``
    — no torch permute/transpose copies). The scale values are unchanged — this only rearranges
    them from plain row-major into the swizzled order the tensor core reads directly (the same
    layout cuBLAS/CUTLASS require for MXFP8/NVFP4 ``scaled_mm``). Plain row-major forces a gather
    that caps the scaled dot below the fp8/fp4 peak; the swizzle removes it.

    ``gather_idx`` (optional): a 1-D ``(padded_rows,)`` index mapping each output (sorted) row to
    its source row in ``scale`` (``-1`` = padding → zero row), folded into the kernel's load — the
    routed/expert-sorted, per-tile-padded layout a grouped GEMM reads affine per BM=128 tile.
    ``padded_rows`` must be a multiple of 128.

    ``rows``/``cols`` are zero-padded to (128, 4) multiples; returns a 1-D tensor of size
    ``32*ceil(rows/128) * 16*ceil(cols/4)``. Bit-identical to CUTLASS's packer (verified).

    ``scale`` is the per-block scale grid: UE8M0 (``float8_e8m0fnu``, group-32 MXFP8) or E4M3
    (``float8_e4m3fn``, group-16 NVFP4); the reorder is dtype-agnostic (moves bytes)."""
    assert scale.ndim == 2, f"expected a 2D (rows, K//group) scale, got {tuple(scale.shape)}"
    assert gather_idx is None or gather_idx.shape[0] % 128 == 0, (
        f"gather_idx rows must be 128-padded, got {None if gather_idx is None else gather_idx.shape[0]}"
    )
    cols = scale.shape[1]
    rows = gather_idx.shape[0] if gather_idx is not None else scale.shape[0]
    nrb = triton.cdiv(rows, 128)
    ncb = triton.cdiv(cols, 4)
    # the reorder is byte-level; view as uint8 so the triton binder accepts e8m0/e4m3 scales
    src = scale.view(torch.uint8)
    out = torch.empty(nrb * ncb * 512, device=scale.device, dtype=torch.uint8)
    with device_context(scale.device):
        compile_time_only_triton_wrap(_swizzle_scales_kernel)[(nrb, ncb)](
            src,
            out,
            gather_idx,  # None = no gather (the is-not-None guard folds the load out)
            rows,
            cols,
            ncb,
            src.stride(0),
        )
    return out.view(scale.dtype)
