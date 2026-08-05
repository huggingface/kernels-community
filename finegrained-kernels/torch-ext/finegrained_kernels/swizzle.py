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



def _swizzle_to_blocks(
    scale: torch.Tensor, gather_idx: torch.Tensor | None, gate: bool
) -> torch.Tensor:
    """One ``(rows, cols)`` scale matrix -> its ``(row_blocks, cols//4, 2, 256)``
    SWIZZLE_32_4_4 block stack (single kernel launch). Under ``gate`` the rows are the
    stacked gate|up slab ``(2N, cols)``: the blocks come out ``[g0..,u0..]`` and are
    block-interleaved to ``[g0,u0,g1,u1,...]`` so a tile reads its gate + up 128-blocks
    as one contiguous descriptor load — which needs ``N % 128 == 0`` (else the gate/up
    split lands mid-block)."""
    cols = scale.shape[1]
    rows = gather_idx.shape[0] if gather_idx is not None else scale.shape[0]
    assert not gate or rows % 256 == 0, (
        f"gate|up swizzle needs N % 128 == 0 (rows = 2N = {rows}); a non-128 N puts "
        f"the gate/up split mid-block — keep those scales affine"
    )
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
    blocks = out.view(scale.dtype).reshape(nrb, ncb, 2, 256)
    if gate:
        blocks = (
            blocks.reshape(2, nrb // 2, ncb, 2, 256).transpose(0, 1).reshape(nrb, ncb, 2, 256)
        )
    return blocks


def swizzle_mx_scales(
    scale: torch.Tensor, gather_idx: torch.Tensor | None = None, *, gate: bool = False
) -> torch.Tensor:
    """Reorder a block-scale tensor into the ``SWIZZLE_32_4_4`` layout the Blackwell tcgen05
    scaled-MMA consumes, one triton launch per matrix (``_swizzle_scales_kernel`` — no torch
    permute/transpose copies). The scale values are unchanged — this only rearranges them from
    plain row-major into the swizzled order the tensor core reads directly (the same layout
    cuBLAS/CUTLASS require for MXFP8/NVFP4 ``scaled_mm``). Plain row-major forces a gather that
    caps the scaled dot below the fp8/fp4 peak; the swizzle removes it. Apply to weight scales
    once at model load; the ops accept the result directly.

    ``scale`` is the per-block scale grid — UE8M0 (``float8_e8m0fnu``, group-32 MXFP8/MXFP4) or
    E4M3 (``float8_e4m3fn``, group-16 NVFP4); the reorder is dtype-agnostic (moves bytes):

    - 2D ``(rows, K // group)``: one matrix (a dense weight's scale).
    - 3D ``(E, rows, K // group)``: an expert stack, each expert swizzled independently so its
      blocks stay ``ceil(rows/128)``-aligned (the reader indexes expert ``e`` at
      ``e * ceil(rows/128)``; byte-identical to a flat swizzle when ``rows % 128 == 0``).

    ``gate``: ``rows`` is the stacked gate|up slab (``2N``): per matrix the gate blocks
    ``[0,N)`` and up blocks ``[N,2N)`` are interleaved to ``[g0,u0,g1,u1,...]`` so a tile reads
    its gate + up 128-blocks contiguously. Requires ``N % 128 == 0``.

    ``gather_idx`` (2D only): a 1-D ``(padded_rows,)`` index mapping each output (sorted) row to
    its source row in ``scale`` (``-1`` = padding → zero row), folded into the kernel's load — the
    routed/expert-sorted, per-tile-padded layout a grouped GEMM reads affine per BM=128 tile.
    ``padded_rows`` must be a multiple of 128.

    ``rows``/``cols`` are zero-padded to (128, 4) multiples; returns the 5D
    ``(1, row_blocks, ceil(cols/4), 2, 256)`` view the ops read (``row_blocks`` sums the
    expert stacks), or 6D ``(1, pairs, 2, ceil(cols/4), 2, 256)`` under ``gate`` — the shape
    carries the interleave so every consumer (fused pair reads, plain remapped reads) takes
    the SAME artifact. Bit-identical to CUTLASS's packer (verified)."""
    assert gather_idx is None or gather_idx.shape[0] % 128 == 0, (
        f"gather_idx rows must be 128-padded, got {None if gather_idx is None else gather_idx.shape[0]}"
    )
    if scale.ndim == 3:
        assert gather_idx is None, "gather_idx applies to a single 2D scale, not an expert stack"
        blocks = torch.cat([_swizzle_to_blocks(e, None, gate) for e in scale]).unsqueeze(0)
    else:
        assert scale.ndim == 2, (
            f"expected a 2D (rows, K//group) or 3D (E, rows, K//group) scale, got {tuple(scale.shape)}"
        )
        assert not (gate and gather_idx is not None), "gate|up interleave and gather_idx are exclusive"
        blocks = _swizzle_to_blocks(scale, gather_idx, gate).unsqueeze(0)
    if gate:
        # the gate|up artifact is 6D — (1, pairs, 2, ncb, 2, 256), a pure view of the same
        # interleaved bytes — so its layout is READABLE from the shape: the fused wrappers
        # flatten it back for their pair-reading descriptors, and a plain (non-gate) consumer
        # sees 6D and remaps its block index in-kernel. One artifact, every path.
        return blocks.reshape(blocks.shape[0], -1, 2, *blocks.shape[2:])
    return blocks
