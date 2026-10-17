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

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from ..compat import (
    add_op_namespace_prefix, compile_time_only_triton_op, compile_time_only_triton_wrap,
    device_context, sm_count,
)
from ..quant import fp8_act_quant_inline, mx_act_quant_inline


@triton.jit
def load_block_fp8_act_tile(
    a_ptrs,
    as_ptrs,
    a_descriptor=0,
    m_off=0,
    k_off=0,
    A_MEMORY_MODE: tl.constexpr = "pointer",
    as_mask=None,
    TRANSPOSED: tl.constexpr = False,
):
    """Block-FP8 counterpart of ``load_mx_act_tile``: load one activation K-tile as
    ``(a_fp8, a_scale_f32)`` — the arm folds off ``A_MEMORY_MODE`` and the pointer dtype at
    compile time. Descriptor mode loads the pre-quantized ``(BM, BK)`` host-TMA box at
    ``(m_off, k_off)`` (``a_ptrs`` unread); pointer fp8 loads the pre-quantized-offline tile;
    raw bf16/fp16 pointers quantize inline (``as_ptrs`` then a constexpr-dead placeholder).
    Either way the per-K-block scales come from ``as_ptrs``. ``TRANSPOSED`` marks a ``(K, M)``
    tile (the swapped pointer arm) so the inline amax reduces the token axis either way.
    Unmasked unless ``as_mask`` given: every caller's rows are %-wrapped, expert-advanced, or
    token-replicated."""
    if A_MEMORY_MODE != "pointer":  # pre-quantized, host-TMA box [BM, BK]
        a = a_descriptor.load([m_off, k_off])
        # as_mask None (batched/grouped: %-wrapped/expert-advanced rows) -> unmasked load;
        # a bounds mask (the 2D affine-scale path) -> masked. `other` is illegal without mask.
        if as_mask is None:
            a_s = tl.load(as_ptrs)
        else:
            a_s = tl.load(as_ptrs, mask=as_mask, other=0.0)
    elif a_ptrs.dtype.element_ty == tl.float8e4nv:  # pre-quantized offline
        a = tl.load(a_ptrs)
        if as_mask is None:
            a_s = tl.load(as_ptrs)
        else:
            a_s = tl.load(as_ptrs, mask=as_mask, other=0.0)
    else:  # raw bf16/fp16 — quantize inline
        a, a_s = fp8_act_quant_inline(tl.load(a_ptrs).to(tl.float32), TRANSPOSED)
    return a, a_s


@triton.jit
def load_swizzled_scale(
    desc, blk_idx, k_idx,
    REP: tl.constexpr, REP_K: tl.constexpr, BLOCK: tl.constexpr, SCALE_COLS: tl.constexpr,
):
    """Bulk-load one scale tile from a ``SWIZZLE_32_4_4`` descriptor and un-swizzle it to the
    ``(BLOCK, SCALE_COLS)`` layout ``tl.dot_scaled`` consumes. ``REP = BLOCK // 128``,
    ``REP_K = SCALE_COLS // 4``; ``blk_idx``/``k_idx`` are the tile's row-block / K-block ids.
    Descriptor is over the swizzled scale viewed ``(1, rows//128, cols//4, 2, 256)``."""
    s = desc.load([0, blk_idx * REP, k_idx * REP_K, 0, 0])
    return s.reshape(REP, REP_K, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK, SCALE_COLS)


@triton.jit
def load_swizzled_scale_tile(
    descriptor,
    ptr,
    group_id,
    pid,
    k_idx,
    rows,
    K,
    BLOCK: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    """One swizzled-scale tile ``(BLOCK, SCALE_COLS)`` for a row-tile of ANY operand — batched
    weight (``group_id = expert``), 2D weight / 2D activation (``group_id = 0``, dense). Scales are
    SWIZZLE_32_4_4 128-row blocks over ``rows`` (the operand's row count, ``rows//128`` blocks per
    group).

    - ``BLOCK`` a multiple of 128 with whole 4-group K bands (``BK % 128 == 0``): the fast path —
      bulk-load the ``REP = BLOCK//128`` row-blocks via the TMA ``descriptor`` (box
      ``[1, REP, rep_k, 2, 256]``), un-swizzle, feed ``dot_scaled``. Today's pruners cap every
      swizzled tile at 128 (REP=1); the REP>1 form is written for a future >128-tile descriptor
      read, not reachable from the current grids.
    - Otherwise (sub-128 tile — fp8 ``scalar`` decode / small-M offline; or ``BK<128``): the block
      layout can't be TMA-sliced, so pointer-GATHER exactly this tile's rows. The swizzle is a fixed
      permutation: logical (row ``r``, K-group ``col``) → byte
      ``(blk*cols4 + col//4)*512 + (r%32)*16 + ((r%128)//32)*4 + col%4``. Reads only the needed
      bytes — no 128-block over-read, no un-swizzle transpose — the row-major fast path's cost with
      the swizzled layout, so ``scalar`` competes on merit instead of eating a TMA penalty."""
    # Per-expert 128-row-block count is a CEIL: a non-128-multiple ``rows`` (e.g. N=2880) still
    # occupies ceil(rows/128) blocks in the buffer (the swizzle builder pads the partial last block;
    # its tail rows read zero-weight via the TMA OOB clamp, so they don't contribute). ``cdiv == floor``
    # when ``rows`` is 128-aligned, so this is inert for every aligned shape.
    nrb = (rows + 127) // 128
    if BLOCK % 128 == 0 and SCALE_COLS >= 4 and SCALE_COLS % 4 == 0:
        REP: tl.constexpr = BLOCK // 128
        # absolute 128-block base = group_id*nrb + pid*REP; load_swizzled_scale multiplies blk by REP.
        # Non-128 ``rows`` (odd ``nrb``) pins REP=1 (BN=128) in the pruner, so group_id*nrb//REP is exact.
        blk = (group_id * nrb // REP + pid).to(tl.int32)
        return load_swizzled_scale(descriptor, blk, k_idx, REP, SCALE_COLS // 4, BLOCK, SCALE_COLS)
    cols4 = (K // SCALE_GROUP_K + 3) // 4  # cdiv: the buffer pads cols to whole 4-group chunks
    r = pid * BLOCK + tl.arange(0, BLOCK)
    lin = r // 128
    blk = group_id * nrb + lin
    row = r % 128
    col = k_idx * SCALE_COLS + tl.arange(0, SCALE_COLS)
    off = (
        (blk[:, None] * cols4 + col[None, :] // 4) * 512
        + (row[:, None] % 32) * 16
        + (row[:, None] // 32) * 4
        + col[None, :] % 4
    )
    return tl.load(ptr + off)


@triton.jit
def load_weight_scale_tile(
    SWIZZLED_SCALES: tl.constexpr,
    bs_descriptor,
    bs_ptr,
    expert_id,
    pid_n,
    k_idx,
    N,
    K,
    stride_bs_e,
    stride_bs_n,
    stride_bs_k,
    BLOCK_SIZE_N: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    GATE: tl.constexpr,
    k_col_off=0,
):
    """One batched-decode weight-scale tile ``(n_width, SCALE_COLS)``, hiding the swizzled vs
    un-swizzled choice behind the ``SWIZZLED_SCALES`` flag — the kernel loop reads scales the same
    way either layout. ``bs_ptr`` is the un-advanced buffer base; the per-expert offset is applied
    here (the swizzled path indexes by 128-row block, the un-swizzled by the row-major stride, so
    it can't be pre-advanced uniformly). Under ``GATE`` the gate|up sub-tiles stack into ``2*BN``.

    - ``SWIZZLED_SCALES``: SWIZZLE_32_4_4 via ``load_swizzled_scale_tile`` (descriptor bulk at BN=128, or
      pointer gather below), scales swizzled over the full ``2N`` rows/expert under GATE.
    - else: affine per-group load off ``(expert, N-tile row, K-group)`` — no in-op swizzle, so an
      un-swizzled caller pays nothing. ``k_col_off`` shifts that read off the tile grid (the
      caller's K-tile was clamped back into bounds for a BK that doesn't divide K); it is 0
      for every aligned caller, and the swizzled layouts don't take it."""
    n_width: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
    if SWIZZLED_SCALES:
        # gate|up needs no arm: the scale grid is row-interleaved with the weight, so the tile's
        # gate and up rows are the contiguous n_width span at the same pid_n, i.e. the plain read
        # over the doubled extent.
        b_s = load_swizzled_scale_tile(
            bs_descriptor, bs_ptr, expert_id, pid_n, k_idx,
            2 * N if GATE else N, K,
            n_width, SCALE_COLS, SCALE_GROUP_K,
        )
    else:
        # affine per-group load off (expert, N-tile row, K-group) from the un-advanced base.
        # Clamp the row in-bounds instead of masking (a masked load's other=0.0 blocks WS
        # lowering): a partial last N-tile reads a clamped scale the epilogue masks off
        # (N_COLS) — without the clamp the GATE arm's up rows (N + offs_bn) run past 2N.
        base = bs_ptr + expert_id * stride_bs_e
        offs_bn = pid_n * n_width + tl.arange(0, n_width)
        offs_bn = tl.minimum(offs_bn, (2 * N if GATE else N) - 1)
        offs_sf = k_idx * SCALE_COLS + k_col_off + tl.arange(0, SCALE_COLS)
        ptrs = base + offs_bn[:, None] * stride_bs_n + offs_sf[None, :] * stride_bs_k
        b_s = tl.load(ptrs)
    return b_s


@triton.jit
def load_mx_act_tile(
    a_ptrs,
    as_ptrs,
    as_global,  # fp32 NVFP4 act global (None off nvfp4); normalizes the raw tile pre-block-quant
    as_global_row,  # element offset into ``as_global`` (the expert's entry; 0 for a shared scalar)
    row_mask,
    a_descriptor,
    m_start,
    ka_off,
    gather_rows,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    A_MEMORY_MODE: tl.constexpr = "pointer",
    A_GATHER: tl.constexpr = False,
    FORMAT: tl.constexpr = "mxfp8",
):
    """Load one MX activation K-tile as ``(a_vals, a_scale)`` — the arm is picked
    off the pointer dtype at compile time: fp8 pointers load pre-quantized E4M3 values +
    UE8M0 scales (``maybe_act_quant``'s offline arm), uint8 pointers load caller-provided
    packed-E2M1 values (W4A4 — the ``a_ptrs`` tile spans ``BLOCK_SIZE_K // 2`` bytes) +
    the same UE8M0 scales, raw bf16/fp16 pointers load and quantize inline onto
    ``FORMAT``'s grid (``mx_act_quant_inline`` — packed E2M1 under the fp4 formats;
    ``as_ptrs`` then points at a dead placeholder and is never read). Under NVFP4
    two-level, ``as_global[as_global_row]`` (the calibrated activation global — per tensor
    or this expert's) normalizes the raw tile before the block quant — bit-identical to the
    offline ``nvfp4_act_quant(x, global_scale=g_a)`` pass. ``row_mask`` may be ``None``
    (unmasked tiles, e.g. the %-wrapped 2D matmul). Callers advance both pointers
    unconditionally."""
    if a_ptrs.dtype.element_ty == tl.float8e4nv or a_ptrs.dtype.element_ty == tl.uint8:
        # pre-quantized (E4M3 offline, or packed E2M1 handed in by the caller); under the
        # descriptor arm the [BM, BK_bytes] box loads the tile's contiguous sorted rows
        # (no gather — tail rows past the tensor clamp to zero and are store-masked)
        if A_MEMORY_MODE != "pointer":
            if A_GATHER:
                # sm_100 tma gather4: bulk-load the tile's ARBITRARY source rows
                a = a_descriptor.gather(gather_rows, ka_off)
            else:
                a = a_descriptor.load([m_start, ka_off])
            if as_ptrs.dtype.element_ty == tl.float8e4nv:
                # NVFP4 scales: 0.0 encodes as byte 0 — padded rows scale to exact 0
                a_scale = tl.load(as_ptrs, mask=row_mask[:, None], other=0.0)
            else:
                # UE8M0 scales: byte 0 decodes to 2^-127 — padded rows can't make 0*inf
                a_scale = tl.load(as_ptrs, mask=row_mask[:, None], other=0)
        elif row_mask is None:
            a = tl.load(a_ptrs)
            a_scale = tl.load(as_ptrs)
        else:
            a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
            if as_ptrs.dtype.element_ty == tl.float8e4nv:
                # NVFP4 scales: 0.0 encodes as byte 0 — padded rows scale to exact 0
                a_scale = tl.load(as_ptrs, mask=row_mask[:, None], other=0.0)
            else:
                # UE8M0 scales: byte 0 decodes to 2^-127 — padded rows can't make 0*inf
                a_scale = tl.load(as_ptrs, mask=row_mask[:, None], other=0)
    else:  # raw bf16/fp16 — quantize inline
        # the descriptor arm mirrors the pre-quantized branch above. Without it every
        # A_MEMORY_MODE != "pointer" config fell through to `tl.load(a_ptrs)`, and under a
        # descriptor arm `operand_tile_ptrs` hands back the bare base pointer -- so that load
        # produced a SCALAR and mx_act_quant_inline's reshape died with "'dtype' object has no
        # attribute 'numel'". Raw-A is exactly the inline-quant (nvfp4/mxfp4 2D) path, so it
        # killed every descriptor config there: 321 dead compiles in one tune.
        if A_MEMORY_MODE != "pointer":
            if A_GATHER:
                a_raw = a_descriptor.gather(gather_rows, ka_off).to(tl.float32)
            else:
                a_raw = a_descriptor.load([m_start, ka_off]).to(tl.float32)
        elif row_mask is None:
            a_raw = tl.load(a_ptrs).to(tl.float32)
        else:
            a_raw = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        if as_global is not None:  # NVFP4 two-level: normalize by the calibrated act global
            a_raw = a_raw / tl.load(as_global + as_global_row).to(tl.float32)
        a, a_scale = mx_act_quant_inline(
            a_raw, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K, FORMAT
        )
    return a, a_scale


@triton.jit
def apply_global_scale(acc, GlobalScale, stride_gs_e, GlobalScaleA, stride_gsa_e, expert_id):
    """Fold the NVFP4 second-level globals onto the fp32 accumulator: the weight global ``g_b``
    and the activation global ``g_a``, each per tensor (stride 0, broadcast) or per expert.
    ``None`` folds the arm out."""
    if GlobalScale is not None:
        acc = acc * tl.load(GlobalScale + expert_id * stride_gs_e).to(tl.float32)
    if GlobalScaleA is not None:
        acc = acc * tl.load(GlobalScaleA + expert_id * stride_gsa_e).to(tl.float32)
    return acc


@triton.jit
def decode_group_scale(scale):
    """Decode a group scale to fp32 by its dtype: ``uint8`` = UE8M0 exponent bits
    (``value = 2^(exp - 127)``, built directly as the fp32 bit pattern), E4M3 = NVFP4's
    direct fp8 value, fp32 (block-dynamic with float scales) passes through. The dtype
    branch is a compile-time constant, so only the taken path is emitted (single return —
    Triton requires all ``return`` statements to share a type)."""
    if scale.dtype == tl.uint8:
        scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    elif scale.dtype == tl.float8e4nv:
        scale = scale.to(tl.float32)
    return scale


@triton.jit
def mx_2d_scale_ptrs(
    As,
    Bs,
    pid_m,
    pid_n,
    M,
    N,
    stride_as_m,
    stride_bs_n,
    stride_bs_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    SWIZZLED_SCALES: tl.constexpr,
    GATE: tl.constexpr = False,
):
    """Prologue 2D MX scale-pointer tiles + bounds masks as ``(as_ptrs, bs_ptrs, as_mask,
    bs_mask)``. Affine arm: per-(row, group) ``as``/``bs`` pointer tiles read off AFFINE
    row/col offsets (the %-wrapped operand offsets would make the scale load a non-affine
    gather) with row/col bounds masks. Swizzled arm: the scales are read via the SA/BS
    descriptors in the loop, so these tiles are dead — return the base scalars + null masks.
    Single return — the arms have divergent types (base scalars + null masks vs affine pointer
    tiles + bounds masks), so an early return can't unify them; the constexpr selects one."""
    if SWIZZLED_SCALES:
        as_ptrs, bs_ptrs, as_mask, bs_mask = As, Bs, None, None
    else:
        offs_am_lin = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        ROWS: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
        offs_bn_lin = pid_n * ROWS + tl.arange(0, ROWS)
        offs_sf = tl.arange(0, SCALE_COLS)
        as_ptrs = As + offs_am_lin[:, None] * stride_as_m + offs_sf[None, :]
        bs_ptrs = Bs + (offs_bn_lin[:, None] * stride_bs_n + offs_sf[None, :] * stride_bs_k)
        as_mask = offs_am_lin < M
        bs_mask = offs_bn_lin < (2 * N if GATE else N)
    return as_ptrs, bs_ptrs, as_mask, bs_mask


# CSF byte-window scales: stream preparation and staged decoding before ordinary GEMMs.


def prepare_nvfp4_csf_scales(
    fixed: torch.Tensor, exceptions: torch.Tensor | Sequence[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare one matrix or an expert stack without expanding or moving its scales.

    `fixed` is uint8 `[rows/16, 16*(1+cols/2)]`, or the same shape with an
    expert dimension. Exceptions are uint32 `[count]` for one matrix, or one
    tensor per expert; low 24 bits locate a scale and high 8 bits replace it.
    Returns (fixed, concatenated exceptions, expert offsets) on the input device.
    Decode on the accelerator at load time for dense residency, or into shared
    scratch immediately before GEMM.
    """
    if fixed.dtype != torch.uint8 or fixed.ndim not in (2, 3):
        raise ValueError("CSF fixed stream must be a 2D or 3D uint8 tensor")
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
    offsets, words = [0], []
    for stream in streams:
        if (
            stream.dtype != torch.uint32
            or stream.ndim != 1
            or stream.device != fixed.device
        ):
            raise ValueError(
                "CSF exception streams must be 1D uint32 tensors on the fixed stream's device"
            )
        values = stream.to(torch.int64)
        positions = values & 0xFFFFFF
        if positions.numel() and (
            int(positions[-1]) >= rows * cols
            or bool((positions[1:] <= positions[:-1]).any())
        ):
            raise ValueError(
                "CSF exception positions must be sorted, unique and in range"
            )
        offsets.append(offsets[-1] + stream.numel())
        words.append(stream)
    if offsets[-1] >= 1 << 32:
        raise ValueError("CSF exception directory exceeds uint32 offsets")
    return (
        fixed.contiguous(),
        torch.cat(words),
        torch.tensor(offsets, dtype=torch.uint32, device=fixed.device),
    )


@triton.jit
def _decode_fixed_kernel(
    Fixed,
    Out,
    E: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK: tl.constexpr,
    ALIGNED_OUTPUT: tl.constexpr,
):
    pairs: tl.constexpr = ROWS * (COLS // 2)
    tiles: tl.constexpr = tl.cdiv(pairs, BLOCK)
    offsets = tl.arange(0, BLOCK)
    # Contiguous pairs avoid padding each row to a power-of-two tile width.
    # Keep indices local to an expert; only the expert base needs 64-bit addressing.
    for tile in range(tl.program_id(0), E * tiles, tl.num_programs(0)):
        expert = tile // tiles
        pair = (tile % tiles) * BLOCK + offsets
        row = pair // (COLS // 2)
        slab = expert.to(tl.int64) * (ROWS // 16) + row // 16
        slab_bytes: tl.constexpr = 16 * (1 + COLS // 2)
        base = tl.load(Fixed + slab * slab_bytes + row % 16, pair < pairs, other=0).to(
            tl.uint16
        )
        packed = tl.load(
            Fixed + slab * slab_bytes + 16 + (row % 16) * (COLS // 2) + pair % (COLS // 2),
            pair < pairs,
            other=0,
        ).to(tl.uint16)
        low = base + (packed & 15)
        high = base + (packed >> 4)
        # A contiguous sliced workspace can start at an odd byte.
        if ALIGNED_OUTPUT:
            output = Out.to(tl.pointer_type(tl.uint16))
            tl.store(
                output + expert.to(tl.int64) * pairs + pair,
                low | (high << 8),
                pair < pairs,
            )
        else:
            values = tl.interleave(low, high).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
            output_offsets = (tile % tiles) * BLOCK * 2 + tl.arange(0, BLOCK * 2)
            tl.store(
                Out + expert.to(tl.int64) * pairs * 2 + output_offsets,
                values,
                output_offsets < pairs * 2,
            )


@triton.jit
def _apply_exceptions_kernel(
    Exceptions,
    ExpertOffsets,
    Out,
    PLANE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(0)
    start = tl.load(ExpertOffsets + expert).to(tl.int64)
    end = tl.load(ExpertOffsets + expert + 1).to(tl.int64)
    for base in range(start, end, BLOCK):
        index = base + tl.arange(0, BLOCK)
        word = tl.load(Exceptions + index, index < end, other=0)
        position = word & 0xFFFFFF
        value = (word >> 24).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        tl.store(Out + expert.to(tl.int64) * PLANE + position, value, index < end)


@compile_time_only_triton_op(
    add_op_namespace_prefix("decompress_nvfp4_csf_scales_into"),
    mutates_args={"out"},
    opaque=True,
)
def _decompress_into(
    fixed: torch.Tensor,
    exceptions: torch.Tensor,
    expert_offsets: torch.Tensor,
    out: torch.Tensor,
) -> None:
    experts, rows, cols = out.shape
    with device_context(fixed.device):
        programs = min(
            experts * triton.cdiv(rows * cols // 2, 4096),
            16 * sm_count(fixed.device.index),
        )
        compile_time_only_triton_wrap(_decode_fixed_kernel)[(programs,)](
            fixed,
            out,
            experts,
            rows,
            cols,
            4096,
            ALIGNED_OUTPUT=out.storage_offset() % 2 == 0,
        )
        if exceptions.numel():
            compile_time_only_triton_wrap(_apply_exceptions_kernel)[(experts,)](
                exceptions, expert_offsets, out, rows * cols, 256
            )


def decompress_nvfp4_csf_scales(
    fixed: torch.Tensor,
    exceptions: torch.Tensor,
    expert_offsets: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode prepared CSF streams to dense E4M3 scales for an ordinary NVFP4 GEMM.

    With no `out`, allocate dense scales on the accelerator at checkpoint loading.
    Pass reusable contiguous [experts, rows, cols] workspace to keep the streams
    compressed between layers. Consume the workspace before overwriting it; it
    must not be retained for backward or shared by concurrent consumers.
    A shared dense workspace saves memory across layers, not within one layer.
    """
    if fixed.dtype != torch.uint8 or fixed.ndim != 3 or not fixed.is_contiguous():
        raise ValueError("CSF fixed stream must be contiguous 3D uint8")
    if fixed.device.type == "cpu":
        raise ValueError("CSF decompression requires an accelerator")
    experts, slabs, slab_bytes = fixed.shape
    if experts == 0 or slabs == 0 or slab_bytes <= 16 or slab_bytes % 16:
        raise ValueError("Invalid CSF slab shape")
    if (
        exceptions.dtype != torch.uint32
        or exceptions.ndim != 1
        or not exceptions.is_contiguous()
        or expert_offsets.dtype != torch.uint32
        or expert_offsets.shape != (experts + 1,)
        or not expert_offsets.is_contiguous()
    ):
        raise ValueError(
            "CSF exceptions and expert offsets must be contiguous uint32 vectors of prepared shape"
        )
    rows, cols = slabs * 16, (slab_bytes // 16 - 1) * 2
    if rows * cols > 1 << 24:
        raise ValueError(
            "CSF exception positions support at most 2**24 scales per matrix"
        )
    shape = (experts, rows, cols)
    if out is None:
        out = torch.empty(shape, dtype=torch.float8_e4m3fn, device=fixed.device)
    if (
        out.shape != shape
        or out.dtype != torch.float8_e4m3fn
        or not out.is_contiguous()
    ):
        raise ValueError(f"CSF workspace must be contiguous E4M3 with shape {shape}")
    if any(t.device != out.device for t in (fixed, exceptions, expert_offsets)):
        raise ValueError("CSF buffers and workspace must share a device")
    _decompress_into(fixed, exceptions, expert_offsets, out)
    return out
