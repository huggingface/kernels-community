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
from .swizzle import *  # noqa: F401,F403
from .tile_layout import *  # noqa: F401,F403
from .quant import *  # noqa: F401,F403



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
      ``[1, REP, rep_k, 2, 256]``), un-swizzle, feed ``dot_scaled``. This is the tutorial's
      ``rep_m``/``rep_n`` load — BN=256 (REP=2) stays on the descriptor instead of falling to gather.
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
    blk = group_id * nrb + r // 128
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
):
    """One batched-decode weight-scale tile ``(n_width, SCALE_COLS)``, hiding the swizzled vs
    un-swizzled choice behind the ``SWIZZLED_SCALES`` flag — the kernel loop reads scales the same
    way either layout. ``bs_ptr`` is the un-advanced buffer base; the per-expert offset is applied
    here (the swizzled path indexes by 128-row block, the un-swizzled by the row-major stride, so
    it can't be pre-advanced uniformly). Under ``GATE`` the gate|up sub-tiles stack into ``2*BN``.

    - ``SWIZZLED_SCALES``: SWIZZLE_32_4_4 via ``load_swizzled_scale_tile`` (descriptor bulk at BN=128, or
      pointer gather below), scales swizzled over the full ``2N`` rows/expert under GATE.
    - else: affine per-group load off ``(expert, N-tile row, K-group)`` — no in-op swizzle, so an
      un-swizzled caller pays nothing."""
    n_width: tl.constexpr = 2 * BLOCK_SIZE_N if GATE else BLOCK_SIZE_N
    if SWIZZLED_SCALES and GATE:
        # scales swizzled over the full 2N rows/expert; gate tile at row-block pid_n, up tile
        # N/BN blocks later. Stack [gate BN; up BN] -> (2*BN, SCALE_COLS).
        gate_s = load_swizzled_scale_tile(
            bs_descriptor, bs_ptr, expert_id, pid_n, k_idx, 2 * N, K,
            BLOCK_SIZE_N, SCALE_COLS, SCALE_GROUP_K,
        )
        up_s = load_swizzled_scale_tile(
            bs_descriptor, bs_ptr, expert_id, N // BLOCK_SIZE_N + pid_n, k_idx, 2 * N, K,
            BLOCK_SIZE_N, SCALE_COLS, SCALE_GROUP_K,
        )
        b_s = tl.reshape(tl.trans(tl.join(gate_s, up_s), 2, 0, 1), (n_width, SCALE_COLS))
    elif SWIZZLED_SCALES:
        b_s = load_swizzled_scale_tile(
            bs_descriptor, bs_ptr, expert_id, pid_n, k_idx, N, K,
            BLOCK_SIZE_N, SCALE_COLS, SCALE_GROUP_K,
        )
    else:
        # affine per-group load off (expert, N-tile row, K-group) from the un-advanced base
        base = bs_ptr + expert_id * stride_bs_e
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_sf = k_idx * SCALE_COLS + tl.arange(0, SCALE_COLS)
        if GATE:
            rows2 = tl.arange(0, 2)[:, None] * N + offs_bn[None, :]
            ptrs = base + rows2[:, :, None] * stride_bs_n + offs_sf[None, None, :] * stride_bs_k
        else:
            ptrs = base + offs_bn[:, None] * stride_bs_n + offs_sf[None, :] * stride_bs_k
        b_s = tl.reshape(tl.load(ptrs), (n_width, SCALE_COLS))
    return b_s



@triton.jit
def load_mx_act_tile(
    a_ptrs,
    as_ptrs,
    as_global,  # (1,) fp32 NVFP4 act global (None off nvfp4); normalizes the raw tile pre-block-quant
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
    RECIPE: tl.constexpr = "mxfp8",
):
    """Load one MX activation K-tile as ``(a_vals, a_scale)`` — the arm is picked
    off the pointer dtype at compile time: fp8 pointers load pre-quantized E4M3 values +
    UE8M0 scales (``maybe_act_quant``'s offline arm), uint8 pointers load caller-provided
    packed-E2M1 values (W4A4 — the ``a_ptrs`` tile spans ``BLOCK_SIZE_K // 2`` bytes) +
    the same UE8M0 scales, raw bf16/fp16 pointers load and quantize inline onto
    ``RECIPE``'s grid (``mx_act_quant_inline`` — packed E2M1 under the fp4 recipes;
    ``as_ptrs`` then points at a dead placeholder and is never read). Under NVFP4
    two-level, ``as_global`` (the calibrated activation global) normalizes the raw tile
    before the block quant — bit-identical to the offline ``nvfp4_act_quant(x,
    global_scale=g_a)`` pass. ``row_mask`` may be ``None`` (unmasked tiles, e.g. the
    %-wrapped 2D matmul). Callers advance both pointers unconditionally."""
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
        if row_mask is None:
            a_raw = tl.load(a_ptrs).to(tl.float32)
        else:
            a_raw = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
        if as_global is not None:  # NVFP4 two-level: normalize by the calibrated act global
            a_raw = a_raw / tl.load(as_global).to(tl.float32)
        a, a_scale = mx_act_quant_inline(
            a_raw, BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K, RECIPE
        )
    return a, a_scale



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
def gate_stacked_block_scale_ptrs(
    Bs, pid_n, N,
    block_n: tl.constexpr, stride_bs_n,
    BLOCK_SIZE_N: tl.constexpr, n_width: tl.constexpr,
):
    """Per-weight-row block-scale pointers for the stacked gate|up weight (``2*BN`` rows): gate
    rows ``[0,N)`` index their own ``block_n`` scale block, up rows ``[N,2N)`` the same block
    offset by ``N // block_n`` scale-blocks (the up projection sits ``N`` rows after gate). The
    ``block_dynamic_dot`` / ``accumulate("static")`` broadcast then folds one scale per weight row,
    exactly as the dense (non-gate) affine gather does. Returns ``(ptrs, mask)`` — the affine gather
    the swizzle ``%``-wrap would otherwise turn non-affine, bounds-masked to the valid rows."""
    proj_row = pid_n * BLOCK_SIZE_N + tl.arange(0, n_width) % BLOCK_SIZE_N
    up = tl.where(tl.arange(0, n_width) < BLOCK_SIZE_N, 0, N // block_n)
    return Bs + (proj_row // block_n + up) * stride_bs_n, proj_row < N



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
        offs_bn_lin = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_sf = tl.arange(0, SCALE_COLS)
        as_ptrs = As + offs_am_lin[:, None] * stride_as_m + offs_sf[None, :]
        bs_ptrs = Bs + (offs_bn_lin[:, None] * stride_bs_n + offs_sf[None, :] * stride_bs_k)
        as_mask = offs_am_lin < M
        bs_mask = offs_bn_lin < N
    return as_ptrs, bs_ptrs, as_mask, bs_mask
