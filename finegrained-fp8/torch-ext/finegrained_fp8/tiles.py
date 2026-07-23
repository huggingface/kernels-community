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
from .scales import *  # noqa: F401,F403
from .mma import *  # noqa: F401,F403
from .scheduling import *  # noqa: F401,F403


# A-sub-tile byte budget for the 1-byte-activation grouped-swizzle depth cap (see
# swizzle_offsets): the co-scheduled rows' A tile per K-step (depth*BM*BK bytes) must stay
# L2-hot to reuse, and past this it thrashes and the win collapses. ~512KB is a wave
# reuse-window, NOT gross L2 (0.4% of the B200's 132MB) — a B200 measurement (2026-07-16), so
# RE-MEASURE per device. Packed fp4 bypasses this (full grouping wins; see swizzle_offsets).
SWIZZLE_GROUP_A_BYTES = tl.constexpr(524288)


@triton.jit
def swizzle_offsets(
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    WEIGHT_VALUES_PER_BYTE: tl.constexpr = 1,
):
    """2D-grid tile scheduling shared by the kernels below: grouped-swizzle the
    ``(pid_m, pid_n)`` program ids for L2 locality on B, then build the operand offset
    vectors. Returns ``(pid_m, pid_n, offs_am, offs_bn, offs_k)`` — the swizzled ids
    (reused by the output store) and the ``%``-wrapped row/col offsets plus the K range.

    The swizzle keeps the B (weight) column-tile L2-hot while the co-scheduled rows reuse it,
    so the depth cap is set by the WEIGHT footprint, capped at ``min(num_pid_m, .)``. With
    1-byte weights the reuse thrashes past ~512KB (a hard cliff): the growing rival is the
    rows' A sub-tile per K-step (``depth * BM * BK`` bytes), so ``SWIZZLE_GROUP_A_BYTES //
    (BM * BK)`` — MEASURED on B200 (2026-07-16), bd BK128->32 (cliffs at 64), MX BK256->16,
    BM64->64 (g*BM*BK ~512KB across BM and BK; BN-independent). Packed-fp4 weights
    (``WEIGHT_VALUES_PER_BYTE==2``) halve that hot set, so it never thrashes and full grouping
    wins outright (monotone, no cliff — measured for both W4A4 and W4A8, i.e. weight-driven not
    activation-driven). Uses an EXPLICIT grouped swizzle (linearize the 2D program ids, then
    group), NOT ``tl.swizzle2d`` (which degrades with group depth, -3pp+ MFU at depth 32). Same
    grid launch, same result set — only the program-id -> tile mapping changes."""
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid = tl.program_id(axis=1) * num_pid_m + tl.program_id(axis=0)
    if WEIGHT_VALUES_PER_BYTE == 2:
        max_group = num_pid_m
    else:
        max_group = SWIZZLE_GROUP_A_BYTES // (BLOCK_SIZE_M * BLOCK_SIZE_K)
    num_pid_in_group = max_group * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * max_group
    group_size_m = min(num_pid_m - first_pid_m, max_group)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    return pid_m, pid_n, offs_am, offs_bn, offs_k


@triton.jit
def operand_tile_descriptor(
    HostDescriptor,
    W,
    N,
    K,
    stride_n,
    stride_k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    B_MEMORY_MODE: tl.constexpr,
):
    """Resolve one operand's tile descriptor once per program (weight OR activation — the bd 2D
    kernel calls it for both): the host-built TMA descriptor as passed, a device-built in-kernel
    tensormap, or 0 under "pointer" (never read — the constexpr branch folds it out of
    ``load_weight_tile``). Single return — only the taken branch compiles."""
    if B_MEMORY_MODE == "host_descriptor":
        descriptor = HostDescriptor
    elif B_MEMORY_MODE == "device_descriptor":
        descriptor = tl.make_tensor_descriptor(
            W,
            shape=(N, K),
            strides=(stride_n, stride_k),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        )
    else:  # pointer
        descriptor = 0
    return descriptor


@triton.jit
def load_grouped_act_tile(
    a_ptrs,
    a_descriptor,
    m_start,
    ka_off,
    row_mask,
    gather_rows,
    A_MEMORY_MODE: tl.constexpr,
    A_GATHER: tl.constexpr = False,
):
    """A plain (scale-less) grouped activation K-tile: the masked pointer load, or —
    under the descriptor arm — sm_100 tma gather4 over the tile's ARBITRARY source rows
    (gathered launches; padded rows read row 0 and are store-masked) or the ``[BM, BK]``
    box at the tile's contiguous sorted-row start (no-gather launches; tail rows past
    the tensor clamp to zero). ``row_mask`` None loads the pointer tile maskless (the
    %-wrapped 2D fast path). Single return — only the taken arm compiles."""
    if A_MEMORY_MODE == "pointer":
        if row_mask is None:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
    elif A_GATHER:
        a = a_descriptor.gather(gather_rows, ka_off)
    else:
        a = a_descriptor.load([m_start, ka_off])
    return a


@triton.jit
def load_grouped_weight_tile(
    w_ptrs,
    w_descriptor,
    row0,
    n_off,
    kb_off,
    BLOCK_SIZE_N: tl.constexpr,
    KB: tl.constexpr,
    GATE: tl.constexpr,
    B_MEMORY_MODE: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
):
    """One K-major (optionally gate|up-stacked) MX weight K-tile for the grouped / batched loop:
    the explicit-pointer tile flattened to the ``[KB, (2|1)*BN]`` rhs (or, under ``SWAP_AB``, the
    ``[(2|1)*BN, KB]`` rows-major lhs — the batched-decode orientation), or the ``[(2|1), BN, KB]``
    descriptor box over the ``(2E|E, N, K_bytes)`` weight view, reshaped and transposed to the same
    form (the fused-era TMA arm: natural orientation + per-iteration trans; grouped/2D never swap).
    Single return — only the taken arm compiles; the caller advances ``w_ptrs`` and passes the box
    offsets either way."""
    if B_MEMORY_MODE == "pointer":
        w = flatten_weight_tile(tl.load(w_ptrs), 2 * BLOCK_SIZE_N, KB, GATE, SWAP_AB)
    else:
        w = tl.trans(
            tl.reshape(
                w_descriptor.load([row0, n_off, kb_off]),
                ((2 if GATE else 1) * BLOCK_SIZE_N, KB),
            )
        )
    return w


@triton.jit
def load_weight_tile(
    w_ptrs, w_descriptor, row_off, k_off, B_MEMORY_MODE: tl.constexpr, SWAP_AB: tl.constexpr = False
):
    """One weight K-tile. Descriptor modes load the ``(BN, BK)`` box at ``(row_off, k_off)``
    and, in the natural orientation, transpose it once to the ``(BK, BN)`` K-major rhs the
    pointer arm builds directly (``SWAP_AB`` keeps the box as-is: the weight rows then sit in
    the MMA M dim). Pointer mode loads the explicit tile in whatever orientation ``w_ptrs``
    was built with. Single return — only the taken branch compiles."""
    if B_MEMORY_MODE == "pointer":
        w = tl.load(w_ptrs)
    else:
        w = w_descriptor.load([row_off, k_off])
        if not SWAP_AB:
            w = tl.trans(w)
    return w


@triton.jit
def matmul_weight_ptrs(
    B,
    offs_n,
    offs_k,
    N,
    stride_b_n,
    stride_b_k,
    GATE: tl.constexpr,
    B_MEMORY_MODE: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
):
    """Prologue weight-tile pointers, folding the gate|up-stack branch: under ``GATE`` the stacked
    (2N, K) gate|up tile (``weight_tile_ptrs``, up block ``N`` rows away), else the plain single
    tile via ``operand_tile_ptrs`` (which also folds the descriptor-vs-pointer arm). The
    weight analogue of the activation's single ``operand_tile_ptrs`` call. SINGLE-EXIT (one
    trailing return): multiple early ``if CONSTEXPR: return`` would type-check the dead arm and
    fail under GATE (Triton 3.7.1)."""
    if GATE:
        ptrs = weight_tile_ptrs(
            B, offs_n, offs_k, N * stride_b_n, stride_b_n, stride_b_k, GATE, SWAP_AB
        )
    else:
        ptrs = operand_tile_ptrs(B, offs_n, offs_k, stride_b_n, stride_b_k, B_MEMORY_MODE, SWAP_AB)
    return ptrs


@triton.jit
def load_act_mx(
    a_ptrs, as_ptrs, as_global, value_mask, scale_mask, a_descriptor, m_off, k_off,
    as_descriptor, as_ptr, gather_rows, stride_as_m, pid_m, k, M, K,
    A_MEMORY_MODE: tl.constexpr, A_GATHER: tl.constexpr, GROUPED: tl.constexpr,
    SWIZZLED_SCALES: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr, INPUT_RECIPE: tl.constexpr,
):
    """MX activation tile + scale. Offline-quantized A under a swizzled weight -> pre-swizzled
    SWIZZLE_32_4_4 scale (tcgen05 fast path); grouped -> gathered per-(row, K-group) affine scale off
    the source-order ``As``; else (2D / decode raw A) -> ``load_mx_act_tile`` inline-quant / in-register.
    Raw bf16 A stays in-register affine even under a swizzled weight (the dot_scaled mix is fine)."""
    SCALE_COLS: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
    A_OFFLINE: tl.constexpr = (
        a_ptrs.dtype.element_ty == tl.float8e4nv or a_ptrs.dtype.element_ty == tl.uint8
    )
    if SWIZZLED_SCALES and A_OFFLINE:  # pre-swizzled SWIZZLE_32_4_4 scale (tcgen05 fast path)
        a = load_grouped_act_tile(
            a_ptrs, a_descriptor, m_off, k_off, value_mask, gather_rows, A_MEMORY_MODE, A_GATHER
        )
        a_s = load_swizzled_scale_tile(
            as_descriptor, as_ptr, 0, pid_m, k, M, K, BLOCK_SIZE_M, SCALE_COLS, SCALE_GROUP_K
        )
    elif GROUPED:  # gathered per-(row, K-group) affine scale off the row-major source-order As
        a = load_grouped_act_tile(
            a_ptrs, a_descriptor, m_off, k_off, value_mask, gather_rows, A_MEMORY_MODE, A_GATHER
        )
        offs_sf = k * SCALE_COLS + tl.arange(0, SCALE_COLS)
        a_s = tl.load(
            as_ptrs + gather_rows[:, None] * stride_as_m + offs_sf[None, :],
            mask=scale_mask[:, None],
            other=0.0,
        )
    else:  # 2D / decode: inline-quant affine or in-register scale
        a, a_s = load_mx_act_tile(
            a_ptrs, as_ptrs, as_global, scale_mask, a_descriptor, m_off, k_off, 0,
            BLOCK_SIZE_M, BLOCK_SIZE_K, SCALE_GROUP_K, A_MEMORY_MODE,
            RECIPE=INPUT_RECIPE,
        )
    return a, a_s


@triton.jit
def load_act_block_dynamic(
    a_ptrs, as_ptrs, value_mask, scale_mask, a_descriptor, m_off, k_off, gather_rows, k,
    A_MEMORY_MODE: tl.constexpr, A_GATHER: tl.constexpr, GROUPED: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    """block_dynamic activation tile + per-row per-K-block scale. Grouped: gathered value + the scale
    read contiguously at ``As + k``; else ``load_block_fp8_act_tile`` (swap-aware, inline-quant 2D)."""
    if GROUPED:  # gathered value + per-row per-K-block scale read contiguously at As + k
        a = load_grouped_act_tile(
            a_ptrs, a_descriptor, m_off, k_off, value_mask, gather_rows, A_MEMORY_MODE, A_GATHER
        )
        a_s = tl.load(as_ptrs + k, mask=scale_mask, other=0.0)
    else:
        a, a_s = load_block_fp8_act_tile(
            a_ptrs, as_ptrs, a_descriptor, m_off, k_off, A_MEMORY_MODE, scale_mask, SWAP_AB
        )
    return a, a_s


@triton.jit
def load_act_static(
    a_ptrs, a_descriptor, m_off, k_off, value_mask, gather_rows, a_s_static,
    A_MEMORY_MODE: tl.constexpr, A_GATHER: tl.constexpr,
):
    """static activation tile. Pre-quantized fp8 A loads as the rows-major MMA lhs; raw bf16/fp16
    (inline arm, small M, pointer-only) is quantized against the scalar ``a_s_static``. ``a_s`` = the
    values (the static scale is a scalar folded post-loop)."""
    if a_ptrs.dtype.element_ty == tl.float8e4nv:  # pre-quantized fp8 A (MMA lhs, rows-major)
        a = load_grouped_act_tile(
            a_ptrs, a_descriptor, m_off, k_off, value_mask, gather_rows, A_MEMORY_MODE, A_GATHER
        )
    else:  # raw bf16/fp16 (inline arm, M<threshold, pointer-only) — quantize vs the static scale
        a = (tl.load(a_ptrs).to(tl.float32) / a_s_static).to(tl.float8e4nv)
    a_s = a
    return a, a_s


@triton.jit
def load_act_plain(
    a_ptrs, a_descriptor, m_off, k_off, value_mask, gather_rows,
    A_MEMORY_MODE: tl.constexpr, A_GATHER: tl.constexpr,
):
    """tensor / full_precision activation: plain value tile, no scale (returns ``(a, a)`` — the
    second slot is dead, kept for the uniform (value, scale) shape)."""
    a = load_grouped_act_tile(
        a_ptrs, a_descriptor, m_off, k_off, value_mask, gather_rows, A_MEMORY_MODE, A_GATHER
    )
    return a, a


@triton.jit
def _weight_scale_mx(
    bs_ptrs, bs_mask, bs_descriptor, bs_ptr, blk_idx, expert_id, pid_n, k, N, K,
    stride_bs_e, stride_bs_n, stride_bs_k,
    GROUPED: tl.constexpr, GATE: tl.constexpr, PER_EXPERT: tl.constexpr,
    SWIZZLED_SCALES: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    SCALE_COLS: tl.constexpr, SCALE_GROUP_K: tl.constexpr,
):
    """MX weight-scale K-tile ``(rows, SCALE_COLS)``. Grouped: pre-swizzled SWIZZLE_32_4_4 via the
    descriptor (``load_swizzled_scale`` — gate|up is one 2*BN tile off the block-interleaved buffer,
    GATE folding into REP/width), else an inlined affine per-group read off the un-swizzled 3D ``Bs``
    (inlined, not ``load_weight_scale_tile``: the nested leaf breaks warp-specialization partitioning
    in the grouped loop). 2D-GATE / batched decode: the per-(expert, N, K) 3D leaf. 2D dense: the
    descriptor bulk-load (BN=128) else the bounds-masked affine pointer."""
    if GROUPED:
        if SWIZZLED_SCALES:
            NREP: tl.constexpr = (2 if GATE else 1) * (BLOCK_SIZE_N // 128)
            NW: tl.constexpr = (2 if GATE else 1) * BLOCK_SIZE_N
            b_s = load_swizzled_scale(
                bs_descriptor, blk_idx, k, NREP, SCALE_COLS // 4, NW, SCALE_COLS
            )
        else:  # affine per-group read off the un-swizzled 3D Bs (num_experts, n_rows, K//g)
            base = bs_ptrs + expert_id * stride_bs_e
            offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_sf = k * SCALE_COLS + tl.arange(0, SCALE_COLS)
            if GATE:
                rows2 = tl.arange(0, 2)[:, None] * N + offs_bn[None, :]
                b_s = tl.reshape(
                    tl.load(base + rows2[:, :, None] * stride_bs_n + offs_sf[None, None, :] * stride_bs_k),
                    (2 * BLOCK_SIZE_N, SCALE_COLS),
                )
            else:
                b_s = tl.load(base + offs_bn[:, None] * stride_bs_n + offs_sf[None, :] * stride_bs_k)
    elif GATE or PER_EXPERT:  # 2D-GATE (expert 0) / batched decode: per-(expert, N, K) scale leaf
        b_s = load_weight_scale_tile(
            SWIZZLED_SCALES, bs_descriptor, bs_ptr, expert_id, pid_n, k, N, K,
            stride_bs_e, stride_bs_n, stride_bs_k, BLOCK_SIZE_N, SCALE_COLS, SCALE_GROUP_K, GATE,
        )
    elif SWIZZLED_SCALES:  # pre-swizzled SWIZZLE_32_4_4 scale — descriptor at BN=128, gather below
        b_s = load_swizzled_scale_tile(
            bs_descriptor, bs_ptr, 0, pid_n, k, N, K, BLOCK_SIZE_N, SCALE_COLS, SCALE_GROUP_K
        )
    else:
        b_s = tl.load(bs_ptrs, mask=bs_mask[:, None], other=0.0)  # 0.0 casts to fp8/uint8
    return b_s


@triton.jit
def _weight_scale_block_dynamic(
    bs_ptrs, up_s_ptr, bs_mask, k, stride_bs_k,
    GROUPED: tl.constexpr, GATE: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """block_dynamic per-K-block weight scale. Grouped reads it at ``bs_ptrs + k*stride`` (GATE folds
    gate on the first ``BN`` columns, up on the rest via ``up_s_ptr``); batched reads the pre-offset
    maskless pointer; 2D reads the bounds-masked advanced pointer."""
    if GROUPED:
        if GATE:  # gate scale on the first BN columns, up on the rest
            b_s = tl.where(
                tl.arange(0, 2 * BLOCK_SIZE_N) < BLOCK_SIZE_N,
                tl.load(bs_ptrs + k * stride_bs_k),
                tl.load(up_s_ptr + k * stride_bs_k),
            )
        else:
            b_s = tl.load(bs_ptrs + k * stride_bs_k)
    elif bs_mask is None:  # batched: bs_ptrs pre-offset (gate/up folded), maskless decode tile
        b_s = tl.load(bs_ptrs)
    else:
        b_s = tl.load(bs_ptrs, mask=bs_mask, other=0.0)
    return b_s


@triton.jit
def _weight_scale_static(
    bs_ptrs, up_s_ptr, bs_mask, k, stride_bs_k,
    GROUPED: tl.constexpr, GATE: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """static weight scale — as block_dynamic, but the grouped non-GATE scale broadcasts to a per-N
    vector so ``accumulate("static")`` applies it in N (the static act scale is a scalar folded
    post-loop)."""
    if GROUPED:
        if GATE:  # gate on the first BN, up on the rest
            b_s = tl.where(
                tl.arange(0, 2 * BLOCK_SIZE_N) < BLOCK_SIZE_N,
                tl.load(bs_ptrs + k * stride_bs_k),
                tl.load(up_s_ptr + k * stride_bs_k),
            )
        else:
            b_s = tl.load(bs_ptrs + k * stride_bs_k) + tl.zeros(
                (BLOCK_SIZE_N,), bs_ptrs.dtype.element_ty
            )
    elif bs_mask is None:  # batched: bs_ptrs pre-offset (gate/up folded), maskless decode tile
        b_s = tl.load(bs_ptrs)
    else:
        b_s = tl.load(bs_ptrs, mask=bs_mask, other=0.0)  # affine col index -> mask OOB last tile
    return b_s


@triton.jit
def _weight_value(
    b_ptrs, b_descriptor, row0, n_off, k_off,
    GATE: tl.constexpr, GROUPED: tl.constexpr, B_MEMORY_MODE: tl.constexpr,
    SWAP_AB: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, KB: tl.constexpr,
):
    """Recipe-agnostic weight value tile: stacked / per-expert-3D box (GATE/GROUPED) or the plain
    swap-aware tile. Shared by every ``load_weight_<recipe>`` — the value load never depends on the
    scale recipe. Single-exit if/else (an early return would type-check the untaken arm — the plain
    ``load_weight_tile``'s 2D descriptor load trips the 3D grouped weight descriptor)."""
    if GATE or GROUPED:
        w = load_grouped_weight_tile(
            b_ptrs, b_descriptor, row0, n_off, k_off, BLOCK_SIZE_N, KB, GATE, B_MEMORY_MODE, SWAP_AB
        )
    else:
        w = load_weight_tile(b_ptrs, b_descriptor, n_off, k_off, B_MEMORY_MODE, SWAP_AB)
    return w


@triton.jit
def load_weight_mx(
    b_ptrs, b_descriptor, bs_ptrs, bs_mask, bs_descriptor, bs_ptr, row0, n_off, k_off,
    blk_idx, expert_id, pid_n, k, N, K, stride_bs_e, stride_bs_n, stride_bs_k,
    GATE: tl.constexpr, GROUPED: tl.constexpr, PER_EXPERT: tl.constexpr,
    B_MEMORY_MODE: tl.constexpr, SWAP_AB: tl.constexpr, SWIZZLED_SCALES: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, SCALE_GROUP_K: tl.constexpr,
    WEIGHT_VALUES_PER_BYTE: tl.constexpr,
):
    """The MX weight path: value tile + pre-swizzled/affine group scale."""
    KB: tl.constexpr = BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE
    SCALE_COLS: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
    w = _weight_value(b_ptrs, b_descriptor, row0, n_off, k_off, GATE, GROUPED, B_MEMORY_MODE, SWAP_AB, BLOCK_SIZE_N, KB)
    w_s = _weight_scale_mx(
        bs_ptrs, bs_mask, bs_descriptor, bs_ptr, blk_idx, expert_id, pid_n, k, N, K,
        stride_bs_e, stride_bs_n, stride_bs_k,
        GROUPED, GATE, PER_EXPERT, SWIZZLED_SCALES, BLOCK_SIZE_N, SCALE_COLS, SCALE_GROUP_K,
    )
    return w, w_s


@triton.jit
def load_weight_block_dynamic(
    b_ptrs, b_descriptor, bs_ptrs, bs_mask, up_s_ptr, row0, n_off, k_off, k, stride_bs_k,
    GATE: tl.constexpr, GROUPED: tl.constexpr, B_MEMORY_MODE: tl.constexpr, SWAP_AB: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, WEIGHT_VALUES_PER_BYTE: tl.constexpr = 1,
):
    """The block_dynamic weight path: value tile + per-K-block scale (gate on the first BN, up on the
    rest via ``up_s_ptr``)."""
    KB: tl.constexpr = BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE
    w = _weight_value(b_ptrs, b_descriptor, row0, n_off, k_off, GATE, GROUPED, B_MEMORY_MODE, SWAP_AB, BLOCK_SIZE_N, KB)
    w_s = _weight_scale_block_dynamic(bs_ptrs, up_s_ptr, bs_mask, k, stride_bs_k, GROUPED, GATE, BLOCK_SIZE_N)
    return w, w_s


@triton.jit
def load_weight_static(
    b_ptrs, b_descriptor, bs_ptrs, bs_mask, up_s_ptr, row0, n_off, k_off, k, stride_bs_k,
    GATE: tl.constexpr, GROUPED: tl.constexpr, B_MEMORY_MODE: tl.constexpr, SWAP_AB: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, WEIGHT_VALUES_PER_BYTE: tl.constexpr = 1,
):
    """The static weight path: value tile + per-K-block scale broadcast to a per-N vector (the static
    act scale is a scalar folded post-loop)."""
    KB: tl.constexpr = BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE
    w = _weight_value(b_ptrs, b_descriptor, row0, n_off, k_off, GATE, GROUPED, B_MEMORY_MODE, SWAP_AB, BLOCK_SIZE_N, KB)
    w_s = _weight_scale_static(bs_ptrs, up_s_ptr, bs_mask, k, stride_bs_k, GROUPED, GATE, BLOCK_SIZE_N)
    return w, w_s


@triton.jit
def load_weight_plain(
    b_ptrs, b_descriptor, row0, n_off, k_off,
    GATE: tl.constexpr, GROUPED: tl.constexpr, B_MEMORY_MODE: tl.constexpr, SWAP_AB: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, WEIGHT_VALUES_PER_BYTE: tl.constexpr = 1,
):
    """The tensor / full_precision weight path: plain value tile, no block scale (the per-tensor
    scale, if any, is applied post-loop). Returns ``(w, w)`` so callers keep the uniform (value,
    scale) shape; the second slot is dead."""
    KB: tl.constexpr = BLOCK_SIZE_K // WEIGHT_VALUES_PER_BYTE
    w = _weight_value(b_ptrs, b_descriptor, row0, n_off, k_off, GATE, GROUPED, B_MEMORY_MODE, SWAP_AB, BLOCK_SIZE_N, KB)
    return w, w


@triton.jit
def advance_ptrs(
    a_ptrs,
    as_ptrs,
    w_ptrs,
    ws_ptrs,
    w_up_ptrs,
    ws_up_ptrs,
    a_step,
    as_step,
    w_step,
    ws_step,
    A_MEMORY_MODE: tl.constexpr,
    W_MEMORY_MODE: tl.constexpr,
    ADVANCE_AS: tl.constexpr,
    ADVANCE_WS: tl.constexpr,
    GATE_STREAMS: tl.constexpr = False,
):
    """Advance the shared GEMM operand pointers one K-step, folding the memory-mode / scale-layout /
    gate|up-stream conditionals out of every loop. The operand set is uniform across the kernels:
    activation (``a_ptrs`` + affine scale ``as_ptrs``) and weight (``w_ptrs`` + affine scale
    ``ws_ptrs``), the weight either a single stream or — under ``GATE_STREAMS`` — the gate|up pair
    (``w_up_ptrs`` + ``ws_up_ptrs``, bumped by the same steps). Value pointers advance only on the
    pointer arm (a descriptor arm re-derives the box K offset from ``k``); scale pointers advance
    only when read affine (``ADVANCE_AS`` / ``ADVANCE_WS`` — swizzled / in-leaf / per-tensor scales
    don't). Pass a dead pointer + step 0 (flag off) for any stream a kernel doesn't carry. Returns
    the six pointers in argument order."""
    if A_MEMORY_MODE == "pointer":
        a_ptrs += a_step
    if W_MEMORY_MODE == "pointer":
        w_ptrs += w_step
        if GATE_STREAMS:
            w_up_ptrs += w_step
    if ADVANCE_AS:
        as_ptrs += as_step
    if ADVANCE_WS:
        ws_ptrs += ws_step
        if GATE_STREAMS:
            ws_up_ptrs += ws_step
    return a_ptrs, as_ptrs, w_ptrs, ws_ptrs, w_up_ptrs, ws_up_ptrs


@triton.jit
def weight_tile_ptrs(
    base,
    offs_n,
    offs_k,
    block_stride,
    stride_n,
    stride_k,
    GATE: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    """Weight-tile pointers oriented by ``SWAP_AB``, gated by ``GATE`` — the gate_up
    counterpart of ``oriented_tile_ptrs``. With ``GATE`` a leading axis indexes the
    {gate, up} row block (up offset by ``block_stride``), placed so
    ``flatten_weight_tile``'s plain reshape yields the 2D stacked tile: swap
    ``[2, N, K]`` (output rows in the MMA M dim), no-swap ``[K, 2, N]`` (K-major, gate|up
    along the MMA N dim — the grouped kernel's combined form). Without ``GATE`` it is the
    plain single 2D tile (``block_stride`` unused), delegated to ``oriented_tile_ptrs``. The
    per-step K-advance is the same scalar stride step in every orientation."""
    if GATE:
        blk = tl.arange(0, 2) * block_stride
        if SWAP_AB:
            ptrs = base + (
                blk[:, None, None]
                + offs_n[None, :, None] * stride_n
                + offs_k[None, None, :] * stride_k
            )
        else:
            ptrs = base + (
                offs_k[:, None, None] * stride_k
                + blk[None, :, None]
                + offs_n[None, None, :] * stride_n
            )
    else:
        ptrs = oriented_tile_ptrs(base, offs_n, offs_k, stride_n, stride_k, SWAP_AB)
    return ptrs


@triton.jit
def flatten_weight_tile(
    w3, N2: tl.constexpr, KB: tl.constexpr, GATE: tl.constexpr, SWAP_AB: tl.constexpr
):
    """Flatten a loaded gate|up weight tile (see ``weight_tile_ptrs``) to the 2D MMA tile. Under
    ``GATE`` the stacked 3D tile (gate half + up half) collapses to the 2D form: swap ``[N2, KB]``
    (rows-major MMA lhs), no-swap ``[KB, N2]`` (K-major rhs), where ``N2 = 2*TN == BN`` — cols
    ``0..TN-1`` gate, ``TN..2TN-1`` up (the epilogue's ``split_gate_up`` undoes it). Without ``GATE``
    the tile is already 2D and passes through unchanged (``N2``/``KB`` unused)."""
    if GATE:
        if SWAP_AB:
            w2 = tl.reshape(w3, (N2, KB))
        else:
            w2 = tl.reshape(w3, (KB, N2))
    else:
        w2 = w3
    return w2


@triton.jit
def oriented_tile_ptrs(
    base, offs_rows, offs_k, stride_rows, stride_k, SWAP_AB: tl.constexpr
):
    """Operand-tile pointers oriented by whether these rows sit in the MMA M dim
    (``SWAP_AB``, from the weight's viewpoint — activation callers pass the flag inverted):
    rows-major ``[rows, K]`` when they do, else K-major ``[K, rows]``. Only the taken
    constexpr branch compiles, so the divergent shapes never meet. The per-step K-advance
    is identical for both layouts, so the caller advances the returned pointer the same
    way regardless of orientation."""
    if SWAP_AB:
        ptrs = base + (offs_rows[:, None] * stride_rows + offs_k[None, :] * stride_k)
    else:
        ptrs = base + (offs_k[:, None] * stride_k + offs_rows[None, :] * stride_rows)
    return ptrs


@triton.jit
def operand_tile_ptrs(
    base,
    offs_rows,
    offs_k,
    stride_rows,
    stride_k,
    MEMORY_MODE: tl.constexpr,
    SWAP_AB: tl.constexpr,
):
    """Prologue operand-tile pointer, folding the per-operand memory-mode branch: the explicit
    oriented ``[rows,K]``/``[K,rows]`` tile on the pointer arm, or ``base`` as a scalar
    placeholder on a descriptor arm (which reads its box via the descriptor — building the index
    tensor there would only stay live across the K-loop and spill registers). ``SWAP_AB`` is the
    orientation from the weight's viewpoint (activation callers pass it inverted). Single
    return — the arms have divergent types (a ``[rows,K]`` tile vs the scalar base), so an early
    return can't unify them; the constexpr selects one."""
    if MEMORY_MODE == "pointer":
        ptrs = oriented_tile_ptrs(base, offs_rows, offs_k, stride_rows, stride_k, SWAP_AB)
    else:
        ptrs = base
    return ptrs
