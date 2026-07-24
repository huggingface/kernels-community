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



# ── Triton-side helpers (inlined by ``@triton.jit`` callers) ──────────────────


@triton.jit
def fp8_act_quant_inline(
    a_raw, TRANSPOSED: tl.constexpr = False, UE8M0: tl.constexpr = False
):
    """Inline FP8 (E4M3) activation quant for the W8A8 block-scale path.

    Per-token amax → fp32 scale ``amax/448`` (floored at 1e-12 against zero rows)
    → cast values to FP8. Returns ``(a_fp8, a_s)`` with ``a_s`` shaped ``(M,)``;
    ``TRANSPOSED`` marks a ``(K, M)`` tile (the swapped descriptor arm), where the
    token axis is 0 instead of 1.

    ``UE8M0`` ceils each scale up to a power of two and returns the E8M0 exponent
    byte (uint8) instead of the fp32 scale — the group format the tcgen05
    ``dot_scaled`` MMA consumes. The ceil mirrors DeepGEMM's ``ceil_to_ue8m0``
    (add ``0x7FFFFF``, clear the mantissa) so our scales match its checkpoints.
    """
    if TRANSPOSED:
        a_s = tl.max(tl.abs(a_raw), axis=0) / 448.0
    else:
        a_s = tl.max(tl.abs(a_raw), axis=1) / 448.0
    denom = tl.maximum(a_s, 1e-12)
    if UE8M0:
        bits = (denom.to(tl.int32, bitcast=True) + 0x7FFFFF) & ~0x7FFFFF
        denom = bits.to(tl.float32, bitcast=True)
    if TRANSPOSED:
        a_fp8 = (a_raw / denom[None, :]).to(tl.float8e4nv)
    else:
        a_fp8 = (a_raw / denom[:, None]).to(tl.float8e4nv)
    if UE8M0:
        a_scale = ((bits >> 23) & 0xFF).to(tl.uint8)
    else:
        a_scale = a_s
    return a_fp8, a_scale



# cvt.e2m1x2.f32 (hardware FP4 pack) exists only on sm_100 (Blackwell). Resolved once at
# import as a compile-time constexpr for the jit helper below; the ALU fallback compiles
# everywhere else. ``is_sm10x`` is driverless-safe, so no import-time guard is needed.
_E2M1_HW_CVT = tl.constexpr(is_sm10x())



@triton.jit
def _quant_e2m1_packed(v, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    """Pack signed, pre-scaled values on the E2M1 grid to ``(M, K//2)`` uint8 (first value
    of each pair in the low nibble). On sm_100 the Blackwell hardware convert
    ``cvt.rn.satfinite.e2m1x2.f32`` (two f32 → one packed byte, first operand → HIGH nibble)
    does it in one instruction; elsewhere a ``>=``-threshold bucketize builds the code. The
    two agree except at exact E2M1 midpoints (0.25, 0.75, …), where the hardware rounds to
    nearest-even and the ALU form rounds half-up. Only the taken arm compiles."""
    if _E2M1_HW_CVT:
        lo, hi = tl.split(tl.reshape(v, (BLOCK_SIZE_M, BLOCK_SIZE_K // 2, 2)))
        packed = tl.inline_asm_elementwise(
            "{ .reg .b8 t8; cvt.rn.satfinite.e2m1x2.f32 t8, $1, $2; cvt.u16.u8 $0, t8; }",
            "=h,f,f",
            [hi, lo],
            dtype=tl.uint16,
            is_pure=True,
            pack=1,
        )
        values = (packed & 0xFF).to(tl.uint8)
    else:
        av = tl.abs(v)
        code = (
            (av >= 0.25).to(tl.int32)
            + (av >= 0.75).to(tl.int32)
            + (av >= 1.25).to(tl.int32)
            + (av >= 1.75).to(tl.int32)
            + (av >= 2.5).to(tl.int32)
            + (av >= 3.5).to(tl.int32)
            + (av >= 5.0).to(tl.int32)
        ) | ((v < 0).to(tl.int32) << 3)
        lo, hi = tl.split(tl.reshape(code, (BLOCK_SIZE_M, BLOCK_SIZE_K // 2, 2)))
        values = (lo | (hi << 4)).to(tl.uint8)
    return values



@triton.jit
def mx_act_quant_inline(
    a_raw,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    RECIPE: tl.constexpr = "mxfp8",
):
    """Inline MX activation quant, one helper for both value grids. Per-row, per-K-group
    amax → UE8M0 scale (ceil to the next power of two via the exponent-bump trick, the
    divisor being the grid's largest magnitude) → values onto the recipe's grid:

    - ``"mxfp8"``: cast to E4M3 — returns ``((M, K) fp8, (M, K // SCALE_GROUP_K) uint8)``.
    - ``"mxfp4"``: round to E2M1 and pack nibble pairs (``_quant_e2m1_packed`` — hardware
      ``cvt.e2m1x2`` on sm_100, else a ``>=``-threshold bucketize; they agree off exact
      midpoints) — returns ``((M, K//2) uint8, (M, K // SCALE_GROUP_K) uint8)``.
    - ``"nvfp4"``: E4M3 scale (amax/6, not a power of two), values divide by the DECODED
      scale before the E2M1 grid — returns ``((M, K//2) uint8, (M, K // SCALE_GROUP_K) E4M3)``.

    Only the taken recipe arm compiles."""
    a_groups = tl.reshape(
        a_raw, (BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_K, SCALE_GROUP_K)
    )
    amax = tl.max(tl.abs(a_groups), axis=2)
    if RECIPE == "nvfp4":
        # E4M3 scale (amax/6 rounded to E4M3, NOT a power of two); values divide by the
        # DECODED scale before hitting the E2M1 grid — the standard NVFP4 two-step
        scales = (amax / 6.0).to(tl.float8e4nv)
        decoded = tl.maximum(scales.to(tl.float32), 1.1754944e-38)
        v = tl.reshape(a_groups / decoded[:, :, None], (BLOCK_SIZE_M, BLOCK_SIZE_K))
        values = _quant_e2m1_packed(v, BLOCK_SIZE_M, BLOCK_SIZE_K)
    elif RECIPE == "mxfp4":
        bits = (amax / 6.0).to(tl.int32, bitcast=True)
        # ceil_to_ue8m0: bump exponent by 1 when mantissa is non-zero.
        exp_ceil = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        exp_ceil = tl.minimum(tl.maximum(exp_ceil, 1), 254)
        exp_ceil = tl.where(amax == 0, 127, exp_ceil)
        scales = exp_ceil.to(tl.uint8)
        a_s_pow2 = (exp_ceil << 23).to(tl.float32, bitcast=True)
        v = tl.reshape(a_groups / a_s_pow2[:, :, None], (BLOCK_SIZE_M, BLOCK_SIZE_K))
        values = _quant_e2m1_packed(v, BLOCK_SIZE_M, BLOCK_SIZE_K)
    else:
        bits = (amax / 448.0).to(tl.int32, bitcast=True)
        # ceil_to_ue8m0: bump exponent by 1 when mantissa is non-zero.
        exp_ceil = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        exp_ceil = tl.minimum(tl.maximum(exp_ceil, 1), 254)
        scales = exp_ceil.to(tl.uint8)
        a_s_pow2 = (exp_ceil << 23).to(tl.float32, bitcast=True)
        values = tl.reshape(
            a_groups / tl.maximum(a_s_pow2[:, :, None], 1e-12),
            (BLOCK_SIZE_M, BLOCK_SIZE_K),
        ).to(tl.float8e4nv)
    return values, scales



@triton.jit
def _e2m1_code_to_e4m3_bits(code):
    """One E2M1 4-bit code -> the E4M3 byte holding the same value, in pure integer
    ops. Every E2M1 magnitude ``{0, .5, 1, 1.5, 2, 3, 4, 6}`` is exact in E4M3, and
    above the 0.5 subnormal the mapping is affine in the code: ``bits = (mag + 12) << 2``
    (exponent re-bias +6, mantissa bit lands at bit 2). No float math, no converts —
    callers bitcast the byte to ``float8e4nv``."""
    code = code.to(tl.int32)
    mag = code & 7
    bits = tl.where(mag == 0, 0, tl.where(mag == 1, 0x30, (mag + 12) << 2))
    return bits | ((code >> 3) << 7)



@triton.jit
def e2m1_cols_to_e4m3(packed):
    """Column-unpack packed E2M1 (two nibbles per byte along the last dim, low nibble
    first) to E4M3: ``(..., C) uint8 -> (..., 2C)`` — the column-axis counterpart of the
    row-doubling ``e2m1_to_e4m3``; lossless (every E2M1 value is exact in E4M3) and
    integer-only: the bytes are built by ``_e2m1_code_to_e4m3_bits`` and bitcast once."""
    bits = tl.interleave(
        _e2m1_code_to_e4m3_bits(packed & 0xF), _e2m1_code_to_e4m3_bits(packed >> 4)
    )
    return bits.to(tl.uint8).to(tl.float8e4nv, bitcast=True)



@triton.jit
def e2m1_to_e4m3(b_packed):
    """Unpack packed MXFP4 (E2M1, two nibbles/byte along K) to E4M3, doubling the K
    (row) dim: ``(R, C) uint8 -> (2R, C) E4M3``. E2M1's 8 magnitudes are all exact in
    E4M3, so this is lossless — it lets the FP8 ``tl.dot`` path stand in for
    ``tl.dot_scaled`` at decode (avoiding its M->128 pad). K order is the low nibble
    first: ``[byte0_lo, byte0_hi, byte1_lo, ...]``."""
    lo = _e2m1_code_to_e4m3_bits(b_packed & 0xF)
    hi = _e2m1_code_to_e4m3_bits(b_packed >> 4)
    # interleave along the K (row) dim via trans -> interleave-last-dim -> trans back
    unpacked = tl.trans(tl.interleave(tl.trans(lo), tl.trans(hi)))
    return unpacked.to(tl.uint8).to(tl.float8e4nv, bitcast=True)



def _quant_block_k_pruner(configs, named_args, **kwargs):
    """Keep configs whose BLOCK_K divides K (the quant grid is K // BLOCK_K programs per row;
    K is always a multiple of 32, so the BLOCK_K=32 configs guarantee a non-empty list). On the
    SWIZZLED path, additionally require BLOCK_K a multiple of ``4 * SCALE_GROUP_K`` so a whole
    SWIZZLE_32_4_4 col-block (4 scale groups) lands inside one K-tile. BLOCK_T is free on the dense
    grid (the per-element store handles any tile height) but pinned to 128 on the GROUPED grid,
    where one program == one 128-row expert-sorted tile (``build_tile_layout``'s pad granularity)."""
    args = {**named_args, **kwargs}
    k = args["K"]
    if not args.get("SWIZZLED"):
        return [c for c in configs if k % c.kwargs["BLOCK_K"] == 0]
    g = args["SCALE_GROUP_K"]
    grouped = args.get("GROUPED", True)  # grouped grid is one program per 128-row block
    return [
        c
        for c in configs
        if k % c.kwargs["BLOCK_K"] == 0
        and c.kwargs["BLOCK_K"] % (4 * g) == 0
        and (not grouped or c.kwargs["BLOCK_T"] == 128)
    ]



@bayesian_autotune(
    [
        triton.Config({"BLOCK_K": bk, "BLOCK_T": bt}, num_warps=w)
        for bk in (32, 64, 128, 256)
        for bt in (8, 16, 32, 64, 128)
        for w in (2, 4, 8)
    ],
    # t_bucket (log2 of the token count) is in the key: at small T the tile is the only
    # parallelism lever while at prefill scale it isn't — same bucketing as the grouped
    # kernels (raw T would retune per unique token count). SWIZZLED keys the swizzled-scale
    # store separately (a disjoint config basin). RECIPE keys the value dtype/packing (E4M3 vs
    # packed E2M1, and SCALE_GROUP_K 32 vs 16) — a dtype-blind key hands packed MXFP4 the E4M3
    # config and mistunes it.
    ["K", "t_bucket", "SWIZZLED", "RECIPE"],
    n_trials=100,
    prune_configs_by={"early_config_prune": _quant_block_k_pruner},
)
@triton.jit
def _mx_act_quant_kernel(
    X,
    Y,
    S,  # (T, K // SCALE_GROUP_K) row-major scales (plain path); dummy on the swizzled path
    SOut,  # flat SWIZZLE_32_4_4 scale buffer (1, n_tiles, cb, 2, 256); dummy int on the plain path
    GatherIdx,  # (S,) int32 sorted position -> source row of X; read only when SWIZZLED and not None
    ExpertStart,  # (NUM_EXPERTS_POW2 + 1,) int32 cumulative sorted-row starts; read iff SWIZZLED
    GlobalScale,  # (1,) fp32 NVFP4 second-level per-tensor global; None ⇒ single-level (arm folds out)
    stride_x_t,
    stride_x_k,
    T,
    t_bucket,  # autotune key only (log2 token-count bucket); unused in body
    K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    # dynamo's triton wrapper appends the tuner's config kwargs after the call kwargs
    # and requires signature order — the tuned axes stay LAST
    RECIPE: tl.constexpr = "mxfp8",
    SWIZZLED: tl.constexpr = False,
    GROUPED: tl.constexpr = True,  # SWIZZLED grid: expert-sorted tiles (True) vs plain dense (False)
    NUM_EXPERTS_POW2: tl.constexpr = 1,
    BLOCK_K: tl.constexpr = 32,
    BLOCK_T: tl.constexpr = 32,
):
    """One-pass activation quant, one launch per recipe (``mx_act_quant_inline`` does
    the math, so the offline and inline forms are bit-identical by construction): E4M3 +
    UE8M0 ("mxfp8"), packed E2M1 + UE8M0 ("mxfp4"), or packed E2M1 + E4M3 group-16
    ("nvfp4"). Group boundaries are identical across forms (SCALE_GROUP_K | BLOCK_K | K).
    Arbitrary input strides.

    Plain path (``SWIZZLED=False``): grid ``(cdiv(T, BLOCK_T), K // BLOCK_K)`` — each program
    quantizes a ``[BLOCK_T, BLOCK_K]`` tile and writes ``S`` row-major (the one-row-per-program
    form starved memory at 1.5-1.8 TB/s on the packed recipes; the row tile coalesces).

    Swizzled grouped path (``SWIZZLED=True``, BLOCK_T pinned 128): grid ``(n_m_tiles,
    K // BLOCK_K)`` over the expert-sorted, 128-padded tile layout (``build_tile_layout``). Each
    program gathers its tile's source rows through ``GatherIdx`` (padding masked), quantizes,
    scatters the VALUES back to source row order (the GEMM still TMA-gathers them; duplicate
    sorted->source writes store identical bytes), and writes the SCALES straight into the
    SWIZZLE_32_4_4 layout the grouped GEMM reads affine (the inverse of ``load_swizzled_scale``)
    — no post-quant gather/swizzle pass. Padding-row scales are quantized zeros (harmless: the
    GEMM masks those rows' values to 0)."""
    kb = tl.program_id(1)
    if SWIZZLED:
        pid_m = tl.program_id(0)
        # scale output-row position (== source row on the dense path); the swizzled block index is
        # so // 128, valid for ANY BLOCK_T (no 128-row pin) so the dense grid autotunes BLOCK_T.
        so = pid_m * BLOCK_T + tl.arange(0, BLOCK_T)
        if GROUPED:
            exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
                ExpertStart, NUM_EXPERTS_POW2, BLOCK_T
            )
            _, sorted_idx, row_mask = resolve_tile_inline(
                pid_m, exp_start, freqs, tile_start_excl, e_offs, BLOCK_T
            )
            if GatherIdx is not None:
                in_row = tl.load(GatherIdx + sorted_idx, mask=row_mask, other=0).to(tl.int64)
            else:
                in_row = sorted_idx.to(tl.int64)
        else:
            in_row = so.to(tl.int64)
            row_mask = so < T
    else:
        pid_t = tl.program_id(0)
        in_row = (pid_t * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
        row_mask = in_row < T
    offs = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    x = tl.load(
        X + in_row[:, None] * stride_x_t + offs[None, :] * stride_x_k,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    if GlobalScale is not None:
        # NVFP4 two-level: normalize by the calibrated per-tensor global before the block
        # quant — block scales are then amax/6 of x/g (the canonical two-step); the GEMM
        # folds g back onto the accumulator (g_a·g_b). None folds the arm out at trace time.
        x = x / tl.load(GlobalScale).to(tl.float32)
    y, s = mx_act_quant_inline(x, BLOCK_T, BLOCK_K, SCALE_GROUP_K, RECIPE)
    width: tl.constexpr = BLOCK_K // 2 if RECIPE != "mxfp8" else BLOCK_K
    y_row: tl.constexpr = K // (BLOCK_K // width)  # per-row element count of Y
    yo = kb * width + tl.arange(0, width)
    # values -> source row order either way (the swizzled grid scatters via the gathered in_row)
    tl.store(Y + in_row[:, None] * y_row + yo[None, :], y, mask=row_mask[:, None])
    if SWIZZLED:
        # scales -> SWIZZLE_32_4_4 as PER-ELEMENT ptr arithmetic (no 128-row reshape): s[t, g] lands
        # at (block*cb_total + kb*REP_K + rep)*512 + r32*16 + outer4*4 + c4, where the output row
        # so[t] = pid_m*BLOCK_T + t gives block = so//128, r32 = so%32, outer4 = (so%128)//32, and the
        # col-block splits g into rep = g//4, c4 = g%4. Byte-identical to the old reshape form at
        # BLOCK_T=128 (grouped grid), but valid for any BLOCK_T -> the dense autotuned grid reuses this
        # exact store. Dense masks tiles past the real row-blocks; grouped over-allocates padding
        # blocks the GEMM never reads, so it writes them all (harmless).
        groups: tl.constexpr = BLOCK_K // SCALE_GROUP_K
        cb_total = K // SCALE_GROUP_K // 4
        lg = tl.arange(0, groups)
        rep = lg // 4
        block = so // 128
        off = (
            (block[:, None] * cb_total + kb * (groups // 4) + rep[None, :]) * 512
            + (so % 32)[:, None] * 16
            + ((so % 128) // 32)[:, None] * 4
            + (lg % 4)[None, :]
        )
        if GROUPED:
            tl.store(SOut + off, s)
        else:
            tl.store(SOut + off, s, mask=block[:, None] < tl.cdiv(T, 128))
    else:
        groups: tl.constexpr = BLOCK_K // SCALE_GROUP_K
        sg = kb * groups + tl.arange(0, groups)
        tl.store(
            S + in_row[:, None] * (K // SCALE_GROUP_K) + sg[None, :],
            s,
            mask=row_mask[:, None],
        )



def mx_act_quant_swizzled_grouped(
    x: torch.Tensor,
    recipe: str,
    scale_group: int,
    scale_dtype: torch.dtype,
    gather_idx: torch.Tensor | None,
    expert_start: torch.Tensor,
    global_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Offline MX act-quant for a grouped GEMM that emits SWIZZLE_32_4_4 scales directly (the
    ``SWIZZLED`` arm of ``_mx_act_quant_kernel``). Returns ``(values, swizzled_scale,
    num_m_tiles)`` — the values in source order (E4M3, or packed E2M1 uint8 for the fp4 recipes)
    and the scales as the ``(1, num_m_tiles, cb, 2, 256)`` swizzled tensor (the caller builds the
    GEMM's scale descriptor from it, like every other operand). Only the ``expert_start[-1]``
    scheduled rows are laid out; expert padding is per 128 (the ``BLOCK_T`` pin).

    ``n_m_tiles`` is a STATIC host-side upper bound (``S//128 + E``) on the padded tile count —
    never ``.item()`` (a CPU-GPU sync / cudagraph break). The buffer/grid over-allocate to it; the
    kernel writes only the real tiles (``build_tile_layout`` in-kernel), extra tile-programs mask
    out, and the GEMM reads only ``pid_m < total_m_tiles``. ``sum(ceil(freq/128)) <= S//128 + E``."""
    T, K = x.shape
    E = expert_start.numel() - 1
    S = gather_idx.numel() if gather_idx is not None else T
    n_m_tiles = S // 128 + E
    packed = recipe != "mxfp8"
    y = torch.empty(
        T, K // 2 if packed else K, device=x.device, dtype=torch.uint8 if packed else FP8_DTYPE
    )
    cb = triton.cdiv(K // scale_group, 4)
    s_sw = torch.empty(1, n_m_tiles, cb, 2, 256, device=x.device, dtype=scale_dtype)
    with device_context(x.device):
        compile_time_only_triton_wrap(_mx_act_quant_kernel)[
            lambda META: (n_m_tiles, K // META["BLOCK_K"])
        ](
            x,
            y,
            expert_start,  # dummy S (row-major scales unused on the swizzled arm)
            s_sw,  # flat SWIZZLE_32_4_4 scale buffer (pointer store; no descriptor)
            gather_idx,  # None = no gather (the is-not-None guard folds the load out)
            expert_start,
            global_scale,  # (1,) fp32 NVFP4 two-level global; None ⇒ single-level (arm folds out)
            x.stride(0),
            x.stride(1),
            T,
            T.bit_length(),
            K=K,
            SCALE_GROUP_K=scale_group,
            RECIPE=recipe,
            SWIZZLED=True,
            GROUPED=True,
            NUM_EXPERTS_POW2=E,
        )
    return y, s_sw, n_m_tiles



@triton.jit
def _swizzle_grouped_scales_kernel(
    SRC,  # (rows, cols) row-major pre-quantized group scales (uint8 / e8m0 / e4m3, 1 byte)
    DST,  # flat SWIZZLE_32_4_4 output buffer (1, n_tiles, NCB, 2, 256)
    GatherIdx,  # (S,) int sorted position -> source row of SRC; read only when not None
    ExpertStart,  # (NUM_EXPERTS_POW2 + 1,) cumulative sorted-row starts, S sentinel
    COLS,
    NCB,  # number of 4-wide column blocks (cols // 4)
    stride_src_m,
    NUM_EXPERTS_POW2: tl.constexpr,
):
    """Gather + expert-sorted 128-pad + swizzle a PRE-QUANTIZED grouped scale into the
    SWIZZLE_32_4_4 layout, in one launch — the padded tile layout is derived in-kernel from
    ``ExpertStart`` (``build_tile_layout``), so no torch index build. One 128x4 block per
    (M-tile, col-block): inverse of ``load_swizzled_scale``. Padding rows write quantized-zero
    scales (the GEMM masks those rows)."""
    exp_start, freqs, tile_start_excl, total_m_tiles, e_offs = build_tile_layout(
        ExpertStart, NUM_EXPERTS_POW2, 128
    )
    pid_m = tl.program_id(0)
    cb = tl.program_id(1)
    _, sorted_idx, row_mask = resolve_tile_inline(
        pid_m, exp_start, freqs, tile_start_excl, e_offs, 128
    )
    if GatherIdx is not None:
        src = tl.load(GatherIdx + sorted_idx, mask=row_mask, other=0)
    else:
        src = sorted_idx
    cj = cb * 4 + tl.arange(0, 4)
    s = tl.load(
        SRC + src[:, None] * stride_src_m + cj[None, :],
        mask=row_mask[:, None] & (cj[None, :] < COLS),
        other=0,
    )
    swizzle_store_block(DST, s, pid_m, cb, NCB)



def swizzle_grouped_mx_scales(
    scale: torch.Tensor,
    expert_start: torch.Tensor,
    gather_idx: torch.Tensor | None = None,
    pad_bm: int = 128,
) -> tuple[torch.Tensor, int]:
    """Fused gather + expert-sorted 128-pad + swizzle of a pre-quantized grouped scale (a
    down projection's externally supplied ``As``), one triton launch — the layout comes from
    ``expert_start`` in-kernel (no torch index build). Returns ``(swizzled_scale, num_m_tiles)``
    as the ``(1, num_m_tiles, cols // 4, 2, 256)`` tensor; the caller builds the GEMM's scale
    descriptor from it. ``gather_idx`` None = activations already expert-sorted.

    ``n_m_tiles`` is a STATIC host-side upper bound (``S//pad_bm + E``) — never ``.item()`` (a
    CPU-GPU sync / cudagraph break); the buffer/grid over-allocate and the extra tile-programs
    mask out (``sum(ceil(freq/pad_bm)) <= S//pad_bm + E``)."""
    E = expert_start.numel() - 1
    S = gather_idx.numel() if gather_idx is not None else scale.shape[0]
    n_m_tiles = S // pad_bm + E
    cols = scale.shape[1]
    cb = triton.cdiv(cols, 4)
    src = scale.view(torch.uint8)  # byte-level; the binder rejects e8m0/e4m3
    out = torch.empty(1, n_m_tiles, cb, 2, 256, device=scale.device, dtype=torch.uint8)
    with device_context(scale.device):
        compile_time_only_triton_wrap(_swizzle_grouped_scales_kernel)[(n_m_tiles, cb)](
            src,
            out,  # flat SWIZZLE_32_4_4 buffer (pointer store; no descriptor)
            gather_idx,  # None = no gather (the is-not-None guard folds the load out)
            expert_start,
            cols,
            cb,
            src.stride(0),
            NUM_EXPERTS_POW2=E,
        )
    return out.view(scale.dtype), n_m_tiles



def maybe_act_quant(x, act_quant, min_m):
    """Row-count-gated offline activation pre-quant. Apply ``act_quant`` (a one-pass
    quant kernel, e.g. ``mxfp8_act_quant``) when the GEMM consuming ``x`` is
    compute-bound (``rows >= min_m``) — the inline form re-quantizes per N-tile there.
    ``min_m`` is the consumer kernel's crossover, defined next to its wrapper with its
    provenance (measured sweep or inherited estimate). Below the threshold return ``x``
    raw: the weight-bandwidth-bound GEMM quantizes its
    one thin tile inline for free (the UE8M0 inline quant is exponent-only), and a
    separate quant kernel is pure added latency (M=1 graph attn decode measured
    0.66-0.85x offline). Bit-exact either way. Returns ``(a, a_scale)``; the consumer
    kernel picks its arm off ``a``'s dtype at compile time (fp8 = pre-quantized, raw
    bf16/fp16 = quantize inline), so in the inline arm ``a_scale`` is a constexpr-dead
    placeholder."""
    if x.shape[0] >= min_m:
        return act_quant(x)
    return x, x



def mxfp8_act_quant(x: torch.Tensor, swizzled: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``(T, K)`` activations to MX once (E4M3 values + UE8M0 group-32 uint8 scales)
    instead of inline per weight-tile — the fused gate_up re-ran the inline quant per N-tile
    (16x redundant amax/convert ALU + 2x act bytes), which held it at ~380 TFLOPS while the
    pre-quantized down kernel ran at ~1080. One pass costs ~50µs at 8k tokens. Bit-exact with
    the inline form (same group boundaries). ``swizzled=True`` emits the scale directly in
    SWIZZLE_32_4_4 for the tcgen05 fast path (same dense grid, per-element store)."""
    return _launch_act_quant(x, "mxfp8", MX_SCALE_GROUP_K, torch.uint8, swizzled)



def mxfp4_act_quant(x: torch.Tensor, swizzled: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``(T, K)`` activations to MXFP4 in one kernel pass: packed-E2M1 values
    (``(T, K//2)`` int8, two codes per byte, first value in the low nibble) + UE8M0
    group-32 uint8 scales (``(T, K//32)``, amax/6 ceil'd to a power of two). Bit-identical
    to the fused epilogues' inline form (shared ``mx_act_quant_inline`` arm). Feeds the
    W4A4 arm of the MX matmul ops; quantizing activations to fp4 at runtime is an
    accuracy call the caller owns — the ops never do it implicitly. ``swizzled=True`` emits the
    scale directly in SWIZZLE_32_4_4 for the tcgen05 fast path."""
    return _launch_act_quant(x, "mxfp4", MX_SCALE_GROUP_K, torch.uint8, swizzled)



def nvfp4_act_quant(
    x: torch.Tensor, swizzled: bool = False, global_scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``(T, K)`` activations to NVFP4 in one kernel pass: packed-E2M1 values
    (``(T, K//2)`` int8) + E4M3 group-16 block scales (``(T, K//16)`` — ``amax/6`` rounded to
    E4M3, NOT a power of two; values divide by the DECODED scale before the E2M1 grid,
    the standard two-step). ``global_scale`` is the CALIBRATED per-tensor second level
    (``(1,)`` fp32, the checkpoint's ``input_scale``): values are normalized by it before
    the block quant, so block scales stay in e4m3 range regardless of the activation's
    dynamic range — the canonical two-level recipe. The GEMM folds ``g_a·g_b`` back onto
    the accumulator (pass ``As = [scales, global_scale]``). ``None`` = single-level
    (``g_a = 1``). ``swizzled=True`` emits the scale directly in SWIZZLE_32_4_4 for the
    tcgen05 fast path."""
    return _launch_act_quant(
        x, "nvfp4", NVFP4_SCALE_GROUP_K, torch.float8_e4m3fn, swizzled, global_scale
    )



def nvfp4_quantize_two_level(
    weight: torch.Tensor, swizzled: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Canonical two-level NVFP4 quant of a ``(N, K)`` (or ``(K,)``-last) tensor. Returns
    ``(packed_e2m1 int8, e4m3 group-16 block scales, fp32 per-tensor global)`` — the block scale
    is the op's ``Bs`` and the global its ``b_global_scale`` (the decoupled API keeps the two levels
    as separate arguments).

    The second level is a per-tensor fp32 global = ``amax / (6 · 448)`` — the smallest global that
    keeps every e4m3 block scale in range (block ``amax/6`` after dividing by the global stays
    ``≤ 448``). Two-level quant IS single-level quant of the globally-normalized tensor, so the block
    values + scales come straight from ``nvfp4_act_quant(weight / global)``. The kernels multiply
    the folded ``g_a · g_b`` onto the accumulator; the e4m3 block scales ride ``dot_scaled`` as
    usual (activations are single-level ⇒ ``g_a = 1``)."""
    global_scale = (weight.abs().amax() / (6.0 * 448.0)).clamp(min=1e-30).float()
    packed, block = nvfp4_act_quant((weight / global_scale).contiguous(), swizzled)
    return packed.view(torch.int8), block, global_scale.reshape(1)



# offline act-quant pass per recipe (keys = ``resolve_input_recipe`` results)
MX_ACT_QUANT = {
    "mxfp8": mxfp8_act_quant,
    "mxfp4": mxfp4_act_quant,
    "nvfp4": nvfp4_act_quant,
}



def _launch_act_quant(x, recipe, scale_group, scale_dtype, swizzled=False, global_scale=None):
    """One-pass activation quant for every recipe (``mxfp8`` = E4M3 values, else packed E2M1) and
    both scale layouts. ``swizzled=True`` writes the scale straight into the SWIZZLE_32_4_4 buffer
    ``(1, cdiv(T, 128), cb, 2, 256)`` (per-element ptr store, dense autotuned ``BLOCK_T`` — same grid
    as the affine path, just the store address flips); ``swizzled=False`` writes row-major
    ``(T, K // scale_group)``. ``global_scale`` (NVFP4 two-level, ``(1,)`` fp32) normalizes the
    values before the block quant. Returns ``(values, scales)``."""
    T, K = x.shape
    packed = recipe != "mxfp8"
    if packed:
        assert K % (2 * scale_group) == 0, (
            f"K (={K}) must be a multiple of {2 * scale_group} to pack E2M1 pairs"
        )
    values = torch.empty(
        T, K // 2 if packed else K, device=x.device,
        dtype=torch.uint8 if packed else FP8_DTYPE,
    )
    if swizzled:
        cb = triton.cdiv(K // scale_group, 4)
        scales = torch.empty(
            1, triton.cdiv(T, 128), cb, 2, 256, device=x.device, dtype=scale_dtype
        )
    else:
        scales = torch.empty(T, K // scale_group, device=x.device, dtype=scale_dtype)
    with device_context(x.device):
        compile_time_only_triton_wrap(_mx_act_quant_kernel)[
            lambda META: (triton.cdiv(T, META["BLOCK_T"]), K // META["BLOCK_K"])
        ](
            x,
            values,
            values if swizzled else scales,  # S: row-major scales (plain) / dummy (swizzled)
            scales if swizzled else 0,  # SOut: SWIZZLE_32_4_4 buffer (swizzled) / dummy
            values,  # dummy GatherIdx (unread on the dense grid)
            values,  # dummy ExpertStart (unread on the dense grid)
            global_scale,  # (1,) fp32 NVFP4 two-level global; None ⇒ single-level (arm folds out)
            x.stride(0),
            x.stride(1),
            T,
            T.bit_length(),
            K=K,
            SCALE_GROUP_K=scale_group,
            RECIPE=recipe,
            SWIZZLED=swizzled,
            GROUPED=False,
        )
    return (values.view(torch.int8) if packed else values), scales



@bayesian_autotune(
    [
        triton.Config({"BLOCK_T": bt}, num_warps=w)
        for bt in (16, 32, 64, 128)
        for w in (1, 2, 4, 8)
    ],
    ["K", "BLOCK_K", "t_bucket"],
    n_trials=100,
)
@triton.jit
def _fp8_act_quant_block_dynamic_kernel(
    X,
    Y,
    S,
    stride_x_t,
    stride_x_k,
    T,
    t_bucket,  # autotune key only (log2 token-count bucket); unused in body
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    UE8M0: tl.constexpr = False,
):
    """One-pass block-FP8 activation quant: rows → E4M3 + one ``amax/448`` scale per
    ``BLOCK_K`` span (fp32, or a UE8M0 exponent byte under ``UE8M0``). Grid
    ``(cdiv(T, BLOCK_T), K // BLOCK_K)`` — each program quantizes a ``[BLOCK_T, BLOCK_K]``
    tile (``BLOCK_T`` rows over one K-block, each row its own scale). The one-row-per-program
    form starved memory (128-byte transactions across ~T*K/BK tiny programs); the row tile
    gives the loads something to coalesce. The span equals the consumer's ``BLOCK_SIZE_K``,
    so results are bit-exact with the inline quant. Arbitrary input strides (no host-side
    copy); ``BLOCK_K`` is fixed by the scale layout, ``BLOCK_T`` and warps are tuned."""
    pid_t = tl.program_id(0)
    kb = tl.program_id(1)
    rows = (pid_t * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
    offs = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = rows < T
    x = tl.load(
        X + rows[:, None] * stride_x_t + offs[None, :] * stride_x_k,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    y, s = fp8_act_quant_inline(x, UE8M0=UE8M0)
    tl.store(Y + rows[:, None] * K + offs[None, :], y, mask=row_mask[:, None])
    tl.store(S + rows * (K // BLOCK_K) + kb, s, mask=row_mask)



def fp8_act_quant_block_dynamic(
    x: torch.Tensor, block_k: int, use_ue8m0: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``(T, K)`` activations to block-FP8 once (E4M3 + per-``block_k`` scales)
    instead of inline per weight-tile — same rationale and layout as ``mxfp8_act_quant`` (a
    GEMM re-reads its activation once per N-tile). Bit-exact with the inline form. Scales are
    fp32 (``amax/448``) or, under ``use_ue8m0``, UE8M0 exponent bytes (power-of-two scales)
    for the tcgen05 ``dot_scaled`` path — the DeepGEMM-Blackwell recipe."""
    T, K = x.shape
    y = torch.empty(T, K, device=x.device, dtype=FP8_DTYPE)
    s_dtype = torch.uint8 if use_ue8m0 else torch.float32
    s = torch.empty(T, K // block_k, device=x.device, dtype=s_dtype)

    def grid(META):
        return (triton.cdiv(T, META["BLOCK_T"]), K // block_k)

    with device_context(x.device):
        compile_time_only_triton_wrap(_fp8_act_quant_block_dynamic_kernel)[grid](
            x, y, s, x.stride(0), x.stride(1), T, T.bit_length(),
            K=K, BLOCK_K=block_k, UE8M0=use_ue8m0,
        )
    return y, s



# ── fp8_act_quant_tensor_wide kernel (used by tensor-mode FP8 wrappers) ───────────────────


@triton.jit
def _fp8_act_quant_kernel(
    x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr, PADDED_BLOCK: tl.constexpr
):
    # ``tl.arange`` needs a power-of-2 length, so iterate over PADDED_BLOCK (the next
    # power of 2) and mask the tail — lets block_size be non-power-of-2 (e.g. a full
    # row K=14336 in tensor-mode). Masked lanes load 0, which can't affect ``amax``.
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, PADDED_BLOCK)
    mask = cols < BLOCK_SIZE
    offs = pid * BLOCK_SIZE + cols
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0  # float8_e4m3fn max
    y = (x / tl.maximum(s, 1e-12)).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid, s)



@compile_time_only_triton_op(
    add_op_namespace_prefix("fp8_act_quant_tensor_wide"), mutates_args=(), opaque=True
)
def fp8_act_quant_tensor_wide(
    x: torch.Tensor, block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous()
    assert x.shape[-1] % block_size == 0
    y = torch.empty_like(x, dtype=FP8_DTYPE)
    grid = (triton.cdiv(x.numel(), block_size),)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)

    with device_context(x.device):
        compile_time_only_triton_wrap(_fp8_act_quant_kernel)[grid](
            x,
            y,
            s,
            BLOCK_SIZE=block_size,
            PADDED_BLOCK=triton.next_power_of_2(block_size),
        )

    return y, s
