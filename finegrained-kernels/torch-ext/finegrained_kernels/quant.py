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
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from .bayesian_autotuner import bayesian_autotune
from .formats import global_scale_stride, int_weight_bits, is_per_expert_global, normalize_global_scale
from .compat import add_op_namespace_prefix, FP8_DTYPE, MX_SCALE_GROUP_K, NVFP4_SCALE_GROUP_K, compile_time_only_triton_op, compile_time_only_triton_wrap, device_context, is_blackwell, is_sm10x, pdl_launch_kwargs, sm_count
from .ptx import e2m1x2_to_e4m3x2, e2m1x2_to_f16x2, f32x2_to_e2m1x2
from .swizzle import swizzle_store_block
from .scheduling import build_tile_layout, resolve_tile_inline

# ── Triton-side helpers (inlined by ``@triton.jit`` callers) ──────────────────


@triton.jit
def quantize_calibrated_act(vals, scale, OUT_DTYPE: tl.constexpr):
    """Static per-tensor activation quantization against a calibrated ``scale``, bit for bit with ModelOpt's input
    quantizer: ``vals / scale`` (a true divide; a reciprocal multiply rounds differently), clamped to the target's range,
    then INT8 rounded half-to-even (negative half-way values included; a cast would truncate) or saturating E4M3."""
    value = vals.to(tl.float32) / scale
    if OUT_DTYPE == tl.int8:
        value = tl.clamp(value, -128.0, 127.0)
        # Round to nearest even, including negative half-way values, on every backend.
        lower = tl.floor(value)
        fraction = value - lower
        value = lower + ((fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))).to(tl.float32)
    else:
        value = tl.clamp(value, -448.0, 448.0)
    return value.to(tl.int8 if OUT_DTYPE == tl.int8 else tl.float8e4nv)


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


# Hardware FP4 conversions exist on every Blackwell family, independently of TMEM MMA.
_E2M1_HW_CVT = tl.constexpr(is_blackwell())


@triton.jit
def _quant_e2m1_packed(v, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    """Pack signed, pre-scaled values on the E2M1 grid to ``(M, K//2)`` uint8 (first value
    of each pair in the low nibble). On Blackwell the hardware convert
    ``cvt.rn.satfinite.e2m1x2.f32`` (two f32 → one packed byte, first operand → HIGH nibble)
    does it in one instruction; elsewhere a ``>=``-threshold bucketize builds the code. The
    two agree except at exact E2M1 midpoints (0.25, 0.75, …), where the hardware rounds to
    nearest-even and the ALU form rounds half-up. Only the taken arm compiles."""
    if _E2M1_HW_CVT:
        lo, hi = tl.split(tl.reshape(v, (BLOCK_SIZE_M, BLOCK_SIZE_K // 2, 2)))
        packed = f32x2_to_e2m1x2(hi, lo)
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
    FORMAT: tl.constexpr = "mxfp8",
):
    """Inline MX activation quant, one helper for both value grids. Per-row, per-K-group
    amax → UE8M0 scale (ceil to the next power of two via the exponent-bump trick, the
    divisor being the grid's largest magnitude) → values onto the format's grid:

    - ``"mxfp8"``: cast to E4M3 — returns ``((M, K) fp8, (M, K // SCALE_GROUP_K) uint8)``.
    - ``"mxfp4"``: round to E2M1 and pack nibble pairs (``_quant_e2m1_packed`` — hardware
      ``cvt.e2m1x2`` on sm_100, else a ``>=``-threshold bucketize; they agree off exact
      midpoints) — returns ``((M, K//2) uint8, (M, K // SCALE_GROUP_K) uint8)``.
    - ``"nvfp4"``: E4M3 scale (amax/6, not a power of two), values divide by the DECODED
      scale before the E2M1 grid — returns ``((M, K//2) uint8, (M, K // SCALE_GROUP_K) E4M3)``.

    Only the taken format arm compiles."""
    a_groups = tl.reshape(
        a_raw, (BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_K, SCALE_GROUP_K)
    )
    amax = tl.max(tl.abs(a_groups), axis=2)
    if FORMAT == "nvfp4":
        # E4M3 scale (amax/6 rounded to E4M3, NOT a power of two); values divide by the
        # DECODED scale before hitting the E2M1 grid — the standard NVFP4 two-step
        scales = (amax / 6.0).to(tl.float8e4nv)
        decoded = tl.maximum(scales.to(tl.float32), 1.1754944e-38)
        v = tl.reshape(a_groups / decoded[:, :, None], (BLOCK_SIZE_M, BLOCK_SIZE_K))
        values = _quant_e2m1_packed(v, BLOCK_SIZE_M, BLOCK_SIZE_K)
    elif FORMAT == "mxfp4":
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
def _e2m1_code_to_bf16_bits(code):
    """One E2M1 4-bit code -> the bf16 bit pattern (uint16) holding the same value, pure int ops.
    The 8 magnitudes ``{0,.5,1,1.5,2,3,4,6}`` are exact in bf16; above the 0.5 subnormal the mapping
    is affine in the code: ``bits = 0x3F80 + (mag - 2) * 0x40`` (each step doubles/1.5x's cleanly on
    the bf16 exponent/mantissa). Callers bitcast the uint16 to bf16 — no float convert, and no E4M3
    intermediate. bf16 is the exact intermediate for every activation dtype, not the final one: the
    caller widens the dequantized tile if the activation is wider."""
    code = code.to(tl.int32)
    mag = code & 7
    bits = tl.where(mag == 0, 0, tl.where(mag == 1, 0x3F00, 0x3F80 + (mag - 2) * 0x40))
    return bits | ((code >> 3) << 15)  # sign at bit 15


@triton.jit
def e2m1_to_bf16(b_packed):
    """Unpack packed MXFP4 (E2M1, two nibbles/byte along K) straight to bf16, doubling the K (row)
    dim: ``(R, C) uint8 -> (2R, C) bf16``. Direct (no E4M3 round-trip), and the low/high nibbles
    interleave along K with a SINGLE 3D axis-swap (``join`` -> ``[R, C, 2]`` -> swap last two ->
    reshape) instead of the three 2D transposes ``e2m1_to_e4m3`` needs — the weight-only unpack hot path
    (fp4 weight upcast in-loop, the caller widening to the activation dtype). K order low-nibble-first, matching
    ``e2m1_to_e4m3``: ``[byte0_lo, byte0_hi, byte1_lo, ...]``. On Blackwell the codes decode through
    the hardware CVT (``ptx.e2m1x2_to_f16x2``, fp16 being exact for the grid); elsewhere the integer
    bit-build. Only the taken arm compiles."""
    if _E2M1_HW_CVT:
        pair = e2m1x2_to_f16x2(b_packed)
        lo = (pair & 0xFFFF).to(tl.uint16)
        hi = (pair >> 16).to(tl.uint16)
        paired = tl.trans(tl.join(lo, hi), 0, 2, 1)
        f16 = tl.reshape(
            paired, (2 * b_packed.shape[0], b_packed.shape[1])
        ).to(tl.float16, bitcast=True)
        return f16.to(tl.bfloat16)
    lo = _e2m1_code_to_bf16_bits(b_packed & 0xF)  # [R, C] int32 (bf16 bits)
    hi = _e2m1_code_to_bf16_bits(b_packed >> 4)   # [R, C]
    paired = tl.join(lo, hi)                       # [R, C, 2]  (pair in last dim)
    paired = tl.trans(paired, 0, 2, 1)             # [R, 2, C]  (pair adjacent in row dim)
    out = tl.reshape(paired, (2 * b_packed.shape[0], b_packed.shape[1]))  # [2R, C]
    return out.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def e2m1_cols_to_bf16(packed):
    """Column-unpack packed E2M1 to bf16: ``(..., C) uint8 -> (..., 2C)``, low nibble first — the
    column-axis counterpart of the row-doubling ``e2m1_to_bf16``, for the swapped weight-only
    tile (output rows leading, K contiguous). Same sm_100 hardware decode; the pair lands in
    adjacent columns, so ``tl.interleave`` places it (no axis swap)."""
    if _E2M1_HW_CVT:
        pair = e2m1x2_to_f16x2(packed)
        lo = (pair & 0xFFFF).to(tl.uint16)
        hi = (pair >> 16).to(tl.uint16)
        f16 = tl.interleave(lo, hi).to(tl.float16, bitcast=True)
        return f16.to(tl.bfloat16)
    bits = tl.interleave(
        _e2m1_code_to_bf16_bits(packed & 0xF), _e2m1_code_to_bf16_bits(packed >> 4)
    )
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def e2m1_cols_to_f32(packed):
    """Column-unpack packed E2M1 straight to fp32, for the CUDA-core reduces: ``(..., C) uint8 ->
    (..., 2C)``. The narrow-float hop its siblings owe their MMA callers (bf16 for ``tl.dot``, E4M3
    for the scaled MMA) is pure work for a reduce that widens to fp32 anyway — measured 12.6 ->
    15.4us on the gpt-oss decode tile for the bf16 round trip alone. Bit-identical to going via
    either: every E2M1 value is exact in f16, bf16, E4M3 and fp32."""
    if _E2M1_HW_CVT:
        pair = e2m1x2_to_f16x2(packed)
        lo = (pair & 0xFFFF).to(tl.uint16)
        hi = (pair >> 16).to(tl.uint16)
        f16 = tl.interleave(lo, hi).to(tl.float16, bitcast=True)
        return f16.to(tl.float32)
    bits = tl.interleave(
        _e2m1_code_to_bf16_bits(packed & 0xF), _e2m1_code_to_bf16_bits(packed >> 4)
    )
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)


@triton.jit
def e2m1_cols_to_e4m3(packed):
    """Column-unpack packed E2M1 (two nibbles per byte along the last dim, low nibble
    first) to E4M3: ``(..., C) uint8 -> (..., 2C)`` — the column-axis counterpart of the
    row-doubling ``e2m1_to_e4m3``; lossless (every E2M1 value is exact in E4M3). Blackwell
    decodes through the CVT unit (``ptx.e2m1x2_to_e4m3x2``), elsewhere the integer bit-build;
    either way the bytes are bitcast once. Only the taken arm compiles."""
    if _E2M1_HW_CVT:
        pair = e2m1x2_to_e4m3x2(packed)
        bits = tl.interleave(pair & 0xFF, pair >> 8)
    else:
        bits = tl.interleave(
            _e2m1_code_to_e4m3_bits(packed & 0xF), _e2m1_code_to_e4m3_bits(packed >> 4)
        )
    return bits.to(tl.uint8).to(tl.float8e4nv, bitcast=True)


@triton.jit
def e2m1_to_f32(b_packed):
    """Row-doubling counterpart of ``e2m1_cols_to_f32``: ``(R, C) uint8 -> (2R, C)`` fp32, for the
    non-swapped CUDA-core reduce. Skips the E4M3 hop ``e2m1_to_e4m3`` owes its ``tl.dot`` callers;
    bit-identical, since every E2M1 value is exact in E4M3, f16 and fp32."""
    if _E2M1_HW_CVT:
        pair = e2m1x2_to_f16x2(b_packed)
        lo = (pair & 0xFFFF).to(tl.uint16)
        hi = (pair >> 16).to(tl.uint16)
        unpacked = tl.trans(tl.interleave(tl.trans(lo), tl.trans(hi)))
        return unpacked.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    lo = _e2m1_code_to_bf16_bits(b_packed & 0xF)
    hi = _e2m1_code_to_bf16_bits(b_packed >> 4)
    unpacked = tl.trans(tl.interleave(tl.trans(lo), tl.trans(hi)))
    return unpacked.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)


@triton.jit
def e2m1_to_e4m3(b_packed):
    """Unpack packed MXFP4 (E2M1, two nibbles/byte along K) to E4M3, doubling the K
    (row) dim: ``(R, C) uint8 -> (2R, C) E4M3``. E2M1's 8 magnitudes are all exact in
    E4M3, so this is lossless — it lets the FP8 ``tl.dot`` path stand in for
    ``tl.dot_scaled`` at decode (avoiding its M->128 pad). K order is the low nibble
    first: ``[byte0_lo, byte0_hi, byte1_lo, ...]``. sm_100 decodes through the CVT unit
    (``ptx.e2m1x2_to_e4m3x2``); elsewhere the integer bit-build. Only the taken arm compiles."""
    if _E2M1_HW_CVT:
        pair = e2m1x2_to_e4m3x2(b_packed)
        lo, hi = pair & 0xFF, pair >> 8
    else:
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
    k, g = args["K"], args["SCALE_GROUP_K"]
    swizzled, grouped = args.get("SWIZZLED", False), args.get("GROUPED", True)
    kept = [
        c
        for c in configs
        if k % c.kwargs["BLOCK_K"] == 0
        and (not swizzled or c.kwargs["BLOCK_K"] % (4 * g) == 0)
        and (not grouped or c.kwargs["BLOCK_T"] == 128)
    ]
    if grouped or not kept:
        return kept
    # The dense grid is cdiv(T, BLOCK_T) x K / BLOCK_K programs; keep the tiles that fill the SMs (or come closest).
    # A tile tall enough to leave most SMs idle can still win a 2-3us tune by noise: MXFP8 6144-wide T=64 crowned
    # BLOCK_T=64 BLOCK_K=256 (24 programs, 3.6us) over BLOCK_T=8 BLOCK_K=128 (384 programs, 2.3us).
    def programs(c):
        return triton.cdiv(args["T"], c.kwargs["BLOCK_T"]) * (k // c.kwargs["BLOCK_K"])

    target = min(sm_count(args["X"].device.index), max(programs(c) for c in kept))
    return [c for c in kept if programs(c) >= target]


@triton.jit
def store_mx_act_scales(
    S,
    SOut,
    out_row,  # per-lane output row (any int vector: contiguous grid rows or scattered dests)
    s,
    row_mask,
    kb,
    T,
    K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SWIZZLED: tl.constexpr,
    MASK: tl.constexpr,  # "rows" (per-lane), "blocks" (whole 128-row blocks past T), "none"
):
    """Store one act-quant scale tile. Row-major (``S``, per ``out_row``) or straight into the
    SWIZZLE_32_4_4 buffer (``SOut``) as PER-ELEMENT ptr arithmetic (no 128-row reshape): s[t, g]
    lands at (block*cb_total + kb*REP_K + rep)*512 + r32*16 + outer4*4 + c4, where block =
    out_row//128, r32 = out_row%32, outer4 = (out_row%128)//32, and the col-block splits g into
    rep = g//4, c4 = g%4. Byte-identical to the old reshape form at BLOCK_T=128 (grouped grid)
    but valid for any out_row vector, so the dense grid, the grouped-sorted grid, and the
    scatter (quantize-once) stores all share it. Masking is the caller\'s store policy: the
    grouped-sorted grid writes its padding rows too ("none" — quantized zeros the GEMM never
    reads), the dense grid masks whole tiles past the real row-blocks ("blocks"), scattered
    and row-major stores mask per lane ("rows")."""
    groups: tl.constexpr = BLOCK_K // SCALE_GROUP_K
    if SWIZZLED:
        cb_total = K // SCALE_GROUP_K // 4
        lg = tl.arange(0, groups)
        rep = lg // 4
        block = out_row // 128
        off = (
            (block[:, None] * cb_total + kb * (groups // 4) + rep[None, :]) * 512
            + (out_row % 32)[:, None] * 16
            + ((out_row % 128) // 32)[:, None] * 4
            + (lg % 4)[None, :]
        )
        if MASK == "rows":
            tl.store(SOut + off, s, mask=row_mask[:, None])
        elif MASK == "blocks":
            tl.store(SOut + off, s, mask=block[:, None] < tl.cdiv(T, 128))
        else:
            tl.store(SOut + off, s)
    else:
        sg = kb * groups + tl.arange(0, groups)
        tl.store(
            S + out_row.to(tl.int64)[:, None] * (K // SCALE_GROUP_K) + sg[None, :],
            s,
            mask=row_mask[:, None],
        )


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
    # store separately (a disjoint config basin), GROUPED the grid — its 128-row expert tiles
    # pin BLOCK_T, so its crowns cannot be shared with the dense grid's. FORMAT keys the value
    # dtype/packing (E4M3 vs packed E2M1, and SCALE_GROUP_K 32 vs 16) — a dtype-blind key hands
    # packed MXFP4 the E4M3 config and mistunes it.
    ["K", "t_bucket", "SWIZZLED", "GROUPED", "FORMAT"],
    n_trials=100,
    prune_configs_by={"early_config_prune": _quant_block_k_pruner},
)
@triton.jit
def _mx_act_quant_kernel(
    X,
    PreQuantScale,
    Y,
    S,  # (T, K // SCALE_GROUP_K) row-major scales; None on the swizzled path
    SOut,  # flat SWIZZLE_32_4_4 scale buffer (1, n_tiles, cb, 2, 256); None on the plain path
    GatherIdx,  # (S,) int32 row -> source row of X (grouped: sorted position); None = the row itself
    ExpertStart,  # (NUM_EXPERTS_POW2 + 1,) int32 cumulative sorted-row starts; read iff GROUPED
    GlobalScale,  # fp32 NVFP4 second-level global, per tensor or per expert; None ⇒ single-level (arm folds out)
    GlobalIdx,  # (T,) row -> expert map, for a per-expert global on the DENSE grid (the batched op's own expert_ids); None on the grouped grid, whose tiles carry an expert
    ScaleRows,  # (T, SCALE_ROWS) dense row -> its rows of the swizzled scale tiles (-1 skips); None = the row itself
    stride_global,  # expert stride of GlobalScale (0 broadcasts a scalar)
    last_global,  # index of GlobalScale's last entry — GlobalIdx is clamped to [0, last_global]: an EP sentinel row names an off-rank expert (masked out of the GEMM, so any in-range global serves it)
    stride_x_t,
    stride_x_k,
    T,
    t_bucket,  # autotune key only (log2 token-count bucket); unused in body
    K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    # dynamo's triton wrapper appends the tuner's config kwargs after the call kwargs
    # and requires signature order — the tuned axes stay LAST
    FORMAT: tl.constexpr = "mxfp8",
    SWIZZLED: tl.constexpr = False,
    GROUPED: tl.constexpr = True,  # grid: expert-sorted tiles (True) vs plain dense rows (False)
    NUM_EXPERTS_POW2: tl.constexpr = 1,  # always passed explicitly; see the dense launch
    SCALE_ROWS: tl.constexpr = 1,  # ScaleRows entries per row: top_k when a token's scales serve each of its routed rows
    BLOCK_K: tl.constexpr = 32,
    BLOCK_T: tl.constexpr = 32,
    PDL: tl.constexpr = False,
):
    """One-pass activation quant, one launch per format (``mx_act_quant_inline`` does
    the math, so the offline and inline forms are bit-identical by construction): E4M3 +
    UE8M0 ("mxfp8"), packed E2M1 + UE8M0 ("mxfp4"), or packed E2M1 + E4M3 group-16
    ("nvfp4"). Group boundaries are identical across forms (SCALE_GROUP_K | BLOCK_K | K).
    Arbitrary input strides.

    Two independent axes. ``GROUPED`` picks the GRID: dense ``(cdiv(T, BLOCK_T), K // BLOCK_K)``
    over plain row tiles (the one-row-per-program form starved memory at 1.5-1.8 TB/s on the
    packed formats; the row tile coalesces), or ``(n_m_tiles, K // BLOCK_K)`` over the
    expert-sorted, 128-padded tile layout (``build_tile_layout``, so BLOCK_T is pinned there).
    A grouped program gathers its tile's source rows through ``GatherIdx`` (padding masked) and
    scatters the VALUES back to source row order — the GEMM still TMA-gathers them, and duplicate
    sorted->source writes store identical bytes — and its tile carries the expert a per-expert
    NVFP4 global is indexed by, which is why only the dense grid needs a row map for that.

    ``SWIZZLED`` picks the SCALE STORE: straight into the SWIZZLE_32_4_4 layout the grouped GEMM
    reads affine (the inverse of ``load_swizzled_scale``), so no post-quant gather/swizzle pass,
    or row-major ``S``. Padding-row scales are quantized zeros (harmless: the GEMM masks those
    rows' values to 0)."""
    if PDL:
        gdc_wait()
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
        expert_id, sorted_idx, row_mask = resolve_tile_inline(
            tl.program_id(0), exp_start, freqs, tile_start_excl, e_offs, BLOCK_T
        )
        if GatherIdx is not None:
            in_row = tl.load(GatherIdx + sorted_idx, mask=row_mask, other=0).to(tl.int64)
        else:
            in_row = sorted_idx.to(tl.int64)
        out_row = in_row
    else:
        expert_id = 0
        rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        row_mask = rows < T
        out_row = rows.to(tl.int64)
        if GatherIdx is not None:
            in_row = tl.load(GatherIdx + rows, mask=row_mask, other=0).to(tl.int64)
        else:
            in_row = out_row
    offs = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    x = tl.load(
        X + in_row[:, None] * stride_x_t + offs[None, :] * stride_x_k,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    if PreQuantScale is not None:
        # SVDQuant rounds the smoothed input to BF16 before either branch.
        x = (x * tl.load(PreQuantScale + offs)[None, :]).to(X.dtype.element_ty).to(tl.float32)
    if GlobalScale is not None:
        # NVFP4 two-level: normalize by the calibrated global before the block quant — block
        # scales are then amax/6 of x/g (the canonical two-step); the GEMM folds g back onto the
        # accumulator (g_a·g_b). A per-expert global is indexed by the tile's OWN expert, which
        # the grouped grid resolved for free — the rows of a tile belong to one expert; the dense
        # grid has no such layout and takes the caller's row map. None folds the arm out.
        if GlobalIdx is not None:
            e = tl.minimum(tl.maximum(tl.load(GlobalIdx + out_row, mask=row_mask, other=0), 0), last_global)
            x = x / tl.load(GlobalScale + e * stride_global).to(tl.float32)[:, None]
        else:
            x = x / tl.load(GlobalScale + expert_id * stride_global).to(tl.float32)
    y, s = mx_act_quant_inline(x, BLOCK_T, BLOCK_K, SCALE_GROUP_K, FORMAT)
    width: tl.constexpr = BLOCK_K // 2 if FORMAT != "mxfp8" else BLOCK_K
    y_row: tl.constexpr = K // (BLOCK_K // width)  # per-row element count of Y
    yo = kb * width + tl.arange(0, width)
    # values -> out_row: the source row on the grouped grid, the row itself on the dense one
    if PDL:
        gdc_launch_dependents()
    tl.store(Y + out_row[:, None] * y_row + yo[None, :], y, mask=row_mask[:, None])
    if SWIZZLED and ScaleRows is not None:
        for j in tl.static_range(SCALE_ROWS):
            dest = tl.load(ScaleRows + out_row * SCALE_ROWS + j, mask=row_mask, other=-1)
            store_mx_act_scales(
                S, SOut, dest, s, row_mask & (dest >= 0), kb, T, K, SCALE_GROUP_K, BLOCK_K, True, "rows"
            )
    elif SWIZZLED:
        store_mx_act_scales(
            S, SOut, so, s, row_mask, kb, T, K, SCALE_GROUP_K, BLOCK_K, True,
            "none" if GROUPED else "blocks",
        )
    else:
        store_mx_act_scales(
            S, SOut, out_row, s, row_mask, kb, T, K, SCALE_GROUP_K, BLOCK_K, False, "rows"
        )


def mx_act_quant_grouped(
    x: torch.Tensor,
    activation_format: str,
    scale_group: int,
    scale_dtype: torch.dtype,
    gather_idx: torch.Tensor | None,
    expert_start: torch.Tensor,
    global_scale: torch.Tensor | None = None,
    swizzled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Offline MX act-quant on the GROUPED grid of ``_mx_act_quant_kernel``: one program per
    expert-sorted tile, so each tile knows its own expert and a per-expert NVFP4 global needs no
    per-row index. Returns ``(values, scales, num_m_tiles)`` — the values in source order (E4M3,
    or packed E2M1 uint8 for the fp4 formats) and the scales either as the ``(1, num_m_tiles, cb,
    2, 256)`` SWIZZLE_32_4_4 tensor the GEMM reads through a descriptor (``swizzled``) or
    row-major ``(T, K // scale_group)`` for the affine arm. Only the ``expert_start[-1]``
    scheduled rows are laid out; expert padding is per 128 (the ``BLOCK_T`` pin).

    ``n_m_tiles`` is a STATIC host-side upper bound (``S//128 + E``) on the padded tile count —
    never ``.item()`` (a CPU-GPU sync / cudagraph break). The buffer/grid over-allocate to it; the
    kernel writes only the real tiles (``build_tile_layout`` in-kernel), extra tile-programs mask
    out, and the GEMM reads only ``pid_m < total_m_tiles``. ``sum(ceil(freq/128)) <= S//128 + E``."""
    T, K = x.shape
    E = expert_start.numel() - 1
    S = gather_idx.numel() if gather_idx is not None else T
    n_m_tiles = S // 128 + E
    packed = activation_format != "mxfp8"
    y = torch.empty(
        T, K // 2 if packed else K, device=x.device, dtype=torch.uint8 if packed else FP8_DTYPE
    )
    if swizzled:
        cb = triton.cdiv(K // scale_group, 4)
        scales = torch.empty(1, n_m_tiles, cb, 2, 256, device=x.device, dtype=scale_dtype)
    else:
        scales = torch.empty(T, K // scale_group, device=x.device, dtype=scale_dtype)
    with device_context(x.device):
        compile_time_only_triton_wrap(_mx_act_quant_kernel)[
            lambda META: (n_m_tiles, K // META["BLOCK_K"])
        ](
            x,
            None,  # PreQuantScale
            y,
            None if swizzled else scales,  # S: row-major scales, affine arm only
            scales if swizzled else None,  # SOut: SWIZZLE_32_4_4 buffer, swizzled arm only
            gather_idx,  # None = no gather (the is-not-None guard folds the load out)
            expert_start,
            global_scale,  # fp32 NVFP4 two-level global; None ⇒ single-level (arm folds out)
            None,  # GlobalIdx: the grouped tiles carry their own expert
            None,  # ScaleRows
            global_scale_stride(global_scale),
            0,  # last_global: unread without a row map
            x.stride(0),
            x.stride(1),
            T,
            T.bit_length(),
            K=K,
            SCALE_GROUP_K=scale_group,
            FORMAT=activation_format,
            SWIZZLED=swizzled,
            GROUPED=True,
            NUM_EXPERTS_POW2=E,
            **pdl_launch_kwargs(),
        )
    return y, scales, n_m_tiles


ROUTED_ROW_MAP_BLOCK = 32


@triton.jit
def _routed_row_map_kernel(
    ExpertStart, RowExpert, ScaleRows, ScatterIdx, S, NUM_EXPERTS_POW2: tl.constexpr, BLOCK: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """For each routed row (expert-sorted): its expert and its row in the grouped GEMM's padded
    128-row expert tiles (-1 past every expert, an EP sentinel). The tile row is stored at the
    routed row, or at its ``ScatterIdx`` slot (``t * top_k + j``), so token ``t``'s ``top_k`` entries
    name its routed rows' tiles."""
    if PDL:
        gdc_wait()
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = rows < S
    exp_start, freqs, tile_start_excl, _, e_offs = build_tile_layout(ExpertStart, NUM_EXPERTS_POW2, 128)
    expert_id = tl.sum((rows[:, None] >= (exp_start + freqs)[None, :]).to(tl.int32), 1)
    if PDL:
        gdc_launch_dependents()
    if RowExpert is not None:
        tl.store(RowExpert + rows, expert_id, mask=mask)
    if ScaleRows is not None:
        in_expert = e_offs[None, :] == expert_id[:, None]
        tile_row = tl.sum(tl.where(in_expert, (tile_start_excl * 128 - exp_start)[None, :], 0), 1) + rows
        tile_row = tl.where(expert_id < NUM_EXPERTS_POW2, tile_row, -1)
        dest = tl.load(ScatterIdx + rows, mask=mask, other=0) if ScatterIdx is not None else rows
        tl.store(ScaleRows + dest, tile_row, mask=mask)


def mx_act_quant_routed_rows(
    x: torch.Tensor,
    activation_format: str,
    scale_group: int,
    scale_dtype: torch.dtype,
    gather_idx: torch.Tensor,
    expert_start: torch.Tensor,
    global_scale: torch.Tensor | None = None,
    swizzled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """The routed rows ``x[gather_idx]`` quantized in expert-sorted order, each against its own
    expert's global — what a per-expert NVFP4 global needs, since a token routed to several experts
    quantizes differently for each. One dense pass, with each row's expert and tile row resolved
    once up front; the scales land in the layout the grouped GEMM reads (SWIZZLE_32_4_4 tiles, or
    row-major ``(S, K // scale_group)``), and the GEMM reads the rows without a gather. Returns
    ``(values, scales, num_m_tiles)`` like ``mx_act_quant_grouped``."""
    K = x.shape[1]
    E = expert_start.numel() - 1
    S = gather_idx.numel()
    n_m_tiles = S // 128 + E
    packed = activation_format != "mxfp8"
    y = torch.empty(S, K // 2 if packed else K, device=x.device, dtype=torch.uint8 if packed else FP8_DTYPE)
    # the quant stores swizzled when each row's scales fill whole 4-column blocks; otherwise
    # row-major, for the swizzle pass to pad
    store_swizzled = swizzled and K % (4 * scale_group) == 0
    if store_swizzled:
        cb = triton.cdiv(K // scale_group, 4)
        scales = torch.empty(1, n_m_tiles, cb, 2, 256, device=x.device, dtype=scale_dtype)
    else:
        scales = torch.empty(S, K // scale_group, device=x.device, dtype=scale_dtype)
    per_expert = is_per_expert_global(global_scale)
    row_expert = torch.empty(S, device=x.device, dtype=torch.int32)
    scale_rows = torch.empty(S, device=x.device, dtype=torch.int32) if store_swizzled else None
    with device_context(x.device):
        compile_time_only_triton_wrap(_routed_row_map_kernel)[(triton.cdiv(S, ROUTED_ROW_MAP_BLOCK),)](
            expert_start, row_expert, scale_rows, None, S,
            NUM_EXPERTS_POW2=E, BLOCK=ROUTED_ROW_MAP_BLOCK, **pdl_launch_kwargs(),
        )
        compile_time_only_triton_wrap(_mx_act_quant_kernel)[
            lambda META: (triton.cdiv(S, META["BLOCK_T"]), K // META["BLOCK_K"])
        ](
            x,
            None,  # PreQuantScale
            y,
            None if store_swizzled else scales,
            scales if store_swizzled else None,
            gather_idx,
            expert_start,
            global_scale,
            row_expert if per_expert else None,
            scale_rows,
            global_scale_stride(global_scale),
            (global_scale.numel() - 1) if per_expert else 0,
            x.stride(0),
            x.stride(1),
            S,
            S.bit_length(),
            K=K,
            SCALE_GROUP_K=scale_group,
            FORMAT=activation_format,
            SWIZZLED=store_swizzled,
            GROUPED=False,
            NUM_EXPERTS_POW2=E,
            **pdl_launch_kwargs(),
        )
    if swizzled and not store_swizzled:
        scales, n_m_tiles = swizzle_grouped_mx_scales(scales, expert_start, None)
    return y, scales, n_m_tiles


# opaque, so torch.compile never reads the PDL inline asm, which its kernel analysis cannot parse
@compile_time_only_triton_op(add_op_namespace_prefix("mx_act_quant_routed_scales"), mutates_args=(), opaque=True)
def mx_act_quant_routed_scales(
    x: torch.Tensor,
    activation_format: str,
    scale_group: int,
    scale_dtype: torch.dtype,
    scatter_idx: torch.Tensor,
    expert_start: torch.Tensor,
    global_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each token of ``x`` quantized once against a shared global, its values in token order for
    the grouped GEMM to gather and its scales stored straight into every one of its routed rows'
    SWIZZLE_32_4_4 expert tiles — the scale gather+swizzle folded into the quant. ``scatter_idx``
    as ``compute_grouped_scheduling`` returns it. Returns ``(values, scales)``."""
    T, K = x.shape
    E = expert_start.numel() - 1
    S = scatter_idx.numel()
    top_k = S // T
    n_m_tiles = S // 128 + E
    packed = activation_format != "mxfp8"
    y = torch.empty(T, K // 2 if packed else K, device=x.device, dtype=torch.uint8 if packed else FP8_DTYPE)
    cb = triton.cdiv(K // scale_group, 4)
    scales = torch.empty(1, n_m_tiles, cb, 2, 256, device=x.device, dtype=scale_dtype)
    scale_rows = torch.empty(S, device=x.device, dtype=torch.int32)  # (T, top_k)
    with device_context(x.device):
        compile_time_only_triton_wrap(_routed_row_map_kernel)[(triton.cdiv(S, ROUTED_ROW_MAP_BLOCK),)](
            expert_start, None, scale_rows, scatter_idx, S,
            NUM_EXPERTS_POW2=E, BLOCK=ROUTED_ROW_MAP_BLOCK, **pdl_launch_kwargs(),
        )
        compile_time_only_triton_wrap(_mx_act_quant_kernel)[
            lambda META: (triton.cdiv(T, META["BLOCK_T"]), K // META["BLOCK_K"])
        ](
            x,
            None,  # PreQuantScale
            y,
            None,
            scales,
            None,  # GatherIdx: one row per token
            None,  # ExpertStart
            global_scale,
            None,  # GlobalIdx: a per-tensor global
            scale_rows,
            global_scale_stride(global_scale),
            0,
            x.stride(0),
            x.stride(1),
            T,
            T.bit_length(),
            K=K,
            SCALE_GROUP_K=scale_group,
            FORMAT=activation_format,
            SWIZZLED=True,
            GROUPED=False,
            NUM_EXPERTS_POW2=1,
            SCALE_ROWS=top_k,
            **pdl_launch_kwargs(),
        )
    return (y.view(torch.int8) if packed else y), scales


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


def row_padded_empty(rows: int, cols: int, row_multiple: int, dtype: torch.dtype, device) -> torch.Tensor:
    """An empty ``(rows, cols)`` tensor whose storage is rounded up to ``row_multiple`` rows, so a
    descriptor box over the padded rows reads in bounds (the extra rows are never written)."""
    padded = triton.cdiv(rows, row_multiple) * row_multiple
    return torch.empty(padded * cols, device=device, dtype=dtype)[: rows * cols].view(rows, cols)


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


def mxfp8_act_quant(
    x: torch.Tensor, swizzled: bool = False, row_multiple: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``(T, K)`` activations to MX once (E4M3 values + UE8M0 group-32 uint8 scales)
    instead of inline per weight-tile — the fused gate_up re-ran the inline quant per N-tile
    (16x redundant amax/convert ALU + 2x act bytes), which held it at ~380 TFLOPS while the
    pre-quantized down kernel ran at ~1080. One pass costs ~50µs at 8k tokens. Bit-exact with
    the inline form (same group boundaries). ``swizzled=True`` emits the scale directly in
    SWIZZLE_32_4_4 for the tcgen05 fast path (same dense grid, per-element store)."""
    return _launch_act_quant(x, "mxfp8", MX_SCALE_GROUP_K, torch.uint8, swizzled, row_multiple=row_multiple)


def mxfp4_act_quant(
    x: torch.Tensor, swizzled: bool = False, row_multiple: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``(T, K)`` activations to MXFP4 in one kernel pass: packed-E2M1 values
    (``(T, K//2)`` int8, two codes per byte, first value in the low nibble) + UE8M0
    group-32 uint8 scales (``(T, K//32)``, amax/6 ceil'd to a power of two). Bit-identical
    to the fused epilogues' inline form (shared ``mx_act_quant_inline`` arm). Feeds the
    W4A4 arm of the MX matmul ops; quantizing activations to fp4 at runtime is an
    accuracy call the caller owns — the ops never do it implicitly. ``swizzled=True`` emits the
    scale directly in SWIZZLE_32_4_4 for the tcgen05 fast path."""
    return _launch_act_quant(x, "mxfp4", MX_SCALE_GROUP_K, torch.uint8, swizzled, row_multiple=row_multiple)


def nvfp4_act_quant(
    x: torch.Tensor,
    swizzled: bool = False,
    global_scale: torch.Tensor | None = None,
    expert_index: torch.Tensor | None = None,
    gather_idx: torch.Tensor | None = None,
    row_multiple: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``(T, K)`` activations to NVFP4 in one kernel pass: packed-E2M1 values
    (``(T, K//2)`` int8) + E4M3 group-16 block scales (``(T, K//16)`` — ``amax/6`` rounded to
    E4M3, NOT a power of two; values divide by the DECODED scale before the E2M1 grid,
    the standard two-step). ``global_scale`` is the CALIBRATED per-tensor second level
    (fp32, the checkpoint's ``input_scale``): values are normalized by it before
    the block quant, so block scales stay in e4m3 range regardless of the activation's
    dynamic range — the canonical two-level format. The GEMM folds ``g_a·g_b`` back onto
    the accumulator (pass ``As=scales`` with ``a_global_scale=global_scale``). Per-expert globals
    need each row's expert, which ``expert_index`` gives where the caller already holds that map
    (the batched op's ``expert_ids``, sentinels and all) — it is read only when the globals are
    per expert, and ``mx_act_quant_grouped`` needs none at all, its tiles carrying an expert.
    ``None`` = single-level (``g_a = 1``). ``swizzled=True`` emits the scale directly in
    SWIZZLE_32_4_4 for the tcgen05 fast path."""
    return _launch_act_quant(
        x, "nvfp4", NVFP4_SCALE_GROUP_K, torch.float8_e4m3fn, swizzled, global_scale, expert_index, gather_idx,
        row_multiple=row_multiple,
    )


def _svdquant_pruner(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    configs = [c for c in configs if args["K"] % c.kwargs["BLOCK_K"] == 0]
    # One K tile per split cannot amortize padded MMA setup for a rank-32 decode matvec.
    single_tile_decode = (args["M"] == 1 and args["RANK"] == 32 and args["Y"] is not None
                          and args["K"] <= args["SPLIT_K"] * max((c.kwargs["BLOCK_K"] for c in configs), default=0))
    return [c for c in configs if (c.kwargs["BLOCK_M"] > 1 or args["M"] <= 16)
            and not (c.kwargs["BLOCK_M"] == 1 and c.kwargs["SWAP_AB"])
            and (not single_tile_decode or c.kwargs["BLOCK_M"] == 1)]


@bayesian_autotune(
    [triton.Config({"BLOCK_M": bm, "BLOCK_K": bk, "SWAP_AB": swap}, num_warps=w, num_stages=s)
     for bm in (1, 16, 32, 64) for bk in (64, 128, 256) for w in (4, 8) for s in (1, 3)
     for swap in (False, True)],
    ["K", "RANK", "m_bucket", "SPLIT_K", "SWIZZLED"],
    n_trials=100,
    prune_configs_by={"early_config_prune": _svdquant_pruner},
)
@triton.jit
def _svdquant_act_quant_kernel(
    X, PreQuantScale, GlobalScale,
    LoraA,
    Y, S, SOut, Down, Partials,
    M, m_bucket,
    stride_x_m, stride_x_k, stride_la_r, stride_la_k,
    K: tl.constexpr, SCALE_GROUP_K: tl.constexpr, RANK: tl.constexpr, SPLIT_K: tl.constexpr, SWIZZLED: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, SWAP_AB: tl.constexpr,
):
    """BF16 LoRA-down with optional fused NVFP4 quantization of the smoothed input."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    split = tl.program_id(1)
    rank = tl.arange(0, triton.next_power_of_2(RANK))
    if SWAP_AB:
        acc = tl.full((triton.next_power_of_2(RANK), BLOCK_M), 0, tl.float32)
    else:
        acc = tl.full((BLOCK_M, triton.next_power_of_2(RANK)), 0, tl.float32)
    if GlobalScale is not None and Y is not None:
        global_scale = tl.load(GlobalScale)
    for kb in range(split, K // BLOCK_K, SPLIT_K):
        cols = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(X + rows[:, None] * stride_x_m + cols[None, :] * stride_x_k,
                    mask=rows[:, None] < M, other=0)
        if PreQuantScale is not None:
            x = (x.to(tl.float32) * tl.load(PreQuantScale + cols)[None, :]).to(X.dtype.element_ty)
        w = tl.load(LoraA + rank[:, None] * stride_la_r + cols[None, :] * stride_la_k,
                    mask=rank[:, None] < RANK, other=0)
        if BLOCK_M == 1:
            acc += tl.sum(x.to(tl.float32)[:, None, :] * w.to(tl.float32)[None, :, :], 2)
        elif SWAP_AB:
            acc = tl.dot(w, tl.trans(x), acc)
        else:
            acc = tl.dot(x, tl.trans(w), acc)
        if Y is not None:
            normalized = x.to(tl.float32)
            if GlobalScale is not None:
                normalized /= global_scale
            y, scales = mx_act_quant_inline(normalized, BLOCK_M, BLOCK_K, SCALE_GROUP_K, "nvfp4")
            packed_cols = kb * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
            tl.store(Y + rows[:, None] * (K // 2) + packed_cols[None, :], y, mask=rows[:, None] < M)
            store_mx_act_scales(S, SOut, rows, scales, rows < M, kb, M, K,
                                SCALE_GROUP_K, BLOCK_K, SWIZZLED, "rows")
    if SWAP_AB:
        acc = tl.trans(acc)
    mask = (rows[:, None] < M) & (rank[None, :] < RANK)
    if SPLIT_K == 1:
        tl.store(Down + rows[:, None] * RANK + rank[None, :], acc, mask=mask)
    else:
        tl.store(Partials + (split * M + rows[:, None]) * RANK + rank[None, :], acc, mask=mask)


@triton.jit
def _svdquant_down_reduce_kernel(Partials, Down, M, RANK: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, triton.next_power_of_2(SPLIT_K))
    parts = tl.load(Partials + splits[:, None] * M * RANK + offsets[None, :],
                    mask=(splits[:, None] < SPLIT_K) & (offsets[None, :] < M * RANK), other=0)
    tl.store(Down + offsets, tl.sum(parts, 0), mask=offsets < M * RANK)


def svdquant_act_quant(x, lora_a, pre_quant_scale, global_scale, swizzled):
    """Quantized residual input and BF16 down projection from the same pre-quantized input."""
    K, rank = x.shape[-1], lora_a.shape[0]
    x = x.reshape(-1, K)
    M = x.shape[0]
    separate_quant = is_sm10x() and rank == 32 and M >= 4096 and M * K >= 1 << 25
    if separate_quant:
        # Independent quantization tiles avoid keeping the large input in both quant and MMA layouts.
        values, scales = _launch_act_quant(
            x, "nvfp4", NVFP4_SCALE_GROUP_K, torch.float8_e4m3fn,
            swizzled, global_scale, pre_quant_scale=pre_quant_scale)
    else:
        values = torch.empty(M, K // 2, device=x.device, dtype=torch.int8)
        scale_shape = (1, triton.cdiv(M, 128), triton.cdiv(K // NVFP4_SCALE_GROUP_K, 4), 2, 256) if swizzled else (M, K // NVFP4_SCALE_GROUP_K)
        scales = torch.empty(scale_shape, device=x.device, dtype=torch.float8_e4m3fn)
    split_k = min(32, max(1, triton.cdiv(256, triton.cdiv(M, 32))))
    down = torch.empty(M, rank, device=x.device, dtype=x.dtype) if M > 1 or split_k == 1 else None
    partials = torch.empty(split_k, M, rank, device=x.device, dtype=torch.float32) if split_k > 1 else None
    with device_context(x.device):
        compile_time_only_triton_wrap(_svdquant_act_quant_kernel)[lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), split_k)](
            x, pre_quant_scale, global_scale, lora_a, None if separate_quant else values,
            None if separate_quant or swizzled else scales, scales if swizzled and not separate_quant else None, down, partials,
            M, int(M).bit_length(), x.stride(0), x.stride(1), lora_a.stride(0), lora_a.stride(1),
            K=K, SCALE_GROUP_K=NVFP4_SCALE_GROUP_K, RANK=rank, SPLIT_K=split_k, SWIZZLED=swizzled,
        )
        if split_k > 1 and M > 1:
            compile_time_only_triton_wrap(_svdquant_down_reduce_kernel)[(triton.cdiv(M * rank, 128),)](
                partials, down, M, RANK=rank, SPLIT_K=split_k, BLOCK=128,
            )
    return values, scales, partials if M == 1 and split_k > 1 else down


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
    # normalize in fp32: bf16 / 0-dim-fp32 stays bf16 under type promotion, and the extra
    # rounding before the block quant flips E2M1 codes vs the exact normalized tensor
    packed, block = nvfp4_act_quant((weight.float() / global_scale).contiguous(), swizzled)
    return packed.view(torch.int8), block, global_scale.reshape(1)


# offline act-quant pass per activation format (keys = ``resolve_activation_format`` results)
MX_ACT_QUANT = {
    "mxfp8": mxfp8_act_quant,
    "mxfp4": mxfp4_act_quant,
    "nvfp4": nvfp4_act_quant,
}


def _launch_act_quant(
    x, activation_format, scale_group, scale_dtype, swizzled=False, global_scale=None,
    expert_index=None, gather_idx=None, pre_quant_scale=None, row_multiple=1,
):
    """One-pass activation quant for every format (``mxfp8`` = E4M3 values, else packed E2M1) and
    both scale layouts. ``swizzled=True`` writes the scale straight into the SWIZZLE_32_4_4 buffer
    ``(1, cdiv(T, 128), cb, 2, 256)`` (per-element ptr store, dense autotuned ``BLOCK_T`` — same grid
    as the affine path, just the store address flips); ``swizzled=False`` writes row-major
    ``(T, K // scale_group)``. ``global_scale`` (NVFP4 two-level, ``(1,)`` fp32) normalizes the
    values before the block quant. ``gather_idx`` quantizes ``x[gather_idx]``, one output row per
    entry, without materializing it — what per-expert globals need for routed rows of one token.
    ``row_multiple`` backs the value rows with storage rounded up to that many rows, as
    ``fp8_act_quant_tensor_wide``. Returns ``(values, scales)``."""
    K = x.shape[1]
    T = gather_idx.numel() if gather_idx is not None else x.shape[0]
    packed = activation_format != "mxfp8"
    per_expert = is_per_expert_global(global_scale)
    if packed:
        assert K % (2 * scale_group) == 0, (
            f"K (={K}) must be a multiple of {2 * scale_group} to pack E2M1 pairs"
        )
    values = row_padded_empty(
        T, K // 2 if packed else K, row_multiple, torch.uint8 if packed else FP8_DTYPE, x.device
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
            pre_quant_scale,  # PreQuantScale
            values,
            None if swizzled else scales,  # S: row-major scales, plain arm only
            scales if swizzled else None,  # SOut: SWIZZLE_32_4_4 buffer, swizzled arm only
            gather_idx,  # None = output row s reads source row s
            None,  # ExpertStart: no expert-sorted tiles on the dense grid
            global_scale,  # fp32 NVFP4 two-level global; None ⇒ single-level (arm folds out)
            expert_index if per_expert else None,  # row -> expert map, read iff the globals are per expert
            None,  # ScaleRows
            global_scale_stride(global_scale),
            (global_scale.numel() - 1) if per_expert else 0,
            x.stride(0),
            x.stride(1),
            T,
            T.bit_length(),
            K=K,
            SCALE_GROUP_K=scale_group,
            FORMAT=activation_format,
            SWIZZLED=swizzled,
            GROUPED=False,
            # explicit even though the dense grid never reads it: inductor's wrap_triton path
            # drops constexprs left to their default, and the arity mismatch that produces kills
            # the whole torch.compile launch ("launcher() missing 1 required positional argument")
            NUM_EXPERTS_POW2=1,
            **pdl_launch_kwargs(),
        )
    return (values.view(torch.int8) if packed else values), scales


@bayesian_autotune(
    [
        triton.Config({"BLOCK_T": bt}, num_warps=w)
        for bt in (16, 32, 64, 128)
        for w in (1, 2, 4, 8)
    ],
    ["K", "BLOCK_K", "t_bucket", "SWIZZLED"],
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
    SWIZZLED: tl.constexpr = False,
    PDL: tl.constexpr = False,
):
    """One-pass block-FP8 activation quant: rows → E4M3 + one ``amax/448`` scale per
    ``BLOCK_K`` span (fp32, or a UE8M0 exponent byte under ``UE8M0``). Grid
    ``(cdiv(T, BLOCK_T), K // BLOCK_K)`` — each program quantizes a ``[BLOCK_T, BLOCK_K]``
    tile (``BLOCK_T`` rows over one K-block, each row its own scale). The one-row-per-program
    form starved memory (128-byte transactions across ~T*K/BK tiny programs); the row tile
    gives the loads something to coalesce. The span equals the consumer's ``BLOCK_SIZE_K``,
    so results are bit-exact with the inline quant. Arbitrary input strides (no host-side
    copy); ``BLOCK_K`` is fixed by the scale layout, ``BLOCK_T`` and warps are tuned.

    ``SWIZZLED`` (UE8M0 only) stores each ``1 x BLOCK_K`` scale into its ``BLOCK_K // 32`` group-32
    slots of the SWIZZLE_32_4_4 layout: the same power-of-two scale repeated, so an MXFP8 GEMM
    reads exactly the block-FP8 operand."""
    if PDL:
        gdc_wait()
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
    if PDL:
        gdc_launch_dependents()
    if SWIZZLED:
        store_mx_act_scales(
            None, S, rows, tl.broadcast_to(s[:, None], (BLOCK_T, BLOCK_K // 32)), row_mask, kb, T, K,
            32, BLOCK_K, True, "blocks",
        )
    else:
        tl.store(S + rows * (K // BLOCK_K) + kb, s, mask=row_mask)


def fp8_act_quant_block_dynamic(
    x: torch.Tensor, block_k: int, use_ue8m0: bool = False, row_multiple: int = 1, swizzled: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``(T, K)`` activations to block-FP8 once (E4M3 + per-``block_k`` scales)
    instead of inline per weight-tile — same rationale and layout as ``mxfp8_act_quant`` (a
    GEMM re-reads its activation once per N-tile). Bit-exact with the inline form. Scales are
    fp32 (``amax/448``) or, under ``use_ue8m0``, UE8M0 exponent bytes (power-of-two scales)
    for the tcgen05 ``dot_scaled`` path — the DeepGEMM-Blackwell format. ``row_multiple`` backs
    the E4M3 rows with storage rounded up to that many rows, as ``fp8_act_quant_tensor_wide``.
    ``swizzled`` (UE8M0 only) emits the scales as the group-32 SWIZZLE_32_4_4 operand of an MXFP8
    GEMM instead, each block scale repeated over its ``block_k // 32`` groups."""
    T, K = x.shape
    assert use_ue8m0 or not swizzled, "only power-of-two (UE8M0) block scales are MX group scales"
    y = row_padded_empty(T, K, row_multiple, FP8_DTYPE, x.device)
    s_dtype = torch.uint8 if use_ue8m0 else torch.float32
    if swizzled:
        cb = triton.cdiv(K // MX_SCALE_GROUP_K, 4)
        s = torch.empty(1, triton.cdiv(T, 128), cb, 2, 256, device=x.device, dtype=s_dtype)
    else:
        s = torch.empty(T, K // block_k, device=x.device, dtype=s_dtype)

    def grid(META):
        return (triton.cdiv(T, META["BLOCK_T"]), K // block_k)

    with device_context(x.device):
        compile_time_only_triton_wrap(_fp8_act_quant_block_dynamic_kernel)[grid](
            x, y, s, x.stride(0), x.stride(1), T, T.bit_length(),
            K=K, BLOCK_K=block_k, UE8M0=use_ue8m0, SWIZZLED=swizzled,
            **pdl_launch_kwargs(),
        )
    return y, s


# ── fp8_act_quant_tensor_wide kernel (used by tensor-mode FP8 wrappers) ───────────────────


@triton.jit
def _fp8_act_quant_kernel(
    x_ptr, y_ptr, s_ptr, stride_xm, blocks_per_row, BLOCK_SIZE: tl.constexpr, PADDED_BLOCK: tl.constexpr,
    AMAX_UPPER_BOUND: tl.constexpr = None,
    PDL: tl.constexpr = False,
):
    if PDL:
        gdc_wait()
    # ``tl.arange`` needs a power-of-2 length, so iterate over PADDED_BLOCK (the next
    # power of 2) and mask the tail — lets block_size be non-power-of-2 (e.g. a full
    # row K=14336 in tensor-mode). Masked lanes load 0, which can't affect ``amax``.
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, PADDED_BLOCK)
    mask = cols < BLOCK_SIZE
    offs = pid * BLOCK_SIZE + cols
    # x's rows may sit apart (a row-strided view), so its offsets can pass int32; y and s are written dense
    x_offs = (pid // blocks_per_row).to(tl.int64) * stride_xm + (pid % blocks_per_row) * BLOCK_SIZE + cols
    x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x))
    if AMAX_UPPER_BOUND is not None:
        # the amax is capped, and the values past it saturate
        amax = tl.minimum(amax, AMAX_UPPER_BOUND)
        x = tl.clamp(x, -amax, amax)
    s = amax / 448.0  # float8_e4m3fn max
    y = (x / tl.maximum(s, 1e-12)).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)
    if PDL:
        gdc_launch_dependents()
    tl.store(s_ptr + pid, s)


def static_expert_act_operands(A, As, num_experts):
    """Activation operands of the per-expert STATIC arms: ``(A_q, As, as_stride)``.

    One calibrated scale for the whole matmul, or one per expert — a MoE calibrates each
    separately. ``as_stride`` 0 makes every expert read the one scale, 1 gives each its own.
    Per expert the kernel quantizes in register against the tile's own scale, because one token
    routes to several experts whose scales differ and so has no single pre-quantized form; one
    scale for all of them pre-quantizes once here instead.
    """
    As = As.reshape(-1).to(torch.float32)
    as_stride = int(As.numel() == num_experts)
    A_q = A if as_stride else (A.to(torch.float32) / As).to(FP8_DTYPE)
    return A_q, As, as_stride


# Host-side operand marshalling, NOT a custom op: it returns its inputs unchanged on the
# pre-quantized and per-expert branches, which `torch.library.custom_op` forbids (the output may
# not alias an input), and it hands back Python strides. The quant kernels it calls carry their
# own compile handling.
def tensor_wide_act_operands(
    A: torch.Tensor,
    As: torch.Tensor | None,
    num_experts: int | None = None,
    row_multiple: int = 1,
    activation_scale_ub: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Activation operands of the tensor-scale FP8 arms: ``(A, As, stride_m, stride_e)``.

    No ``As`` derives one scale per row here; a pre-quantized (E4M3) ``A`` brings its own. An
    ``As`` on a raw ``A`` is the CALIBRATED (static) scale that replaces them: one value, which
    ``A`` quantizes against here so every row reads the one entry, or — where the op routes —
    one per expert, which stays raw for the kernel to quantize each tile against its own expert
    — a gathered row serves several experts, so it has no single pre-quantized form until the
    routed rows are laid out (``quantize_routed_rows_per_expert``). ``num_experts`` ``None`` on
    the 2D op, whose rows belong to no expert. ``row_multiple`` pads the derived rows' storage
    (``fp8_act_quant_tensor_wide``). ``activation_scale_ub`` caps their dynamic amax.
    """
    if As is None:
        A, As = fp8_act_quant_tensor_wide(
            A, A.shape[-1], amax_upper_bound=activation_scale_ub, row_multiple=row_multiple
        )
        As = As.reshape(-1)
        return A, As, As.stride(0), 0
    if A.dtype == FP8_DTYPE:
        return A, As, As.stride(0), 0
    As = As.reshape(-1).float()
    assert As.numel() in (1, num_experts), (
        f"a calibrated activation scale is one value, or one per expert where the op routes; "
        f"got {As.numel()}"
    )
    if As.numel() == 1:
        return quantize_rows_static(A, As), As, 0, 0
    return A, As, 0, 1


@triton.jit
def _static_row_scale(S, ExpertStart, row, stride_s, NUM_EXPERTS: tl.constexpr, PER_EXPERT: tl.constexpr):
    """This row's calibrated scale. ``PER_EXPERT`` resolves it from the routing itself — the
    rows are expert-sorted, so the count of expert ends at or below the row IS its expert,
    which costs one ``(NUM_EXPERTS,)`` compare and spares the host materializing a per-row
    vector (a gather or a repeat_interleave, either of which outweighs this whole kernel)."""
    if PER_EXPERT:
        ends = tl.load(ExpertStart + 1 + tl.arange(0, NUM_EXPERTS))
        # an EP-sentinel row past every expert takes the last scale; nothing reads its output
        s = tl.load(S + tl.minimum(tl.sum((ends <= row).to(tl.int32)), NUM_EXPERTS - 1))
    else:
        s = tl.load(S + row * stride_s)
    return s


@triton.jit
def _fp8_quant_static_kernel(
    X, Y, S, GatherIdx, ExpertStart, stride_x_m, stride_y_m, stride_s, K,
    NUM_EXPERTS: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GATHER: tl.constexpr, PER_EXPERT: tl.constexpr,
):
    """One row tile against a PROVIDED scale: gather, divide in fp32, store E4M3. The fp32
    divide is the in-register static arm's arithmetic, so both forms of ``A`` round alike.
    ``stride_s`` 0 broadcasts one scale over every row; ``GATHER`` reads ``X`` through
    ``GatherIdx`` so the routed rows land laid out."""
    row = tl.program_id(0)
    src = tl.load(GatherIdx + row) if GATHER else row
    offs = tl.program_id(1) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    mask = offs < K
    x = tl.load(X + src.to(tl.int64) * stride_x_m + offs, mask=mask, other=0.0)
    y = (x.to(tl.float32) / _static_row_scale(S, ExpertStart, row, stride_s, NUM_EXPERTS, PER_EXPERT)).to(tl.float8e4nv)
    tl.store(Y + row * stride_y_m + offs, y, mask=mask)


def quantize_rows_static(A, scales, gather_idx=None, expert_start=None, num_experts=None):
    """``A`` as E4M3 against ``scales`` — one per row, one for all of them, or (with
    ``expert_start``) one per expert resolved from the routing. ``gather_idx`` lays the routed
    rows out in the same pass. One read and one write, where the torch spelling costs four."""
    assert A.stride(-1) == 1, "the static row quant reads K contiguous"
    rows = gather_idx.shape[0] if gather_idx is not None else A.shape[0]
    K = A.shape[-1]
    y = A.new_empty(rows, K, dtype=FP8_DTYPE)
    block_k = min(triton.next_power_of_2(K), 1024)
    with device_context(A.device):
        compile_time_only_triton_wrap(_fp8_quant_static_kernel)[(rows, triton.cdiv(K, block_k))](
            A, y, scales, gather_idx, expert_start,
            A.stride(0), y.stride(0), 0 if scales.numel() == 1 else scales.stride(0), K,
            NUM_EXPERTS=num_experts or 1, BLOCK_SIZE_K=block_k,
            GATHER=gather_idx is not None, PER_EXPERT=expert_start is not None,
        )
    return y


def quantize_routed_rows_per_expert(A, As, gather_idx, expert_start, num_experts):
    """The routed rows as E4M3, each quantized against its own expert's calibrated scale.

    A per-expert scale otherwise hands the kernel raw rows to quantize per tile, which reads
    ``A`` at bf16 width through every N-tile and holds the tile at ``RAW_ACT_MAX_BN``. Laying
    the rows out once buys back both, at the rounding the in-register arm produces, because a
    row's expert is known once the rows are expert-sorted — ``gather_idx`` orders them, or the
    intermediate already is. ``As`` is untouched: the kernel dequantizes per expert whichever
    form ``A`` arrives in.

    Returns the laid-out rows; the caller's ``gather_idx`` is spent by that layout.
    """
    return quantize_rows_static(A, As, gather_idx, expert_start, num_experts)


def fp8_act_quant_tensor_wide(
    x: torch.Tensor,
    block_size: int = 128,
    amax_upper_bound: float | None = None,
    row_multiple: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 + one ``amax / 448`` fp32 scale per ``block_size`` span (``block_size = K``: per
    token). ``x``'s rows may be strided (unit stride within a row). ``amax_upper_bound`` caps the amax (values
    past it saturate); ``None`` leaves it uncapped. ``row_multiple`` backs the E4M3 rows with storage rounded up
    to that many rows (left unwritten), so a TMA box over the padded extent reads in bounds."""
    assert x.stride(-1) == 1, f"x's rows must be contiguous, got strides {tuple(x.stride())}"
    assert x.shape[-1] % block_size == 0
    rows = x.view(-1, x.shape[-1])
    y = row_padded_empty(rows.shape[0], x.shape[-1], row_multiple, FP8_DTYPE, x.device).view(x.shape)
    grid = (triton.cdiv(x.numel(), block_size),)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)

    with device_context(x.device):
        compile_time_only_triton_wrap(_fp8_act_quant_kernel)[grid](
            rows,
            y,
            s,
            rows.stride(0),
            x.shape[-1] // block_size,
            BLOCK_SIZE=block_size,
            PADDED_BLOCK=triton.next_power_of_2(block_size),
            AMAX_UPPER_BOUND=amax_upper_bound,
            **pdl_launch_kwargs(),
        )

    return y, s




# ── int4 checkpoint repacks (load time) ──────────────────────────────────────


@triton.jit
def _repack_int4_kernel(
    QWeight,  # GPTQ (K // 8, N) words packed along K; AWQ (K, N // 8) words packed along N
    QZeros,  # (K // group, N // 8) words packed along N; read iff not SYMMETRIC
    Scales,  # (K // group, N)
    Perm,  # (K,) int32 act-order K order: output K row j is source row Perm[j]; None: identity (GPTQ only)
    W,  # out: (N, K // 8) uint32, eight K values per word in the interleaved order (interleave_int_words)
    Ws,  # out: (N, K // group)
    Wz,  # out: (N, K // group) uint8; written iff not SYMMETRIC
    N,
    stride_qw_0,
    stride_qw_1,
    stride_qz_0,
    stride_qz_1,
    stride_s_0,
    stride_s_1,
    ZEROS_OFFSET: tl.constexpr,
    AWQ: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUPS: tl.constexpr,  # scale groups per K tile
):
    """One ``[BLOCK_SIZE_K, BLOCK_SIZE_N]`` tile of a GPTQ or AWQ int4 linear into the GEMMs' layout, its groups'
    scales and zero points with it: each value's nibble out of the source word, then eight per word along K in the
    kernels' interleaved order."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    n_mask = (offs_n < N)[None, :]
    # AWQ interleaves a word's eight columns: column c sits in nibble c // 2 + 4 * (c % 2)
    n_shift = 4 * ((offs_n % 8) // 2 + 4 * (offs_n % 2)) if AWQ else 4 * (offs_n % 8)
    if AWQ:
        src = tl.load(QWeight + offs_k[:, None] * stride_qw_0 + (offs_n // 8)[None, :] * stride_qw_1, mask=n_mask)
        q = (src >> n_shift[None, :]) & 0xF
    else:
        src_k = offs_k if Perm is None else tl.load(Perm + offs_k)
        src = tl.load(QWeight + (src_k // 8)[:, None] * stride_qw_0 + offs_n[None, :] * stride_qw_1, mask=n_mask)
        q = (src >> (4 * (src_k % 8))[:, None]) & 0xF
    # the kernels' interleaved word (interleave_int_words): K offset j of the word in nibble j // 2 + 4 * (j % 2)
    j = tl.arange(0, 8)
    words = tl.sum(
        q.reshape(BLOCK_SIZE_K // 8, 8, BLOCK_SIZE_N) << (4 * (j // 2 + 4 * (j % 2)))[None, :, None], axis=1
    )
    offs_w = pid_k * (BLOCK_SIZE_K // 8) + tl.arange(0, BLOCK_SIZE_K // 8)
    K_WORDS = tl.num_programs(1) * (BLOCK_SIZE_K // 8)
    tl.store(W + offs_n[None, :] * K_WORDS + offs_w[:, None], words, mask=n_mask)

    offs_g = pid_k * GROUPS + tl.arange(0, GROUPS)
    n_groups = tl.num_programs(1) * GROUPS
    out_g = offs_n[None, :] * n_groups + offs_g[:, None]
    s = tl.load(Scales + offs_g[:, None] * stride_s_0 + offs_n[None, :] * stride_s_1, mask=n_mask)
    tl.store(Ws + out_g, s, mask=n_mask)
    if not SYMMETRIC:
        zw = tl.load(QZeros + offs_g[:, None] * stride_qz_0 + (offs_n // 8)[None, :] * stride_qz_1, mask=n_mask)
        z = (((zw >> n_shift[None, :]) & 0xF) + ZEROS_OFFSET) & 0xF  # a zero of 0 packs as 0xF under offset 1
        tl.store(Wz + out_g, z.to(tl.uint8), mask=n_mask)


def _repack_int4(qweight, qzeros, scales, K, awq, zeros_offset, symmetric, perm=None):
    groups, N = scales.shape
    group = K // groups
    assert group * groups == K and group % 8 == 0, f"K={K} is not a whole number of groups of {group}"
    block_k = group * max(1, 64 // group)  # a K tile of whole groups, at least 64 values
    assert K % block_k == 0, f"K={K} must be a multiple of {block_k}"
    W = torch.empty(N, K // 8, dtype=torch.uint32, device=qweight.device)
    Ws = torch.empty(N, groups, dtype=scales.dtype, device=scales.device)
    Wz = None if symmetric else torch.empty(N, groups, dtype=torch.uint8, device=qweight.device)
    grid = (triton.cdiv(N, 64), K // block_k)
    with device_context(qweight.device):
        _repack_int4_kernel[grid](
            qweight, None if symmetric else qzeros, scales, perm, W, Ws, Wz, N,
            qweight.stride(0), qweight.stride(1),
            *((0, 0) if symmetric else qzeros.stride()),
            scales.stride(0), scales.stride(1),
            ZEROS_OFFSET=zeros_offset, AWQ=awq, SYMMETRIC=symmetric,
            BLOCK_SIZE_N=64, BLOCK_SIZE_K=block_k, GROUPS=block_k // group,
        )
    return W, Ws, Wz


@triton.jit
def _permute_cols_kernel(X, Perm, Scale, QuantScale, UnitScale, Out, M, K, stride_x_m, stride_o_m, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, UNIT_GROUP: tl.constexpr):
    """Gather original input columns and optionally apply their AWQ scales in the same pass."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (rows < M)[:, None] & (cols < K)[None, :]
    src = cols if Perm is None else tl.load(Perm + cols, mask=cols < K, other=0)
    rows64 = rows[:, None].to(tl.int64)
    vals = tl.load(X + rows64 * stride_x_m + src[None, :], mask=mask, other=0)
    if Scale is not None:
        vals = (vals.to(tl.float32) * tl.load(Scale + src, mask=cols < K, other=0)[None, :]).to(X.dtype.element_ty)
    if QuantScale is not None:
        vals = quantize_calibrated_act(vals, tl.load(QuantScale), Out.dtype.element_ty)
    tl.store(Out + rows64 * stride_o_m + cols[None, :], vals, mask=mask)
    if UnitScale is not None:
        if tl.program_id(0) == 0:
            group = tl.program_id(1) * (BLOCK_K // UNIT_GROUP) + tl.arange(0, BLOCK_K // UNIT_GROUP)
            tl.store(UnitScale + group, 127 if UnitScale.dtype.element_ty == tl.uint8 else 1.0, mask=group < K // UNIT_GROUP)


def permute_act_cols(
    x: torch.Tensor, perm: torch.Tensor | None, pre_quant_scale: torch.Tensor | None = None,
    quant_scale: torch.Tensor | None = None,
    quant_dtype: torch.dtype = FP8_DTYPE,
    unit_scale: torch.Tensor | None = None,
    unit_group: int = MX_SCALE_GROUP_K,
    storage_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``x[..., perm]`` for an act-order weight's K order (``b_perm``), one launch: the activation the natural-order
    kernels (TMA arms included) then read. ``pre_quant_scale`` optionally multiplies the original columns
    before permutation, rounded in the activation dtype; ``perm=None`` only applies the transform."""
    K = x.shape[-1]
    if pre_quant_scale is not None:
        assert pre_quant_scale.shape == (K,) and pre_quant_scale.is_contiguous(), "pre_quant_scale must be contiguous (K,)"
        assert pre_quant_scale.device == x.device and pre_quant_scale.dtype == x.dtype, "pre_quant_scale must match the activation device and dtype"
    x2 = x.reshape(-1, K)
    out = torch.empty_like(x2, dtype=(storage_dtype or quant_dtype) if quant_scale is not None else x.dtype)
    M = x2.shape[0]
    # one row per program; a long column span per program once rows are many (B200, K=12288: M=1 1.4 vs 2.3us at
    # 512 vs 4096 columns, M=2048 73.0 vs 46.7us)
    block_k = min(512 if M <= 64 else 4096, triton.next_power_of_2(K))
    with device_context(x.device):
        compile_time_only_triton_wrap(_permute_cols_kernel)[(M, triton.cdiv(K, block_k))](
            x2, perm, pre_quant_scale, quant_scale, unit_scale, out, M, K, x2.stride(0), out.stride(0),
            BLOCK_M=1, BLOCK_K=block_k, UNIT_GROUP=unit_group, num_warps=4,
        )
    return out.view(x.shape)


@triton.jit
def _permute_routed_rows_kernel(
    X, Perm, Scale, QuantScale, UnitScale, GatherIdx, ExpertStart, ExpertIds, Out, K, stride_x_m, stride_perm_e, stride_scale_e, stride_quant_e, num_experts,
    NUM_EXPERTS_POW2: tl.constexpr, BLOCK_K: tl.constexpr,
    UNIT_GROUP: tl.constexpr,
):
    """``Out[s, j] = X[src(s), Perm[e(s), j]]`` over one ``BLOCK_K`` column span of routed row ``s``: ``src`` from
    ``GatherIdx`` (None: ``s``), the expert from ``ExpertIds`` (batched) or the sorted ranges of ``ExpertStart``
    (grouped). An EP-sentinel row (off-rank expert) is left unwritten, as the GEMM leaves its output."""
    s = tl.program_id(0)
    if UnitScale is not None:
        if s == 0:
            group = tl.program_id(1) * (BLOCK_K // UNIT_GROUP) + tl.arange(0, BLOCK_K // UNIT_GROUP)
            tl.store(UnitScale + group, 127 if UnitScale.dtype.element_ty == tl.uint8 else 1.0, mask=group < K // UNIT_GROUP)
    if ExpertIds is not None:
        e = tl.load(ExpertIds + s)
    else:
        ends = tl.load(ExpertStart + 1 + tl.arange(0, NUM_EXPERTS_POW2), mask=tl.arange(0, NUM_EXPERTS_POW2) < num_experts, other=2**31 - 1)
        e = tl.sum((ends <= s).to(tl.int32))
    if (e >= 0) & (e < num_experts):
        src = s if GatherIdx is None else tl.load(GatherIdx + s)
        cols = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = cols < K
        idx = cols if Perm is None else tl.load(Perm + e.to(tl.int64) * stride_perm_e + cols, mask=mask, other=0)
        vals = tl.load(X + src.to(tl.int64) * stride_x_m + idx, mask=mask)
        if Scale is not None:
            scale = tl.load(Scale + e.to(tl.int64) * stride_scale_e + idx, mask=mask, other=0)
            vals = (vals.to(tl.float32) * scale).to(X.dtype.element_ty)
        if QuantScale is not None:
            denom = tl.load(QuantScale + e * stride_quant_e)
            vals = quantize_calibrated_act(vals, denom, Out.dtype.element_ty)
        tl.store(Out + s.to(tl.int64) * K + cols, vals, mask=mask)


def permute_routed_rows(
    x: torch.Tensor, perm: torch.Tensor | None, num_rows: int, gather_idx: torch.Tensor | None = None,
    expert_start: torch.Tensor | None = None, expert_ids: torch.Tensor | None = None,
    pre_quant_scale: torch.Tensor | None = None, num_experts: int | None = None,
    quant_scale: torch.Tensor | None = None,
    quant_dtype: torch.dtype = FP8_DTYPE,
    unit_scale: torch.Tensor | None = None,
    unit_group: int = MX_SCALE_GROUP_K,
    storage_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """The ``num_rows`` routed activation rows, each read from ``x`` (through ``gather_idx``) in its expert's
    act-order K order, ``perm`` ``(E, K)``: ``out[s] = x[gather_idx[s], perm[expert(s)]]`` with the expert from
    ``expert_ids`` (batched) or ``expert_start``'s sorted ranges (grouped). One launch; the routed GEMMs then read
    contiguous natural-order rows. ``pre_quant_scale`` additionally multiplies each original column by
    its shared or per-expert AWQ input scale, rounded in the activation dtype; ``perm=None`` retains K order."""
    K = x.shape[-1]
    out = x.new_empty(num_rows, K, dtype=(storage_dtype or quant_dtype) if quant_scale is not None else x.dtype)
    if num_experts is None:
        num_experts = perm.shape[0]
    stride_scale_e = 0
    if pre_quant_scale is not None:
        assert pre_quant_scale.shape in ((K,), (1, K), (num_experts, K)) and pre_quant_scale.is_contiguous(), (
            "pre_quant_scale must be contiguous (K,), (1, K), or (E, K)"
        )
        assert pre_quant_scale.device == x.device and pre_quant_scale.dtype == x.dtype, "pre_quant_scale must match the activation device and dtype"
        stride_scale_e = K if pre_quant_scale.numel() > K else 0
    block_k = min(512, triton.next_power_of_2(K))
    num_warps = 4
    if num_rows > 64 and is_sm10x():
        block_k = min(4096 if perm is not None else 2048, triton.next_power_of_2(K))
        num_warps = 2 if block_k > 2048 else 1
    with device_context(x.device):
        compile_time_only_triton_wrap(_permute_routed_rows_kernel)[(num_rows, triton.cdiv(K, block_k))](
            x, perm, pre_quant_scale, quant_scale, unit_scale, gather_idx, expert_start, expert_ids, out, K, x.stride(0),
            perm.stride(0) if perm is not None else 0, stride_scale_e, global_scale_stride(quant_scale), num_experts,
            NUM_EXPERTS_POW2=triton.next_power_of_2(num_experts), BLOCK_K=block_k, UNIT_GROUP=unit_group, num_warps=num_warps,
        )
    return out


def prepare_fp8_mx_inputs(
    A, input_scale, scale_dtype, group, num_experts=1, num_rows=None,
    gather_idx=None, expert_start=None, expert_ids=None, pre_quant_scale=None, storage_dtype=FP8_DTYPE,
):
    """Calibrated tensor-FP8 inputs for FP4/MX loops: quantized rows, unit group scales, and calibration.

    Constant unit group scales preserve ordinary FP8 quantization (one calibrated scale per tensor/expert),
    rather than re-quantizing each block as MXFP8. Routed quantization and the sorted copy share one pass.
    BF16 stores FP8-rounded inputs exactly, as well as every finite E2M1 x E4M3 weight-scale product.
    This lets the wide-K weight-upcast GEMM use the same real operands without unit group scales.
    """
    assert input_scale is not None and A.dtype in (torch.bfloat16, torch.float16, torch.float32), (
        "FP4 with ordinary FP8 activations requires raw inputs and a calibrated input_scale"
    )
    assert storage_dtype in (FP8_DTYPE, torch.bfloat16)
    calibration = normalize_global_scale(input_scale, num_experts)
    K = A.shape[-1]
    unit = None if storage_dtype == torch.bfloat16 else torch.empty((1, K // group), device=A.device, dtype=scale_dtype)
    if num_rows is None:
        values = permute_act_cols(A.reshape(-1, K), None, pre_quant_scale, quant_scale=calibration, unit_scale=unit, unit_group=group, storage_dtype=storage_dtype)
    else:
        values = permute_routed_rows(
            A, None, num_rows, gather_idx, expert_start, expert_ids,
            num_experts=num_experts, quant_scale=calibration, unit_scale=unit, unit_group=group, pre_quant_scale=pre_quant_scale, storage_dtype=storage_dtype,
        )
    # One row, written by the preparation pass, broadcasts across tokens without a full scale grid.
    scales = None if storage_dtype == torch.bfloat16 else unit.expand(values.shape[0], K // group)
    return values.reshape(A.shape) if num_rows is None else values, scales, calibration


def interleave_int_words(words: torch.Tensor, bits: int) -> torch.Tensor:
    """Natural-order integer weight words (``(..., K // VPW)`` int32, ``VPW = 32 // bits`` unsigned values per word
    along K, the first lowest: compressed-tensors' ``pack-quantized`` layout) into the integer GEMMs' interleaved
    layout, uint32 words (the dtype marks the layout): within each word, the value at K offset ``j`` moves to slot
    ``j // 2 + (j % 2) * VPW // 2``, so one shift and one mask yield K-adjacent value pairs in a lane's two 16-bit
    halves (``int_word_pairs``). A loader-time conversion, done once per weight (the GPTQ / AWQ repacks emit this
    layout directly)."""
    assert words.dtype == torch.int32 and bits in (2, 4, 8), f"int32 words of 2/4/8-bit values, got {words.dtype}/{bits}"
    vpw = 32 // bits
    mask = (1 << bits) - 1
    j = torch.arange(vpw, device=words.device, dtype=torch.int32)
    values = (words.unsqueeze(-1) >> (bits * j)) & mask  # (..., words, vpw), natural slot j
    slot = j // 2 + (j % 2) * (vpw // 2)
    return (values << (bits * slot)).sum(-1, dtype=torch.int32).view(torch.uint32)


@triton.jit
def _dequantize_int_kernel(
    W,  # (rows, K // VPW) uint32 words, interleaved (interleave_int_words)
    Ws,  # (rows, K // SCALE_GROUP_K) float group scales
    Wz,  # (rows, K // SCALE_GROUP_K) integer zero points; None for the symmetric 2 ** (BITS - 1)
    Out,  # (rows, K)
    rows,
    K_WORDS,
    BITS: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """One ``[BLOCK_SIZE_R, BLOCK_SIZE_W]`` word tile back to its ``[BLOCK_SIZE_R, BLOCK_SIZE_W * VPW]`` values:
    ``((q - zero) * scale)`` rounded in the scale dtype, then cast to ``Out``'s."""
    VPW: tl.constexpr = 32 // BITS
    HALF: tl.constexpr = VPW // 2
    MASK: tl.constexpr = (1 << BITS) - 1
    offs_r = tl.program_id(0) * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    offs_w = tl.program_id(1) * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    r_mask = (offs_r < rows)[:, None]
    w = tl.load(W + offs_r[:, None].to(tl.int64) * K_WORDS + offs_w[None, :], mask=r_mask & (offs_w < K_WORDS)[None, :])
    shifts = (tl.arange(0, HALF) * BITS)[None, None, :]
    # slot s < HALF holds K offset 2s, slot s + HALF offset 2s + 1
    lo = (w[:, :, None] >> shifts) & MASK
    hi = (w[:, :, None] >> (shifts + HALF * BITS)) & MASK
    q = tl.join(lo, hi).reshape(BLOCK_SIZE_R, BLOCK_SIZE_W * VPW).to(tl.int32)
    offs_k = tl.program_id(1) * BLOCK_SIZE_W * VPW + tl.arange(0, BLOCK_SIZE_W * VPW)
    mask = r_mask & (offs_k < K_WORDS * VPW)[None, :]
    groups = K_WORDS * VPW // SCALE_GROUP_K
    g_ptrs = offs_r[:, None].to(tl.int64) * groups + (offs_k // SCALE_GROUP_K)[None, :]
    scale = tl.load(Ws + g_ptrs, mask=mask)
    zero = (1 << (BITS - 1)) if Wz is None else tl.load(Wz + g_ptrs, mask=mask).to(tl.int32)
    v = (q - zero).to(scale.dtype) * scale
    out_ptrs = Out + offs_r[:, None].to(tl.int64) * (K_WORDS * VPW) + offs_k[None, :]
    tl.store(out_ptrs, v.to(Out.dtype.element_ty), mask=mask)


def dequantize_int_weight(
    words: torch.Tensor, scales: torch.Tensor, K: int, zeros: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """An integer weight in the GEMMs' interleaved layout (``(..., N, K // VPW)`` uint32, ``interleave_int_words``)
    back to ``(..., N, K)`` the way the GEMMs dequantize it: ``(q - zero)`` in the scale dtype times the per-group
    scale, rounded once, then cast to ``dtype`` (``None``: the scale dtype). The width follows from ``K`` and the
    word count; ``zeros`` ``None`` is the symmetric zero point ``2 ** (bits - 1)``. One fused launch, used by the
    large-prefill path that dequantizes once and runs the unquantized GEMM."""
    bits = int_weight_bits(K, words)
    k_words = words.shape[-1]
    group = K // scales.shape[-1]
    rows = words.numel() // k_words
    assert group * scales.shape[-1] == K and group % (32 // bits) == 0, f"{scales.shape[-1]} groups cannot split K={K}"
    assert zeros is None or zeros.shape == scales.shape, "zero points are shaped like the scales"
    out = torch.empty(*words.shape[:-1], K, device=words.device, dtype=dtype or scales.dtype)

    grid = (triton.cdiv(rows, 32), triton.cdiv(k_words, 16))
    with device_context(words.device):
        compile_time_only_triton_wrap(_dequantize_int_kernel)[grid](
            words.contiguous(), scales.contiguous(), None if zeros is None else zeros.contiguous(), out,
            rows, k_words, BITS=bits, SCALE_GROUP_K=group, BLOCK_SIZE_R=32, BLOCK_SIZE_W=16,
        )
    return out


def repack_gptq_int4(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor | None = None,
    zeros_offset: int = 1,
    symmetric: bool = False,
) -> tuple[torch.Tensor, ...]:
    """A GPTQ int4 linear (``qweight`` ``(K // 8, N)``, ``qzeros`` ``(K // group, N // 8)``, ``scales``
    ``(K // group, N)``) as the int4 GEMMs' ``(weight, scale, zero)``: ``(N, K // 8)`` uint32, ``(N, K // group)``,
    ``(N, K // group)`` uint8 or ``None`` for a ``symmetric`` checkpoint (``sym=True``: every zero point 8).
    ``zeros_offset`` is what the checkpoint subtracted from each zero point before packing: 1 in the original GPTQ
    format, 0 in ``"gptq_v2"``. An act-order ``g_idx`` (``desc_act=True``: groups scattered across K) sorts K by
    group, so the weight holds ordinary contiguous groups, and returns that order too, ``(weight, scale, zero,
    perm)``: ``perm`` ``(K,)`` int32 is the ops' ``b_perm`` (the GEMM contracts ``A[..., perm]``). An identity
    ``g_idx`` (or none) keeps the three-tensor form."""
    K = qweight.shape[0] * 8
    group = K // scales.shape[0]
    natural = torch.arange(K, device=qweight.device, dtype=torch.int32) // group
    if g_idx is None or torch.equal(g_idx.to(torch.int32), natural):
        return _repack_int4(qweight, qzeros, scales, K, False, zeros_offset, symmetric)
    perm = torch.argsort(g_idx, stable=True).to(torch.int32)
    if not torch.equal(g_idx.to(torch.int32)[perm.long()], natural):
        raise ValueError(f"g_idx does not assign exactly {group} K rows to each of its {K // group} groups")
    return (*_repack_int4(qweight, qzeros, scales, K, False, zeros_offset, symmetric, perm), perm)


def fold_act_order_into_gate_up(
    gate_up: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_zero: torch.Tensor | None,
    gate_up_bias: torch.Tensor | None,
    down_perm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Fold an act-order down projection's K order into the gate|up projection that feeds it, at load: the down
    contracts its input in ``down_perm`` order, so gate|up output pair ``j`` (rows ``2j`` gate, ``2j + 1`` up,
    interleaved) takes rows ``2 * down_perm[j]`` and ``2 * down_perm[j] + 1``, with their scales, zero points and
    bias. The GLU output then comes out already in the down's order and the down runs with ``b_perm=None``.
    ``(E, 2I, ...)`` stacks with ``down_perm`` ``(E, I)``, or one linear: ``(2I, ...)`` with ``(I,)``."""
    rows = torch.stack([2 * down_perm, 2 * down_perm + 1], dim=-1).flatten(-2).long()  # (..., 2I)

    def take_rows(t):
        return None if t is None else torch.gather(t, rows.ndim - 1, rows[..., None].expand(*rows.shape, t.shape[-1]))

    bias = None if gate_up_bias is None else torch.gather(gate_up_bias, rows.ndim - 1, rows)
    return take_rows(gate_up), take_rows(gate_up_scale), take_rows(gate_up_zero), bias


def repack_awq_int4(
    qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """An AWQ (GEMM) int4 linear (``qweight`` ``(K, N // 8)``, ``qzeros`` ``(K // group, N // 8)``, both packed along N
    in AWQ's interleaved nibble order, ``scales`` ``(K // group, N)``) as the int4 GEMMs' ``(weight, scale, zero)``."""
    return _repack_int4(qweight, qzeros, scales, qweight.shape[0], True, 0, False)


@triton.jit
def _pack_interleaved_int_tile(q, BITS: tl.constexpr):
    VPW: tl.constexpr = 32 // BITS
    j = tl.arange(0, VPW)
    slot = j // 2 + (j % 2) * (VPW // 2)
    return tl.sum(q << (BITS * slot)[None, None, :], axis=2)


@triton.jit
def _repack_modelopt_int_kernel(
    QWeight, W, N, K, stride_e, stride_n, stride_k,
    BITS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr,
):
    """Signed ModelOpt values to offset-binary, K-interleaved words without an unpacked weight bank."""
    VPW: tl.constexpr = 32 // BITS
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    w = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    j = tl.arange(0, VPW)
    k = w[:, None] * VPW + j[None, :]
    if BITS == 4:
        src_n = n // 2
    else:
        src_n = n
    raw = tl.load(
        QWeight + tl.program_id(2).to(tl.int64) * stride_e
        + src_n[:, None, None].to(tl.int64) * stride_n + k[None, :, :] * stride_k,
        mask=(n < N)[:, None, None] & (k < K)[None, :, :], other=0,
    ).to(tl.uint32)
    if BITS == 4:
        raw >>= (4 * (n % 2))[:, None, None]
    q = (raw & ((1 << BITS) - 1)) ^ (1 << (BITS - 1))
    packed = _pack_interleaved_int_tile(q, BITS)
    tl.store(
        W + (tl.program_id(2).to(tl.int64) * N + n[:, None]) * (K // VPW) + w[None, :],
        packed, mask=(n < N)[:, None] & (w < K // VPW)[None, :],
    )


def _repack_modelopt_int(qweight, scales, bits):
    expected_dtype = torch.uint8 if bits == 4 else torch.int8
    assert qweight.dtype == expected_dtype and qweight.ndim in (2, 3), (
        f"ModelOpt INT{bits} weights must be 2D or 3D {expected_dtype}, got {qweight.dtype}/{qweight.ndim}D"
    )
    assert scales.device == qweight.device and scales.dtype in (torch.bfloat16, torch.float16, torch.float32), (
        "ModelOpt weight scales must be bf16/fp16/fp32 on the weight device"
    )
    N = qweight.shape[-2] * (2 if bits == 4 else 1)
    K = qweight.shape[-1]
    leading = qweight.shape[:-2]
    if bits == 8 and scales.shape == (*leading, N):
        scales = scales.unsqueeze(-1)
    assert scales.ndim == qweight.ndim and scales.shape[:-1] == (*leading, N), (
        f"ModelOpt weight scales must have shape {(*leading, N, 'K/group')}, got {tuple(scales.shape)}"
    )
    groups = scales.shape[-1]
    vpw = 32 // bits
    assert groups > 0 and K % groups == 0 and (K // groups) % vpw == 0, (
        f"K={K} must contain whole scale groups aligned to {vpw} values"
    )
    assert bits == 4 or groups == 1, "ModelOpt INT8 weight-only uses one scale per output channel"
    W = torch.empty(*leading, N, K // vpw, dtype=torch.uint32, device=qweight.device)
    experts = qweight.shape[0] if qweight.ndim == 3 else 1
    with device_context(qweight.device):
        _repack_modelopt_int_kernel[(triton.cdiv(N, 16), triton.cdiv(K // vpw, 16), experts)](
            qweight, W, N, K, qweight.stride(0) if leading else 0, *qweight.stride()[-2:],
            BITS=bits, BLOCK_N=16, BLOCK_W=16, num_warps=4,
        )
    return W, scales.contiguous(), None


def repack_modelopt_int4(
    qweight: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """ModelOpt ``INT4_AWQ`` / blockwise weight-only export to ``(weight, scale, None)`` for the integer GEMMs.

    ``qweight`` is uint8 ``(..., N // 2, K)``: output row ``2n`` is the low signed two's-complement nibble,
    row ``2n+1`` the high nibble. ``scales`` is ``(..., N, K // group)``. Returns K-interleaved uint32
    words ``(..., N, K // 8)`` with implicit zero point 8; scales keep their dtype and calibrated values.
    Pass the checkpoint's ``pre_quant_scale``, when present, to the GEMM as ``a_pre_quant_scale`` (or fuse it
    upstream), exactly as in the exported model. This conversion neither applies it nor requantizes weights.
    ``W4A8_AWQ`` uses the same packing, plus ``activation_format='fp8'``, ``As=input_scale``, and
    ``b_global_scale=weight_scale_2`` on the GEMM; a repack alone does not apply that recipe.
    """
    return _repack_modelopt_int(qweight, scales, 4)


def repack_modelopt_int8(
    qweight: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """ModelOpt ``INT8_WEIGHT_ONLY`` signed int8 ``(..., N, K)`` to the integer GEMMs' K-interleaved uint32
    words, implicit zero point 128. Per-channel ``scales`` is ``(..., N)`` or ``(..., N, 1)``; returned as
    ``(..., N, 1)`` without changing its dtype or values. This is W8A16, not INT8 SmoothQuant W8A8.
    """
    return _repack_modelopt_int(qweight, scales, 8)


@triton.jit
def _repack_ggml_int_kernel(
    Blocks, W, Scales, ROWS, GROUPS, BITS: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_W: tl.constexpr,
):
    VPW: tl.constexpr = 32 // BITS
    BLOCK_BYTES: tl.constexpr = 2 + 32 * BITS // 8
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    w = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    k = w[:, None] * VPW + tl.arange(0, VPW)[None, :]
    block = r[:, None, None].to(tl.int64) * GROUPS + k[None, :, :] // 32
    qoffset = k % 16 if BITS == 4 else k % 32
    raw = tl.load(
        Blocks + block * BLOCK_BYTES + 2 + qoffset[None, :, :],
        mask=(r < ROWS)[:, None, None] & (k < GROUPS * 32)[None, :, :], other=0,
    ).to(tl.uint32)
    if BITS == 4:
        q = (raw >> (4 * ((k % 32) // 16))[None, :, :]) & 15
    else:
        q = raw ^ 128  # signed byte to offset-binary
    words = _pack_interleaved_int_tile(q, BITS)
    tl.store(W + r[:, None].to(tl.int64) * (GROUPS * 32 // VPW) + w[None, :], words,
             mask=(r < ROWS)[:, None] & (w < GROUPS * 32 // VPW)[None, :])
    group = w // (32 // VPW)
    scale_ptr = Blocks + (r[:, None].to(tl.int64) * GROUPS + group[None, :]) * BLOCK_BYTES
    mask = (r < ROWS)[:, None] & (group < GROUPS)[None, :]
    lo = tl.load(scale_ptr, mask=mask, other=0).to(tl.uint16)
    hi = tl.load(scale_ptr + 1, mask=mask, other=0).to(tl.uint16)
    d = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    tl.store(Scales + r[:, None].to(tl.int64) * GROUPS + group[None, :], d,
             mask=mask & (w % (32 // VPW) == 0)[None, :])


def repack_ggml_int(packed: torch.Tensor, quant_type: str) -> tuple[torch.Tensor, torch.Tensor, None]:
    """GGML ``Q4_0`` / ``Q8_0`` payload blocks to the existing integer GEMMs, without a GGUF parser.

    ``packed`` is uint8 ``(..., N, K // 32, block_bytes)`` (18 for Q4_0, 34 for Q8_0), in canonical
    little-endian GGML block order. A GGUF loader reshapes a tensor's raw payload to this shape first.
    Q4_0 stores the first 16 values in the low nibbles, the next 16 in the high ones; its zero is 8.
    Q8_0 stores signed int8 values; its zero becomes 128. Both use one FP16 scale per 32 values.
    Returns ``(interleaved_uint32, fp32_scales, None)``. Scales expand to FP32 because GGML's
    reference reconstructs weights in FP32 before the GEMM casts them to its activation dtype.
    No requantization, dense weight bank, or codebook-format approximation is performed.
    """
    assert quant_type in ("Q4_0", "Q8_0"), "only affine symmetric GGML Q4_0 and Q8_0 blocks are supported"
    bits = 4 if quant_type == "Q4_0" else 8
    block_bytes = 2 + 4 * bits
    assert packed.dtype == torch.uint8 and packed.ndim in (3, 4) and packed.shape[-1] == block_bytes, (
        f"{quant_type} needs uint8 (..., N, K // 32, {block_bytes}) payload blocks"
    )
    shape = packed.shape[:-2]
    groups = packed.shape[-2]
    rows = packed.numel() // (groups * block_bytes)
    W = torch.empty(*shape, groups * bits, device=packed.device, dtype=torch.uint32)
    scales = torch.empty(*shape, groups, device=packed.device, dtype=torch.float32)
    with device_context(packed.device):
        _repack_ggml_int_kernel[(triton.cdiv(rows, 16), triton.cdiv(groups * bits, 16))](
            packed.contiguous(), W, scales, rows, groups, BITS=bits, BLOCK_R=16, BLOCK_W=16, num_warps=4,
        )
    return W, scales, None
