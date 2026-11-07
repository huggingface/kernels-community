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

import triton
import triton.language as tl

from .quant import e2m1_cols_to_bf16, e2m1_cols_to_e4m3, e2m1_cols_to_f32, e2m1_to_bf16, e2m1_to_e4m3, e2m1_to_f32
from .loading.scales import decode_group_scale
from .compat import is_nvidia, is_sm10x
from .ptx import (
    add_mul_x2, add_x2, dp4a, e2m1x8_to_f16x2, e4m3x4_to_f16x2, f16x2_dot4_f32, int4_nibbles_to_16bit_lanes, int8_bytes_to_f16_lanes, int_words_fma_rows,
    nibbles_to_table_bytes, offset_int8_to_bf16_lanes, round_f32_to_bf16_f32,
)


@triton.jit
def mx_dot_scaled(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    SWAP_AB: tl.constexpr = False,
    BLOCK_SIZE_K: tl.constexpr = 0,
    SCALE_GROUP_K: tl.constexpr = 0,
):
    """MX 'dot_scaled' step: the tcgen05 scaled MMA folds the group scales into the tensor core;
    each operand's format is its tile's dtype (``uint8`` = packed E2M1, E4M3, or bf16 for the
    weight-only unscaled lhs). fp4 on BOTH operands lowers to the native ``kind::mxf4`` MMA (2x the
    fp8 rate; native iff the M operand is 128 — same gate as mxf8f6f4). Plain: ``a`` [BM, BK] x
    ``w`` [BK, BN] (the caller pre-shapes ``w``/``w_scale``, e.g. ``tl.trans(gu)``). ``SWAP_AB``
    (decode): the weight ``w`` [BN, BK] is the lhs (output rows in M) and the single [BK] token is
    padded to the N=16 rhs (col 0 real — 16 is Triton's tcgen05-selection gate, N=8 drops to the
    bf16-upcast fallback, bare-1 was 1.83x); ``acc`` is the persistent ``[BN, MMA_N_ATOM]``
    accumulator the caller takes column 0 of (never a fresh per-step init, which trips the sm_100
    accumulator-init pass). A packed ``a`` stays packed (the E4M3-scaled mxf4nvf4 kind is fp4 x fp4
    only) and the token's group scale broadcasts to the rhs columns."""
    a_fmt: tl.constexpr = (
        "e2m1" if a.dtype == tl.uint8
        else ("e4m3" if a.dtype == tl.float8e4nv else "bf16")
    )
    w_fmt: tl.constexpr = "e2m1" if w.dtype == tl.uint8 else "e4m3"
    if SWAP_AB:
        rhs = swap_pad_rhs(a, BLOCK_SIZE_K // 2 if a.dtype == tl.uint8 else BLOCK_SIZE_K)
        if a_scale.dtype == tl.uint8:  # UE8M0 broadcast via the zero-add idiom
            asc = tl.trans(
                a_scale[:, None]
                + tl.zeros((BLOCK_SIZE_K // SCALE_GROUP_K, MMA_N_ATOM), tl.uint8)
            )
        else:  # E4M3 (NVFP4) — no fp8 arithmetic; materialize the broadcast directly
            asc = tl.trans(
                tl.broadcast_to(
                    a_scale[:, None], (BLOCK_SIZE_K // SCALE_GROUP_K, MMA_N_ATOM)
                )
            )
        acc = tl.dot_scaled(w, w_scale, w_fmt, rhs, asc, a_fmt, acc)
    else:
        acc = tl.dot_scaled(a, a_scale, a_fmt, w, w_scale, w_fmt, acc)
    return acc


@triton.jit
def mx_dot_rescale(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    SWAP_AB: tl.constexpr = False,
    BLOCK_SIZE_K: tl.constexpr = 0,
):
    """MX 'dot' step (BK == one scale group): fp8 ``tl.dot`` on E4M3-decoded operands + per-group
    software rescale (both UE8M0 scales decoded here), accumulating into ``acc``. Plain: ``a``
    [BM, BK] x ``w`` [BK, BN] with the weight's per-column scale transposed onto the product (the
    batched gate_up kernel passes the INTERLEAVED gate|up 2*BN tile — per-column independence keeps
    that bit-exact). ``SWAP_AB`` (decode): the weight ``w`` [ROWS, BK] is the lhs (E2M1 column-unpacked,
    K order low nibble first) and the [BK] token padded to the N=16 atom is the rhs — the
    well-shaped fp8 MMA at M=1 (M quantizes to 64/128, N only to 8, so weight rows fill the big
    atom); the weight's per-output-row scale broadcasts down the acc columns and the token's single
    group scale is a scalar; ``acc`` is the persistent ``[ROWS, MMA_N_ATOM]`` accumulator."""
    aq = e2m1_cols_to_e4m3(a) if a.dtype == tl.uint8 else a
    a_s = decode_group_scale(a_scale)
    w_s = decode_group_scale(w_scale)
    if SWAP_AB:
        wq = e2m1_cols_to_e4m3(w) if w.dtype == tl.uint8 else w
        acc = acc + tl.dot(wq, swap_pad_rhs(aq, BLOCK_SIZE_K)) * w_s * a_s
    else:
        wq = e2m1_to_e4m3(w) if w.dtype == tl.uint8 else w
        acc = acc + tl.dot(aq, wq) * a_s * tl.trans(w_s)
    return acc


@triton.jit
def mx_weight_upcast(w, w_scale, BLOCK_SIZE_K: tl.constexpr, N: tl.constexpr, SCALE_GROUP_K: tl.constexpr,
                     OUT_DTYPE: tl.constexpr = tl.bfloat16, ROWS_MAJOR: tl.constexpr = False):
    """Upcast one MXFP4/MXFP8 weight K-tile ``[BK, N]`` to ``OUT_DTYPE`` for the weight-only path: unpack
    E2M1 -> bf16 directly (fp8 passes through a cast) and apply each ``SCALE_GROUP_K``-row K-group's
    UE8M0 group scale. ``ROWS_MAJOR`` decodes and returns ``[N, BK]`` directly for the
    computed MMA lhs; the default returns ``[BK, N]``. ``w_scale`` is ``[N, BK // SCALE_GROUP_K]`` (per-N-row, per-group); transposed
    and broadcast across each group's rows to ``[BK, N]``. Unlike the ``dot`` arm (BK == group, scale
    folded onto the [M,N] product) this dequantizes IN the tile, so BK spans any number of groups — a
    full-BK bf16 ``tl.dot`` (the matmul_ogs format: fp4 weight, bf16 acts).

    The scale is applied as a plain bf16 multiply, NOT a hand-rolled exponent-add: the UE8M0 scale is
    a power of two so a ``bf16 * 2^k`` is already an exact exponent shift (a cheap FMA), and folding it
    into the bits as an int add + zero-mask ``tl.where`` measured ~13% SLOWER (gate_up 4.68->5.29ms) —
    the mask + uint16<->int32 bitcast chain costs more than the multiply it replaces."""
    ng: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
    if ROWS_MAJOR:
        wq = e2m1_cols_to_bf16(w) if w.dtype == tl.uint8 else w.to(tl.bfloat16)
        ws = decode_group_scale(w_scale).to(tl.bfloat16)[:, :, None]
        out = (wq.reshape(N, ng, SCALE_GROUP_K) * ws).reshape(N, BLOCK_SIZE_K)
    else:
        wq = e2m1_to_bf16(w) if w.dtype == tl.uint8 else w.to(tl.bfloat16)
        ws = tl.trans(decode_group_scale(w_scale)).to(tl.bfloat16)[:, None, :]
        out = (wq.reshape(ng, SCALE_GROUP_K, N) * ws).reshape(BLOCK_SIZE_K, N)
    # Decode and group scaling round in bf16 before the activation-dtype cast, in either orientation.
    return out.to(OUT_DTYPE)


@triton.jit
def mx_weight_only_compute(
    acc,
    a,
    w,
    w_scale,
    COMPUTE_MODE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    N_WIDTH: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
):
    """Weight-only MMA step with raw activations and MXFP4/MXFP8 weights.

    ``dot_scaled`` delegates weight upcasting and group scaling to Triton; an unscaled
    BF16 activation does not imply native block-scaled MMA. ``dot`` explicitly upcasts
    the weight tile before ``tl.dot``; the tuner chooses between these lowerings.
    ``SWAP_AB`` puts weight rows on the MMA lhs; scalar decode instead reduces
    each weight row against one token. Callers restore output orientation after the K loop.
    """
    if SWAP_AB and COMPUTE_MODE == "scalar":
        acc = mx_weight_only_scalar_swapped(
            acc, tl.reshape(a, (BLOCK_SIZE_K,)), w, w_scale,
            N_WIDTH, BLOCK_SIZE_K, SCALE_GROUP_K,
        )
    elif COMPUTE_MODE == "dot_scaled":
        acc = mx_dot_scaled(acc, a, None, w, w_scale)
    elif SWAP_AB:
        lhs = mx_weight_upcast(w, w_scale, BLOCK_SIZE_K, N_WIDTH, SCALE_GROUP_K, a.dtype, ROWS_MAJOR=True)
        acc = tl.dot(lhs, tl.trans(a), acc)
    else:
        acc = acc + tl.dot(
            a, mx_weight_upcast(w, w_scale, BLOCK_SIZE_K, N_WIDTH, SCALE_GROUP_K, a.dtype)
        )
    return acc


@triton.jit
def mx_weight_only_scalar_swapped(
    acc,
    a,
    w,
    w_scale,
    ROWS_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
):
    """Swapped weight-only reduce: CUDA-core FMA GEVM against RAW bf16 activations (no act
    scale — the weight-only counterpart of ``mx_scalar_reduce_swapped``). Weight
    ``w`` is output-rows-major ``[ROWS_W, BK]``, ``a`` the ``[BK]`` token; the UE8M0 group
    scale factors out of the inner sum, so it costs one multiply per group instead of per
    element. Returns ``acc + [1, ROWS_W]``.

    This is the decode structure for W4A16: with no MMA there is no M->16 pad to waste, and
    the loop runs at the dequantized weight stream's own rate (gpt-oss gate_up 32.0 -> 13.7us,
    down 19.8 -> 9.1us, against a 12.3us/6us dequant-inclusive load floor)."""
    NG: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
    wq = e2m1_cols_to_f32(w) if w.dtype == tl.uint8 else w.to(tl.float32)  # [ROWS_W, BK]
    prod = wq * a.to(tl.float32)[None, :]
    grp = tl.sum(tl.reshape(prod, (ROWS_W, NG, SCALE_GROUP_K)), axis=2)  # [ROWS_W, NG]
    return acc + tl.reshape(
        tl.sum(grp * decode_group_scale(w_scale), axis=1), (1, ROWS_W)
    )


# PTX inline asm (round_f32_to_bf16_f32, the packed zero-point subtract) on NVIDIA; elsewhere plain Triton
_PTX = tl.constexpr(is_nvidia())
# Keep architecture-specific integer fast paths on the measured sm_10x target.
SM10X = tl.constexpr(is_sm10x())


@triton.jit
def int_word_pairs(w, BITS: tl.constexpr, K_MAJOR: tl.constexpr):
    """The value pairs of uint32 words in the kernels' INTERLEAVED layout (``interleave_int_words``): of a word's
    ``32 // BITS`` values, position ``s`` holds the value at K offset ``2s`` and position ``s + 16 // BITS`` the one
    at ``2s + 1``, so one shift and one mask put K-adjacent values in the low and high 16-bit halves of a lane,
    the register pair an MMA operand holds. ``w`` ``[R, C]`` words along C (``K_MAJOR``: along R); returns the raw
    pairs ``[R, C, 16 // BITS]`` (``K_MAJOR``: ``[R, 16 // BITS, C]``), pair ``s`` = K offsets ``(2s, 2s + 1)``."""
    HALF: tl.constexpr = 16 // BITS
    MASK: tl.constexpr = (1 << BITS) - 1
    shifts = tl.arange(0, HALF) * BITS
    if K_MAJOR:
        pairs = (w[:, None, :] >> shifts[None, :, None]) & (MASK | (MASK << 16))
    else:
        pairs = (w[:, :, None] >> shifts[None, None, :]) & (MASK | (MASK << 16))
    return pairs


@triton.jit
def int8_words_to_int8(w, K_MAJOR: tl.constexpr):
    """Offset-binary interleaved words to signed INT8, in natural K order."""
    pairs = int_word_pairs(w ^ 0x80808080, 8, K_MAJOR)
    v = tl.join((pairs & 0xFF).to(tl.int8), (pairs >> 16).to(tl.int8))
    if K_MAJOR:
        out = tl.trans(v, 0, 1, 3, 2).reshape(4 * w.shape[0], w.shape[1])
    else:
        out = v.reshape(w.shape[0], 4 * w.shape[1])
    return out


@triton.jit
def int_words_to_offset(w, zero_words, BITS: tl.constexpr, K_MAJOR: tl.constexpr, WHOLE_WORD: tl.constexpr = False,
                        FP16: tl.constexpr = False, scale_words=None):
    """Interleaved uint32 words (``int_word_pairs``) to their values as ``OFFSET + q``, exactly, with no integer-to-float
    conversion: a value ORed under the offset's exponent bits IS that float, bf16 ``128 + q`` for 2- and 4-bit values
    (q < 128), fp16 ``1024 + q`` for FP16 scales or 8-bit values (bf16 cannot hold 128 + 255). ``zero_words`` (``None``: no
    zero point) is one word per weight word ``[R, C]`` carrying its group's zero point ``z`` in both 16-bit lanes
    (``int_zero_words``): on CUDA the float ``-(OFFSET + z)``, added to each pair by one packed FMA, so the values
    are ``q - z``; elsewhere the integer lift ``2 ** BITS - z`` added before the OR, so they are
    ``OFFSET + 2 ** BITS + q - z``. Either way the zero point costs no per-value float op of its own. ``scale_words``
    (CUDA, with a zero point) multiplies the pairs by their group scale in the same packed asm, ``(q - z) * s`` rounded
    once as ``int_dequant`` rounds it: unpacked lanes would leave the multiply to scalar ``mul.bf16``. Output-rows
    ``[R, C]`` -> ``[R, (32 // BITS) C]`` (``K_MAJOR``: ``[R, C]`` -> ``[(32 // BITS) R, C]``), natural K order."""
    MAGIC: tl.constexpr = 0x64006400 if FP16 or BITS == 8 else 0x43004300
    if _PTX and BITS == 8:
        # Keep both byte pairs with their word before orienting the MMA tile.
        lo, hi = int8_bytes_to_f16_lanes(w)
        pairs = tl.join(lo, hi)
        if K_MAJOR:
            pairs = tl.trans(pairs, 0, 2, 1)
    elif WHOLE_WORD:
        # Keep all four pairs in the word's lane.
        p0, p1, p2, p3 = int4_nibbles_to_16bit_lanes(w, "0x64006400" if FP16 else "0x43004300")
        pairs = tl.join(tl.join(p0, p2), tl.join(p1, p3)).reshape(w.shape[0], w.shape[1], 4)
        if K_MAJOR:
            pairs = tl.trans(pairs, 0, 2, 1)
    else:
        pairs = int_word_pairs(w, BITS, K_MAJOR)
    if zero_words is not None:
        zero_words = tl.broadcast_to(zero_words[:, None, :] if K_MAJOR else zero_words[:, :, None], pairs.shape)
    if zero_words is not None and not _PTX:
        pairs += zero_words
    if not WHOLE_WORD:
        pairs = pairs | MAGIC
    if zero_words is not None and _PTX and scale_words is not None:
        # (pair - (OFFSET + z)) * s: the exact difference, then the product's one rounding
        scale_words = tl.broadcast_to(scale_words[:, None, :] if K_MAJOR else scale_words[:, :, None], pairs.shape)
        pairs = add_mul_x2(pairs, zero_words, scale_words, FP16 or BITS == 8)
    elif zero_words is not None and _PTX:
        pairs = add_x2(pairs, zero_words, FP16 or BITS == 8)  # pair - (OFFSET + z), exact
    ODT: tl.constexpr = tl.float16 if FP16 or BITS == 8 else tl.bfloat16
    lo = (pairs & 0xFFFF).to(tl.uint16).to(ODT, bitcast=True)
    hi = (pairs >> 16).to(tl.uint16).to(ODT, bitcast=True)
    v = tl.join(lo, hi)  # [.., pair, half] (K_MAJOR: [R, pair, C, half])
    if K_MAJOR:
        out = tl.trans(v, 0, 1, 3, 2).reshape((32 // BITS) * w.shape[0], w.shape[1])
    else:
        out = v.reshape(w.shape[0], (32 // BITS) * w.shape[1])
    return out


@triton.jit
def int8_words_to_bf16(w, K_MAJOR: tl.constexpr):
    """Interleaved INT8 words to exact bf16 ``q - 128``.
    ``K_MAJOR`` maps ``[K // 4, N]`` to ``[K, N]``; otherwise ``[N, K // 4]`` to ``[N, K]``.
    Return both K-adjacent pairs together to keep a whole word per lane and enable ldmatrix shared loads."""
    p0, p1 = offset_int8_to_bf16_lanes(w)  # exact, no fp16 -> fp32 -> bf16 conversion
    pairs = tl.join(p0, p1)
    if K_MAJOR:
        pairs = tl.trans(pairs, 0, 2, 1)
    lo = (pairs & 0xffff).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    hi = (pairs >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    v = tl.join(lo, hi)
    if K_MAJOR:
        out = tl.trans(v, 0, 1, 3, 2).reshape(4 * w.shape[0], w.shape[1])
    else:
        out = v.reshape(w.shape[0], 4 * w.shape[1])
    return out


@triton.jit
def int_scale_words(w_scale, WORDS_PER_GROUP: tl.constexpr, K_MAJOR: tl.constexpr):
    """``[R, G]`` bf16/fp16 group scales to ``int_words_to_offset``'s ``[R, G * WORDS_PER_GROUP]`` scale words (``K_MAJOR``:
    ``[G, R]`` to ``[G * WORDS_PER_GROUP, R]``), the scale in both 16-bit lanes."""
    lane = w_scale.to(tl.uint16, bitcast=True).to(tl.uint32) * 0x00010001
    if K_MAJOR:
        G: tl.constexpr = w_scale.shape[0]
        R: tl.constexpr = w_scale.shape[1]
        out = tl.broadcast_to(lane[:, None, :], (G, WORDS_PER_GROUP, R)).reshape(G * WORDS_PER_GROUP, R)
    else:
        R: tl.constexpr = w_scale.shape[0]
        G: tl.constexpr = w_scale.shape[1]
        out = tl.broadcast_to(lane[:, :, None], (R, G, WORDS_PER_GROUP)).reshape(R, G * WORDS_PER_GROUP)
    return out


@triton.jit
def int_zero_words(w_zero, WORDS_PER_GROUP: tl.constexpr, BITS: tl.constexpr, K_MAJOR: tl.constexpr, FP16: tl.constexpr = False):
    """``[R, G]`` per-group zero points to ``int_words_to_offset``'s ``[R, G * WORDS_PER_GROUP]`` zero words
    (``K_MAJOR``: ``[G, R]`` to ``[G * WORDS_PER_GROUP, R]``): CUDA, the packed float ``-(OFFSET + z)`` pair (bf16
    ``0xC300 | z``, fp16 ``0xE400 | z``); elsewhere the lift ``2 ** BITS - z`` in both lanes."""
    z = w_zero.to(tl.uint32)
    if _PTX:
        lane = (z | (0xC300 if BITS < 8 and not FP16 else 0xE400)) * 0x00010001
    else:
        lane = ((1 << BITS) - z) * 0x00010001
    if K_MAJOR:
        G: tl.constexpr = w_zero.shape[0]
        R: tl.constexpr = w_zero.shape[1]
        out = tl.broadcast_to(lane[:, None, :], (G, WORDS_PER_GROUP, R)).reshape(G * WORDS_PER_GROUP, R)
    else:
        R: tl.constexpr = w_zero.shape[0]
        G: tl.constexpr = w_zero.shape[1]
        out = tl.broadcast_to(lane[:, :, None], (R, G, WORDS_PER_GROUP)).reshape(R, G * WORDS_PER_GROUP)
    return out


@triton.jit
def int_dequant(q, zero, scale, OUT_DTYPE: tl.constexpr, divisor=None):
    """``((q - zero) * scale)`` rounded in the scale's dtype, then cast to ``OUT_DTYPE``: the dequantize-to-a-float
    reference (compressed-tensors' ``_dequantize``, what transformers and Marlin users compare against), bit for bit.
    ``q`` holds ``q - z`` plus a constant offset and ``zero`` (a scalar) that offset, both exact in ``q``'s dtype;
    their difference is exact in every float dtype, and a bf16/fp16 scale's product with it is exact in fp32, so its
    one rounding is the reference's."""
    qz = (q - zero).to(scale.dtype)
    value = qz * scale
    if divisor is not None:
        # ModelOpt's second sequential quantizer rounds the reconstructed INT4 weight to E4M3.
        value = tl.clamp(value.to(tl.float32) / divisor, -448.0, 448.0)
    return value.to(OUT_DTYPE)


@triton.jit
def bytes_to_words(x):
    """A ``[R, C]`` tile of a byte dtype as ``[R, C // 4]`` uint32 words, four consecutive values a word, the first in
    the low byte."""
    R: tl.constexpr = x.shape[0]
    C: tl.constexpr = x.shape[1]
    b = x.to(tl.uint8, bitcast=True).to(tl.uint32).reshape(R, C // 4, 4)
    return tl.sum(b << (tl.arange(0, 4) * 8)[None, None, :], axis=2)


@triton.jit
def e4m3_word_products(a_words, w_words):
    """The CUDA-core dot of E4M3 rows held four K values a word: ``[BM, C]`` activation words against ``[R, C]``
    weight words gives ``[BM, R, C]`` fp32, each word pair's four products summed (``f16x2_dot4_f32``). Both sides
    widen to exact fp16 pairs once, before the broadcast, so the inner loop is one fused multiply-add per value
    instead of a convert to fp32 and a multiply."""
    BM: tl.constexpr = a_words.shape[0]
    R: tl.constexpr = w_words.shape[0]
    C: tl.constexpr = w_words.shape[1]
    a01, a23 = e4m3x4_to_f16x2(a_words.reshape(BM, 1, C))
    w01, w23 = e4m3x4_to_f16x2(w_words.reshape(1, R, C))
    shape: tl.constexpr = (BM, R, C)
    return f16x2_dot4_f32(
        tl.broadcast_to(a01, shape), tl.broadcast_to(a23, shape), tl.broadcast_to(w01, shape), tl.broadcast_to(w23, shape)
    )


@triton.jit
def e2m1_e4m3_word_products(a_words, w_words):
    """``e4m3_word_products`` for E2M1 weights, eight K values a weight word: ``[BM, 2C]`` E4M3 activation words
    against ``[R, C]`` E2M1 weight words gives ``[BM, R, C]`` fp32, each weight word's eight products summed. Every
    E2M1 and E4M3 value is exact in fp16, so the products are too."""
    BM: tl.constexpr = a_words.shape[0]
    R: tl.constexpr = w_words.shape[0]
    C: tl.constexpr = w_words.shape[1]
    a_lo, a_hi = tl.split(a_words.reshape(BM, 1, C, 2))  # K 0-3 and 4-7 of each weight word
    a01, a23 = e4m3x4_to_f16x2(a_lo)
    a45, a67 = e4m3x4_to_f16x2(a_hi)
    w01, w23, w45, w67 = e2m1x8_to_f16x2(w_words.reshape(1, R, C))
    shape: tl.constexpr = (BM, R, C)
    low = f16x2_dot4_f32(
        tl.broadcast_to(a01, shape), tl.broadcast_to(a23, shape), tl.broadcast_to(w01, shape), tl.broadcast_to(w23, shape)
    )
    high = f16x2_dot4_f32(
        tl.broadcast_to(a45, shape), tl.broadcast_to(a67, shape), tl.broadcast_to(w45, shape), tl.broadcast_to(w67, shape)
    )
    return low + high


@triton.jit
def word_group_dot(a, w, GROUPS: tl.constexpr):
    """CUDA-core dot of one E4M3 token row ``a`` (``BK`` values) against weight rows ``w`` ``[R, BK // VPB]`` (E4M3,
    or E2M1 two to a byte), as ``[R, GROUPS]`` fp32 sums over equal K spans (the scale groups), reduced a word at a
    time (``e4m3_word_products``, ``e2m1_e4m3_word_products``); off NVIDIA, value by value in fp32."""
    R: tl.constexpr = w.shape[0]
    BK: tl.constexpr = a.numel
    if _PTX:
        a_words = bytes_to_words(tl.reshape(a, (1, BK)))
        w_words = bytes_to_words(w)
        if w.dtype == tl.uint8:
            products = e2m1_e4m3_word_products(a_words, w_words)
        else:
            products = e4m3_word_products(a_words, w_words)
        products = tl.reshape(products, (R, w_words.shape[1]))
    else:
        wf = e2m1_cols_to_f32(w) if w.dtype == tl.uint8 else w.to(tl.float32)
        products = tl.reshape(a, (1, BK)).to(tl.float32) * wf
    return tl.sum(tl.reshape(products, (R, GROUPS, products.shape[1] // GROUPS)), axis=2)


@triton.jit
def int4_e4m3_table(w_scale, w_zero, divisor=None):
    """Per (row, group), the E4M3 value of each of the 16 nibbles by ``int_dequant``'s own arithmetic (so the lookup
    is bit for bit), packed four entries a word: ``[N, NG]`` scales to ``[N, NG, 4]`` words, entries 0-3, ..., 12-15.
    Symmetric INT4 sits at zero point 8."""
    N: tl.constexpr = w_scale.shape[0]
    NG: tl.constexpr = w_scale.shape[1]
    nibble = tl.arange(0, 16).to(w_scale.dtype)[None, None, :]
    zero = 8.0 if w_zero is None else w_zero.to(w_scale.dtype)[:, :, None]
    entries = int_dequant(nibble, zero, w_scale[:, :, None], tl.float8e4nv, divisor)
    return bytes_to_words(entries.reshape(N, NG * 16)).reshape(N, NG, 4)


@triton.jit
def int4_e4m3_lookup(w, table, WORDS_PER_GROUP: tl.constexpr, ROWS_MAJOR: tl.constexpr):
    """Each weight word's eight nibbles through its group's ``int4_e4m3_table``: ``[R, C]`` words (``ROWS_MAJOR``
    ``[N, BK // 8]``, else ``[BK // 8, N]``) to ``[R, C, 2]`` words of four K-ordered E4M3 bytes."""
    N: tl.constexpr = table.shape[0]
    NG: tl.constexpr = table.shape[1]
    table = tl.broadcast_to(table[:, :, None, :], (N, NG, WORDS_PER_GROUP, 4)).reshape(N, NG * WORDS_PER_GROUP, 4)
    if not ROWS_MAJOR:
        table = tl.trans(table, 1, 0, 2)
    t02, t13 = tl.split(table.reshape(table.shape[0], table.shape[1], 2, 2))  # split takes the last axis
    t0, t2 = tl.split(t02)
    t1, t3 = tl.split(t13)
    even, odd = nibbles_to_table_bytes(w, t0, t1, t2, t3)
    return tl.join(even, odd)


@triton.jit
def e4m3_words_to_tile(words, ROWS_MAJOR: tl.constexpr):
    """``int4_e4m3_lookup``'s ``[R, C, 2]`` words as the K-major ``[BK, N]`` E4M3 operand."""
    R: tl.constexpr = words.shape[0]
    C: tl.constexpr = words.shape[1]
    shifts = tl.arange(0, 4) * 8
    if ROWS_MAJOR:
        out = ((words.reshape(R, 2 * C)[:, :, None] >> shifts[None, None, :]) & 0xFF).to(tl.uint8)
        out = tl.trans(out.reshape(R, 8 * C))
    else:
        words = tl.trans(words, 0, 2, 1).reshape(2 * R, C)
        out = tl.trans(((words[:, :, None] >> shifts[None, None, :]) & 0xFF).to(tl.uint8), 0, 2, 1)
        out = out.reshape(8 * R, C)
    return out.to(tl.float8e4nv, bitcast=True)


@triton.jit
def e4m3_words_to_f16_tile(words, ROWS_MAJOR: tl.constexpr):
    """The same values as ``e4m3_words_to_tile`` in fp16 (exact), converted in registers as K-adjacent pairs: the
    operand of the 16-bit ``mma.sync`` that serves fp8 below the tcgen05 M. Left as E4M3, that MMA's own upcast relays
    the tile through shared memory a byte at a time (128 ``ld.shared.b8`` per thread at BM16 BN64 BK128)."""
    R: tl.constexpr = words.shape[0]
    C: tl.constexpr = words.shape[1]
    p01, p23 = e4m3x4_to_f16x2(words)
    pairs = tl.join(p01, p23)  # [R, C, 2 words, 2 pairs], K order
    lo = (pairs & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    hi = (pairs >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    v = tl.join(lo, hi)
    if ROWS_MAJOR:
        out = tl.trans(v.reshape(R, 8 * C))
    else:
        out = tl.trans(v.reshape(R, C, 8), 0, 2, 1).reshape(8 * R, C)
    return out


@triton.jit
def int4_e4m3_tile(w, w_scale, w_zero, ROWS_MAJOR: tl.constexpr, SCALE_GROUP_K: tl.constexpr, divisor=None,
                   F16: tl.constexpr = False):
    """``int_dequant_tile`` for INT4 weights into E4M3 by table lookup, bit for bit: a group's 16 possible values come
    from ``int_dequant`` once (``int4_e4m3_table``), then every nibble is a byte permute (``int4_e4m3_lookup``), about
    two integer ops per weight instead of a per-value subtract, scale, divide, clamp and convert, which left the M=1
    W4A8 GEMV ALU-bound (ModelOpt W4A8, 6144x4096: 0.47 TB/s). ``F16`` hands the values over as fp16 pairs
    (``e4m3_words_to_f16_tile``)."""
    words = int4_e4m3_lookup(w, int4_e4m3_table(w_scale, w_zero, divisor), SCALE_GROUP_K // 8, ROWS_MAJOR)
    if F16:
        out = e4m3_words_to_f16_tile(words, ROWS_MAJOR)
    else:
        out = e4m3_words_to_tile(words, ROWS_MAJOR)
    return out


@triton.jit
def int4_e4m3_scalar(acc, a, w, w_scale, w_zero, SCALE_GROUP_K: tl.constexpr, divisor=None):
    """W4A8's CUDA-core reduce for a few token rows ``a`` ``[BM, BK]`` (E4M3) against output-rows-major INT4 words: the
    table-lookup weight words (``int4_e4m3_lookup``) line up four K values a word with the activation's bytes, so
    ``e4m3_word_products`` takes both as is. At M=1 it skips the 16-row ``mma.sync`` (one real row) and its operand
    staging."""
    words = int4_e4m3_lookup(w, int4_e4m3_table(w_scale, w_zero, divisor), SCALE_GROUP_K // 8, True)
    return acc + tl.sum(e4m3_word_products(bytes_to_words(a), words.reshape(w.shape[0], a.shape[1] // 4)), axis=2)


@triton.jit
def int_dequant_tile(
    w, w_scale, w_zero, ROWS_MAJOR: tl.constexpr, SCALE_GROUP_K: tl.constexpr, BITS: tl.constexpr,
    OUT_DTYPE: tl.constexpr, WHOLE_WORD: tl.constexpr = False, divisor=None,
):
    """One ``BITS``-wide integer weight K-tile of interleaved uint32 words (``int_word_pairs``), output-rows-major
    ``[N, BK // VPW]`` (``ROWS_MAJOR``) or K-major ``[BK // VPW, N]`` (VPW = ``32 // BITS`` values per word),
    dequantized (``int_dequant``) into the K-major ``[BK, N]`` MMA rhs. The tile spans ``BK // SCALE_GROUP_K`` whole
    groups: ``w_scale`` and ``w_zero`` (``None``: symmetric, zero point ``2 ** (BITS - 1)``) are
    ``[N, BK // SCALE_GROUP_K]``. Rows-major INT4 into E4M3 takes the table lookup (``int4_e4m3_tile``)."""
    if _PTX and BITS == 4 and OUT_DTYPE == tl.float8e4nv and ROWS_MAJOR:
        out = int4_e4m3_tile(w, w_scale, w_zero, ROWS_MAJOR, SCALE_GROUP_K, divisor)
    else:
        out = int_offset_dequant_tile(w, w_scale, w_zero, ROWS_MAJOR, SCALE_GROUP_K, BITS, OUT_DTYPE, WHOLE_WORD, divisor)
    return out


@triton.jit
def int_offset_dequant_tile(
    w, w_scale, w_zero, ROWS_MAJOR: tl.constexpr, SCALE_GROUP_K: tl.constexpr, BITS: tl.constexpr,
    OUT_DTYPE: tl.constexpr, WHOLE_WORD: tl.constexpr = False, divisor=None,
):
    """``int_dequant_tile`` by arithmetic: the values ``int_words_to_offset`` lifts under a float exponent, then
    ``int_dequant`` per value."""
    VPW: tl.constexpr = 32 // BITS
    FP16: tl.constexpr = w_scale.dtype == tl.float16
    OFFSET: tl.constexpr = 128.0 if BITS < 8 and not FP16 else 1024.0  # int_words_to_offset
    # what int_words_to_offset leaves on top of q - z
    if w_zero is None:
        ZERO: tl.constexpr = OFFSET + (1 << (BITS - 1))
    elif _PTX:
        ZERO: tl.constexpr = 0.0
    else:
        ZERO: tl.constexpr = OFFSET + (1 << BITS)
    WPG: tl.constexpr = SCALE_GROUP_K // VPW  # words per scale group
    # a zero-point tile already in the scale's dtype takes its scale inside int_words_to_offset's packed asm
    PACKED_SCALE: tl.constexpr = (_PTX and w_zero is not None and divisor is None and BITS < 8
                                  and OUT_DTYPE == w_scale.dtype)
    if ROWS_MAJOR:
        N: tl.constexpr = w.shape[0]
        BK: tl.constexpr = VPW * w.shape[1]
        NG: tl.constexpr = BK // SCALE_GROUP_K
        if SM10X and BITS == 8 and w_scale.dtype == tl.bfloat16:
            q = int8_words_to_bf16(w, False).reshape(N, NG, SCALE_GROUP_K)
            center = 0 if w_zero is None else (w_zero.to(tl.bfloat16) - 128)[:, :, None]
            v = int_dequant(q, center, w_scale[:, :, None], OUT_DTYPE, divisor)
        else:
            zero_words = None if w_zero is None else int_zero_words(w_zero, WPG, BITS, False, FP16)
            if PACKED_SCALE:
                v = int_words_to_offset(w, zero_words, BITS, False, WHOLE_WORD, FP16, int_scale_words(w_scale, WPG, False))
            else:
                q = int_words_to_offset(w, zero_words, BITS, False, WHOLE_WORD, FP16).reshape(N, NG, SCALE_GROUP_K)
                v = int_dequant(q, ZERO, w_scale[:, :, None], OUT_DTYPE, divisor)

        out = tl.trans(v.reshape(N, BK))
    else:
        BK: tl.constexpr = VPW * w.shape[0]
        N: tl.constexpr = w.shape[1]
        NG: tl.constexpr = BK // SCALE_GROUP_K
        if SM10X and BITS == 8 and w_scale.dtype == tl.bfloat16:
            q = int8_words_to_bf16(w, True).reshape(NG, SCALE_GROUP_K, N)
            center = 0 if w_zero is None else (tl.trans(w_zero).to(tl.bfloat16) - 128)[:, None, :]
            v = int_dequant(q, center, tl.trans(w_scale)[:, None, :], OUT_DTYPE, divisor)
        else:
            zero_words = None if w_zero is None else int_zero_words(tl.trans(w_zero), WPG, BITS, True, FP16)
            if PACKED_SCALE:
                v = int_words_to_offset(w, zero_words, BITS, True, WHOLE_WORD, FP16,
                                        int_scale_words(tl.trans(w_scale), WPG, True))
            else:
                q = int_words_to_offset(w, zero_words, BITS, True, WHOLE_WORD, FP16).reshape(NG, SCALE_GROUP_K, N)
                v = int_dequant(q, ZERO, tl.trans(w_scale)[:, None, :], OUT_DTYPE, divisor)
        out = v.reshape(BK, N)
    return out


@triton.jit
def _bcast(x, shape: tl.constexpr):
    return tl.broadcast_to(x[None, :], shape)


@triton.jit
def _row_split(p, BM: tl.constexpr):
    """``[BM, KW]`` -> ``BM`` tensors ``[KW]`` (row order), BM in {1, 2, 4}."""
    if BM == 1:
        out = (p.reshape(p.shape[1]),)
    elif BM == 2:
        r0, r1 = tl.split(tl.trans(p))
        out = (r0, r1)
    else:
        e, o = tl.split(tl.trans(p).reshape(p.shape[1], 2, 2))  # row = 2 * hi + lo: split lo
        r0, r2 = tl.split(e)
        r1, r3 = tl.split(o)
        out = (r0, r1, r2, r3)
    return out


@triton.jit
def int_words_dot(w, zero2, scale2, a, BM: tl.constexpr, BITS: tl.constexpr):
    """Interleaved 2/4/8-bit words ``[R, KW]`` against ``BM`` bf16/fp16 token rows. Dequantize each pair once for all
    rows with the scale's rounding, then accumulate the products in fp32 on sm_100.
    ``zero2`` holds packed ``offset + zero`` (128 for bf16, 1024 for fp16); ``scale2`` holds each word's packed group scale.
    Returns fp32 ``[BM, R]`` for ``BM`` in {1, 2, 4}."""
    R: tl.constexpr = w.shape[0]
    KW: tl.constexpr = w.shape[1]
    HALF: tl.constexpr = 16 // BITS
    shape: tl.constexpr = (R, KW)
    lo, hi = tl.split(a.to(tl.uint16, bitcast=True).to(tl.uint32).reshape(BM, KW, HALF, 2))
    packed = lo | (hi << 16)
    if BITS == 8:
        p0, p1 = tl.split(packed.reshape(BM, KW, 2))
        pairs = (p0, p1)
    elif BITS == 4:
        even, odd = tl.split(packed.reshape(BM, KW, 2, 2))
        p0, p2 = tl.split(even)
        p1, p3 = tl.split(odd)
        pairs = (p0, p1, p2, p3)
    else:
        even, odd = tl.split(packed.reshape(BM, KW, 4, 2))
        ee, eo = tl.split(even.reshape(BM, KW, 2, 2))
        oe, oo = tl.split(odd.reshape(BM, KW, 2, 2))
        p0, p4 = tl.split(ee)
        p2, p6 = tl.split(eo)
        p1, p5 = tl.split(oe)
        p3, p7 = tl.split(oo)
        pairs = (p0, p1, p2, p3, p4, p5, p6, p7)
    row_pairs = ()
    for p in tl.static_range(HALF):
        row_pairs += (_row_split(pairs[p], BM),)
    operands = (w.to(tl.uint32, bitcast=True), zero2, scale2)
    for m in tl.static_range(BM):
        for p in tl.static_range(HALF):
            operands += (_bcast(row_pairs[p][m], shape),)
    out = int_words_fma_rows(operands, BITS, BM, a.dtype == tl.float16)
    sums = ()
    for m in tl.static_range(BM):
        sums += (tl.sum(out[m].to(tl.float32, bitcast=True), axis=1),)
    if BM == 1:
        result = sums[0][None, :]
    elif BM == 2:
        result = tl.trans(tl.join(sums[0], sums[1]))
    else:
        result = tl.permute(tl.join(tl.join(sums[0], sums[1]), tl.join(sums[2], sums[3])), (2, 1, 0)).reshape(4, R)
    return result


@triton.jit
def int8_words_dot(w, a):
    """Signed INT8 dot from offset-binary interleaved words, accumulating exactly in INT32."""
    BM: tl.constexpr = a.shape[0]
    KW: tl.constexpr = a.shape[1] // 4
    if SM10X:
        # Weight bytes hold K offsets (0, 2, 1, 3); pack the activation bytes in the same order.
        even, odd = tl.split(a.to(tl.uint8, bitcast=True).to(tl.uint32).reshape(BM, KW, 2, 2))
        a0, a2 = tl.split(even)
        a1, a3 = tl.split(odd)
        packed_a = a0 | (a2 << 8) | (a1 << 16) | (a3 << 24)
        signed_w = w.to(tl.uint32, bitcast=True) ^ 0x80808080
        prod = dp4a(signed_w[None, :, :], packed_a[:, None, :])
    else:
        rhs = int8_words_to_int8(w, False)
        prod = a.to(tl.int32)[:, None, :] * rhs.to(tl.int32)[None, :, :]
    return tl.sum(prod, axis=2)


@triton.jit
def int_weight_only_scalar(
    acc,
    a,
    w,
    w_scale,
    w_zero,
    ROWS_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    BITS: tl.constexpr,
):
    """The integer weight-only CUDA-core reduce: ``w`` is output-rows-major ``[ROWS_W, BK // VPW]`` interleaved int32
    words, dequantized to ``a``'s dtype with ``int_dequant``'s rounding once for all ``BM`` token rows of ``a``
    ``[BM, BK]``, and reduced in fp32. Matching bf16/fp16 on sm_100 (2/4 bits, plus 8 bits for fp16) runs ``int_words_dot``; otherwise the dequant is
    computed in fp32 (the reduce's operand dtype). Returns ``acc + [BM, ROWS_W]``."""
    NG: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
    VPW: tl.constexpr = 32 // BITS
    KW: tl.constexpr = BLOCK_SIZE_K // VPW
    BM: tl.constexpr = a.shape[0]
    if (
        SM10X and (BITS == 2 or BITS == 4 or a.dtype == tl.float16) and (BM == 1 or BM == 2 or BM == 4)
        and a.dtype == w_scale.dtype and (a.dtype == tl.bfloat16 or a.dtype == tl.float16)
    ):
        WPG: tl.constexpr = SCALE_GROUP_K // VPW  # words per scale group
        s16 = w_scale.to(tl.uint16, bitcast=True).to(tl.uint32)
        scale2 = tl.broadcast_to((s16 | (s16 << 16))[:, :, None], (ROWS_W, NG, WPG)).reshape(ROWS_W, KW)
        OFFSET: tl.constexpr = 1024.0 if a.dtype == tl.float16 else 128.0
        MAGIC: tl.constexpr = 0x6400 if a.dtype == tl.float16 else 0x4300
        if w_zero is None:
            zero2 = tl.full((ROWS_W, KW), (MAGIC | (1 << (BITS - 1))) * 0x10001, tl.uint32)
        else:
            z16 = (w_zero.to(tl.float32) + OFFSET).to(a.dtype).to(tl.uint16, bitcast=True).to(tl.uint32)
            zero2 = tl.broadcast_to((z16 | (z16 << 16))[:, :, None], (ROWS_W, NG, WPG)).reshape(ROWS_W, KW)
        return acc + int_words_dot(w, zero2, scale2, a, BM, BITS)
    # a value ORed under fp32 2^23's exponent bits IS 2^23 + q: no int-to-float conversion
    pairs = int_word_pairs(w, BITS, False)
    lo = ((pairs & 0xFFFF) | 0x4B000000).to(tl.float32, bitcast=True)
    hi = ((pairs >> 16) | 0x4B000000).to(tl.float32, bitcast=True)
    v = tl.join(lo, hi).reshape(ROWS_W, NG, SCALE_GROUP_K)
    zero = 8388608.0 + (1 << (BITS - 1)) if w_zero is None else (w_zero.to(tl.float32) + 8388608.0)[:, :, None]
    # q - zero is exact, its product with a bf16/fp16 scale exact in fp32: one rounding, in the scale dtype
    wt = (v - zero) * w_scale.to(tl.float32)[:, :, None]
    if _PTX and w_scale.dtype == tl.bfloat16 and a.dtype == tl.bfloat16:
        wt = round_f32_to_bf16_f32(wt)
    else:
        wt = wt.to(w_scale.dtype).to(a.dtype).to(tl.float32)
    prod = wt.reshape(ROWS_W, BLOCK_SIZE_K)[None] * a.to(tl.float32)[:, None, :]
    return acc + tl.sum(prod, axis=2)


@triton.jit
def int_weight_only_compute(
    acc,
    a,
    w,
    w_scale,
    w_zero,
    BLOCK_SIZE_K: tl.constexpr,
    N_WIDTH: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    COMPUTE_MODE: tl.constexpr = "dot",
    ROWS_MAJOR: tl.constexpr = False,
    BITS: tl.constexpr = 4,
    SWAP_AB: tl.constexpr = False,
    weight_global=None,
    K_SLICES: tl.constexpr = 1,
):
    """Integer-weight GEMM step: dequantize a ``BITS``-wide tile to the activation dtype (``int_dequant``),
    then accumulate with the MMA (``"dot"``) or the small-token CUDA-core reduce (``"scalar"``).
    Floating operands accumulate in FP32; SmoothQuant keeps its signed integer dot in INT32 until the epilogue.
    ``ROWS_MAJOR`` is the weight tile's orientation (``int_dequant_tile``); the scalar arm's is always
    output-rows-major. ``SWAP_AB`` puts token rows on the MMA's N axis; keep that accumulator orientation
    across the K-loop and transpose it once before the epilogue."""
    # A tile inside a wider scale group reads one scale and zero-point column.
    TILE_GROUP: tl.constexpr = min(SCALE_GROUP_K, BLOCK_SIZE_K)
    if COMPUTE_MODE == "scalar" and a.dtype == tl.int8:
        acc += int8_words_dot(w, a)
    elif COMPUTE_MODE == "scalar" and a.dtype == tl.float8e4nv:
        acc = int4_e4m3_scalar(acc, a, w, w_scale, w_zero, TILE_GROUP, weight_global)
    elif COMPUTE_MODE == "scalar":
        acc = int_weight_only_scalar(acc, a, w, w_scale, w_zero, N_WIDTH, BLOCK_SIZE_K, TILE_GROUP, BITS)
    else:
        # Keep a complete INT4 word with its owner through either orientation.
        WHOLE_WORD: tl.constexpr = SM10X and BITS == 4 and (
            (SWAP_AB and ROWS_MAJOR) or (not ROWS_MAJOR and w_zero is None)
        )
        if a.dtype == tl.int8:
            # SmoothQuant's per-channel scale applies after the integer dot, not to its INT8 operands.
            rhs = int8_words_to_int8(w, not ROWS_MAJOR)
            if ROWS_MAJOR:
                rhs = tl.trans(rhs)
        elif _PTX and BITS == 4 and a.dtype == tl.float8e4nv and ROWS_MAJOR:
            # below the tcgen05 M the fp8 dot is a 16-bit mma.sync: hand it fp16 operands (exact E4M3 values)
            rhs = int4_e4m3_tile(w, w_scale, w_zero, ROWS_MAJOR, TILE_GROUP, weight_global, F16=True)
            a = a.to(tl.float16)
        else:
            rhs = int_dequant_tile(w, w_scale, w_zero, ROWS_MAJOR, TILE_GROUP, BITS, a.dtype, WHOLE_WORD, weight_global)
        DOT_DTYPE: tl.constexpr = tl.int32 if a.dtype == tl.int8 else tl.float32
        if SWAP_AB:
            # Put the token rows on the MMA's smaller N axis (8 lanes instead of 16).
            acc = tl.dot(tl.trans(rhs), tl.trans(a), acc, out_dtype=DOT_DTYPE)
        else:
            acc = plain_dot(acc, a, rhs, False, K_SLICES, DOT_DTYPE)
    return acc


@triton.jit
def mx_scalar_reduce(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    ROWS_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
):
    """MX 'scalar' step: CUDA-core FMA GEMV, unpacking MXFP4 to fp32 and dequantizing activation +
    weight by their group scales, reducing over K into ``acc``. No tensor core (so no M->16 MMA
    pad) — wins the memory-bound decode GEMV (M=1). The gate_up kernels pass the INTERLEAVED
    gate|up tile (gate even, up odd; ROWS_W = 2*BN). The UE8M0 scale is constant within each
    ``SCALE_GROUP_K`` group, so it
    factors out of the inner sum: reduce the raw products per group, then apply ONE combined
    (act x weight) scale per group — ``SCALE_GROUP_K``x fewer scale-muls (~18% faster, bit-identical
    to the expanded form). Plain: ``a`` [BM, BK] against ``w`` [BK, ROWS_W]. ``SWAP_AB`` (decode):
    ``w`` output-rows-major [ROWS_W, BK] against the [BK] token, MXFP4 unpacked along columns (K),
    no transposes. Both return ``acc + [1, ROWS_W]``. E4M3 activations against E4M3 or E2M1 weights (MXFP8 W8A8,
    MXFP4 W4A8) reduce a word at a time (``e4m3_word_products``, ``e2m1_e4m3_word_products``) in either orientation."""
    NG: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
    if a.dtype == tl.float8e4nv:
        grp = word_group_dot(a, w if SWAP_AB else tl.trans(w), NG)
        scale = tl.reshape(decode_group_scale(a_scale), (1, NG)) * tl.reshape(decode_group_scale(w_scale), (ROWS_W, NG))
        acc = acc + tl.reshape(tl.sum(grp * scale, axis=1), (1, ROWS_W))
    elif SWAP_AB:
        aq = e2m1_cols_to_f32(a) if a.dtype == tl.uint8 else a.to(tl.float32)
        wq = e2m1_cols_to_f32(w) if w.dtype == tl.uint8 else w.to(tl.float32)  # [ROWS_W, BK]
        grp = tl.sum(tl.reshape(aq[None, :] * wq, (ROWS_W, NG, SCALE_GROUP_K)), axis=2)
        scale = decode_group_scale(a_scale)[None, :] * decode_group_scale(w_scale)  # [ROWS_W, NG]
        acc = acc + tl.reshape(tl.sum(grp * scale, axis=1), (1, ROWS_W))
    else:
        aq = e2m1_cols_to_f32(a) if a.dtype == tl.uint8 else a.to(tl.float32)
        wq = e2m1_to_f32(w) if w.dtype == tl.uint8 else w.to(tl.float32)  # [BK, ROWS_W]
        grp = tl.sum(tl.reshape(tl.trans(aq) * wq, (NG, SCALE_GROUP_K, ROWS_W)), axis=1)
        scale = tl.trans(decode_group_scale(a_scale)) * tl.trans(decode_group_scale(w_scale))
        acc = acc + tl.sum(grp * scale, axis=0)[None, :]
    return acc


@triton.jit
def mx_compute(
    acc,
    a,
    a_scale,
    w,
    w_scale,
    COMPUTE_MODE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SCALE_GROUP_K: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
):
    """Single-projection MMA step, dispatched on ``COMPUTE_MODE`` (scaled-MMA on the raw weight,
    fp8 ``tl.dot`` + per-group rescale, or the scalar reduce); ``SWAP_AB`` selects each leaf's
    decode form — weight output rows in the MMA M dim, the single token flattened to the [BK] rhs
    (a packed-E2M1 token flattens to its BYTE length; dot_scaled consumes it packed, dot/scalar
    column-unpack it losslessly). The acc shapes diverge across modes and orientations, but only
    the taken constexpr branch compiles, so the single return never has to unify them.
    ``BLOCK_SIZE_N`` is the weight tile's row count — the gate_up kernels pass ``2*BN`` with the
    INTERLEAVED gate|up tile (gate on even rows, up on odd; split back via ``split_gate_up``): one load and one MMA for both
    projections keeps the native microscaled-MMA M=128 operand at BN=64, doubling the CTAs on the
    parallelism-starved decode grid (dsv4 gate_up 1.34x, bit-exact)."""
    if SWAP_AB:
        if a.dtype == tl.uint8:
            a1 = tl.reshape(a, (BLOCK_SIZE_K // 2,))
        else:
            a1 = tl.reshape(a, (BLOCK_SIZE_K,))
        as1 = tl.reshape(a_scale, (BLOCK_SIZE_K // SCALE_GROUP_K,))
    else:
        a1 = a
        as1 = a_scale
    if COMPUTE_MODE == "dot_scaled":
        acc = mx_dot_scaled(acc, a1, as1, w, w_scale, SWAP_AB, BLOCK_SIZE_K, SCALE_GROUP_K)
    elif COMPUTE_MODE == "dot":
        acc = mx_dot_rescale(acc, a1, as1, w, w_scale, SWAP_AB, BLOCK_SIZE_K)
    else:  # scalar
        acc = mx_scalar_reduce(acc, a1, as1, w, w_scale, BLOCK_SIZE_N, BLOCK_SIZE_K, SCALE_GROUP_K, SWAP_AB)
    return acc


# ── swap-AB decode compute: M=1 batched GEMV with output rows in the MMA M dim ───────
#
# The batched (decode) kernels are structurally M=1, where the sm_100 scaled MMA pads M→128.
# Putting the WEIGHT's output rows in M (fully used) and the single token in N cuts that pad to
# the N-atom (16) — a ~1.5× decode win for fp4 (dot_scaled), neutral-to-worse for fp8 (scalar
# stays memory-bound and wins, so the tuner keeps it). Weight is loaded output-rows-major
# ``[BN, BK]`` for every mode, so the kernel does ONE load; each helper returns ``[1, BN]``.


# The sm_100 MMA's minimum N tile (16). In the swap path the single decode token sits in the MMA's
# N dim, so it must be padded up to this width (col 0 = the token, cols 1..15 = zero). It is NOT a
# block size — BLOCK_SIZE_M stays 1 under swap; this is the token's *padded N extent*, fixed by the
# hardware. The plain int is the source of truth: host-side consumers (the autotune pruners, which
# filter configs in ordinary Python) compare against it directly, while kernels read the
# tl.constexpr view derived from it — the only module-global form a @triton.jit fn can read.
MMA_N_ATOM_WIDTH = 16
MMA_N_ATOM = tl.constexpr(MMA_N_ATOM_WIDTH)


@triton.jit
def swap_pad_rhs(a, BLOCK_SIZE_K: tl.constexpr):
    """Pad the ``[BLOCK_SIZE_K]`` M=1 token to the ``[BLOCK_SIZE_K, MMA_N_ATOM]`` swap-AB MMA rhs —
    only column 0 is the real token (16 is measured, not a hardware floor — see acc_init). Used
    by the M=1 batched / fused-MoE fp8 ``tl.dot`` swap paths (weight output rows in the MMA M dim);
    the caller takes column 0 of the ``[BN, MMA_N_ATOM]`` result after the K-loop."""
    return tl.where(
        tl.arange(0, MMA_N_ATOM)[None, :] == 0,
        a[:, None],
        tl.zeros((BLOCK_SIZE_K, MMA_N_ATOM), a.dtype),
    )


@triton.jit
def plain_dot(acc, a, b, SWAP_AB: tl.constexpr = False, K_SLICES: tl.constexpr = 1, OUT_DTYPE: tl.constexpr = tl.float32):
    """Plain GEMM step with a logical [M, N] accumulator in either operand orientation.

    Swapping uses row-major weights [N, K] as the MMA lhs and transposes the
    [M, K] activations, avoiding the padded token columns of single-token helpers.
    ``K_SLICES`` > 1 splits the K tile across warps: one batched dot over ``[S, M, BK / S]`` and
    ``[S, BK / S, N]`` into the ``[S, M, N]`` accumulator ``acc_init`` gives it, summed once by ``k_slices_sum``.
    """
    if K_SLICES > 1:
        M: tl.constexpr = a.shape[0]
        N: tl.constexpr = b.shape[1]
        KS: tl.constexpr = a.shape[1] // K_SLICES
        a3 = tl.permute(tl.reshape(a, (M, K_SLICES, KS)), (1, 0, 2))
        out = tl.dot(a3, tl.reshape(b, (K_SLICES, KS, N)), acc, out_dtype=OUT_DTYPE)
    elif SWAP_AB:
        out = tl.trans(tl.dot(b, tl.trans(a), tl.trans(acc), out_dtype=OUT_DTYPE))
    else:
        out = tl.dot(a, b, acc, out_dtype=OUT_DTYPE)
    return out


@triton.jit
def k_slices_sum(acc, K_SLICES: tl.constexpr = 1):
    """The ``[M, N]`` accumulator of an in-CTA K split (``plain_dot``): its K slices summed, through shared memory
    across the warps that own them; the plain accumulator passes through."""
    if K_SLICES > 1:
        acc = tl.sum(acc, axis=0)
    return acc


@triton.jit
def fp8_dot(a, b, SWAP_AB: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    """Swap-aware plain ``tl.dot`` for the fp8 (block/tensor-dynamic) paths — no scaling; the caller
    applies its own per-block/per-tensor scales. Swap: weight ``b`` ``[N, BK]`` × the single token
    padded to the N=16 atom → ``[N, MMA_N_ATOM]`` (col 0 real). No-swap: token ``a`` ``[M, BK]`` ×
    weight ``b`` ``[BK, N]`` → ``[M, N]``. ``BLOCK_SIZE_K`` is the contraction tile (the down
    projection passes its intermediate tile). Single return: only the taken branch compiles."""
    if SWAP_AB:
        out = tl.dot(b, swap_pad_rhs(tl.reshape(a, (BLOCK_SIZE_K,)), BLOCK_SIZE_K))
    else:
        out = tl.dot(a, b)
    return out


@triton.jit
def block_dynamic_dot(
    acc, a, a_s, b, b_s,
    block_k: tl.constexpr, SWAP_AB: tl.constexpr, USE_DOT_SCALED: tl.constexpr,
    FAKE_BATCH: tl.constexpr = False,
):
    """Accumulate one block-dynamic (1x128/128x128) fp8 K-tile into ``acc``, oriented by
    ``SWAP_AB``. ``USE_DOT_SCALED`` (UE8M0 scales on a native-M tile): fold the group scales
    into the tcgen05 MMA — the tile's single 128-group scale broadcasts in-register to the
    ``block_k // 32`` group-32 columns ``dot_scaled`` consumes, identical to a 128-group
    rescale but with no 4x scale memory and no software multiply. Else: plain fp8 ``tl.dot``
    + per-group software rescale (``decode_group_scale`` is a no-op on fp32 scales, decodes
    UE8M0). ``FAKE_BATCH`` (single-token decode): ``fp8_dot`` pads the lone token to the MMA N
    atom and both scales broadcast down the weight-row (M) dim. Single-exit if/else so only the
    taken arm type-checks (a trailing fall-through arm would be checked even when an earlier
    branch is taken)."""
    if USE_DOT_SCALED:
        reps: tl.constexpr = block_k // 32
        a_sg = a_s[:, None].broadcast_to(a_s.shape[0], reps)
        b_sg = b_s[:, None].broadcast_to(b_s.shape[0], reps)
        if SWAP_AB:
            acc = tl.dot_scaled(b, b_sg, "e4m3", a, a_sg, "e4m3", acc)
        else:
            acc = tl.dot_scaled(a, a_sg, "e4m3", b, b_sg, "e4m3", acc)
    else:
        # plain fp8 tl.dot + per-group decoded scales, oriented by SWAP_AB (weight rows in the MMA
        # M dim under swap). decode_group_scale: fp32 passthrough, UE8M0 -> 2^(e-127).
        a_sd = decode_group_scale(a_s)
        b_sd = decode_group_scale(b_s)
        if FAKE_BATCH:
            d = fp8_dot(a, b, SWAP_AB, block_k)
            if SWAP_AB:  # [BN, N-atom]: weight-row scale down the M=BN dim, act scalar broadcasts
                acc = acc + d * a_sd[:, None] * b_sd[:, None]
            else:  # no-swap [BM, BN] GEVM: act scale down M, weight-N scale across N (upstream form)
                acc = acc + d * a_sd[:, None] * b_sd[None, :]
        elif SWAP_AB:
            acc = acc + tl.dot(b, a) * b_sd[:, None] * a_sd[None, :]
        else:
            acc = acc + tl.dot(a, b) * a_sd[:, None] * b_sd[None, :]
    return acc


@triton.jit
def static_dot(acc, a, b, b_s, SWAP_AB: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, FAKE_BATCH: tl.constexpr):
    """static format K-step: plain (swap-aware) fp8 dot + per-K-block weight rescale. FAKE_BATCH
    (single-token decode) routes the rescale down the weight-row (M) dim; else it broadcasts across
    the N columns. The per-tensor activation scale is applied post-loop."""
    b_sd = decode_group_scale(b_s)
    d = fp8_dot(a, b, SWAP_AB, BLOCK_SIZE_K)
    if FAKE_BATCH:
        acc = acc + d * b_sd[:, None]
    else:
        acc = acc + d * b_sd[None, :]
    return acc


