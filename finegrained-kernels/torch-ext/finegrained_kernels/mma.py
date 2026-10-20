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
from .compat import get_active_device_type, is_sm10x


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


# PTX inline asm (round_f32_to_bf16_f32, the packed zero-point subtract) on the CUDA backend; elsewhere plain Triton
_PTX = tl.constexpr(get_active_device_type() == "cuda")
# Keep architecture-specific integer fast paths on the measured sm_10x target.
_SM10X = tl.constexpr(is_sm10x())


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
def int_words_to_offset(w, zero_words, BITS: tl.constexpr, K_MAJOR: tl.constexpr, WHOLE_WORD: tl.constexpr = False, FP16: tl.constexpr = False):
    """Interleaved uint32 words (``int_word_pairs``) to their values as ``OFFSET + q``, exactly, with no integer-to-float
    conversion: a value ORed under the offset's exponent bits IS that float, bf16 ``128 + q`` for 2- and 4-bit values
    (q < 128), fp16 ``1024 + q`` for FP16 scales or 8-bit values (bf16 cannot hold 128 + 255). ``zero_words`` (``None``: no
    zero point) is one word per weight word ``[R, C]`` carrying its group's zero point ``z`` in both 16-bit lanes
    (``int_zero_words``): on CUDA the float ``-(OFFSET + z)``, added to each pair by one packed FMA, so the values
    are ``q - z``; elsewhere the integer lift ``2 ** BITS - z`` added before the OR, so they are
    ``OFFSET + 2 ** BITS + q - z``. Either way the zero point costs no per-value float op of its own. Output-rows
    ``[R, C]`` -> ``[R, (32 // BITS) C]`` (``K_MAJOR``: ``[R, C]`` -> ``[(32 // BITS) R, C]``), natural K order."""
    MAGIC: tl.constexpr = 0x64006400 if FP16 or BITS == 8 else 0x43004300
    if _PTX and BITS == 8:
        # Keep both byte pairs with their word before orienting the MMA tile.
        lo, hi = tl.inline_asm_elementwise(
            "prmt.b32 $0, $2, 0x64646464, 0x5250; prmt.b32 $1, $2, 0x64646464, 0x5351;",
            "=r,=r,r", [w], dtype=(tl.uint32, tl.uint32), is_pure=True, pack=1,
        )
        pairs = tl.join(lo, hi)
        if K_MAJOR:
            pairs = tl.trans(pairs, 0, 2, 1)
    elif WHOLE_WORD:
        # Keep all four pairs in the word's lane. A shared mask avoids materializing one per asm expansion.
        mask = tl.full(w.shape, 0x000f000f, tl.uint32)
        exponent: tl.constexpr = "0x64006400" if FP16 else "0x43004300"
        p0, p1, p2, p3 = tl.inline_asm_elementwise(
            "{ .reg .b32 t1, t2, t3, mask; mov.b32 mask, $5; "
            "shr.u32 t1, $4, 4; shr.u32 t2, $4, 8; shr.u32 t3, $4, 12; "
            "lop3.b32 $0, $4, mask, " + exponent + ", 0xEA; "
            "lop3.b32 $1, t1, mask, " + exponent + ", 0xEA; "
            "lop3.b32 $2, t2, mask, " + exponent + ", 0xEA; "
            "lop3.b32 $3, t3, mask, " + exponent + ", 0xEA; }",
            "=r,=r,=r,=r,r,r", [w, mask], dtype=(tl.uint32,) * 4, is_pure=True, pack=1,
        )
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
    if zero_words is not None and _PTX:
        # fma(pair, 1, -(OFFSET + z)): the exact difference; fma.rn.bf16x2 / f16x2 lower from sm_80 / sm_53
        pairs = tl.inline_asm_elementwise(
            "{ .reg .b32 one; mov.b32 one, 0x3F803F80; fma.rn.bf16x2 $0, $1, one, $2; }" if BITS < 8 and not FP16
            else "{ .reg .b32 one; mov.b32 one, 0x3C003C00; fma.rn.f16x2 $0, $1, one, $2; }",
            "=r,r,r", [pairs, zero_words], dtype=tl.uint32, is_pure=True, pack=1,
        )
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
    # 128+(q&127) minus 128 or 256 is exact, without fp16 -> fp32 -> bf16 conversion.
    p0, p1 = tl.inline_asm_elementwise(
        "{ .reg .b32 q, z, t; shr.u32 t, $2, 8; lop3.b32 q, $2, 0x007f007f, 0x43004300, 0xEA; "
        "lop3.b32 z, $2, 0x00800080, 0x43804380, 0x6A; sub.rn.bf16x2 $0, q, z; "
        "lop3.b32 q, t, 0x007f007f, 0x43004300, 0xEA; "
        "lop3.b32 z, t, 0x00800080, 0x43804380, 0x6A; sub.rn.bf16x2 $1, q, z; }",
        "=r,=r,r", [w], dtype=(tl.uint32, tl.uint32), is_pure=True, pack=1,
    )
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
def int_dequant_tile(
    w, w_scale, w_zero, ROWS_MAJOR: tl.constexpr, SCALE_GROUP_K: tl.constexpr, BITS: tl.constexpr,
    OUT_DTYPE: tl.constexpr, WHOLE_WORD: tl.constexpr = False, divisor=None,
):
    """One ``BITS``-wide integer weight K-tile of interleaved uint32 words (``int_word_pairs``), output-rows-major
    ``[N, BK // VPW]`` (``ROWS_MAJOR``) or K-major ``[BK // VPW, N]`` (VPW = ``32 // BITS`` values per word),
    dequantized (``int_dequant``) into the K-major ``[BK, N]`` MMA rhs. The tile spans ``BK // SCALE_GROUP_K`` whole
    groups: ``w_scale`` and ``w_zero`` (``None``: symmetric, zero point ``2 ** (BITS - 1)``) are
    ``[N, BK // SCALE_GROUP_K]``."""
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
    if ROWS_MAJOR:
        N: tl.constexpr = w.shape[0]
        BK: tl.constexpr = VPW * w.shape[1]
        NG: tl.constexpr = BK // SCALE_GROUP_K
        if _SM10X and BITS == 8 and w_scale.dtype == tl.bfloat16:
            q = int8_words_to_bf16(w, False).reshape(N, NG, SCALE_GROUP_K)
            center = 0 if w_zero is None else (w_zero.to(tl.bfloat16) - 128)[:, :, None]
            v = int_dequant(q, center, w_scale[:, :, None], OUT_DTYPE, divisor)
        else:
            zero_words = None if w_zero is None else int_zero_words(w_zero, WPG, BITS, False, FP16)
            q = int_words_to_offset(w, zero_words, BITS, False, WHOLE_WORD, FP16).reshape(N, NG, SCALE_GROUP_K)
            v = int_dequant(q, ZERO, w_scale[:, :, None], OUT_DTYPE, divisor)

        out = tl.trans(v.reshape(N, BK))
    else:
        BK: tl.constexpr = VPW * w.shape[0]
        N: tl.constexpr = w.shape[1]
        NG: tl.constexpr = BK // SCALE_GROUP_K
        if _SM10X and BITS == 8 and w_scale.dtype == tl.bfloat16:
            q = int8_words_to_bf16(w, True).reshape(NG, SCALE_GROUP_K, N)
            center = 0 if w_zero is None else (tl.trans(w_zero).to(tl.bfloat16) - 128)[:, None, :]
            v = int_dequant(q, center, tl.trans(w_scale)[:, None, :], OUT_DTYPE, divisor)
        else:
            zero_words = None if w_zero is None else int_zero_words(tl.trans(w_zero), WPG, BITS, True, FP16)
            q = int_words_to_offset(w, zero_words, BITS, True, WHOLE_WORD, FP16).reshape(NG, SCALE_GROUP_K, N)
            v = int_dequant(q, ZERO, tl.trans(w_scale)[:, None, :], OUT_DTYPE, divisor)
        out = v.reshape(BK, N)
    return out


@triton.jit
def round_f32_to_bf16_f32(x):
    """fp32 ``x`` rounded to bf16 (round-to-nearest-even) and widened back, two lanes per packed convert
    (``cvt.rn.bf16x2.f32``) instead of one ``F2F`` per value."""
    return tl.inline_asm_elementwise(
        "{ .reg .b32 d; cvt.rn.bf16x2.f32 d, $3, $2; shl.b32 $0, d, 16; and.b32 $1, d, 0xFFFF0000; }",
        "=r,=r,r,r",
        [x.to(tl.uint32, bitcast=True)],
        dtype=tl.uint32,
        is_pure=True,
        pack=2,
    ).to(tl.float32, bitcast=True)


def _int_word_dot_asm(rows: int, bits: int, fp16: bool = False) -> str:
    """Pair-unpack interleaved 2/4/8-bit words, dequantize in bf16/fp16, then FMA into each token row's fp32 sum.
    Operands: row sums, word, packed zero/scale, then ``16 // bits`` activation pairs per row."""
    w, z, sc = rows, rows + 1, rows + 2
    dtype = "f16" if fp16 else "bf16"
    magic = "0x64006400" if fp16 else "0x43004300"
    lines = [
        "{",
        ".reg .b32 t, x;",
        ".reg .b16 wl, wh, al, ah;",
        ".reg .b32 " + ", ".join(f"c{m}" for m in range(rows)) + ";",
        f"mov.b32 t, ${w};",
    ] + [f"mov.b32 c{m}, 0;" for m in range(rows)]
    for p in range(16 // bits):
        lines += [
            f"lop3.b32 x, t, {((1 << bits) - 1) * 0x10001:#010x}, {magic}, 0xEA;",
            f"sub.rn.{dtype}x2 x, x, ${z}; mul.rn.{dtype}x2 x, x, ${sc}; mov.b32 {{wl, wh}}, x;",
        ]
        for m in range(rows):
            lines += [
                f"mov.b32 {{al, ah}}, ${rows + 3 + (16 // bits) * m + p};",
                f"fma.rn.f32.{dtype} c{m}, wl, al, c{m}; fma.rn.f32.{dtype} c{m}, wh, ah, c{m};",
            ]
        if p < 16 // bits - 1:
            lines.append(f"shr.u32 t, t, {bits};")
    lines += [f"mov.b32 ${m}, c{m};" for m in range(rows)] + ["}"]
    return "\n".join(lines)


# BF16 supports widths (2, 4), FP16 (2, 4, 8), each with token-row counts (1, 2, 4).
_INT_WORD_DOT_ASMS = tl.constexpr(tuple(
    _int_word_dot_asm(rows, bits, fp16)
    for fp16 in (False, True) for bits in ((2, 4, 8) if fp16 else (2, 4)) for rows in (1, 2, 4)
))
_INT_WORD_DOT_CONSTRAINTS = tl.constexpr(tuple(
    ",".join(["=r"] * rows + ["r"] * (3 + (16 // bits) * rows))
    for bits in (2, 4, 8) for rows in (1, 2, 4)
))


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
    IDX: tl.constexpr = (0 if BITS == 2 else (3 if BITS == 4 else 6)) + (0 if BM == 1 else (1 if BM == 2 else 2))
    ASM_IDX: tl.constexpr = IDX + (6 if a.dtype == tl.float16 else 0)
    out = tl.inline_asm_elementwise(
        _INT_WORD_DOT_ASMS[ASM_IDX], _INT_WORD_DOT_CONSTRAINTS[IDX], operands,
        dtype=(tl.uint32,) * BM, is_pure=True, pack=1,
    )
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
    if _SM10X:
        # Weight bytes hold K offsets (0, 2, 1, 3); pack the activation bytes in the same order.
        even, odd = tl.split(a.to(tl.uint8, bitcast=True).to(tl.uint32).reshape(BM, KW, 2, 2))
        a0, a2 = tl.split(even)
        a1, a3 = tl.split(odd)
        packed_a = a0 | (a2 << 8) | (a1 << 16) | (a3 << 24)
        signed_w = w.to(tl.uint32, bitcast=True) ^ 0x80808080
        prod = tl.inline_asm_elementwise(
            "dp4a.s32.s32 $0, $1, $2, 0;", "=r,r,r", [signed_w[None, :, :], packed_a[:, None, :]],
            dtype=tl.int32, is_pure=True, pack=1,
        )
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
        _SM10X and (BITS == 2 or BITS == 4 or a.dtype == tl.float16) and (BM == 1 or BM == 2 or BM == 4)
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
    elif COMPUTE_MODE == "scalar":
        acc = int_weight_only_scalar(acc, a, w, w_scale, w_zero, N_WIDTH, BLOCK_SIZE_K, TILE_GROUP, BITS)
    else:
        # Keep a complete INT4 word with its owner through either orientation.
        WHOLE_WORD: tl.constexpr = _SM10X and BITS == 4 and (
            (SWAP_AB and ROWS_MAJOR) or (not ROWS_MAJOR and w_zero is None)
        )
        if a.dtype == tl.int8:
            # SmoothQuant's per-channel scale applies after the integer dot, not to its INT8 operands.
            rhs = int8_words_to_int8(w, not ROWS_MAJOR)
            if ROWS_MAJOR:
                rhs = tl.trans(rhs)
        else:
            rhs = int_dequant_tile(w, w_scale, w_zero, ROWS_MAJOR, TILE_GROUP, BITS, a.dtype, WHOLE_WORD, weight_global)
        DOT_DTYPE: tl.constexpr = tl.int32 if a.dtype == tl.int8 else tl.float32
        if SWAP_AB:
            # Put the token rows on the MMA's smaller N axis (8 lanes instead of 16).
            acc = tl.dot(tl.trans(rhs), tl.trans(a), acc, out_dtype=DOT_DTYPE)
        else:
            acc = tl.dot(a, rhs, acc, out_dtype=DOT_DTYPE)
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
    no transposes. Both return ``acc + [1, ROWS_W]``."""
    NG: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_K
    aq = e2m1_cols_to_f32(a) if a.dtype == tl.uint8 else a.to(tl.float32)
    if SWAP_AB:
        wq = e2m1_cols_to_f32(w) if w.dtype == tl.uint8 else w.to(tl.float32)  # [ROWS_W, BK]
        grp = tl.sum(tl.reshape(aq[None, :] * wq, (ROWS_W, NG, SCALE_GROUP_K)), axis=2)
        scale = decode_group_scale(a_scale)[None, :] * decode_group_scale(w_scale)  # [ROWS_W, NG]
        acc = acc + tl.reshape(tl.sum(grp * scale, axis=1), (1, ROWS_W))
    else:
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
def plain_dot(acc, a, b, SWAP_AB: tl.constexpr = False):
    """Plain GEMM step with a logical [M, N] accumulator in either operand orientation.

    Swapping uses row-major weights [N, K] as the MMA lhs and transposes the
    [M, K] activations, avoiding the padded token columns of single-token helpers.
    """
    if SWAP_AB:
        out = tl.trans(tl.dot(b, tl.trans(a), tl.trans(acc)))
    else:
        out = tl.dot(a, b, acc)
    return out


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


