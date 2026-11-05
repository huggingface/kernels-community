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

"""The kernels' inline PTX: one wrapper per instruction idiom, elementwise over packed 32-bit words, with no layout
of its own. Callers decide where a word comes from and how its lanes map to a tile; these only say what the
instructions compute. Every wrapper needs an NVIDIA target (callers guard on ``is_nvidia``)."""

import triton
import triton.language as tl


@triton.jit
def int8_bytes_to_f16_lanes(w):
    """Bytes ``(b0, b1, b2, b3)`` of each word into two words of fp16 lanes ``1024 + b`` (``(b0, b2)`` and
    ``(b1, b3)``): each byte lands under the ``0x64`` exponent, which makes it that float exactly."""
    return tl.inline_asm_elementwise(
        "prmt.b32 $0, $2, 0x64646464, 0x5250; prmt.b32 $1, $2, 0x64646464, 0x5351;",
        "=r,=r,r", [w], dtype=(tl.uint32, tl.uint32), is_pure=True, pack=1,
    )


@triton.jit
def int4_nibbles_to_16bit_lanes(w, EXPONENT: tl.constexpr):
    """The eight nibbles of each word as four words of 16-bit lanes ``EXPONENT | nibble`` (nibbles ``(0, 4)``,
    ``(1, 5)``, ``(2, 6)``, ``(3, 7)``): with the bf16 ``0x4300`` / fp16 ``0x6400`` exponent, ``128 + q`` /
    ``1024 + q`` exactly. A shared mask avoids materializing one per asm expansion."""
    mask = tl.full(w.shape, 0x000f000f, tl.uint32)
    return tl.inline_asm_elementwise(
        "{ .reg .b32 t1, t2, t3, mask; mov.b32 mask, $5; "
        "shr.u32 t1, $4, 4; shr.u32 t2, $4, 8; shr.u32 t3, $4, 12; "
        "lop3.b32 $0, $4, mask, " + EXPONENT + ", 0xEA; "
        "lop3.b32 $1, t1, mask, " + EXPONENT + ", 0xEA; "
        "lop3.b32 $2, t2, mask, " + EXPONENT + ", 0xEA; "
        "lop3.b32 $3, t3, mask, " + EXPONENT + ", 0xEA; }",
        "=r,=r,=r,=r,r,r", [w, mask], dtype=(tl.uint32,) * 4, is_pure=True, pack=1,
    )


@triton.jit
def add_x2(a, b, FP16: tl.constexpr):
    """``a + b`` per 16-bit lane, bf16x2 (``FP16``: f16x2), as ``fma(a, 1, b)``: exact when the sum is
    representable, as it is for the offset-lifted integers it subtracts zero points from."""
    return tl.inline_asm_elementwise(
        "{ .reg .b32 one; mov.b32 one, 0x3C003C00; fma.rn.f16x2 $0, $1, one, $2; }" if FP16
        else "{ .reg .b32 one; mov.b32 one, 0x3F803F80; fma.rn.bf16x2 $0, $1, one, $2; }",
        "=r,r,r", [a, b], dtype=tl.uint32, is_pure=True, pack=1,
    )


@triton.jit
def add_mul_x2(a, b, c, FP16: tl.constexpr):
    """``(a + b) * c`` per 16-bit lane, bf16x2 (``FP16``: f16x2): the exact ``add_x2`` then one rounding of the product
    (``mul.rn.bf16x2`` lowers from sm_90)."""
    return tl.inline_asm_elementwise(
        "{ .reg .b32 one, d; mov.b32 one, 0x3C003C00; fma.rn.f16x2 d, $1, one, $2; mul.rn.f16x2 $0, d, $3; }" if FP16
        else "{ .reg .b32 one, d; mov.b32 one, 0x3F803F80; fma.rn.bf16x2 d, $1, one, $2; mul.rn.bf16x2 $0, d, $3; }",
        "=r,r,r,r", [a, b, c], dtype=tl.uint32, is_pure=True, pack=1,
    )


@triton.jit
def offset_int8_to_bf16_lanes(w):
    """Offset-binary bytes ``(b0, b1, b2, b3)`` of each word as two words of exact bf16 lanes ``b - 128``
    (``(b0, b2)`` and ``(b1, b3)``): ``128 + (b & 127)`` lifted under the exponent, minus 128 or 256."""
    return tl.inline_asm_elementwise(
        "{ .reg .b32 q, z, t; shr.u32 t, $2, 8; lop3.b32 q, $2, 0x007f007f, 0x43004300, 0xEA; "
        "lop3.b32 z, $2, 0x00800080, 0x43804380, 0x6A; sub.rn.bf16x2 $0, q, z; "
        "lop3.b32 q, t, 0x007f007f, 0x43004300, 0xEA; "
        "lop3.b32 z, t, 0x00800080, 0x43804380, 0x6A; sub.rn.bf16x2 $1, q, z; }",
        "=r,=r,r", [w], dtype=(tl.uint32, tl.uint32), is_pure=True, pack=1,
    )


# Nibbles 0-3 of a word index table entries 0-7 ($3, $4) and 8-15 ($5, $6) by two prmt; the high nibble bit picks
# between them through a byte mask from a sign-replicating prmt (selector 8: replicate the sign of 0x80; 4: zero).
_NIBBLE_LOOKUP_HALF = (
    "and.b32 s, {f}, 0x7777; prmt.b32 lo, $3, $4, s; prmt.b32 hi, $5, $6, s; "
    "and.b32 t, {f}, 0x8888; shr.b32 t, t, 1; add.u32 t, t, 0x4444; prmt.b32 m, sgn, zero, t; "
    "lop3.b32 {out}, hi, lo, m, 0xE4; "
)
_NIBBLE_LOOKUP = tl.constexpr(
    "{ .reg .b32 s, t, lo, hi, m, f, a, b, sgn, zero; mov.b32 sgn, 0x80; mov.b32 zero, 0; "
    + _NIBBLE_LOOKUP_HALF.format(f="$2", out="a") + "shr.b32 f, $2, 16; " + _NIBBLE_LOOKUP_HALF.format(f="f", out="b")
    + "prmt.b32 $0, a, b, 0x5140; prmt.b32 $1, a, b, 0x7362; }"
)


@triton.jit
def nibbles_to_table_bytes(w, t0, t1, t2, t3):
    """Each of a word's eight nibbles replaced by its entry of a 16-byte table (``t0`` entries 0-3, ..., ``t3`` 12-15,
    byte ``i`` of word ``j`` entry ``4j + i``): two words, the bytes of nibbles ``(0, 4, 1, 5)`` and ``(2, 6, 3, 7)``,
    which for the kernels' interleaved INT4 words (``int_word_pairs``) are K offsets 0-3 and 4-7."""
    return tl.inline_asm_elementwise(
        _NIBBLE_LOOKUP, "=r,=r,r,r,r,r,r", [w, t0, t1, t2, t3], dtype=(tl.uint32, tl.uint32), is_pure=True, pack=1,
    )


@triton.jit
def e4m3x4_to_f16x2(words):
    """Each word's four E4M3 bytes as two words of fp16 pairs (bytes ``(0, 1)`` and ``(2, 3)``), exact: every E4M3 value
    is an fp16 value."""
    return tl.inline_asm_elementwise(
        "{ .reg .b16 l, h; mov.b32 {l, h}, $2; cvt.rn.f16x2.e4m3x2 $0, l; cvt.rn.f16x2.e4m3x2 $1, h; }",
        "=r,=r,r", [words], dtype=(tl.uint32, tl.uint32), is_pure=True, pack=1,
    )


@triton.jit
def f16x2_dot4_f32(x01, x23, y01, y23):
    """The sum of four products, ``x`` and ``y`` each two words of fp16 pairs, chained in order through sm_100's
    mixed-precision ``fma.rn.f32.f16``: fp16 operands into an fp32 accumulator, so no operand is widened first. A
    product of two fp16-exact E4M3 values is exact before the fp32 rounding."""
    return tl.inline_asm_elementwise(
        """{ .reg .f16 x0, x1, x2, x3, y0, y1, y2, y3;
        mov.b32 {x0, x1}, $1; mov.b32 {x2, x3}, $2; mov.b32 {y0, y1}, $3; mov.b32 {y2, y3}, $4;
        fma.rn.f32.f16 $0, x0, y0, 0f00000000;
        fma.rn.f32.f16 $0, x1, y1, $0;
        fma.rn.f32.f16 $0, x2, y2, $0;
        fma.rn.f32.f16 $0, x3, y3, $0; }""",
        "=r,r,r,r,r", [x01, x23, y01, y23], dtype=tl.float32, is_pure=True, pack=1,
    )


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


@triton.jit
def dp4a(a, b):
    """The signed 4-way byte dot product of each pair of words, accumulated into int32 from zero."""
    return tl.inline_asm_elementwise(
        "dp4a.s32.s32 $0, $1, $2, 0;", "=r,r,r", [a, b], dtype=tl.int32, is_pure=True, pack=1,
    )


@triton.jit
def f32x2_to_e2m1x2(hi, lo):
    """Two f32 rounded to one packed E2M1 byte (``hi`` in the high nibble), nearest-even and saturating, in a uint16
    (PTX has no 8-bit operand)."""
    return tl.inline_asm_elementwise(
        "{ .reg .b8 t8; cvt.rn.satfinite.e2m1x2.f32 t8, $1, $2; cvt.u16.u8 $0, t8; }",
        "=h,f,f", [hi, lo], dtype=tl.uint16, is_pure=True, pack=1,
    )


@triton.jit
def e2m1x2_to_f16x2(packed):
    """One packed E2M1 byte (in a 16-bit register; PTX has no 8-bit operand) as its two fp16 values in a uint32,
    low half the low nibble, exact: ONE convert per pair against the bit-twiddle form's ~10 integer ops (gpt-oss W4A16
    decode weight stream: 39.6us bit-twiddle, either axis, vs 11.6us here, against an 8.0us load-only floor)."""
    return tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.u8.u16 t, $1; cvt.rn.f16x2.e2m1x2 $0, t; }",
        "=r,h", [packed.to(tl.uint16)], dtype=tl.uint32, is_pure=True, pack=1,
    )


@triton.jit
def e2m1x8_to_f16x2(words):
    """Each word's eight E2M1 values (byte ``i`` holds values ``2i`` and ``2i + 1``, low nibble first) as four words of
    fp16 pairs, exact, one convert per byte."""
    return tl.inline_asm_elementwise(
        "{ .reg .b8 b0, b1, b2, b3; mov.b32 {b0, b1, b2, b3}, $4; cvt.rn.f16x2.e2m1x2 $0, b0; "
        "cvt.rn.f16x2.e2m1x2 $1, b1; cvt.rn.f16x2.e2m1x2 $2, b2; cvt.rn.f16x2.e2m1x2 $3, b3; }",
        "=r,=r,=r,=r,r", [words], dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32), is_pure=True, pack=1,
    )


@triton.jit
def e2m1x2_to_e4m3x2(packed):
    """One packed E2M1 byte as its two E4M3 values in a uint16, low byte the low nibble, through fp16; both hops are
    exact, two converts against the bit-build's ~8 integer ops per value."""
    return tl.inline_asm_elementwise(
        "{ .reg .b8 t; .reg .b32 f; cvt.u8.u16 t, $1; cvt.rn.f16x2.e2m1x2 f, t; "
        "cvt.rn.satfinite.e4m3x2.f16x2 $0, f; }",
        "=h,h", [packed.to(tl.uint16)], dtype=tl.uint16, is_pure=True, pack=1,
    )


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
def int_words_fma_rows(operands, BITS: tl.constexpr, BM: tl.constexpr, FP16: tl.constexpr):
    """``_int_word_dot_asm`` over ``operands`` = (words, packed zeros, packed scales, then each of the ``BM`` token
    rows' ``16 // BITS`` activation pairs): ``BM`` words of fp32 bits, each row's sum over the word's values."""
    IDX: tl.constexpr = (0 if BITS == 2 else (3 if BITS == 4 else 6)) + (0 if BM == 1 else (1 if BM == 2 else 2))
    return tl.inline_asm_elementwise(
        _INT_WORD_DOT_ASMS[IDX + (6 if FP16 else 0)], _INT_WORD_DOT_CONSTRAINTS[IDX], operands,
        dtype=(tl.uint32,) * BM, is_pure=True, pack=1,
    )
