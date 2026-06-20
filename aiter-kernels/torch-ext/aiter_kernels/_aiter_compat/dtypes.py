"""Minimal local replacement for ``aiter.utility.dtypes``.

Upstream parses a generated C header (``csrc/include/aiter_enum.h``) to derive
its dtype table and depends on ``aiter.ops.enum`` (which in turn pulls the
JIT C++ extension). For a Triton-only Hub kernel we only need the dtype
constants the Triton ops actually reference — none of the enum / header
machinery is required.
"""

from __future__ import annotations

import torch

from .chip_info import get_gfx


_FP8_BY_ARCH = {
    "gfx942": getattr(torch, "float8_e4m3fnuz", None),
    "gfx950": getattr(torch, "float8_e4m3fn", None),
    "gfx1200": getattr(torch, "float8_e4m3fn", None),
    "gfx1201": getattr(torch, "float8_e4m3fn", None),
    "gfx1250": getattr(torch, "float8_e4m3fn", None),
}

_8BIT_FALLBACK = torch.uint8


def get_dtype_fp8() -> torch.dtype:
    return _FP8_BY_ARCH.get(get_gfx()) or _8BIT_FALLBACK


i4x2 = getattr(torch, "int4", _8BIT_FALLBACK)
fp4x2 = getattr(torch, "float4_e2m1fn_x2", _8BIT_FALLBACK)
fp8 = get_dtype_fp8()
fp8_e8m0 = getattr(torch, "float8_e8m0fnu", _8BIT_FALLBACK)
fp16 = torch.float16
bf16 = torch.bfloat16
fp32 = torch.float32
u32 = torch.uint32
i32 = torch.int32
i16 = torch.int16
i8 = torch.int8
u8 = torch.uint8
i64 = torch.int64
u64 = torch.uint64
