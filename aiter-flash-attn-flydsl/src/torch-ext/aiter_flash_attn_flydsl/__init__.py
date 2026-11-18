"""AITER gfx1201 FlyDSL attention and FP8 quantization."""

from .attention import attention
from .fmha import flydsl_flash_attn_func as flash_attn_func
from .fp8_quant import flydsl_fp8_quant as fp8_quant

__kernel_metadata__ = {"license": "mit"}
__all__ = ["attention", "flash_attn_func", "fp8_quant"]
