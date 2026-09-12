"""Flash Attention CUTE (CUDA Template Engine) implementation."""

__version__ = "4.0.0.beta30"

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
