"""Flash Attention CUTE (CUDA Template Engine) implementation."""

__version__ = "4.0.0.beta30"

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

# Nothing else in the kernel imports this module, so it is not reachable as
# `<kernel>.compute_block_sparsity` unless it is imported here.
from . import compute_block_sparsity  # noqa: F401

__all__ = [
    "compute_block_sparsity",
    "flash_attn_func",
    "flash_attn_varlen_func",
]
