"""Flash Attention CUTE (CUDA Template Engine) implementation."""

__version__ = "4.0.0.beta30"

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

# Internal symbols the tests need to reach. Not public API -- see the module
# docstring.
from . import _private_for_testing  # noqa: F401

__all__ = [
    "_private_for_testing",
    "flash_attn_func",
    "flash_attn_varlen_func",
]
