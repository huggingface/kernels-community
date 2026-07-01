"""AITER Composable-Kernel FlashAttention kernels for AMD ROCm (compiled HIP).

Compiled (Composable Kernel / HIP) counterpart to the Triton-based
``aiter-flash-attn`` package, repackaged from the CK FMHA forward path of the
`ROCm/aiter <https://github.com/ROCm/aiter>`_ project as a self-contained
Hugging Face Hub kernel. Exposes the FlashAttention forward entry points with
learnable attention sink support (e.g. gpt-oss).
"""

from .mha import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)


__kernel_metadata__ = {
    "license": "mit",
}


__all__ = [
    "__kernel_metadata__",
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
]
