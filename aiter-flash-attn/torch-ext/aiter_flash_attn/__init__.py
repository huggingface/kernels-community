"""AITER Flash Attention kernels for AMD ROCm (Triton MHA).

Repackaged from the `aiter.ops.triton.attention.mha` module of the
`ROCm/aiter <https://github.com/ROCm/aiter>`_ project as a self-contained
Hugging Face Hub kernel. Exposes the two FlashAttention entry points used by
the `transformers` ROCm fallback.
"""

from .mha import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    mha_set_impl,
    mha_set_use_fused_bwd_kernel,
    mha_set_use_int64_strides,
)


__kernel_metadata__ = {
    "license": "mit",
}


__all__ = [
    "__kernel_metadata__",
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "mha_set_impl",
    "mha_set_use_fused_bwd_kernel",
    "mha_set_use_int64_strides",
]
