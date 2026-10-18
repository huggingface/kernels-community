"""AITER RoPE kernels for AMD ROCm (Triton).

Triton-based RoPE (Rotary Position Embedding) kernels repackaged from the
[ROCm/aiter](https://github.com/ROCm/aiter) project. Exposes
``apply_rotary_transformers`` — drop-in replacement for the same-named
function in ``kernels-community/rotary`` — so transformers can use this as
the ROCm entry of ``_KERNEL_MAPPING["rotary_pos_emb"]`` with zero model-side
changes.
"""

from . import layers
from .layers import _apply_rotary
from .rope import (
    RotateStyle,
    rope_cached_fwd,
    rope_cached_fwd_inplace,
)


__kernel_metadata__ = {
    "license": "mit",
}


def apply_rotary_transformers(q, k, cos, sin, unsqueeze_dim=1):
    """Apply NEOX-style RoPE to ``q`` and ``k``.

    Signature mirrors ``kernels-community/rotary``'s ``apply_rotary_transformers``
    so this kernel can be a drop-in replacement under the ``"rocm"`` entry of
    transformers' ``_KERNEL_MAPPING["rotary_pos_emb"]``.

    Args:
        q, k: ``(batch, heads, seq, head_dim)``.
        cos, sin: ``(batch, seq, head_dim)`` or ``(batch, seq, head_dim // 2)`` — pre-``unsqueeze``
            form. Rows may hold different positions (e.g. left padding).
        unsqueeze_dim: accepted for API parity; the kernel reads positions from
            the already-computed ``cos`` / ``sin``.

    Returns:
        ``(q_embed, k_embed)`` in the same shape as ``q``, ``k``.
    """
    return _apply_rotary(q, k, cos, sin)


# Add torch compile support for functions
apply_rotary_transformers.can_torch_compile = True


__all__ = [
    "__kernel_metadata__",
    "RotateStyle",
    "apply_rotary_transformers",
    "layers",
    "rope_cached_fwd",
    "rope_cached_fwd_inplace",
]
