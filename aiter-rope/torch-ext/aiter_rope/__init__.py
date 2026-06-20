"""AITER RoPE kernels for AMD ROCm (Triton).

Triton-based RoPE (Rotary Position Embedding) kernels repackaged from the
[ROCm/aiter](https://github.com/ROCm/aiter) project. Exposes
``apply_rotary_transformers`` — drop-in replacement for the same-named
function in ``kernels-community/rotary`` — so transformers can use this as
the ROCm entry of ``_KERNEL_MAPPING["rotary_pos_emb"]`` with zero model-side
changes.
"""

import torch

from .rope import (
    RotateStyle,
    rope_cached_fwd,
    rope_cached_fwd_inplace,
)


__kernel_metadata__ = {
    "license": "mit",
}


def _to_sbhd(t: torch.Tensor) -> torch.Tensor:
    # transformers passes attention tensors as (batch, heads, seq, head_dim);
    # AITER's kernels operate in (seq, batch, heads, head_dim).
    return t.permute(2, 0, 1, 3).contiguous()


def _to_bhsd(t: torch.Tensor) -> torch.Tensor:
    return t.permute(1, 2, 0, 3)


def apply_rotary_transformers(q, k, cos, sin, unsqueeze_dim=1):
    """Apply NEOX-style RoPE to ``q`` and ``k``.

    Signature mirrors ``kernels-community/rotary``'s ``apply_rotary_transformers``
    so this kernel can be a drop-in replacement under the ``"rocm"`` entry of
    transformers' ``_KERNEL_MAPPING["rotary_pos_emb"]``.

    Args:
        q, k: ``(batch, heads, seq, head_dim)``.
        cos, sin: ``(batch, seq, head_dim // 2)`` — pre-``unsqueeze`` form.
        unsqueeze_dim: accepted for API parity; the kernel reads positions from
            the already-computed ``cos`` / ``sin``.

    Returns:
        ``(q_embed, k_embed)`` in the same shape as ``q``, ``k``.
    """
    q_sbhd = _to_sbhd(q)
    k_sbhd = _to_sbhd(k)

    # Cached path expects (seq, 1, 1, head_dim/2). Take batch index 0 — for the
    # standard (unpadded, equal-length) inference case all batch rows share the
    # same cos/sin since they come from the same position_ids.
    cos_freqs = cos[0].unsqueeze(1).unsqueeze(1).contiguous()
    sin_freqs = sin[0].unsqueeze(1).unsqueeze(1).contiguous()

    common = dict(
        rotate_style=int(RotateStyle.NEOX),
        reuse_freqs_front_part=True,
        nope_first=False,
    )
    q_out_sbhd = rope_cached_fwd(q_sbhd, cos_freqs, sin_freqs, **common)
    k_out_sbhd = rope_cached_fwd(k_sbhd, cos_freqs, sin_freqs, **common)

    return _to_bhsd(q_out_sbhd), _to_bhsd(k_out_sbhd)


# Add torch compile support for functions
apply_rotary_transformers.can_torch_compile = True


__all__ = [
    "__kernel_metadata__",
    "RotateStyle",
    "apply_rotary_transformers",
    "rope_cached_fwd",
    "rope_cached_fwd_inplace",
]
