# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER Triton RoPE kernels.

Mirrors ``aiter/ops/triton/rope/`` upstream and additionally exposes
``apply_rotary_transformers`` — a transformers-compatible shim that lets
this submodule serve as a drop-in replacement for the standalone
``kernels-community/aiter-rope`` repo (and for the ROCm entry of
transformers' ``_KERNEL_MAPPING["rotary_pos_emb"]``).
"""

from __future__ import annotations

import torch

from .rope import (
    RotateStyle,
    rope_cached_fwd,
    rope_cached_fwd_inplace,
)


def _to_sbhd(t: torch.Tensor) -> torch.Tensor:
    # transformers passes attention tensors as (batch, heads, seq, head_dim);
    # AITER's kernels operate in (seq, batch, heads, head_dim).
    return t.permute(2, 0, 1, 3).contiguous()


def _to_bhsd(t: torch.Tensor) -> torch.Tensor:
    return t.permute(1, 2, 0, 3)


def apply_rotary_transformers(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply NEOX-style RoPE to ``q`` and ``k``.

    Drop-in replacement for ``kernels-community/rotary``'s
    ``apply_rotary_transformers`` and for the same-named function in
    ``kernels-community/aiter-rope``.

    Args:
        q, k: ``(batch, heads, seq, head_dim)``.
        cos, sin: ``(batch, seq, head_dim // 2)`` — pre-``unsqueeze`` form.
        position_ids, unsqueeze_dim: accepted for API parity; the kernel reads
            positions from the already-computed ``cos`` / ``sin``.

    Returns:
        ``(q_embed, k_embed)`` in the same shape as ``q``, ``k``.
    """
    q_sbhd = _to_sbhd(q)
    k_sbhd = _to_sbhd(k)

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


__all__ = [
    "RotateStyle",
    "apply_rotary_transformers",
    "rope_cached_fwd",
    "rope_cached_fwd_inplace",
]
