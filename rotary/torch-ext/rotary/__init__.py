from typing import Optional, Tuple
import torch

from ._ops import ops


def apply_rotary(
    x1: torch.Tensor,
    x2: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    conj: bool,
) -> None:
    ops.apply_rotary(x1, x2, cos, sin, out1, out2, conj)


def apply_rotary_transformers(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotary kernel implementation wrapper
    Adapts rotary kernel implementation to match transformers apply_rotary_pos_emb signature
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_rotated = q.clone()
    k_rotated = k.clone()

    # Get half dimension for rotation
    half_dim = q.shape[-1] // 2
    q1 = q_rotated[..., :half_dim]
    q2 = q_rotated[..., half_dim:]
    k1 = k_rotated[..., :half_dim]
    k2 = k_rotated[..., half_dim:]
    if cos.shape[-1] != half_dim:
        # Trim cos/sin to match half_dim
        cos = cos[..., :half_dim]
        sin = sin[..., :half_dim]

    apply_rotary(q1, q2, cos, sin, q1, q2, False)
    apply_rotary(k1, k2, cos, sin, k1, k2, False)
    return q_rotated, k_rotated


__all__ = ["apply_rotary", "apply_rotary_transformers"]
