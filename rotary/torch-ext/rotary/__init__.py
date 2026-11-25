import torch

from . import layers
from ._ops import ops
from .layers import _ApplyRotary


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotary kernel implementation wrapper
    Adapts rotary kernel implementation to match transformers apply_rotary_pos_emb signature
    """
    half_dim = q.shape[-1] // 2
    cos = cos.unsqueeze(unsqueeze_dim)[..., :half_dim]
    sin = sin.unsqueeze(unsqueeze_dim)[..., :half_dim]
    return _ApplyRotary.apply(q, cos, sin), _ApplyRotary.apply(k, cos, sin)


# Add torch compile and backward support for functions
apply_rotary_transformers.can_torch_compile = True
apply_rotary_transformers.has_backward = True


# Keeping `apply_rotary` and `apply_rotary_transformers` for BC
__all__ = ["apply_rotary", "apply_rotary_transformers", "layers"]
