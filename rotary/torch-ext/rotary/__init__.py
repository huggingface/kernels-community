from typing import Tuple
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
):
    ops.apply_rotary(x1, x2, cos, sin, out1, out2, conj)


__all__ = ["apply_rotary"]
