from typing import List
import torch

from ._ops import ops
from . import layers


def ms_deform_attn_backward(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_loc: torch.Tensor,
    attn_weight: torch.Tensor,
    grad_output: torch.Tensor,
    im2col_step: int,
) -> List[torch.Tensor]:
    return ops.ms_deform_attn_backward(
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc,
        attn_weight,
        grad_output,
        im2col_step,
    )


def ms_deform_attn_forward(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_loc: torch.Tensor,
    attn_weight: torch.Tensor,
    im2col_step: int,
) -> torch.Tensor:
    return ops.ms_deform_attn_forward(
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc,
        attn_weight,
        im2col_step,
    )


__all__ = ["layers", "ms_deform_attn_forward", "ms_deform_attn_backward"]
