from typing import List, Union, Tuple

from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.nn as nn

from ._ops import ops


class MultiScaleDeformableAttentionFunction(Function):
    @staticmethod
    def forward(
        context,
        value: Tensor,
        value_spatial_shapes: Tensor,
        value_level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        im2col_step: int,
    ):
        context.im2col_step = im2col_step
        output = ops.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            context.im2col_step,
        )
        context.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(context, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = context.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = ops.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            context.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


class MultiScaleDeformableAttention(nn.Module):
    def forward(
        self,
        value: Tensor,
        value_spatial_shapes: Tensor,
        value_spatial_shapes_list: List[Tuple],
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        im2col_step: int,
    ):
        return MultiScaleDeformableAttentionFunction.apply(
            value,
            value_spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )


__all__ = ["MultiScaleDeformableAttention"]
