from typing import List, Union, Tuple

import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.nn as nn

from ._ops import add_op_namespace_prefix, ops


# Fake implementations, so that `torch.compile` can trace the ops.
@torch.library.register_fake(add_op_namespace_prefix("ms_deform_attn_forward"))
def _ms_deform_attn_forward_fake(
    value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step
):
    batch, _, num_heads, channels = value.shape
    return value.new_empty(batch, sampling_loc.shape[1], num_heads * channels)


@torch.library.register_fake(add_op_namespace_prefix("ms_deform_attn_backward"))
def _ms_deform_attn_backward_fake(
    value,
    spatial_shapes,
    level_start_index,
    sampling_loc,
    attn_weight,
    grad_output,
    im2col_step,
):
    return [t.new_empty(t.shape) for t in (value, sampling_loc, attn_weight)]


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
            # The kernel requires a contiguous gradient, which autograd does not
            # guarantee (e.g. the expanded gradient of `.sum()`).
            grad_output.contiguous(),
            context.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


class MultiScaleDeformableAttention(nn.Module):
    can_torch_compile: bool = True

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
        # The kernel needs a single dtype. Under autocast, Transformers passes
        # a reduced precision `value` with float32 sampling locations and
        # attention weights, for which the reference implementation (with
        # `grid_sample`) computes and returns float32. Do the same.
        dtype = torch.promote_types(
            torch.promote_types(value.dtype, sampling_locations.dtype),
            attention_weights.dtype,
        )
        value, sampling_locations, attention_weights = (
            t.to(dtype) for t in (value, sampling_locations, attention_weights)
        )
        return MultiScaleDeformableAttentionFunction.apply(
            value,
            value_spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )


__all__ = ["MultiScaleDeformableAttention"]
