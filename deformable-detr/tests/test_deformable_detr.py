"""Tests for the deformable-detr kernels, checked against a pure-PyTorch reference."""

import kernels
import pytest
import torch
import torch.nn as nn


deformable_detr = kernels.get_kernel("kernels-community/deformable-detr", version=1)

DEVICE = "cuda"


# Copied from `MultiScaleDeformableAttention.forward` in
# transformers/models/deformable_detr/modeling_deformable_detr.py
# (Copyright 2022 SenseTime and The HuggingFace Inc. team, Apache-2.0).
def ms_deform_attn_ref(
    value,
    value_spatial_shapes_list,
    sampling_locations,
    attention_weights,
):
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([height * width for height, width in value_spatial_shapes_list], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes_list):
        # batch_size, height*width, num_heads, hidden_dim
        # -> batch_size, height*width, num_heads*hidden_dim
        # -> batch_size, num_heads*hidden_dim, height*width
        # -> batch_size*num_heads, hidden_dim, height, width
        value_l_ = (
            value_list[level_id]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        # batch_size*num_heads, hidden_dim, num_queries, num_points
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


def make_inputs(
    spatial_shapes_list,
    batch=2,
    num_heads=2,
    channels=8,
    num_queries=5,
    num_points=4,
    dtype=torch.float32,
    seed=0,
):
    g = torch.Generator().manual_seed(seed)
    num_levels = len(spatial_shapes_list)
    spatial_size = sum(h * w for h, w in spatial_shapes_list)

    value = torch.randn(batch, spatial_size, num_heads, channels, generator=g)

    # The gradient w.r.t. the sampling locations is discontinuous at integer
    # pixel coordinates (the kernel samples at `loc * size - 0.5`). Keep the
    # sampling points at least 0.1 px away from those, so that rounding in
    # half precision cannot move a point across a discontinuity. Points are
    # also placed (partially) outside the feature map to exercise the zero
    # padding.
    sampling_locations = torch.empty(
        batch, num_queries, num_heads, num_levels, num_points, 2
    )
    for level, (height, width) in enumerate(spatial_shapes_list):
        for coord, size in enumerate((width, height)):
            shape = (batch, num_queries, num_heads, num_points)
            pixel = torch.randint(-2, size + 1, shape, generator=g) + (
                0.1 + 0.8 * torch.rand(shape, generator=g)
            )
            sampling_locations[:, :, :, level, :, coord] = (pixel + 0.5) / size
    attention_weights = torch.rand(
        batch, num_queries, num_heads, num_levels, num_points, generator=g
    )
    attention_weights = attention_weights / attention_weights.sum(
        (-2, -1), keepdim=True
    )

    spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.long)
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )

    return (
        value.to(DEVICE, dtype),
        spatial_shapes.to(DEVICE),
        level_start_index.to(DEVICE),
        sampling_locations.to(DEVICE, dtype),
        attention_weights.to(DEVICE, dtype),
    )


def reference_with_grads(
    value, spatial_shapes_list, sampling_locations, attention_weights, grad_output
):
    """Reference output and input gradients, computed in float64."""
    value = value.double().requires_grad_()
    sampling_locations = sampling_locations.double().requires_grad_()
    attention_weights = attention_weights.double().requires_grad_()
    output = ms_deform_attn_ref(
        value, spatial_shapes_list, sampling_locations, attention_weights
    )
    output.backward(grad_output.double())
    return output.detach(), value.grad, sampling_locations.grad, attention_weights.grad


# Tolerances are relative to the largest magnitude in the expected tensor, since
# e.g. the sampling-location gradients scale with the feature map size.
TOLERANCES = {
    torch.float64: 1e-10,
    torch.float32: 1e-5,
    torch.float16: 1e-2,
    torch.bfloat16: 5e-2,
}


def assert_close(actual, expected, dtype):
    tol = TOLERANCES[dtype]
    scale = max(expected.abs().max().item(), 1.0)
    torch.testing.assert_close(
        actual.double(), expected, atol=tol * scale, rtol=tol
    )


SHAPES = {
    "single-level": [(4, 4)],
    "multi-level": [(8, 8), (4, 4), (2, 2), (1, 1)],
    "non-square": [(6, 10), (3, 5)],
}

DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]

# The CI subset: one multi-level configuration in a full- and half-precision
# dtype is enough to catch most breakage.
CI_PARAMS = {("multi-level", torch.float32), ("multi-level", torch.float16)}


def shape_dtype_params():
    return [
        pytest.param(
            shapes,
            dtype,
            id=f"{name}-{str(dtype).removeprefix('torch.')}",
            marks=[pytest.mark.kernels_ci] if (name, dtype) in CI_PARAMS else [],
        )
        for name, shapes in SHAPES.items()
        for dtype in DTYPES
    ]


@pytest.mark.parametrize("im2col_step", [1, 2, 64])
@pytest.mark.parametrize("spatial_shapes_list,dtype", shape_dtype_params())
def test_forward(spatial_shapes_list, dtype, im2col_step):
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = (
        make_inputs(spatial_shapes_list, batch=4, dtype=dtype)
    )

    out = deformable_detr.ms_deform_attn_forward(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    )
    expected = ms_deform_attn_ref(
        value.double(),
        spatial_shapes_list,
        sampling_locations.double(),
        attention_weights.double(),
    )

    assert out.dtype == dtype
    assert out.shape == expected.shape
    assert_close(out, expected, dtype)


@pytest.mark.parametrize("im2col_step", [1, 2, 64])
@pytest.mark.parametrize("spatial_shapes_list,dtype", shape_dtype_params())
def test_backward(spatial_shapes_list, dtype, im2col_step):
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = (
        make_inputs(spatial_shapes_list, batch=4, dtype=dtype)
    )
    num_queries = sampling_locations.shape[1]
    grad_output = torch.randn(
        value.shape[0],
        num_queries,
        value.shape[2] * value.shape[3],
        generator=torch.Generator().manual_seed(1),
    ).to(DEVICE, dtype)

    grad_value, grad_sampling_loc, grad_attn_weight = (
        deformable_detr.ms_deform_attn_backward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            im2col_step,
        )
    )
    _, ref_grad_value, ref_grad_sampling_loc, ref_grad_attn_weight = (
        reference_with_grads(
            value, spatial_shapes_list, sampling_locations, attention_weights, grad_output
        )
    )

    assert_close(grad_value, ref_grad_value, dtype)
    assert_close(grad_sampling_loc, ref_grad_sampling_loc, dtype)
    assert_close(grad_attn_weight, ref_grad_attn_weight, dtype)


# The backward kernel picks a different implementation depending on the number
# of channels per head: power-of-two sizes up to 1024 (templated block sizes),
# other sizes below/above 64, multiples of 1024 above 1024 (multi-block shared
# memory reduction), and other sizes above 1024 (global memory).
@pytest.mark.parametrize(
    "channels",
    [
        pytest.param(1, marks=pytest.mark.kernels_ci),
        pytest.param(3, marks=pytest.mark.kernels_ci),
        32,
        pytest.param(64, marks=pytest.mark.kernels_ci),
        pytest.param(96, marks=pytest.mark.kernels_ci),
        1024,
        pytest.param(1536, marks=pytest.mark.kernels_ci),
        pytest.param(2048, marks=pytest.mark.kernels_ci),
    ],
)
def test_backward_channels(channels):
    dtype = torch.float32
    spatial_shapes_list = SHAPES["multi-level"]
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = (
        make_inputs(
            spatial_shapes_list,
            batch=2,
            num_heads=2,
            channels=channels,
            num_queries=3,
            dtype=dtype,
        )
    )
    grad_output = torch.randn(
        2, 3, 2 * channels, generator=torch.Generator().manual_seed(1)
    ).to(DEVICE, dtype)

    grads = deformable_detr.ms_deform_attn_backward(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        grad_output,
        2,
    )
    _, *ref_grads = reference_with_grads(
        value, spatial_shapes_list, sampling_locations, attention_weights, grad_output
    )

    for grad, ref_grad in zip(grads, ref_grads):
        assert_close(grad, ref_grad, dtype)


@pytest.mark.kernels_ci
def test_layer():
    dtype = torch.float32
    spatial_shapes_list = SHAPES["multi-level"]
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = (
        make_inputs(spatial_shapes_list, batch=4, dtype=dtype)
    )
    value.requires_grad_()
    sampling_locations.requires_grad_()
    attention_weights.requires_grad_()

    layer = deformable_detr.layers.MultiScaleDeformableAttention()
    out = layer(
        value,
        spatial_shapes,
        spatial_shapes_list,
        level_start_index,
        sampling_locations,
        attention_weights,
        2,
    )
    grad_output = torch.randn(
        out.shape, generator=torch.Generator().manual_seed(1)
    ).to(DEVICE, dtype)
    out.backward(grad_output)

    ref_out, ref_grad_value, ref_grad_sampling_loc, ref_grad_attn_weight = (
        reference_with_grads(
            value.detach(),
            spatial_shapes_list,
            sampling_locations.detach(),
            attention_weights.detach(),
            grad_output,
        )
    )

    assert_close(out.detach(), ref_out, dtype)
    assert_close(value.grad, ref_grad_value, dtype)
    assert_close(sampling_locations.grad, ref_grad_sampling_loc, dtype)
    assert_close(attention_weights.grad, ref_grad_attn_weight, dtype)


@pytest.mark.kernels_ci
def test_batch_not_divisible_by_im2col_step():
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = (
        make_inputs(SHAPES["multi-level"], batch=3)
    )
    with pytest.raises(RuntimeError, match="must divide im2col_step"):
        deformable_detr.ms_deform_attn_forward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            2,
        )


@pytest.mark.kernels_ci
def test_non_contiguous_input():
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = (
        make_inputs(SHAPES["multi-level"], batch=2, num_heads=2, channels=8)
    )
    # Same shape, but non-contiguous.
    value = value.transpose(2, 3).contiguous().transpose(2, 3)
    with pytest.raises(RuntimeError, match="value tensor has to be contiguous"):
        deformable_detr.ms_deform_attn_forward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            2,
        )
