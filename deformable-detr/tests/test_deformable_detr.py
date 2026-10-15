import kernels
import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"

# The col2im (backward) launcher picks a different kernel depending on the
# number of channels, see `ms_deformable_col2im_cuda`.
CHANNELS = {
    "blocksize_aware_v1_small": 8,
    "blocksize_aware_v1": 32,
    "shm_reduce_v1": 48,
    "shm_reduce_v2": 96,
    "blocksize_aware_v2": 256,
    "gm": 1040,
    "shm_reduce_v2_multi_blocks": 2048,
}


@pytest.fixture(scope="module")
def deformable_detr():
    return kernels.get_kernel("kernels-community/deformable-detr", version=2)


def make_inputs(
    batch=2,
    num_heads=2,
    channels=32,
    spatial_shapes=((8, 8), (4, 6), (2, 3)),
    num_query=10,
    num_point=4,
    dtype=torch.float32,
    requires_grad=False,
    seed=0,
):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=DEVICE)
    level_start_index = torch.cat(
        (shapes.new_zeros(1), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1])
    )
    num_levels = len(spatial_shapes)
    spatial_size = sum(h * w for h, w in spatial_shapes)

    value = torch.randn(
        batch, spatial_size, num_heads, channels, device=DEVICE, generator=generator
    )
    # Some locations fall outside of [0, 1] to exercise the zero padding.
    sampling_loc = (
        torch.rand(
            batch,
            num_query,
            num_heads,
            num_levels,
            num_point,
            2,
            device=DEVICE,
            generator=generator,
        )
        * 1.2
        - 0.1
    )
    attn_weight = torch.rand(
        batch,
        num_query,
        num_heads,
        num_levels,
        num_point,
        device=DEVICE,
        generator=generator,
    )
    attn_weight = attn_weight / attn_weight.sum((-1, -2), keepdim=True)

    value, sampling_loc, attn_weight = (
        t.to(dtype).requires_grad_(requires_grad)
        for t in (value, sampling_loc, attn_weight)
    )
    return value, shapes, level_start_index, sampling_loc, attn_weight


def ms_deform_attn_ref(value, spatial_shapes, sampling_loc, attn_weight):
    """Reference implementation with `grid_sample`, as in the Deformable DETR repo."""
    batch, _, num_heads, channels = value.shape
    _, num_query, _, num_levels, num_point, _ = sampling_loc.shape
    shapes = [(int(h), int(w)) for h, w in spatial_shapes.tolist()]
    value_list = value.split([h * w for h, w in shapes], dim=1)
    sampling_grids = 2 * sampling_loc - 1
    sampled = []
    for level, (h, w) in enumerate(shapes):
        value_l = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch * num_heads, channels, h, w)
        )
        grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled.append(
            F.grid_sample(
                value_l,
                grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    attn = attn_weight.transpose(1, 2).reshape(
        batch * num_heads, 1, num_query, num_levels * num_point
    )
    out = (torch.stack(sampled, dim=-2).flatten(-2) * attn).sum(-1)
    return out.view(batch, num_heads * channels, num_query).transpose(1, 2).contiguous()


TOLERANCES = {
    torch.float64: dict(rtol=1e-9, atol=1e-9),
    torch.float32: dict(rtol=1e-4, atol=1e-4),
    torch.float16: dict(rtol=1e-2, atol=1e-2),
    torch.bfloat16: dict(rtol=5e-2, atol=5e-2),
}


def run_kernel(
    deformable_detr,
    value,
    shapes,
    level_start_index,
    sampling_loc,
    attn_weight,
    im2col_step,
):
    return deformable_detr.layers.MultiScaleDeformableAttention()(
        value,
        shapes,
        shapes.tolist(),
        level_start_index,
        sampling_loc,
        attn_weight,
        im2col_step,
    )


def check_against_ref(deformable_detr, dtype, im2col_step=64, **kwargs):
    value, shapes, level_start_index, sampling_loc, attn_weight = make_inputs(
        dtype=dtype, requires_grad=True, **kwargs
    )
    out = run_kernel(
        deformable_detr,
        value,
        shapes,
        level_start_index,
        sampling_loc,
        attn_weight,
        im2col_step,
    )
    assert out.dtype == dtype

    # The reference runs in at least float32 on the same (rounded) inputs.
    ref_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    ref_inputs = [
        t.detach().to(ref_dtype).requires_grad_()
        for t in (value, sampling_loc, attn_weight)
    ]
    ref = ms_deform_attn_ref(ref_inputs[0], shapes, ref_inputs[1], ref_inputs[2])
    torch.testing.assert_close(out.to(ref_dtype), ref, **TOLERANCES[dtype])

    grad_out = torch.randn_like(ref)
    out.backward(grad_out.to(dtype))
    ref.backward(grad_out)
    for name, actual, expected in zip(
        ("value", "sampling_loc", "attn_weight"),
        (value, sampling_loc, attn_weight),
        ref_inputs,
    ):
        assert actual.grad.dtype == dtype
        # Gradients accumulate many atomic adds in the input precision.
        scale = expected.grad.abs().max().item()
        tol = dict(
            rtol=TOLERANCES[dtype]["rtol"],
            atol=TOLERANCES[dtype]["atol"] * max(scale, 1.0),
        )
        if name == "sampling_loc" and dtype in (torch.float16, torch.bfloat16):
            # The kernel computes sampling coordinates in the input precision.
            # A point close to a pixel boundary can then be rounded into a
            # neighbouring bilinear cell, where the (piecewise) gradient with
            # respect to the location differs. Allow a few such outliers.
            close = torch.isclose(actual.grad.to(ref_dtype), expected.grad, **tol)
            assert close.float().mean().item() >= 0.99, f"{(~close).sum()} mismatches"
        else:
            torch.testing.assert_close(actual.grad.to(ref_dtype), expected.grad, **tol)


@pytest.mark.kernels_ci
@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16], ids=str
)
def test_forward_backward(deformable_detr, dtype):
    check_against_ref(deformable_detr, dtype)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("channels", list(CHANNELS.values()), ids=list(CHANNELS))
def test_backward_kernel_variants(deformable_detr, channels):
    check_against_ref(
        deformable_detr, torch.float32, channels=channels, num_heads=1, num_query=6
    )


@pytest.mark.parametrize("channels", list(CHANNELS.values()), ids=list(CHANNELS))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
def test_backward_kernel_variants_half(deformable_detr, channels, dtype):
    check_against_ref(
        deformable_detr, dtype, channels=channels, num_heads=1, num_query=6
    )


@pytest.mark.kernels_ci
@pytest.mark.parametrize("im2col_step", [1, 2, 64])
def test_im2col_step(deformable_detr, im2col_step):
    check_against_ref(deformable_detr, torch.float32, batch=4, im2col_step=im2col_step)


@pytest.mark.kernels_ci
@pytest.mark.parametrize(
    "batch, im2col_step",
    # Transformers always passes im2col_step=64. The kernel uses the largest
    # divisor of the batch that does not exceed it.
    [(3, 2), (66, 64), (67, 64), (130, 64)],
)
def test_batch_not_divisible_by_im2col_step(deformable_detr, batch, im2col_step):
    check_against_ref(
        deformable_detr,
        torch.float32,
        batch=batch,
        im2col_step=im2col_step,
        num_query=4,
        channels=8,
        num_point=2,
    )


@pytest.mark.kernels_ci
def test_empty_batch(deformable_detr):
    value, shapes, level_start_index, sampling_loc, attn_weight = make_inputs(batch=0)
    out = run_kernel(
        deformable_detr, value, shapes, level_start_index, sampling_loc, attn_weight, 64
    )
    assert out.shape == (0, sampling_loc.shape[1], value.shape[2] * value.shape[3])


@pytest.mark.kernels_ci
@pytest.mark.parametrize("value_dtype", [torch.float16, torch.bfloat16], ids=str)
def test_layer_mixed_dtypes(deformable_detr, value_dtype):
    # Under autocast, Transformers passes a reduced precision value with
    # float32 sampling locations and attention weights. Like the reference,
    # the layer then computes and returns float32.
    value, shapes, level_start_index, sampling_loc, attn_weight = make_inputs(
        requires_grad=True
    )
    value = value.detach().to(value_dtype).requires_grad_()
    out = run_kernel(
        deformable_detr, value, shapes, level_start_index, sampling_loc, attn_weight, 64
    )
    assert out.dtype == torch.float32

    ref_value = value.detach().float().requires_grad_()
    ref_loc, ref_weight = (
        t.detach().clone().requires_grad_() for t in (sampling_loc, attn_weight)
    )
    ref = ms_deform_attn_ref(ref_value, shapes, ref_loc, ref_weight)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    grad_out = torch.randn_like(ref)
    out.backward(grad_out)
    ref.backward(grad_out)
    assert value.grad.dtype == value_dtype
    torch.testing.assert_close(
        value.grad, ref_value.grad.to(value_dtype), rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(sampling_loc.grad, ref_loc.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(attn_weight.grad, ref_weight.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.kernels_ci
def test_functional_api(deformable_detr):
    value, shapes, level_start_index, sampling_loc, attn_weight = make_inputs()
    out = deformable_detr.ms_deform_attn_forward(
        value, shapes, level_start_index, sampling_loc, attn_weight, 64
    )
    torch.testing.assert_close(
        out,
        ms_deform_attn_ref(value, shapes, sampling_loc, attn_weight),
        rtol=1e-4,
        atol=1e-4,
    )

    grad_output = torch.randn_like(out)
    grads = deformable_detr.ms_deform_attn_backward(
        value, shapes, level_start_index, sampling_loc, attn_weight, grad_output, 64
    )
    assert [g.shape for g in grads] == [
        value.shape,
        sampling_loc.shape,
        attn_weight.shape,
    ]


@pytest.mark.kernels_ci
def test_non_contiguous_grad_output(deformable_detr):
    # `.sum()` produces an expanded, non-contiguous gradient.
    value, shapes, level_start_index, sampling_loc, attn_weight = make_inputs(
        requires_grad=True
    )
    run_kernel(
        deformable_detr, value, shapes, level_start_index, sampling_loc, attn_weight, 64
    ).sum().backward()
    assert (
        value.grad is not None
        and sampling_loc.grad is not None
        and attn_weight.grad is not None
    )


@pytest.mark.kernels_ci
def test_double_backward_raises(deformable_detr):
    value, shapes, level_start_index, sampling_loc, attn_weight = make_inputs(
        requires_grad=True
    )
    out = run_kernel(
        deformable_detr, value, shapes, level_start_index, sampling_loc, attn_weight, 64
    )
    (grad_value,) = torch.autograd.grad(out.square().sum(), value, create_graph=True)
    with pytest.raises(RuntimeError, match="differentiate twice"):
        grad_value.sum().backward()


@pytest.mark.kernels_ci
def test_input_validation(deformable_detr):
    value, shapes, level_start_index, sampling_loc, attn_weight = make_inputs()
    op = deformable_detr.ms_deform_attn_forward
    with pytest.raises(RuntimeError, match="contiguous"):
        op(
            value.transpose(1, 2).contiguous().transpose(1, 2),
            shapes,
            level_start_index,
            sampling_loc,
            attn_weight,
            64,
        )
    with pytest.raises(RuntimeError, match="CUDA"):
        op(value.cpu(), shapes, level_start_index, sampling_loc, attn_weight, 64)
    with pytest.raises(RuntimeError, match="scalar type"):
        op(value, shapes, level_start_index, sampling_loc.half(), attn_weight, 64)
    with pytest.raises(RuntimeError, match="im2col_step"):
        op(value, shapes, level_start_index, sampling_loc, attn_weight, 0)
