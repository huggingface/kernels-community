from dataclasses import dataclass

import kernels
import pytest
import torch
import torch.nn as nn

from test_deformable_detr import CHANNELS, make_inputs, ms_deform_attn_ref


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def deformable_detr():
    return kernels.get_kernel("kernels-community/deformable-detr", version=2)


@pytest.fixture(autouse=True)
def reset_compiler():
    # Every test starts without compiled graphs from earlier tests.
    torch.compiler.reset()


def clone_inputs(inputs):
    return [
        t.detach().clone().requires_grad_(t.requires_grad)
        if t.is_floating_point()
        else t
        for t in inputs
    ]


def run_and_grad(fn, inputs):
    inputs = clone_inputs(inputs)
    out = fn(*inputs)
    out.float().square().mean().backward()
    return out, [t.grad for t in inputs if t.requires_grad]


# The backward launcher picks a different kernel depending on the number of
# channels, see `CHANNELS`. The default of 32 channels runs in CI.
@pytest.mark.parametrize(
    "channels",
    [
        pytest.param(c, id=name, marks=pytest.mark.kernels_ci if c == 32 else ())
        for name, c in CHANNELS.items()
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=str)
@torch._dynamo.config.patch(error_on_recompile=True)
def test_layer_compile(deformable_detr, dtype, channels):
    module = deformable_detr.layers.MultiScaleDeformableAttention()
    value, shapes, level_start_index, sampling_loc, attn_weight = make_inputs(
        channels=channels, dtype=dtype, requires_grad=True
    )
    shapes_list = shapes.tolist()

    def fn(value, sampling_loc, attn_weight):
        # Surrounding ops so that the kernel is fused into a larger graph.
        out = module(
            value * 2,
            shapes,
            shapes_list,
            level_start_index,
            sampling_loc.sigmoid(),
            attn_weight.softmax(-1),
            64,
        )
        return out.relu()

    compiled = torch.compile(fn, fullgraph=True)
    inputs = [value, sampling_loc, attn_weight]
    # Inductor fuses the surrounding ops and keeps their intermediates in
    # float32, so bfloat16 results differ from eager by a few ulps.
    tol = dict(rtol=2e-2, atol=2e-3) if dtype == torch.bfloat16 else {}
    for _ in range(2):
        actual = run_and_grad(compiled, inputs)
        expected = run_and_grad(fn, inputs)
        torch.testing.assert_close(actual, expected, **tol)


@pytest.mark.kernels_ci
def test_functional_compile_matches_reference(deformable_detr):
    value, shapes, level_start_index, sampling_loc, attn_weight = make_inputs()

    def fn(value, sampling_loc, attn_weight):
        return deformable_detr.ms_deform_attn_forward(
            value, shapes, level_start_index, sampling_loc, attn_weight, 64
        )

    compiled = torch.compile(fn, fullgraph=True)
    actual = compiled(value, sampling_loc, attn_weight)
    expected = ms_deform_attn_ref(value, shapes, sampling_loc, attn_weight)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.kernels_ci
@torch._dynamo.config.patch(error_on_recompile=True)
def test_layer_compile_autocast(deformable_detr):
    # Under autocast the layer receives a bfloat16 value from a linear layer
    # but float32 sampling locations and attention weights.
    module = deformable_detr.layers.MultiScaleDeformableAttention()
    proj = torch.nn.Linear(32, 32, device="cuda")
    value, shapes, level_start_index, sampling_loc, attn_weight = make_inputs(
        requires_grad=True
    )
    shapes_list = shapes.tolist()

    def fn(value, sampling_loc, attn_weight):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            value = proj(value)
            return module(
                value,
                shapes,
                shapes_list,
                level_start_index,
                sampling_loc,
                attn_weight.softmax(-1),
                64,
            )

    compiled = torch.compile(fn, fullgraph=True)
    inputs = [value, sampling_loc, attn_weight]
    for _ in range(2):
        actual = run_and_grad(compiled, inputs)
        expected = run_and_grad(fn, inputs)
        assert actual[0].dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)


@pytest.mark.kernels_ci
# Sizes 0 and 1 are always specialized, and duck sizing ties sizes that are
# equal to another dimension (2 heads, 3 levels, 4 points).
@pytest.mark.parametrize(
    "batch, num_query", [(6, 11), (7, 13), (9, 17)], ids=lambda x: str(x)
)
def test_dynamic_shapes(deformable_detr, batch, num_query):
    module = deformable_detr.layers.MultiScaleDeformableAttention()
    compiled = torch.compile(module, fullgraph=True, dynamic=True)

    def check(batch, num_query):
        value, shapes, level_start_index, sampling_loc, attn_weight = make_inputs(
            batch=batch, num_query=num_query, requires_grad=True
        )
        args = (shapes, shapes.tolist(), level_start_index)

        def fn(f):
            return lambda v, s, a: f(v, *args, s, a, 64)

        actual = run_and_grad(fn(compiled), [value, sampling_loc, attn_weight])
        expected = run_and_grad(fn(module), [value, sampling_loc, attn_weight])
        torch.testing.assert_close(actual, expected)

    # Compile with another shape first, so that the shape under test must
    # reuse the graph.
    check(5, 10)
    with torch._dynamo.config.patch(error_on_recompile=True):
        check(batch, num_query)


@dataclass(frozen=True)
class KernelLayer:
    """The kernel layer, through `get_kernel` (which uses LOCAL_KERNELS in CI)."""

    layer_name: str = "MultiScaleDeformableAttention"

    def load(self):
        kernel = kernels.get_kernel("kernels-community/deformable-detr", version=2)
        return getattr(kernel.layers, self.layer_name)


@kernels.use_kernel_forward_from_hub("MultiScaleDeformableAttention")
class MultiScaleDeformableAttention(nn.Module):
    """A model layer that kernels can replace, as in Transformers."""

    def forward(
        self,
        value,
        value_spatial_shapes,
        value_spatial_shapes_list,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        return ms_deform_attn_ref(
            value, value_spatial_shapes, sampling_locations, attention_weights
        )


@pytest.mark.kernels_ci
@pytest.mark.parametrize(
    "mode",
    [
        kernels.Mode.INFERENCE | kernels.Mode.TORCH_COMPILE,
        kernels.Mode.TRAINING | kernels.Mode.TORCH_COMPILE,
    ],
    ids=["inference", "training"],
)
def test_kernelize_torch_compile_mode(deformable_detr, mode):
    # kernels falls back to the original layer under Mode.TORCH_COMPILE unless
    # the kernel layer declares that it supports torch.compile.
    model = nn.Sequential(MultiScaleDeformableAttention())
    model.train(kernels.Mode.TRAINING in mode)
    with kernels.use_kernel_mapping(
        {"MultiScaleDeformableAttention": {"cuda": KernelLayer()}},
        inherit_mapping=False,
    ):
        kernels.kernelize(model, mode=mode, device="cuda")
    expected = deformable_detr.layers.MultiScaleDeformableAttention.forward
    assert model[0].forward.__func__ is expected


def _ops(deformable_detr):
    return deformable_detr.layers.ops


# opcheck checks the schema and the fake implementations against the real
# kernel. Autograd goes through `MultiScaleDeformableAttentionFunction`.
@pytest.mark.kernels_ci
def test_opcheck_forward(deformable_detr):
    value, shapes, level_start_index, sampling_loc, attn_weight = make_inputs()
    torch.library.opcheck(
        _ops(deformable_detr).ms_deform_attn_forward.default,
        (value, shapes, level_start_index, sampling_loc, attn_weight, 64),
    )


@pytest.mark.kernels_ci
def test_opcheck_backward(deformable_detr):
    value, shapes, level_start_index, sampling_loc, attn_weight = make_inputs()
    grad_output = torch.randn(
        value.shape[0],
        sampling_loc.shape[1],
        value.shape[2] * value.shape[3],
        device=value.device,
    )
    torch.library.opcheck(
        _ops(deformable_detr).ms_deform_attn_backward.default,
        (value, shapes, level_start_index, sampling_loc, attn_weight, grad_output, 64),
    )
