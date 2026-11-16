import copy

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import kernels

liger_kernels = kernels.get_kernel("kernels-community/liger-kernels", version=4)
apply_tiled_mlp = liger_kernels.layers.apply_tiled_mlp
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


@pytest.mark.kernels_ci
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("compute_params", [False, True])
@pytest.mark.parametrize("functional_grad", [False, True])
def test_tiled_parameter_gradients(requires_grad, compute_params, functional_grad):
    torch.manual_seed(0)
    reference = nn.Linear(8, 12).to(DEVICE)
    tiled = copy.deepcopy(reference)
    x = torch.randn(2, 17, 8, device=DEVICE)
    x_ref = x.clone().requires_grad_(requires_grad)
    x_tiled = x.clone().requires_grad_(requires_grad)
    expected = reference(x_ref)
    actual = apply_tiled_mlp(
        lambda module, inputs: module(inputs),
        tiled,
        x_tiled,
        num_shards=4,
        compute_params=list(tiled.parameters()) if compute_params else None,
    )
    torch.testing.assert_close(actual, expected)
    inputs_ref = list(reference.parameters()) + ([x_ref] if requires_grad else [])
    inputs_tiled = list(tiled.parameters()) + ([x_tiled] if requires_grad else [])
    if functional_grad:
        gradients_ref = torch.autograd.grad(expected.sum(), inputs_ref)
        gradients_tiled = torch.autograd.grad(actual.sum(), inputs_tiled)
    else:
        expected.sum().backward()
        actual.sum().backward()
        gradients_ref = [value.grad for value in inputs_ref]
        gradients_tiled = [value.grad for value in inputs_tiled]
    for actual, expected in zip(gradients_tiled, gradients_ref):
        torch.testing.assert_close(actual, expected)


@pytest.mark.kernels_ci
def test_tiled_autocast_recomputation():
    if DEVICE == "cpu":
        pytest.skip("needs an accelerator")
    torch.manual_seed(0)
    reference = nn.Linear(16, 32).to(DEVICE)
    tiled = copy.deepcopy(reference)
    x = torch.randn(2, 41, 16, device=DEVICE)
    # Both forwards use the same tiles, isolating recomputation from GEMM rounding.
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        expected = torch.cat([reference(tile) for tile in torch.tensor_split(x, 3, dim=-2)], dim=-2)
        actual = apply_tiled_mlp(lambda module, inputs: module(inputs), tiled, x, num_shards=3)
    expected.float().sum().backward()
    actual.float().sum().backward()
    torch.testing.assert_close(actual, expected)
    for actual, expected in zip(tiled.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("outer_checkpoint", [False, True])
def test_tiled_rng_and_gradient_accumulation(outer_checkpoint):
    torch.manual_seed(0)
    reference = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5)).to(DEVICE)
    tiled = copy.deepcopy(reference)
    # Non-contiguous inputs also exercise checkpointing without the old flattening.
    x = torch.randn(2, 8, 17, device=DEVICE).transpose(-1, -2)
    for step in range(2):
        torch.manual_seed(step)
        expected = torch.cat([reference(tile) for tile in torch.tensor_split(x, 3, dim=-2)], dim=-2)
        torch.manual_seed(step)

        def forward(inputs):
            return apply_tiled_mlp(lambda module, tile: module(tile), tiled, inputs, num_shards=3)

        actual = checkpoint(forward, x, use_reentrant=False) if outer_checkpoint else forward(x)
        expected.sum().backward()
        actual.sum().backward()
        torch.testing.assert_close(actual, expected)
        for actual, expected in zip(tiled.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("activation", ["SwiGLU", "GEGLU"])
def test_tiled_mlp_layer(activation):
    if DEVICE == "cpu":
        pytest.skip("needs an accelerator")

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(16, 32)
            self.up_proj = nn.Linear(16, 32)
            self.down_proj = nn.Linear(32, 16)

    torch.manual_seed(0)
    reference = MLP().to(DEVICE)
    tiled = copy.deepcopy(reference)
    reference_forward = getattr(liger_kernels.layers, f"Liger{activation}MLP").forward
    tiled_forward = getattr(liger_kernels.layers, f"LigerTiled{activation}MLP").forward
    x_ref = torch.randn(2, 41, 16, device=DEVICE, requires_grad=True)
    x_tiled = x_ref.detach().clone().requires_grad_()
    expected = reference_forward(reference, x_ref)
    actual = tiled_forward(tiled, x_tiled)
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x_tiled.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    for actual, expected in zip(tiled.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-4, rtol=1e-4)


@pytest.mark.kernels_ci
def test_tiled_compile_with_frozen_inputs():
    class TiledLinear(nn.Linear):
        def forward(self, x):
            return apply_tiled_mlp(
                lambda module, tile: nn.functional.linear(tile, module.weight, module.bias),
                self,
                x,
                num_shards=3,
            )

    torch.manual_seed(0)
    reference = nn.Linear(8, 12).to(DEVICE)
    tiled = TiledLinear(8, 12).to(DEVICE)
    tiled.load_state_dict(reference.state_dict())
    compiled = torch.compile(tiled, backend="aot_eager", fullgraph=True)
    x = torch.randn(2, 17, 8, device=DEVICE)
    expected = reference(x)
    actual = compiled(x)
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(actual, expected)
    for actual, expected in zip(tiled.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)
