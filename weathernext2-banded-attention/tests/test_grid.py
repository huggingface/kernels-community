"""Compare blocked grid layers and their Triton epilogues with original forwards."""

import copy
from types import MethodType, SimpleNamespace

import kernels
import pytest
import torch
from torch import nn


kernel = kernels.get_kernel(
    "kernels-community/weathernext2-banded-attention", version=2
)
DEVICES = list(dict.fromkeys(["cpu", kernel.infer_device()]))


class ConditionedNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-5, elementwise_affine=False)
        self.linear = nn.Linear(4, 2 * channels)

    def forward(self, values, conditioning):
        scale, offset = self.linear(conditioning).chunk(2, dim=-1)
        return self.norm(values) * (1.0 + scale[:, None]) + offset[:, None]


class GridEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 37)
        self.fc2 = nn.Linear(37, 37)
        self.activation_fn = nn.SiLU()
        self.norm = ConditionedNorm(37)

    def forward(self, grid_features, spatial_features, conditioning):
        spatial = spatial_features[None].expand(grid_features.shape[0], -1, -1)
        values = torch.cat(
            [
                spatial.to(self.fc1.weight.dtype),
                grid_features.to(self.fc1.weight.dtype),
            ],
            dim=-1,
        )
        return self.norm(self.fc2(self.activation_fn(self.fc1(values))), conditioning)


class ForecastHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(grid_latitudes=3, grid_longitudes=7)
        self.decoder_proj = nn.Linear(37, 37)
        self.output_proj = nn.Linear(37, 11)
        self.act_fn = nn.SiLU()
        self.register_buffer("sigmoid_gate", torch.arange(11) % 3 == 0)
        self.register_buffer("sigmoid_shift", torch.full((11,), -6.0))

    def forward(self, grid_states):
        values = self.output_proj(self.act_fn(self.decoder_proj(grid_states)))
        values = torch.where(
            self.sigmoid_gate, torch.sigmoid(values - self.sigmoid_shift), values
        )
        return values.transpose(1, 2).reshape(values.shape[0], 11, 3, 7)


def make_layer(name, device, dtype, batch):
    torch.manual_seed(7)
    if name == "encoder":
        layer = GridEncoder()
        # A transpose models the noncontiguous channels-first atmospheric inputs.
        args = (
            torch.randn(batch, 13, 21).transpose(1, 2),
            torch.randn(21, 3),
            torch.randn(batch, 4),
        )
        forward = kernel.WeatherNext2GridEncoder.forward
        projection = "fc1"
    else:
        layer = ForecastHead()
        args = (torch.randn(batch, 37, 21).transpose(1, 2),)
        forward = kernel.WeatherNext2ForecastHead.forward
        projection = "decoder_proj"
    layer = layer.to(device=device, dtype=dtype).eval()
    blocked = copy.deepcopy(layer)
    blocked.forward = MethodType(forward, blocked)
    return (
        layer,
        blocked,
        tuple(value.to(device=device, dtype=dtype) for value in args),
        projection,
    )


@pytest.mark.kernels_ci
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("name", ["encoder", "head"])
def test_chunked_grid_matches_original(device, dtype, batch, name, monkeypatch):
    monkeypatch.setattr(kernel.grid, "GRID_CHUNK_SIZE", 8)
    reference, blocked, args, projection = make_layer(name, device, dtype, batch)
    sizes = []
    hook = getattr(blocked, projection).register_forward_pre_hook(
        lambda module, inputs: sizes.append(inputs[0].shape[1])
    )
    with torch.no_grad():
        expected = reference(*args)
        actual = blocked(*args)
    hook.remove()
    assert sizes == [8, 8, 5]
    # A chunked bf16 matmul rounds differently from a whole-grid one, by up to a few bf16 ulps.
    tolerances = {torch.float32: {}, torch.bfloat16: {"atol": 1e-1, "rtol": 1e-2}}
    torch.testing.assert_close(actual, expected, **tolerances[dtype])


@pytest.mark.kernels_ci
@pytest.mark.parametrize("name", ["encoder", "head"])
@pytest.mark.parametrize("reason", ["training", "autocast", "gradients"])
def test_grid_fallback_preserves_outputs_and_gradients(name, reason, monkeypatch):
    monkeypatch.setattr(kernel.grid, "GRID_CHUNK_SIZE", 8)
    reference, blocked, args, projection = make_layer(name, "cpu", torch.float32, 2)
    if reason == "training":
        reference.train()
        blocked.train()
    sizes = []
    hook = getattr(blocked, projection).register_forward_pre_hook(
        lambda module, inputs: sizes.append(inputs[0].shape[1])
    )
    with (
        torch.set_grad_enabled(reason == "gradients"),
        torch.autocast("cpu", dtype=torch.bfloat16, enabled=reason == "autocast"),
    ):
        expected = reference(*args)
        actual = blocked(*args)
        torch.testing.assert_close(actual, expected)
        if reason == "gradients":
            expected.sum().backward()
            actual.sum().backward()
            for original, replacement in zip(
                reference.parameters(), blocked.parameters()
            ):
                assert replacement.grad is not None
                torch.testing.assert_close(replacement.grad, original.grad)
    hook.remove()
    assert sizes == [21]


@pytest.mark.kernels_ci
@pytest.mark.skipif(
    kernel.infer_device() not in ("cuda", "xpu"),
    reason="Triton requires an accelerator.",
)
def test_fused_conditioning_preserves_pytorch_rounding():
    device = kernel.infer_device()
    torch.manual_seed(7)
    values = torch.randn(2, 9, 37, device=device)
    film = torch.randn(2, 74, device=device)
    output = torch.empty(2, 21, 37, device=device)[:, 5:14]
    scale, offset = film.chunk(2, dim=-1)
    expected = values * (1.0 + scale[:, None]) + offset[:, None]
    with kernel.grid.device_context(values.device):
        kernel.grid._conditioning[(18,)](
            values,
            film,
            output,
            values.stride(0),
            values.stride(1),
            output.stride(0),
            output.stride(1),
            9,
            37,
            64,
            enable_fp_fusion=False,
        )
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
