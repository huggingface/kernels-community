import copy
from contextlib import nullcontext
from types import SimpleNamespace

import kernels
import pytest
import torch
from torch import nn


kernel = kernels.get_kernel("kernels-community/weathernext2-banded-attention", version=3)
DEVICE = kernel.infer_device()


class _ConditionedNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.linear = nn.Linear(4, 2 * channels)

    def forward(self, states, conditioning):
        scale, offset = self.linear(conditioning).chunk(2, dim=-1)
        return self.norm(states) * (1 + scale[:, None]) + offset[:, None]


class _Mlp(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.chunk_size = 7
        self.fc1 = nn.Linear(inputs, 16)
        self.fc2 = nn.Linear(16, outputs)
        self.activation_fn = torch.nn.functional.gelu
        self.norm = _ConditionedNorm(outputs)

    def forward(self, states, conditioning, spatial_features=None):
        states = states.to(dtype=self.fc1.weight.dtype)
        if spatial_features is not None:
            if states.ndim == 2:
                states = states.unsqueeze(1).expand(-1, spatial_features.shape[0], -1)
            spatial_features = spatial_features.to(states).unsqueeze(0).expand(states.shape[0], -1, -1)
            states = torch.cat([spatial_features, states], dim=-1)
        return self.norm(self.fc2(self.activation_fn(self.fc1(states))), conditioning)


class _EdgeUpdate(nn.Module):
    def __init__(self, grid_to_mesh):
        super().__init__()
        self.edge_proj = nn.Linear(8, 16)
        self.sender_proj = nn.Linear(16, 16, bias=False)
        self.receiver_proj = nn.Linear(16, 16, bias=False) if not grid_to_mesh else None
        self.out_proj = nn.Linear(16, 16)
        self.norm = _ConditionedNorm(16)

    def forward(self, edges, sender_states, *args, projected_senders=None):
        if projected_senders is None:
            projected_senders = self.sender_proj(sender_states)
        if self.receiver_proj is None:
            senders, conditioning = args
            messages = self.edge_proj(edges) + projected_senders[:, senders]
        else:
            states, senders, receivers, conditioning = args
            messages = self.edge_proj(edges) + projected_senders[:, senders]
            messages = messages + self.receiver_proj(states)[:, receivers]
        return self.norm(self.out_proj(torch.nn.functional.gelu(messages)), conditioning), projected_senders


class _Graph(nn.Module):
    def __init__(self, grid_to_mesh):
        super().__init__()
        self.grid_to_mesh = grid_to_mesh
        self.config = SimpleNamespace(chunk_size=7)
        self.aggregate_normalization = 3 if grid_to_mesh else None
        self.edge_encoder = _Mlp(4, 8)
        self.edge_update = _EdgeUpdate(grid_to_mesh)
        self.grid_node_update = _Mlp(16 if grid_to_mesh else 32, 16)
        self.mesh_node_update = _Mlp(32 if grid_to_mesh else 16, 16)

    def forward(self, grid, mesh, edges, senders, receivers, conditioning):
        source, target = (grid, mesh) if self.grid_to_mesh else (mesh, grid)
        edges = edges.unsqueeze(0).expand(grid.shape[0], -1, -1).to(self.edge_encoder.fc1.weight.dtype)
        encoded = self.edge_encoder(edges, conditioning)
        args = (senders, conditioning) if self.grid_to_mesh else (target, senders, receivers, conditioning)
        messages, _ = self.edge_update(encoded, source, *args)
        aggregated = torch.zeros(target.shape, dtype=torch.float32, device=target.device)
        aggregated.index_add_(1, receivers, messages.float())
        if self.aggregate_normalization is not None:
            aggregated = aggregated / self.aggregate_normalization
        if self.grid_to_mesh:
            mesh = mesh + self.mesh_node_update(torch.cat([mesh, aggregated.to(mesh.dtype)], dim=-1), conditioning)
            grid = grid + self.grid_node_update(grid, conditioning)
        else:
            grid = grid + self.grid_node_update(torch.cat([grid, aggregated.to(grid.dtype)], dim=-1), conditioning)
            mesh = mesh + self.mesh_node_update(mesh, conditioning)
        return grid, mesh


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(chunk_size=7, grid_latitudes=1, grid_longitudes=19)
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 5)
        self.activation_fn = torch.nn.functional.gelu
        self.register_buffer("sigmoid_gate", torch.tensor([False, True, False, True, False]))
        self.register_buffer("sigmoid_shift", torch.tensor([0.0, 2.0, 0.0, 3.0, 0.0]))

    def forward(self, states):
        values = self.fc2(self.activation_fn(self.fc1(states)))
        values = torch.where(self.sigmoid_gate, torch.sigmoid(values - self.sigmoid_shift), values)
        return values.transpose(1, 2).reshape(states.shape[0], 5, 1, 19)


def _grid_case(layer_name, dtype=torch.float32):
    torch.manual_seed(7)
    conditioning = torch.randn(2, 4, device=DEVICE, dtype=dtype)
    grid = torch.randn(2, 38, 16, device=DEVICE, dtype=dtype)[:, ::2]
    if layer_name in ("mlp", "grid_encoder", "mesh_encoder"):
        spatial = None if layer_name == "mlp" else torch.randn(19, 3, device=DEVICE)
        states = grid if layer_name != "mesh_encoder" else grid[:, 0]
        if spatial is not None:
            states = states.float()
        layer, replacement = _Mlp(16 if spatial is None else 19, 16), kernel.WeatherNext2ConditionedMlp
        args = (states, conditioning, spatial)
    elif layer_name == "head":
        layer, replacement = _Head(), kernel.WeatherNext2ForecastHead
        args = (grid,)
    else:
        grid_to_mesh = layer_name == "grid_to_mesh"
        layer = _Graph(grid_to_mesh)
        replacement = kernel.WeatherNext2GridToMesh if grid_to_mesh else kernel.WeatherNext2MeshToGrid
        mesh = torch.randn(2, 11, 16, device=DEVICE, dtype=dtype)
        edges = torch.randn(57, 4, device=DEVICE, dtype=dtype)
        senders = torch.randint(19 if grid_to_mesh else 11, (57,), device=DEVICE)
        receivers = (
            torch.randint(11, (57,), device=DEVICE).sort().values
            if grid_to_mesh
            else torch.arange(19, device=DEVICE).repeat_interleave(3)
        )
        args = (grid, mesh, edges, senders, receivers, conditioning)
    return layer.to(device=DEVICE, dtype=dtype).eval(), replacement, args


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layer_name", ["mlp", "grid_encoder", "mesh_encoder", "grid_to_mesh", "mesh_to_grid", "head"])
def test_tiled_grid_preserves_reference(layer_name, dtype):
    layer, replacement, args = _grid_case(layer_name, dtype)
    with torch.inference_mode():
        expected = layer(*args)
        actual = replacement.forward(layer, *args)
    tolerance = {} if dtype == torch.float32 else {"atol": 1e-1, "rtol": 1e-2}
    torch.testing.assert_close(actual, expected, **tolerance)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("layer_name", ["mlp", "grid_encoder", "mesh_encoder", "grid_to_mesh", "mesh_to_grid", "head"])
def test_gradients_use_reference(training, layer_name):
    layer, replacement, args = _grid_case(layer_name)
    layer.train(training)
    reference = copy.deepcopy(layer)
    actual = replacement.forward(layer, *args)
    expected = reference(*args)
    actual_outputs = actual if isinstance(actual, tuple) else (actual,)
    expected_outputs = expected if isinstance(expected, tuple) else (expected,)
    sum(value.sum() for value in actual_outputs).backward()
    sum(value.sum() for value in expected_outputs).backward()
    torch.testing.assert_close(actual, expected)
    for parameter, original in zip(layer.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, original.grad)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("fallback", ["training", "autocast"])
@pytest.mark.parametrize("layer_name", ["mlp", "grid_encoder", "mesh_encoder", "grid_to_mesh", "mesh_to_grid", "head"])
def test_inference_fallback_preserves_inputs(fallback, layer_name):
    layer, replacement, args = _grid_case(layer_name)
    layer.train(fallback == "training")
    original = args[0].clone()
    context = torch.autocast(DEVICE, dtype=torch.bfloat16) if fallback == "autocast" else nullcontext()
    with torch.inference_mode(), context:
        expected = layer(*args)
        actual = replacement.forward(layer, *args)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(args[0], original, rtol=0, atol=0)


@pytest.mark.skipif(DEVICE == "cpu", reason="requires an accelerator")
@pytest.mark.parametrize("layer_name", ["mlp", "grid_encoder", "mesh_encoder", "grid_to_mesh", "mesh_to_grid", "head"])
def test_tiled_epilogue_compile(layer_name):
    layer, replacement, args = _grid_case(layer_name)
    compiled = torch.compile(replacement.forward, fullgraph=True)
    try:
        with torch.inference_mode():
            expected = layer(*args)
            torch.testing.assert_close(compiled(layer, *args), expected)
    finally:
        torch._dynamo.reset()
