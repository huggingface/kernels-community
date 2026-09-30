"""Compare blocked graph execution with the original all-edge computation."""

import copy
from types import MethodType

import kernels
import pytest
import torch
from torch import nn


kernel = kernels.get_kernel(
    "kernels-community/weathernext2-banded-attention", version=2
)
DEVICE = kernel.infer_device()


class ConditionedMlp(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.condition = nn.Linear(4, out_features)

    def forward(self, values, conditioning):
        return self.fc(values).tanh() + self.condition(conditioning)[:, None]


class EdgeUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_proj = nn.Linear(8, 16)
        self.sender_proj = nn.Linear(16, 16, bias=False)
        self.receiver_proj = nn.Linear(16, 16, bias=False)
        self.out_proj = nn.Linear(16, 16)
        self.act_fn = nn.SiLU()
        self.norm = ConditionedMlp(16, 16)

    def forward(self, edges, mesh, grid, senders, receivers, conditioning):
        messages = self.edge_proj(edges) + self.sender_proj(mesh)[:, senders]
        if self.receiver_proj is not None:
            messages = messages + self.receiver_proj(grid)[:, receivers]
        return self.norm(self.out_proj(self.act_fn(messages)), conditioning)


class Graph(nn.Module):
    def __init__(self, grid_to_mesh=False):
        super().__init__()
        self.grid_to_mesh = grid_to_mesh
        self.aggregate_normalization = 4.0 if grid_to_mesh else None
        self.edge_encoder = ConditionedMlp(4, 8)
        self.edge_update = EdgeUpdate()
        if grid_to_mesh:
            self.edge_update.receiver_proj = None
        self.grid_node_update = ConditionedMlp(16 if grid_to_mesh else 32, 16)
        self.mesh_node_update = ConditionedMlp(32 if grid_to_mesh else 16, 16)

    def forward(
        self, grid_states, mesh_states, edge_features, senders, receivers, conditioning
    ):
        edges = self.edge_encoder(edge_features, conditioning)
        sender_states, receiver_states = (
            (grid_states, mesh_states)
            if self.grid_to_mesh
            else (mesh_states, grid_states)
        )
        messages = self.edge_update(
            edges, sender_states, receiver_states, senders, receivers, conditioning
        )
        aggregated = torch.zeros_like(receiver_states, dtype=torch.float32).index_add(
            1, receivers, messages.float()
        )
        if self.aggregate_normalization is not None:
            aggregated = aggregated / self.aggregate_normalization
        inputs = torch.cat([receiver_states, aggregated.to(messages.dtype)], dim=-1)
        if self.grid_to_mesh:
            return (
                grid_states + self.grid_node_update(grid_states, conditioning),
                mesh_states + self.mesh_node_update(inputs, conditioning),
            )
        return (
            grid_states + self.grid_node_update(inputs, conditioning),
            mesh_states + self.mesh_node_update(mesh_states, conditioning),
        )


def inputs(device, dtype, batch=2, grid_to_mesh=False):
    generator = torch.Generator(device=device).manual_seed(7)

    def rand(*shape):
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)

    return (
        rand(batch, 19, 16),
        rand(batch, 11, 16),
        rand(batch, 57, 4),
        torch.randint(
            19 if grid_to_mesh else 11, (57,), device=device, generator=generator
        ),
        (
            torch.randint(11, (57,), device=device, generator=generator).sort().values
            if grid_to_mesh
            else torch.arange(19, device=device).repeat_interleave(3)
        ),
        rand(batch, 4),
    )


@pytest.mark.kernels_ci
@pytest.mark.parametrize("device", list(dict.fromkeys(["cpu", DEVICE])))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("grid_to_mesh", [False, True])
def test_chunked_graph_matches_reference(device, dtype, grid_to_mesh, monkeypatch):
    monkeypatch.setattr(kernel.graph, "GRID_CHUNK_SIZE", 8)
    monkeypatch.setattr(kernel.graph, "EDGE_CHUNK_SIZE", 16)
    torch.manual_seed(0)
    reference = Graph(grid_to_mesh=grid_to_mesh).to(device=device, dtype=dtype).eval()
    blocked = copy.deepcopy(reference)
    blocked.forward = MethodType(
        kernel.WeatherNext2BipartiteGraphNetwork.forward, blocked
    )
    sizes = []
    hook = blocked.edge_encoder.register_forward_pre_hook(
        lambda module, args: sizes.append(args[0].shape[1])
    )
    args = inputs(device, dtype, grid_to_mesh=grid_to_mesh)
    with torch.no_grad():
        expected = reference(*args)
        actual = blocked(*args)
    hook.remove()
    expected_sizes = [16, 16, 16, 9] if grid_to_mesh else [24, 24, 9]
    assert sizes == (expected_sizes if dtype == torch.float32 else [57])
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("grid_to_mesh", [False, True])
def test_gradients_use_original_graph(grid_to_mesh):
    reference = Graph(grid_to_mesh=grid_to_mesh).eval()
    blocked = copy.deepcopy(reference)
    blocked.forward = MethodType(
        kernel.WeatherNext2BipartiteGraphNetwork.forward, blocked
    )
    args = inputs("cpu", torch.float32, grid_to_mesh=grid_to_mesh)
    for model in (reference, blocked):
        sum(value.sum() for value in model(*args)).backward()
    for actual, expected in zip(blocked.parameters(), reference.parameters()):
        assert actual.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("reason", ["placeholder", "training", "autocast"])
def test_ineligible_graph_uses_original_forward(reason, monkeypatch):
    monkeypatch.setattr(kernel.graph, "GRID_CHUNK_SIZE", 8)
    model = Graph().eval()
    args = list(inputs("cpu", torch.float32))
    if reason == "placeholder":
        args[4] = torch.zeros_like(args[4])
    elif reason == "training":
        model.train()
    sizes = []
    with (
        torch.no_grad(),
        torch.autocast("cpu", dtype=torch.bfloat16, enabled=reason == "autocast"),
    ):
        expected = model(*args)
        model.forward = MethodType(
            kernel.WeatherNext2BipartiteGraphNetwork.forward, model
        )
        hook = model.edge_encoder.register_forward_pre_hook(
            lambda module, args: sizes.append(args[0].shape[1])
        )
        actual = model(*args)
    hook.remove()
    assert sizes == [57]
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target)
