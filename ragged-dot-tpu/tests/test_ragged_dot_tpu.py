"""ragged_dot and experts_forward against fp32 CPU references, on the shapes
one expert-parallel rank (EP8) of Qwen3-30B-A3B and Gemma 4 26B-A4B sees."""

import kernels
import pytest
import torch

accelerator = torch.accelerator.current_accelerator(check_available=True)
pytestmark = pytest.mark.skipif(
    accelerator is None or accelerator.type != "tpu", reason="the kernel needs a TPU"
)

EP_SIZE = 8
TOP_K = 8
TOL = 2e-2


class GELUTanh(torch.nn.Module):
    """transformers' `gelu_pytorch_tanh` activation; the kernel picks the gate
    activation by the class name of `act_fn`."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x, approximate="tanh")


# name: (hidden size, intermediate size, total experts, activation, routing weights dtype)
MODELS = {
    "qwen3_moe": (2048, 768, 128, torch.nn.SiLU, torch.bfloat16),
    "gemma4": (2816, 704, 128, GELUTanh, torch.float32),
}


class Experts(torch.nn.Module):
    """The attributes transformers' experts interface hands experts_forward."""

    has_bias = False
    is_transposed = False
    has_gate = True
    is_concatenated = True

    def __init__(self, num_experts, hidden, inter, act_cls):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.randn(num_experts, 2 * inter, hidden) * 0.02)
        self.down_proj = torch.nn.Parameter(torch.randn(num_experts, hidden, inter) * 0.02)
        self.act_fn = act_cls()


@pytest.fixture(scope="module")
def kernel():
    return kernels.get_kernel("kernels-community/ragged-dot-tpu", version=1)


def rel_max_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.float() - ref).abs().max() / ref.abs().max()).item()


def ep_routing(tokens, num_experts, local, weights_dtype):
    """Top-k routing over all experts, then one rank's slice: picks of experts
    held by other ranks get the sentinel id `local` and weight 0. The rank is
    the one owning token 0's first pick, so a single token has local work."""
    idx = torch.rand(tokens, num_experts).topk(TOP_K).indices
    weights = torch.softmax(torch.randn(tokens, TOP_K), -1).to(weights_dtype)
    is_local = idx // local == idx[0, 0] // local
    return torch.where(is_local, idx % local, local), torch.where(is_local, weights, 0)


def experts_reference(experts, hidden_states, top_k_index, top_k_weights):
    gate_up_proj = experts.gate_up_proj.detach().float()
    down_proj = experts.down_proj.detach().float()
    h = hidden_states.float()
    out = torch.zeros_like(h)
    for e in range(gate_up_proj.shape[0]):
        token, k = (top_k_index == e).nonzero(as_tuple=True)
        gate, up = (h[token] @ gate_up_proj[e].T).chunk(2, dim=-1)
        y = (experts.act_fn(gate) * up) @ down_proj[e].T
        out.index_add_(0, token, y * top_k_weights[token, k, None].float())
    return out


@pytest.mark.kernels_ci
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tokens", [1, 256])
def test_ragged_dot(kernel, model, tokens):
    torch.manual_seed(0)
    hidden, inter, num_experts, _, weights_dtype = MODELS[model]
    local = num_experts // EP_SIZE
    idx, _ = ep_routing(tokens, num_experts, local, weights_dtype)
    expert_ids = torch.sort(idx.reshape(-1)).values
    group_sizes = torch.stack([(expert_ids == e).sum() for e in range(local)]).int()
    valid = int(group_sizes.sum())
    w = (torch.randn(local, 2 * inter, hidden) * 0.02).to(torch.bfloat16)
    x = torch.randn(tokens * TOP_K, hidden, dtype=torch.bfloat16)

    out = kernel.ragged_dot(x.to("tpu"), w.to("tpu"), group_sizes.to("tpu")).cpu()

    ref = torch.cat(
        [xs.float() @ w[e].float().T for e, xs in enumerate(x[:valid].split(group_sizes.tolist()))]
    )
    assert rel_max_err(out[:valid], ref) < TOL


@pytest.mark.kernels_ci
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tokens", [1, 256])
def test_experts_forward(kernel, model, tokens):
    torch.manual_seed(0)
    hidden, inter, num_experts, act_cls, weights_dtype = MODELS[model]
    local = num_experts // EP_SIZE
    experts = Experts(local, hidden, inter, act_cls).to(torch.bfloat16)
    idx, weights = ep_routing(tokens, num_experts, local, weights_dtype)
    h = torch.randn(tokens, hidden, dtype=torch.bfloat16)

    ref = experts_reference(experts, h, idx, weights)
    experts = experts.to("tpu")
    out = kernel.experts_forward(experts, h.to("tpu"), idx.to("tpu"), weights.to("tpu")).cpu()

    assert rel_max_err(out, ref) < TOL


def test_rejects_unsupported_activation(kernel):
    experts = Experts(2, 128, 128, torch.nn.ReLU)
    with pytest.raises(NotImplementedError):
        kernel.experts_forward(
            experts,
            torch.randn(1, 128),
            torch.zeros(1, TOP_K, dtype=torch.long),
            torch.ones(1, TOP_K),
        )
