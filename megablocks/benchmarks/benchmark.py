import torch
import torch.nn.functional as F
from collections import namedtuple

from kernels.benchmark import Benchmark


def moe_mlp_reference(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    top_k: int = 4,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    in_shape = x.shape
    num_experts = router_weight.shape[0]
    hidden_size = x.shape[-1]

    # Flatten to (num_tokens, hidden_size)
    hidden_states = x.view(-1, hidden_size)
    num_tokens = hidden_states.shape[0]

    # Router: compute logits and get top-k experts
    logits = F.linear(hidden_states, router_weight, router_bias)
    expert_weights, router_indices = torch.topk(logits, top_k, dim=-1)
    routing_weights = F.softmax(expert_weights, dim=-1)

    # Initialize output
    next_states = torch.zeros_like(hidden_states)

    # Create expert mask using one_hot
    with torch.no_grad():
        expert_mask = F.one_hot(router_indices, num_classes=num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, num_tokens)
        # Find which experts are hit
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    # Process each expert that has tokens
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        with torch.no_grad():
            top_k_idx, token_idx = torch.where(expert_mask[expert_idx])

        current_state = hidden_states[token_idx]

        # Up projection
        gate_up = (
            current_state @ gate_up_proj[expert_idx] + gate_up_proj_bias[expert_idx]
        )

        # Split into gate and up
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]

        # Clamp
        gate = gate.clamp(min=None, max=limit)
        up = up.clamp(min=-limit, max=limit)

        # SwiGLU-like activation
        glu = gate * torch.sigmoid(gate * alpha)
        gated_output = (up + 1) * glu

        # Down projection
        out = gated_output @ down_proj[expert_idx] + down_proj_bias[expert_idx]

        # Get the routing weight for this expert at the correct top_k position
        weights_for_expert = routing_weights[token_idx, top_k_idx]
        weighted_output = out * weights_for_expert[:, None]
        next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))

    return next_states.view(in_shape), routing_weights


class MegaBlocksMoeBenchmark(Benchmark):
    seed: int = 42

    def setup(self):
        # Config matching readme_example.py
        ne, hs, isz = 128, 1152, 3072
        batch, seq = 8, 1

        # Router
        self.router_weight = torch.randn(
            ne, hs, device=self.device, dtype=torch.float32
        )
        torch.nn.init.kaiming_uniform_(self.router_weight)
        self.router_bias = torch.zeros(ne, device=self.device, dtype=torch.float32)

        # Expert weights
        self.gate_up_proj = (
            torch.randn(ne, hs, isz, device=self.device, dtype=torch.float32) * 0.02
        )
        self.gate_up_proj_bias = torch.zeros(
            ne, isz, device=self.device, dtype=torch.float32
        )
        self.down_proj = (
            torch.randn(ne, isz // 2, hs, device=self.device, dtype=torch.float32)
            * 0.02
        )
        self.down_proj_bias = torch.zeros(
            ne, hs, device=self.device, dtype=torch.float32
        )

        # Input
        self.x = (
            torch.randn(seq, batch, hs, device=self.device, dtype=torch.float32) * 0.1
        )

        # Setup the model
        self.model = self.kernel.layers.MegaBlocksMoeMLP()
        self.model.router = torch.nn.Linear(hs, ne, device=self.device)
        self.model.router.weight.data = self.router_weight.clone()
        self.model.router.bias.data = self.router_bias.clone()

        Experts = namedtuple(
            "Experts",
            [
                "gate_up_proj",
                "gate_up_proj_bias",
                "down_proj",
                "down_proj_bias",
                "hidden_size",
                "num_experts",
            ],
        )
        self.model.experts = Experts(
            gate_up_proj=torch.nn.Parameter(self.gate_up_proj.clone()),
            gate_up_proj_bias=torch.nn.Parameter(self.gate_up_proj_bias.clone()),
            down_proj=torch.nn.Parameter(self.down_proj.clone()),
            down_proj_bias=torch.nn.Parameter(self.down_proj_bias.clone()),
            hidden_size=hs,
            num_experts=ne,
        )

        self.out = torch.empty(seq, batch, hs, device=self.device, dtype=torch.float32)

    def benchmark_base(self):
        self.out, self.expert_weights = self.model(self.x)

    def verify_base(self) -> torch.Tensor:
        ref_out, _ = moe_mlp_reference(
            self.x,
            self.router_weight,
            self.router_bias,
            self.gate_up_proj,
            self.gate_up_proj_bias,
            self.down_proj,
            self.down_proj_bias,
            top_k=4,
        )
        return ref_out

    def setup_large(self):
        # Larger config with more tokens
        ne, hs, isz = 128, 1152, 3072
        batch, seq = 32, 16

        # Router
        self.router_weight = torch.randn(
            ne, hs, device=self.device, dtype=torch.float32
        )
        torch.nn.init.kaiming_uniform_(self.router_weight)
        self.router_bias = torch.zeros(ne, device=self.device, dtype=torch.float32)

        # Expert weights
        self.gate_up_proj = (
            torch.randn(ne, hs, isz, device=self.device, dtype=torch.float32) * 0.02
        )
        self.gate_up_proj_bias = torch.zeros(
            ne, isz, device=self.device, dtype=torch.float32
        )
        self.down_proj = (
            torch.randn(ne, isz // 2, hs, device=self.device, dtype=torch.float32)
            * 0.02
        )
        self.down_proj_bias = torch.zeros(
            ne, hs, device=self.device, dtype=torch.float32
        )

        # Input
        self.x = (
            torch.randn(seq, batch, hs, device=self.device, dtype=torch.float32) * 0.1
        )

        # Setup the model
        self.model = self.kernel.layers.MegaBlocksMoeMLP()
        self.model.router = torch.nn.Linear(hs, ne, device=self.device)
        self.model.router.weight.data = self.router_weight.clone()
        self.model.router.bias.data = self.router_bias.clone()

        Experts = namedtuple(
            "Experts",
            [
                "gate_up_proj",
                "gate_up_proj_bias",
                "down_proj",
                "down_proj_bias",
                "hidden_size",
                "num_experts",
                "capacity_factor",
            ],
        )
        self.model.experts = Experts(
            gate_up_proj=torch.nn.Parameter(self.gate_up_proj.clone()),
            gate_up_proj_bias=torch.nn.Parameter(self.gate_up_proj_bias.clone()),
            down_proj=torch.nn.Parameter(self.down_proj.clone()),
            down_proj_bias=torch.nn.Parameter(self.down_proj_bias.clone()),
            hidden_size=hs,
            num_experts=ne,
            capacity_factor=4.0,  # Higher capacity to avoid token dropping
        )

        self.out = torch.empty(seq, batch, hs, device=self.device, dtype=torch.float32)

    def benchmark_large(self):
        self.out, self.expert_weights = self.model(self.x)

    def verify_large(self) -> torch.Tensor:
        ref_out, _ = moe_mlp_reference(
            self.x,
            self.router_weight,
            self.router_bias,
            self.gate_up_proj,
            self.gate_up_proj_bias,
            self.down_proj,
            self.down_proj_bias,
            top_k=4,
        )
        return ref_out
