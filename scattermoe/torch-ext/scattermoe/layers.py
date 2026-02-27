import torch
from torch.nn import functional as F
from torch import nn

from . import parallel_linear, flatten_sort_count

class ScatterMoEGatedMLP(nn.Module):
    def forward(self, layer_input):
        """
        Forward pass of the mixture of experts layer.

        Args:
            layer_input (Tensor):
                Input tensor.

        Returns:
            Tensor:
                Output tensor.
            Tensor:
                Router logits.
        """
        bsz, length, emb_size = layer_input.size()
        layer_input = layer_input.reshape(-1, emb_size)
        # compute the top_k routing decision
        router_logits = self.router.layer(layer_input)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.router.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(layer_input.dtype)
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = \
            flatten_sort_count(selected_experts, num_experts=self.router.num_experts)

        # compute experts
        gates, h = parallel_linear(
            layer_input, self.input_linear.weight.transpose(2, 1),
            self.router.top_k,
            sorted_expert_idxs, sorted_scattered_idxs,
            expert_offsets,
            grouped_in=False, grouped_out=True,
        ).chunk(2, dim=-1)
        h = self.activation(gates) * h
        layer_output = parallel_linear(
            h, self.output_linear.weight.transpose(2, 1),
            1,
            sorted_expert_idxs, sorted_scattered_idxs,
            expert_offsets,
            grouped_in=True, grouped_out=False,
            gates=routing_weights
        )
        layer_output = layer_output.view(bsz, length, emb_size)
        return layer_output




class HFScatterMoEGatedMLP(nn.Module):
    """
    ScatterMoE-accelerated forward pass for HF MoEs based on Qwen2MoE.

    This class adapts the ScatterMoE kernel to work with standard Qwen2MoE parameter names:
    - Uses existing `gate_up_proj` and `down_proj` parameters
    """

    def forward(
            self: nn.Module,
            layer_input: torch.Tensor
    ):
        """
        Forward pass using ScatterMoE kernels with standard Qwen2MoE parameter names.

        Args:
            layer_input: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, sequence_length, hidden_dim = layer_input.shape
        hidden_states_flat = layer_input.view(-1, hidden_dim)

        # ============================================================================
        # Shared Expert (if present)
        # ============================================================================
        if hasattr(self, 'shared_expert') and self.shared_expert is not None:
            shared_expert_output = self.shared_expert(hidden_states_flat)
            shared_expert_gate_output = F.sigmoid(
                self.shared_expert_gate(hidden_states_flat)
            )
            shared_expert_output = shared_expert_output * shared_expert_gate_output
        else:
            shared_expert_output = None

        # ============================================================================
        # Router Computation
        # ============================================================================
        # Standard Qwen2MoE router: self.gate.weight is [num_experts, hidden_size]
        router_logits = F.linear(hidden_states_flat, self.gate.weight)  # [num_tokens, num_experts]
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # Get top-k experts
        top_k = self.gate.top_k
        num_experts = self.gate.num_experts
        routing_weights, selected_experts = torch.topk(
            routing_weights, top_k, dim=-1
        )  # [num_tokens, top_k]

        # Normalize top-k weights if required
        if self.gate.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states_flat.dtype)

        # Flatten and sort for ScatterMoE kernel
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count(
            selected_experts, num_experts=num_experts
        )

        # compute experts - Input linear (gate + up projections)
        gates, h = parallel_linear(
            hidden_states_flat,
            self.experts.gate_up_proj.transpose(2, 1),  # [num_experts, hidden, 2*intermediate]
            top_k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_in=False,
            grouped_out=True,
        ).chunk(2, dim=-1)

        # Activation
        h = self.experts.act_fn(gates) * h

        # experts - Output linear
        expert_output = parallel_linear(
            h,
            self.experts.down_proj.transpose(2, 1),  # [num_experts, intermediate, hidden]
            1,  # Each token goes to 1 expert for the output (already routed)
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_in=True,
            grouped_out=False,
            gates=routing_weights,
        )

        # Combine with Shared Expert (if present)
        if shared_expert_output is not None:
            expert_output = expert_output + shared_expert_output

        # Reshape to original dimensions
        expert_output = expert_output.view(batch_size, sequence_length, hidden_dim)

        return expert_output
