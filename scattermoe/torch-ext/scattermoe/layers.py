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

