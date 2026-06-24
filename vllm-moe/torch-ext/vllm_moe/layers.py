import torch
import torch.nn as nn

from .fused_moe import fused_moe


def _llama4_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    topk_weights, topk_ids = torch.topk(gating_output, topk, dim=1)
    topk_weights = torch.sigmoid(topk_weights.float()).to(hidden_states.dtype)
    return topk_weights, topk_ids


def _fix_llama4_experts(hidden_states: torch.Tensor, experts: nn.Module):
    if experts.gate_up_proj.shape[-1] != hidden_states.shape[-1]:
        experts.gate_up_proj = nn.Parameter(
            experts.gate_up_proj.transpose(1, 2).contiguous()
        )
        experts.down_proj = nn.Parameter(experts.down_proj.transpose(1, 2).contiguous())


class Llama4TextMoe(nn.Module):
    has_backward = False

    experts: nn.Module
    router: nn.Linear
    shared_expert: nn.Module
    top_k: int

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        _fix_llama4_experts(hidden_states, self.experts)

        router_logits = self.router(hidden_states)

        extra_kwargs = {}
        use_fp8_w8a8 = False
        if hasattr(self.experts, "gate_up_proj_scale"):
            use_fp8_w8a8 = True
            extra_kwargs["w1_scale"] = self.experts.gate_up_proj_scale
            extra_kwargs["w2_scale"] = self.experts.down_proj_scale

        out = fused_moe(
            hidden_states,
            w1=self.experts.gate_up_proj,
            w2=self.experts.down_proj,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=False,
            custom_routing_function=_llama4_topk,
            apply_router_weight_on_input=True,
            use_fp8_w8a8=use_fp8_w8a8,
            **extra_kwargs
        )

        out += self.shared_expert(hidden_states)

        return out, router_logits.t()
