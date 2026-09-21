import torch
import torch.nn as nn

from .causal_conv1d_interface import causal_conv1d_fn as kernel_causal_conv1d_fn
from .causal_conv1d_interface import causal_conv1d_update as kernel_causal_conv1d_update


class causal_conv1d_fn(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
        **kwargs,
    ):
        # For varlen
        seq_idx = kwargs.pop("seq_idx", None)

        return kernel_causal_conv1d_fn(
            x=hidden_states,
            weight=weight,
            bias=bias,
            activation=activation,
            seq_idx=seq_idx,
        )


class causal_conv1d_update(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
    ):
        return kernel_causal_conv1d_update(
            x=hidden_states,
            conv_state=conv_state,
            weight=weight,
            bias=bias,
            activation=activation,
        )


__all__ = [
    "causal_conv1d_fn",
    "causal_conv1d_update",
]
