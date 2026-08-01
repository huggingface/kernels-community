import torch.nn as nn

from ._causal_conv1d import causal_conv1d_fn as cuda_causal_conv1d_fn
from ._causal_conv1d import causal_conv1d_update as cuda_causal_conv1d_update


class causal_conv1d_fn(nn.Module):
    def forward(
        self,
        hidden_states,
        weight,
        bias=None,
        activation=None,
        **kwargs,
    ):
        # For varlen
        seq_idx = kwargs.pop("seq_idx", None)

        return cuda_causal_conv1d_fn(
            x=hidden_states,
            weight=weight,
            bias=bias,
            activation=activation,
            seq_idx=seq_idx,
        )


class causal_conv1d_update(nn.Module):
    def forward(
        self,
        hidden_states,
        conv_state,
        weight,
        bias=None,
        activation=None,
    ):
        return cuda_causal_conv1d_update(
            x=hidden_states,
            conv_state=conv_state,
            weight=weight,
            bias=bias,
            activation=activation,
        )


__all__ = ["causal_conv1d_fn", "causal_conv1d_update"]
