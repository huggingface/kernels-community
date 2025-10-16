import torch
import torch.nn as nn

from ._ops import ops


class LayerNorm(nn.Module):
    weight: torch.Tensor
    variance_epsilon: float

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = ops.dropout_add_ln_fwd(
            hidden_states.view(-1, hidden_states.shape[-1]),
            gamma = self.weight,
            beta = None,
            rowscale = None,
            colscale = None,
            x0_subset = None,
            z_subset = None,
            dropout_p = 0,
            epsilon = self.variance_epsilon,
            rowscale_const = 1.0,
            z_numrows = hidden_states.shape[1],
            gen = None,
            residual_in_fp32 = False,
            is_rms_norm = False,
        )
        return output[0].view(hidden_states.shape)

class LlamaRMSNorm(nn.Module):
    weight: torch.Tensor
    variance_epsilon: float

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = ops.dropout_add_ln_fwd(
            hidden_states.view(-1, hidden_states.shape[-1]),
            gamma = self.weight,
            beta = None,
            rowscale = None,
            colscale = None,
            x0_subset = None,
            z_subset = None,
            dropout_p = 0,
            epsilon = self.variance_epsilon,
            rowscale_const = 1.0,
            z_numrows = hidden_states.shape[1],
            gen = None,
            residual_in_fp32 = False,
            is_rms_norm = True,
        )
        return output[0].view(hidden_states.shape)