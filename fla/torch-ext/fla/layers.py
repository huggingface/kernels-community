import torch.nn as nn

from .modules.fused_norm_gate import rms_norm_gated


class FusedRMSNormGated(nn.Module):
    def forward(self, hidden_states, gate=None):
        return rms_norm_gated(
            hidden_states,
            gate,
            self.weight,
            None,  # bias
            self.activation,
            residual=None,
            eps=self.eps,
            prenorm=False,
            residual_in_fp32=False,
        )


__all__ = ["FusedRMSNormGated"]
