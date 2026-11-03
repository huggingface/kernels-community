import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._ops import ops


class GRTPearlNorm(nn.Module):
    weight: torch.Tensor
    eps: float

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out, _ = ops.fused_norm_forward(hidden_states, self.weight, float(self.eps))
        return out


class GRTRadialGELU(nn.Module):
    eps: float

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return ops.fused_gelu_forward(hidden_states, float(self.eps))


class GRTEmbedding(nn.Module):
    phase_angles: torch.Tensor
    log_magnitude: torch.Tensor

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return ops.fused_embedding_forward(self.phase_angles, self.log_magnitude, input_ids)
