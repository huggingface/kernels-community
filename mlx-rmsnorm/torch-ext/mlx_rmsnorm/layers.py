from .functions import rmsnorm_forward
import torch

class RMSNorm(torch.nn.Module):
    weight: torch.Tensor
    variance_epsilon: float

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_forward(x, self.weight, self.variance_epsilon)


