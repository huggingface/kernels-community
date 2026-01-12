from typing import Optional, Tuple

import torch

from ._ops import ops


def tinygrad_rms_norm(
    x: torch.Tensor,
    epsilon: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RMSNorm using tinygrad-style CUDA kernels.

    RMSNorm(x) = x * (1 / sqrt(mean(x^2) + epsilon))

    This implementation uses a two-kernel approach:
    1. Compute 1/sqrt(mean(x^2) + epsilon) for each row
    2. Multiply input by the computed factor

    Args:
        x: Input tensor of shape (..., hidden_size)
        epsilon: Small constant for numerical stability
        out: Optional pre-allocated output tensor

    Returns:
        Tuple of (output tensor, rms_inv tensor)
    """
    if out is None:
        out = torch.empty_like(x)

    hidden_size = x.size(-1)
    num_rows = x.numel() // hidden_size
    rms_inv = torch.empty(num_rows, dtype=x.dtype, device=x.device)

    ops.tinygrad_rms_norm(out, rms_inv, x, epsilon)
    return out, rms_inv


def tinygrad_rms_norm_simple(
    x: torch.Tensor,
    epsilon: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute RMSNorm using tinygrad-style CUDA kernels.

    This is a simpler interface that only returns the normalized output.

    Args:
        x: Input tensor of shape (..., hidden_size)
        epsilon: Small constant for numerical stability
        out: Optional pre-allocated output tensor

    Returns:
        Normalized output tensor
    """
    if out is None:
        out = torch.empty_like(x)

    ops.tinygrad_rms_norm_inplace(out, x, epsilon)
    return out
