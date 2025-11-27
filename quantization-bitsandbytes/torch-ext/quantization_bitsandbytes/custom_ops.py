import torch
from ._ops import ops

def gemm_4bit_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: int,
) -> torch.Tensor:
    return ops.gemm_4bit_forward(input, weight, absmax, blocksize, quant_type)
