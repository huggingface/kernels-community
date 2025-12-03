import torch
from ._ops import ops

def gemm_int4_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    zeros: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
) -> torch.Tensor:
    original_dtype = input.dtype
    if original_dtype != torch.bfloat16:
        input = input.to(torch.bfloat16)

    output = ops.gemm_int4_forward(input, weight, zeros, absmax, blocksize)
    if original_dtype != torch.bfloat16:
        output = output.to(original_dtype)

    return output
