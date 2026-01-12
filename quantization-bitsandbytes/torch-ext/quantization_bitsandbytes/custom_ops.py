import torch
from ._ops import ops

def gemm_4bit_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: int,
) -> torch.Tensor:
    original_dtype = input.dtype
    if original_dtype != torch.bfloat16:
        input = input.to(torch.bfloat16)

    output = ops.gemm_4bit_forward(input, weight, absmax, blocksize, quant_type)
    if original_dtype != torch.bfloat16:
        output = output.to(original_dtype)

    return output
