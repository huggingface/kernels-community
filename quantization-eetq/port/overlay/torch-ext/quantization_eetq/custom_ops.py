from typing import List
import torch

from ._ops import ops


def w8_a16_gemm(
    input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return ops.w8_a16_gemm(input, weight, scale)


def w8_a16_gemm_(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
    m: int,
    n: int,
    k: int,
) -> torch.Tensor:
    return ops.w8_a16_gemm_(input, weight, scale, output, m, n, k)


def preprocess_weights(origin_weight: torch.Tensor, is_int4: bool) -> torch.Tensor:
    return ops.preprocess_weights(origin_weight, is_int4)


def quant_weights(
    origin_weight: torch.Tensor,
    quant_type: torch.dtype,
    return_unprocessed_quantized_tensor: bool,
) -> List[torch.Tensor]:
    return ops.quant_weights(
        origin_weight, quant_type, return_unprocessed_quantized_tensor
    )
