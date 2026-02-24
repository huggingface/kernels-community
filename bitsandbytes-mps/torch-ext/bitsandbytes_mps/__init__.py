from typing import Optional, Tuple

import torch

from ._ops import ops

# Quant type constants (match bitsandbytes DataType_t)
FP4 = 1
NF4 = 2


def quantize_4bit(
    input: torch.Tensor,
    blocksize: int = 64,
    quant_type: int = NF4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Blockwise 4-bit quantization using NF4 or FP4 codebook.

    Args:
        input: Input tensor on MPS device (float16, bfloat16, or float32).
        blocksize: Number of elements per quantization block (64 or 128).
        quant_type: FP4 (1) or NF4 (2).

    Returns:
        Tuple of (packed, absmax):
            packed: uint8 tensor of packed 4-bit values [numel/2].
            absmax: float32 tensor of per-block max absolute values.
    """
    return ops.bnb_quantize_4bit(input, blocksize, quant_type)


def dequantize_4bit(
    packed: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int = 64,
    quant_type: int = NF4,
    numel: int = -1,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Blockwise 4-bit dequantization using NF4 or FP4 codebook.

    Args:
        packed: uint8 tensor of packed 4-bit values.
        absmax: float32 tensor of per-block max absolute values.
        blocksize: Number of elements per quantization block (64 or 128).
        quant_type: FP4 (1) or NF4 (2).
        numel: Number of elements in the original tensor.
               If -1, inferred as packed.numel() * 2.
        output_dtype: Output scalar type.

    Returns:
        Dequantized tensor.
    """
    if numel < 0:
        numel = packed.numel() * 2
    return ops.bnb_dequantize_4bit(
        packed, absmax, blocksize, quant_type, numel, output_dtype
    )


def gemv_4bit(
    x: torch.Tensor,
    w: torch.Tensor,
    absmax: torch.Tensor,
    output_features: int,
    blocksize: int = 64,
    quant_type: int = NF4,
) -> torch.Tensor:
    """Fused matrix-vector multiply with 4-bit quantized weights.

    Computes y = dequant(W) @ x, where W is blockwise NF4/FP4 quantized.

    Args:
        x: Input vector [..., K] on MPS device.
        w: Packed weight matrix [N, K/2] (uint8) on MPS device.
        absmax: Per-block scales [N, ceil(K/blocksize)] (float32).
        output_features: Number of output features (N).
        blocksize: Quantization block size (64 or 128).
        quant_type: FP4 (1) or NF4 (2).

    Returns:
        Output tensor [..., N].
    """
    return ops.bnb_gemv_4bit(x, w, absmax, blocksize, quant_type, output_features)


def gemm_4bit(
    x: torch.Tensor,
    w: torch.Tensor,
    absmax: torch.Tensor,
    output_features: int,
    blocksize: int = 64,
    quant_type: int = NF4,
) -> torch.Tensor:
    """Fused matrix-matrix multiply with 4-bit quantized transposed weights.

    Computes Y = X @ dequant(W).T, where W is blockwise NF4/FP4 quantized.

    Args:
        x: Input matrix [..., M, K] on MPS device.
        w: Packed weight matrix [N, K/2] (uint8) on MPS device.
        absmax: Per-block scales [N, ceil(K/blocksize)] (float32).
        output_features: Number of output features (N).
        blocksize: Quantization block size (64 or 128).
        quant_type: FP4 (1) or NF4 (2).

    Returns:
        Output tensor [..., M, N].
    """
    return ops.bnb_gemm_4bit(x, w, absmax, blocksize, quant_type, output_features)


def linear_4bit(
    x: torch.Tensor,
    w: torch.Tensor,
    absmax: torch.Tensor,
    output_features: int,
    blocksize: int = 64,
    quant_type: int = NF4,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """4-bit quantized linear layer (auto-selects GEMV or GEMM).

    Args:
        x: Input tensor on MPS device.
        w: Packed weight [N, K/2] (uint8).
        absmax: Scales [N, ceil(K/blocksize)] (float32).
        output_features: N.
        blocksize: 64 or 128.
        quant_type: FP4 (1) or NF4 (2).
        bias: Optional bias [N].

    Returns:
        Output tensor.
    """
    input_1d = x.dim() == 1
    if input_1d or (x.dim() >= 2 and x.size(-2) == 1):
        x_flat = x.view(x.size(-1)) if input_1d else x.squeeze(-2)
        y = gemv_4bit(
            x_flat,
            w,
            absmax,
            output_features,
            blocksize,
            quant_type,
        )
        if input_1d:
            y = y.squeeze(0)
        elif x.dim() >= 2:
            y = y.unsqueeze(-2)
    else:
        y = gemm_4bit(x, w, absmax, output_features, blocksize, quant_type)

    if bias is not None:
        y = y + bias

    return y

__all__ = [
    "quantize_4bit",
    "dequantize_4bit",
    "gemv_4bit",
    "gemm_4bit",
    "linear_4bit",
]