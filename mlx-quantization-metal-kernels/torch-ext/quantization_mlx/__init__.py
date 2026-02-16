from typing import Optional

import torch

from ._ops import ops


# =============================================================================
# FP-quantized (MXFP4) operations
# =============================================================================


def mxfp4_qmm_n(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    output_features: int,
) -> torch.Tensor:
    """Matrix-matrix multiply with MXFP4 quantized non-transposed weight.

    Computes y = x @ dequantize(w, scales).
    x: [..., M, K], w: [K_packed, N_packed] (uint32), y: [..., M, output_features]
    """
    return ops.mxfp4_qmm_n(x, w, scales, output_features)


def mxfp4_qmv(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    output_features: int,
) -> torch.Tensor:
    """Matrix-vector multiply with MXFP4 quantized weight.

    Computes y = dequantize(w, scales) @ x.
    x: [..., K], w: [N, K_packed] (uint32), y: [..., output_features]
    """
    return ops.mxfp4_qmv(x, w, scales, output_features)


# =============================================================================
# Affine quantized operations (scales + biases)
# =============================================================================


def affine_qmv(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    output_features: int,
    group_size: int = 128,
    bits: int = 4,
) -> torch.Tensor:
    """Matrix-vector multiply with affine quantized weight.

    x: [..., K], w: [N, K_packed], y: [..., output_features]
    """
    return ops.affine_qmv(x, w, scales, biases, group_size, bits, output_features)


def affine_qmm_t(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int = 128,
    bits: int = 4,
) -> torch.Tensor:
    """Matrix-matrix multiply with affine quantized transposed weight.

    Computes y = x @ dequantize(w, scales, biases).T
    x: [..., M, K], w: [N, K_packed], y: [..., M, N]
    N is inferred from w.size(0).
    """
    return ops.affine_qmm_t(x, w, scales, biases, group_size, bits)


def affine_qmm_n(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    output_features: int,
    group_size: int = 128,
    bits: int = 4,
) -> torch.Tensor:
    """Matrix-matrix multiply with affine quantized non-transposed weight.

    Computes y = x @ dequantize(w, scales, biases)
    x: [..., M, K], w: [K_packed, N_packed], y: [..., M, output_features]
    """
    return ops.affine_qmm_n(x, w, scales, biases, group_size, bits, output_features)


# =============================================================================
# Affine quantized NAX operations (MetalPerformancePrimitives accelerated)
# =============================================================================


def affine_qmm_t_nax(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int = 128,
    bits: int = 4,
) -> torch.Tensor:
    """NAX-accelerated matrix-matrix multiply with transposed quantized weight.

    x: [..., M, K], w: [N, K_packed], y: [..., M, N]
    """
    return ops.affine_qmm_t_nax(x, w, scales, biases, group_size, bits)


def affine_qmm_n_nax(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    output_features: int,
    group_size: int = 128,
    bits: int = 4,
) -> torch.Tensor:
    """NAX-accelerated matrix-matrix multiply with non-transposed quantized weight.

    x: [..., M, K], w: [K_packed, N_packed], y: [..., M, output_features]
    """
    return ops.affine_qmm_n_nax(x, w, scales, biases, group_size, bits, output_features)


def affine_gather_qmm_rhs_nax(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    indices: torch.Tensor,
    output_features: int,
    group_size: int = 128,
    bits: int = 4,
    transpose: bool = True,
) -> torch.Tensor:
    """NAX-accelerated gather + matrix-matrix multiply.

    Gathers weight rows using indices, then computes matmul.
    x: [M, K], w: [num_experts, ...], indices: [M], y: [M, output_features]
    """
    return ops.affine_gather_qmm_rhs_nax(
        x, w, scales, biases, indices, group_size, bits, output_features, transpose
    )
