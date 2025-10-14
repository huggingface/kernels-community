# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch

from ._ops import ops

from .grouped_gemm import backend as gg_backend
from .grouped_gemm import ops as gg_ops


from ._layers.arguments import Arguments
from ._layers.dmoe import ParallelDroplessMLP, dMoE
from ._layers.glu import SparseGLU
from ._layers.mlp import MLP, SparseMLP
from ._layers.moe import MoE, ParallelMLP, get_load_balancing_loss

from . import layers

# This section contains the direct kernel exports (not inlcuded in the original code)
def exclusive_cumsum(x: torch.Tensor, dim: int, out: torch.Tensor) -> torch.Tensor:
    """
    Compute exclusive cumulative sum along the specified dimension.

    Args:
        x: Input tensor
        dim: Dimension along which to compute cumsum
        out: Output tensor (modified in-place)

    Returns:
        The output tensor
    """
    result = ops.exclusive_cumsum(x, dim)
    out.copy_(result)
    return out


def inclusive_cumsum(x: torch.Tensor, dim: int, out: torch.Tensor) -> torch.Tensor:
    """
    Compute inclusive cumulative sum along the specified dimension.

    Args:
        x: Input tensor
        dim: Dimension along which to compute cumsum
        out: Output tensor (modified in-place)

    Returns:
        The output tensor
    """
    result = ops.inclusive_cumsum(x, dim)
    out.copy_(result)
    return out


def histogram(x: torch.Tensor, num_bins: int) -> torch.Tensor:
    """
    Compute histogram of input tensor values.

    Args:
        x: Input tensor
        num_bins: Number of histogram bins

    Returns:
        Histogram tensor with counts for each bin
    """
    return ops.histogram(x, num_bins)


def indices(
    padded_bins: torch.Tensor,
    block_size: int,
    output_block_rows: int,
    output_block_columns: int,
) -> torch.Tensor:
    """
    Construct indices from padded bins for sparse operations.

    Args:
        padded_bins: Tensor containing bin boundaries
        block_size: Size of each block
        output_block_rows: Number of rows in output blocks
        output_block_columns: Number of columns in output blocks

    Returns:
        Tensor containing constructed indices
    """
    return ops.indices(padded_bins, block_size, output_block_rows, output_block_columns)


def replicate_forward(
    x: torch.Tensor, bins: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """
    Forward pass of replicate operation - replicate values according to bin sizes.

    Args:
        x: Input tensor with values to replicate
        bins: Tensor containing bin sizes
        out: Output tensor (modified in-place)

    Returns:
        The output tensor
    """
    return ops.replicate_forward(x, bins, out)


def replicate_backward(
    grad: torch.Tensor, bins: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """
    Backward pass of replicate operation - reduce gradients back to bins.

    Args:
        grad: Gradient tensor to reduce
        bins: Tensor containing bin sizes
        out: Output tensor (modified in-place)

    Returns:
        The output tensor
    """
    return ops.replicate_backward(grad, bins, out)


def sort(
    x: torch.Tensor, end_bit: int, x_out: torch.Tensor, iota_out: torch.Tensor
) -> torch.Tensor:
    """
    Radix sort with index tracking.

    Args:
        x: Input tensor to sort
        end_bit: Number of bits to consider in sorting
        x_out: Output tensor for sorted values
        iota_out: Output tensor for sorted indices

    Returns:
        The sorted values tensor
    """
    return ops.sort(x, end_bit, x_out, iota_out)


# Convenience functions for common use cases
def cumsum(x: torch.Tensor, dim: int = -1, exclusive: bool = False) -> torch.Tensor:
    """
    Compute cumulative sum with automatic output allocation.

    Args:
        x: Input tensor
        dim: Dimension along which to compute cumsum (default: last dimension)
        exclusive: Whether to compute exclusive (True) or inclusive (False) cumsum

    Returns:
        New tensor containing the cumulative sum
    """
    out = torch.empty_like(x)
    if exclusive:
        return exclusive_cumsum(x, dim, out)
    else:
        return inclusive_cumsum(x, dim, out)


def argsort(x: torch.Tensor, end_bit: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sort tensor and return both sorted values and indices.

    Args:
        x: Input tensor to sort
        end_bit: Number of bits to consider in sorting

    Returns:
        Tuple of (sorted_values, sorted_indices)
    """
    x_out = torch.empty_like(x)
    iota_out = torch.empty_like(x)
    sort(x, end_bit, x_out, iota_out)
    return x_out, iota_out


# Export public API
__all__ = [
    "MyReplacementLayer",
    # Direct kernel exports
    "exclusive_cumsum",
    "inclusive_cumsum",
    "histogram",
    "indices",
    "replicate_forward",
    "replicate_backward",
    "sort",
    "cumsum",
    "argsort",
    # Original exports
    "Arguments",
    "ParallelDroplessMLP",
    "dMoE",
    "SparseGLU",
    "MLP",
    "SparseMLP",
    "MoE",
    "ParallelMLP",
    "get_load_balancing_loss",
]
