"""MLX's quantized ops for torch tensors on MPS.

The Metal kernels are MLX's own, compiled as they ship; `vendor/UPSTREAM` pins the revision. The
host-side dispatch transcribes MLX's, so each call runs the kernels `mlx.core` would run for the same
inputs on the same GPU, and the functions take MLX's arguments, defaults and layouts:

- `quantize` / `dequantize`: MLX's layout, which `mx.quantize` produces and `mx.dequantize` reads.
- `quantized_matmul`: `x @ dequantize(w).T` (or `x @ dequantize(w)` with `transpose=False`).
- `gather_qmm`: `quantized_matmul` with the weights (and inputs) picked per batch element by index,
  as mixture-of-experts layers use it.

Modes: "affine" (scales and biases in the activation dtype; group_size 32/64/128, bits
2/3/4/5/6/8, default 64/4) and the fp formats "mxfp4" (32/4), "mxfp8" (32/8) and "nvfp4" (16/4),
whose scales are uint8 and which take no biases.

The functions further down are this kernel's version 1 API, kept for callers that used it.
"""

from typing import Optional, Tuple, Union

import torch

from ._ops import ops


__all__ = [
    "dequantize",
    "gather_qmm",
    "quantize",
    "quantized_matmul",
    # version 1
    "affine_gather_qmm_rhs_nax",
    "affine_qmm_n",
    "affine_qmm_n_nax",
    "affine_qmm_t",
    "affine_qmm_t_nax",
    "affine_qmv",
    "mxfp4_qmm_n",
    "mxfp4_qmv",
]


def quantize(
    w: torch.Tensor,
    group_size: Optional[int] = None,
    bits: Optional[int] = None,
    mode: str = "affine",
    global_scale: Optional[torch.Tensor] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """`mx.quantize`: quantize the last axis of `w` ([..., K], at least 2D) in groups.

    Returns `(w_q, scales, biases)` for affine and `(w_q, scales)` for the fp modes. `w_q` is
    [..., K * bits / 32] uint32; scales/biases are [..., K / group_size], in `w`'s dtype for affine
    and uint8 for the fp modes. `global_scale` (a float32 scalar) is nvfp4 only.
    """
    return tuple(ops.quantize(w, group_size, bits, mode, global_scale))


def dequantize(
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: Optional[torch.Tensor] = None,
    group_size: Optional[int] = None,
    bits: Optional[int] = None,
    mode: str = "affine",
    global_scale: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """`mx.dequantize`: the inverse of `quantize`, as a [..., K] tensor.

    `dtype` defaults to the scales'/biases' dtype for affine and to bfloat16 for the fp modes.
    """
    return ops.dequantize(w, scales, biases, group_size, bits, mode, global_scale, dtype)


def quantized_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: Optional[torch.Tensor] = None,
    transpose: bool = True,
    group_size: Optional[int] = None,
    bits: Optional[int] = None,
    mode: str = "affine",
) -> torch.Tensor:
    """`mx.quantized_matmul`: `x @ dequantize(w).T`, or `x @ dequantize(w)` with `transpose=False`.

    With `transpose=True` (an `nn.Linear` weight), `w` is [..., N, K * bits / 32] and the result
    [..., N]; with `transpose=False`, `w` is [..., K, N * bits / 32]. Batch dimensions broadcast
    when both `x` and `w` have them.
    """
    return ops.quantized_matmul(x, w, scales, biases, transpose, group_size, bits, mode)


def gather_qmm(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: Optional[torch.Tensor] = None,
    lhs_indices: Optional[torch.Tensor] = None,
    rhs_indices: Optional[torch.Tensor] = None,
    transpose: bool = True,
    group_size: Optional[int] = None,
    bits: Optional[int] = None,
    mode: str = "affine",
    global_scale: Optional[torch.Tensor] = None,
    sorted_indices: bool = False,
) -> torch.Tensor:
    """`mx.gather_qmm`: `quantized_matmul` of `x[lhs_indices]` with `w[rhs_indices]`.

    `x` is [..., M, K] and `w` is [E, ...] (one quantized matrix per expert); the indices select
    along their batch dimensions and broadcast against each other, and the result is
    `indices.shape + [M, N]`. With `sorted_indices=True` and no `lhs_indices`, `rhs_indices` must
    be sorted, which enables a faster kernel. `global_scale` (nvfp4) is one float32 per expert.
    """
    return ops.gather_qmm(
        x, w, scales, biases, lhs_indices, rhs_indices, transpose, group_size, bits, mode, global_scale, sorted_indices
    )


# ------------------------------------------------------------------------------------------------
# Version 1 API: the same signatures, on the functions above.
# ------------------------------------------------------------------------------------------------


def mxfp4_qmm_n(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    output_features: int,
) -> torch.Tensor:
    """`x @ dequantize(w, scales)` for an MXFP4 weight [K, output_features * 4 / 32] (uint8 scales)."""
    y = quantized_matmul(x, w, scales, None, transpose=False, mode="mxfp4")
    _check_features(y, output_features)
    return y


def mxfp4_qmv(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    output_features: int,
) -> torch.Tensor:
    """`x @ dequantize(w, scales).T` for an MXFP4 weight [output_features, K * 4 / 32] (uint8 scales)."""
    y = quantized_matmul(x, w, scales, None, transpose=True, mode="mxfp4")
    _check_features(y, output_features)
    return y


def affine_qmv(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    output_features: int,
    group_size: int = 128,
    bits: int = 4,
) -> torch.Tensor:
    """`x @ dequantize(w, scales, biases).T`; `w` is [output_features, K * bits / 32]."""
    y = quantized_matmul(x, w, scales, biases, transpose=True, group_size=group_size, bits=bits)
    _check_features(y, output_features)
    return y


def affine_qmm_t(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int = 128,
    bits: int = 4,
) -> torch.Tensor:
    """`x @ dequantize(w, scales, biases).T`; `w` is [N, K * bits / 32]."""
    return quantized_matmul(x, w, scales, biases, transpose=True, group_size=group_size, bits=bits)


def affine_qmm_n(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    output_features: int,
    group_size: int = 128,
    bits: int = 4,
) -> torch.Tensor:
    """`x @ dequantize(w, scales, biases)`; `w` is [K, output_features * bits / 32]."""
    y = quantized_matmul(x, w, scales, biases, transpose=False, group_size=group_size, bits=bits)
    _check_features(y, output_features)
    return y


def affine_qmm_t_nax(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int = 128,
    bits: int = 4,
) -> torch.Tensor:
    """`affine_qmm_t`. MLX's dispatch already uses the NAX kernels on GPUs that have them (M5)."""
    return affine_qmm_t(x, w, scales, biases, group_size, bits)


def affine_qmm_n_nax(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    output_features: int,
    group_size: int = 128,
    bits: int = 4,
) -> torch.Tensor:
    """`affine_qmm_n`. MLX's dispatch already uses the NAX kernels on GPUs that have them (M5)."""
    return affine_qmm_n(x, w, scales, biases, output_features, group_size, bits)


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
    """Row `m` of `x` [M, K] times expert `indices[m]` of `w`: [M, output_features], via `gather_qmm`."""
    y = gather_qmm(
        x.unsqueeze(-2), w, scales, biases, rhs_indices=indices, transpose=transpose, group_size=group_size, bits=bits
    ).squeeze(-2)
    _check_features(y, output_features)
    return y


def _check_features(y: torch.Tensor, output_features: int) -> None:
    if y.shape[-1] != output_features:
        raise ValueError(f"output_features={output_features} does not match the weight, which has {y.shape[-1]}")
