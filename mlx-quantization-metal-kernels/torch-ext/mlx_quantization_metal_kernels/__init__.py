"""MLX's quantized matmul kernels, for torch tensors on MPS.

The Metal kernels are MLX's own, compiled as they ship; `vendor/UPSTREAM` pins the revision. The
host-side dispatch transcribes MLX's `QuantizedMatmul::eval_gpu`, so a given shape runs the same
kernel it would under `mlx.core.quantized_matmul`.

The weight layout is MLX's affine one, the same `mx.quantize` produces: `w` packs `bits`-wide values
little-endian into uint32, and each `group_size` run of a row shares a scale and a bias, so
`w_float = scale * q + bias`.
"""

import torch

from ._ops import ops


__all__ = ["affine_qmm_t"]


def affine_qmm_t(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int = 64,
    bits: int = 4,
) -> torch.Tensor:
    """`x @ dequantize(w, scales, biases).T`, i.e. a quantized `nn.Linear` without its bias.

    x: [..., K] float32/float16/bfloat16, w: [N, K * bits / 32] uint32,
    scales/biases: [N, K / group_size] in x's dtype. Returns [..., N].

    group_size is 32, 64 or 128; bits is 2, 3, 4, 5, 6 or 8. A few rows (decode) run a
    matrix-vector kernel and many rows (prefill) a tiled matmul, with the crossover MLX uses for
    this GPU.
    """
    return ops.affine_qmm_t(x, w, scales, biases, group_size, bits)

