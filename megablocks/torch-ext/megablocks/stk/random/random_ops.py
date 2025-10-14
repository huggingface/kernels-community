import numpy as np
import torch
from ..ops import matrix_ops


@torch.no_grad()
def dense_mask(rows, cols, sparsity, blocking=1):
  assert sparsity >= 0.0 and sparsity <= 1.0
  assert rows % blocking == 0 and cols % blocking == 0

  block_rows, block_cols = (rows // blocking, cols // blocking)
  nnz = round(block_rows * block_cols * (1 - sparsity))

  out = np.ones(block_rows * block_cols)
  mask = np.random.choice(out.size, out.size - nnz, replace=False)
  out[mask] = 0.0

  out = np.tile(
    np.reshape(out, [block_rows, 1, block_cols, 1]),
    (1, blocking, 1, blocking))
  out = np.reshape(out, [rows, cols])
  return torch.from_numpy(out.astype(np.float32))


@torch.no_grad()
def mask(m, n, sparsity, blocking=1):
    out = dense_mask(m, n, sparsity, blocking).type(torch.float16)
    return matrix_ops.to_sparse(out, blocking=blocking)


@torch.no_grad()
def randn(shape, sparsity, blocking=1):
  shape_2d = (np.prod(shape[:-1]), shape[-1])
  out = mask(*shape_2d, sparsity, blocking)
  out.data.copy_(torch.randn(*out.data.shape))
  return out.view(*shape)
