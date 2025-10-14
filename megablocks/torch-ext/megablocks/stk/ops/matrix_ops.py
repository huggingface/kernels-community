from ..backend import sputnik
from ..matrix import Matrix
import torch
import numpy as np


@torch.no_grad()
def row_indices(shape, data, offsets, column_indices):
    return sputnik.row_indices(shape, data, offsets, column_indices)


# TODO(tgale): Replace this helper with a custom kernel. This operation
# is much simpler to do than how it's currently implemented.
@torch.no_grad()
def _expand_for_blocking(idxs, blocking):
    # Duplicate for block column dimension.
    idxs = torch.reshape(idxs, [idxs.size()[0], 1, 2]).repeat(1, blocking, 1)

    # Update the column indices.
    idxs[:, :, 1] *= blocking
    idxs[:, :, 1] += torch.reshape(torch.arange(blocking, device=idxs.device), [1, blocking])

    # Duplicate for block row dimension.
    idxs = torch.reshape(idxs, [idxs.size()[0], 1, blocking, 2])
    idxs = idxs.repeat(1, blocking, 1, 1)

    # Update the row indices.
    idxs[:, :, :, 0] *= blocking
    idxs[:, :, :, 0] += torch.reshape(torch.arange(blocking, device=idxs.device), [1, blocking, 1])
    idxs = torch.reshape(idxs, [-1, 2])
    return idxs


# TODO(tgale): Add input type checking.
@torch.no_grad()
def to_dense(x):
    assert isinstance(x, Matrix)

    shape = (np.prod(x.shape[:-1]), x.shape[-1])
    row_idxs = x.row_indices.type(torch.int32)
    col_idxs = x.column_indices.type(torch.int32)
    indices = _expand_for_blocking(torch.stack([row_idxs, col_idxs], dim=1), x.blocking)
    indices = (indices[:, 0] * shape[1] + indices[:, 1]).type(torch.int64)

    out = torch.zeros(shape[0] * shape[1], dtype=x.dtype, device=x.device)
    out.scatter_(0, indices, x.data.flatten())
    return out.reshape(x.size())


@torch.no_grad()
def _mask(x, blocking=1):
    assert x.dim() == 2
    assert x.size()[0] % blocking == 0
    assert x.size()[1] % blocking == 0
    block_rows = x.size()[0] // blocking
    block_cols = x.size()[1] // blocking
    x = torch.reshape(x, [block_rows, blocking, block_cols, blocking])
    x = torch.sum(torch.abs(x), dim=(1, 3))
    return x != 0


# TODO(tgale): Add input type checking.
@torch.no_grad()
def to_sparse(x, blocking=1):
    m = _mask(x, blocking)

    # TODO(tgale): Set to appropriate type for input matrix.
    row_nnzs = torch.sum(m, dim=1).type(torch.int32)
    zeros = torch.zeros((1,), dtype=row_nnzs.dtype, device=row_nnzs.device)
    offsets = torch.cat([zeros, torch.cumsum(row_nnzs, dim=0)])
    offsets = offsets.type(torch.int32)

    indices = torch.nonzero(m).type(torch.int16)
    row_indices = indices[:, 0]
    column_indices = indices[:, 1]

    # Nonzero indices in the dense matrix.
    nonzero_indices = torch.nonzero(m)
    nonzero_indices = _expand_for_blocking(nonzero_indices, blocking)
    nonzero_indices = nonzero_indices[:, 0] * x.size()[1] + nonzero_indices[:, 1]

    # Gather the data and construct the sparse matrix.
    data = torch.gather(x.flatten(), dim=0, index=nonzero_indices)
    data = torch.reshape(data, [-1, blocking, blocking])
    return Matrix(x.size(), data, row_indices, column_indices, offsets)


@torch.no_grad()
def ones_like(x):
    return Matrix(x.size(),
                  torch.ones_like(x.data),
                  x.row_indices,
                  x.column_indices, x.offsets)


def sum(x):
    assert isinstance(x, Matrix)
    return x.data.sum()
