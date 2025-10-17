from ._ops import ops
import torch

def index_max(index_vals: torch.Tensor, indices: torch.Tensor, A_num_block: int, B_num_block: int):
    return ops.index_max(index_vals, indices, A_num_block, B_num_block)

def mm_to_sparse(dense_A: torch.Tensor, dense_B: torch.Tensor, indices: torch.Tensor):
    return ops.mm_to_sparse(dense_A, dense_B, indices)

def sparse_dense_mm(sparse_A: torch.Tensor, indices: torch.Tensor, dense_B: torch.Tensor, A_num_block: int):
    return ops.sparse_dense_mm(sparse_A, indices, dense_B, A_num_block)

def reduce_sum(sparse_A: torch.Tensor, indices: torch.Tensor, A_num_block: int, B_num_block: int):
    return ops.reduce_sum(sparse_A, indices, A_num_block, B_num_block)

def scatter(dense_A: torch.Tensor, indices: torch.Tensor, B_num_block: int):
    return ops.scatter(dense_A, indices, B_num_block)

__all__ = [
    "index_max",
    "mm_to_sparse",
    "sparse_dense_mm",
    "reduce_sum",
    "scatter",
]