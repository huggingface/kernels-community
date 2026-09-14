"""Compute directly on the packed blocks of a GGUF checkpoint.

A GGUF weight is stored as blocks of 32 or 256 values sharing a scale. These kernels read that
layout as it is, so a quantized model can be loaded and run without ever materializing a dense
copy — which is the whole memory saving.

Ported from llama.cpp's ggml-cuda; `vendor/UPSTREAM` pins the revision. The ops are named for what
they do rather than for a backend: each backend registers its own implementation of the same
schema, so calls dispatch on the tensor's device.
"""

import torch

from ._ops import add_op_namespace_prefix, ops


__all__ = ["GEMV_TYPES", "MAX_GEMV_ROWS", "dequantize", "get_rows", "mul_mat_id", "mul_mat_vec"]

# Upstream's MMVQ_MAX_BATCH_SIZE: `mul_mat_vec` has no implementation beyond this many rows, so a
# caller with more (prefill) dequantizes and uses an ordinary matmul.
MAX_GEMV_ROWS = 8

# ggml type ids `mul_mat_vec` implements, as this build reports them -- it is backend-specific, and
# a type routed into a gemv that has no kernel for it is a fault, not a fallback. CUDA covers
# Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, the K quants, the IQ quants, MXFP4/NVFP4 and Q1_0/Q2_0; Metal is the
# same minus NVFP4/Q1_0/Q2_0, which its metallib has no kernels for. `dequantize` covers more, so
# check this before choosing the fused path.
try:
    GEMV_TYPES = frozenset(ops.gemv_types())
except AttributeError:
    # A build published before `gemv_types` existed; those are CUDA-only, so its list is theirs.
    GEMV_TYPES = frozenset({2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 39, 40, 41, 42})


def get_rows(
    blocks: torch.Tensor, indices: torch.Tensor, ggml_type: int, cols: int, dtype: torch.dtype
) -> torch.Tensor:
    """The rows `indices` names, unpacked: `(rows, bytes_per_row)` uint8 -> `(len(indices), cols)`.

    ggml's `get_rows`, which dequantizes as it gathers -- so reading a few rows out of a large table
    never touches the rest.
    """
    return ops.get_rows(blocks, indices, ggml_type, cols, dtype)


def dequantize(
    blocks: torch.Tensor, ggml_type: int, rows: int, cols: int, dtype: torch.dtype
) -> torch.Tensor:
    """`(rows, bytes_per_row)` uint8 blocks -> `(rows, cols)` values of `dtype`."""
    return ops.dequantize(blocks, ggml_type, rows, cols, dtype)


def mul_mat_vec(
    blocks: torch.Tensor, x: torch.Tensor, ggml_type: int, out_features: int
) -> torch.Tensor:
    """Fused dequantize-gemv: `x @ blocks.T` without unpacking `blocks`.

    `x` is `(rows, in_features)` and `rows` must be at most `MAX_GEMV_ROWS`. The result is f32
    whatever `x`'s dtype was, since the kernel writes an f32 destination; cast it at the call site,
    where the cast can be fused into whatever consumes it.
    """
    return ops.mul_mat_vec(blocks, x, ggml_type, out_features)


def mul_mat_id(
    blocks: torch.Tensor, x: torch.Tensor, ids: torch.Tensor, ggml_type: int, out_features: int
) -> torch.Tensor:
    """One dispatch for a bank of routed experts: ggml's `mul_mv_id`.

    `blocks` is `(n_experts, out_features, bytes_per_row)`, `x` is `(n_tokens, in_features)`, and
    `ids` is `(n_tokens, n_used)` naming the expert each of a token's slots picked. The result is
    `(n_tokens, n_used, out_features)` f32, one row per slot.

    The alternative is a gemv per expert per layer, whose arithmetic is dwarfed by the dispatch
    around it -- which is most of what a MoE decode step costs.
    """
    return ops.mul_mat_id(blocks, x, ids, ggml_type, out_features)


# Without these, torch.compile cannot trace the ops and breaks the graph at every call.
@torch.library.register_fake(add_op_namespace_prefix("get_rows"))
def _get_rows_fake(blocks, indices, ggml_type, cols, dtype):
    return blocks.new_empty((indices.numel(), cols), dtype=dtype)


@torch.library.register_fake(add_op_namespace_prefix("dequantize"))
def _dequantize_fake(blocks, ggml_type, rows, cols, dtype):
    return blocks.new_empty((rows, cols), dtype=dtype)


@torch.library.register_fake(add_op_namespace_prefix("mul_mat_vec"))
def _mul_mat_vec_fake(blocks, x, ggml_type, out_features):
    return x.new_empty((x.shape[0], out_features), dtype=torch.float32)


@torch.library.register_fake(add_op_namespace_prefix("mul_mat_id"))
def _mul_mat_id_fake(blocks, x, ids, ggml_type, out_features):
    return x.new_empty((x.shape[0], ids.shape[1], out_features), dtype=torch.float32)
