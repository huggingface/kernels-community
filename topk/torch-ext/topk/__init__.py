"""Top-k over a small row, as a torch op.

For a MoE router. torch's MPS `topk` is a full sort, and so is ggml's -- `GGML_OP_TOP_K` dispatches
`kernel_argsort_f32_i32_desc` -- which is why this is written rather than ported: selecting the
largest 8 of 256 logits does not need the row ordered. 26us against 71us, once per layer per token.
"""

import os
from pathlib import Path

import torch


# A local (non-nix) build leaves the metallib on disk beside this module instead of embedding it, and
# `dispatch.mm` looks it up through this variable when nothing is embedded. Set before the extension
# is imported, and never over an existing value, so an explicit choice still wins.
_METALLIB = Path(__file__).parent / "ggml-metal.metallib"
if _METALLIB.is_file():
    os.environ.setdefault("TOPK_METALLIB", str(_METALLIB))

from ._ops import add_op_namespace_prefix, ops  # noqa: E402
from .layers import SoftmaxTopKRouter  # noqa: E402,F401


def top_k(logits: torch.Tensor, k: int, softmax: bool = False):
    """`(values, indices)` of the largest `k` per row of a 2D f32 tensor.

    `indices` is int32, which is what an expert-routed matmul wants, so routing them onward costs no
    cast. With `softmax`, `values` is softmaxed over the selected k.
    """
    return ops.top_k(logits, k, softmax)


@torch.library.register_fake(add_op_namespace_prefix("top_k"))
def _top_k_fake(logits, k, softmax):
    rows = logits.shape[0]
    return [logits.new_empty((rows, k)), logits.new_empty((rows, k), dtype=torch.int32)]


__all__ = ["SoftmaxTopKRouter", "top_k"]
