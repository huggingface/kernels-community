"""ggml's normalisation kernels, as torch ops.

Upstream ships `norm.metal` as its own file and this package is the matching unit. Eager torch spells
an RMS norm as five dispatches -- square, mean, add, rsqrt, and two multiplies -- where ggml fuses the
weight multiply into the normalisation and reads the row as float4. A model normalises at least twice
per layer, so at decode the difference is launch overhead rather than arithmetic.
"""

import os
from pathlib import Path

import torch


# A local (non-nix) build leaves the metallib on disk beside this module instead of embedding it, and
# `ggml_dispatch.mm` looks it up through this variable when nothing is embedded. Set before the extension
# is imported, and never over an existing value, so an explicit choice still wins.
_METALLIB = Path(__file__).parent / "ggml-metal.metallib"
if _METALLIB.is_file():
    os.environ.setdefault("GGML_NORM_METALLIB", str(_METALLIB))

from ._ops import add_op_namespace_prefix, ops  # noqa: E402
from .layers import RMSNormZeroCentered  # noqa: E402,F401


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`rms_norm(x) * weight` over the last axis of `x`, in one dispatch.

    `weight` is one row, taken as the kernel will use it: a model whose weight is zero-centered
    passes `1 + w`, not `w`.
    """
    return ops.rms_norm(x, weight, eps)


@torch.library.register_fake(add_op_namespace_prefix("rms_norm"))
def _rms_norm_fake(x, weight, eps):
    return torch.empty_like(x)


__all__ = ["RMSNormZeroCentered", "rms_norm"]
