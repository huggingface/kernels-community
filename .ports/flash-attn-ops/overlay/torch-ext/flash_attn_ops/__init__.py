"""Triton ops vendored from `flash-attn` (`flash_attn/ops/triton` and
`flash_attn/losses`), packaged as a self-contained, `kernels`-compliant kernel.

Upstream: https://github.com/Dao-AILab/flash-attention
Pinned commit: b02b07e1a10238fe12831b80a8937ed59b1353a5

These ops are pure Triton + PyTorch: there is nothing to compile ahead of time,
and the package does not import `flash_attn` at runtime.
"""

from .cross_entropy import cross_entropy_loss
from .losses import CrossEntropyLoss
from .rotary import apply_rotary
from .layer_norm import (
    LayerNormFn,
    RMSNorm,
    layer_norm_fn,
    layer_norm_linear_fn,
    rms_norm_fn,
)

__all__ = [
    # cross entropy
    "cross_entropy_loss",
    "CrossEntropyLoss",
    # rotary
    "apply_rotary",
    # layer / rms norm
    "layer_norm_fn",
    "rms_norm_fn",
    "layer_norm_linear_fn",
    "LayerNormFn",
    "RMSNorm",
]
