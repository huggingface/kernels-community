from . import layers

from ._ops import ops


def apply_rms_norm(input, weight, eps):
    return ops.apply_rms_norm(
            input,
            weight,
            eps,
    )

__all__ = ["layers", "apply_rms_norm"]

