from . import layers

from ._ops import ops


def apply_rms_norm(input, weight, eps):
    # ops.apply_rms_norm returns [output, rstd]
    return ops.apply_rms_norm(
            input,
            weight,
            eps,
    )[0]

def apply_rms_norm_backward(grad_output, input, weight, output, rstd, eps, input_requires_grad=True, weight_requires_grad=True):
    return ops.apply_rms_norm_backward(
            grad_output,
            input,
            weight,
            output,
            rstd,
            eps,
            input_requires_grad,
            weight_requires_grad
    )

__all__ = ["layers", "apply_rms_norm_forward", "apply_rms_norm_backward"]

