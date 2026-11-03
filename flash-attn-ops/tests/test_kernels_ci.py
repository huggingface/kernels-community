"""Fast subset of the vendored tests for the kernels-community CI runner."""

import pytest
import torch

from . import test_cross_entropy as _cross_entropy
from . import test_layer_norm as _layer_norm
from . import test_rotary as _rotary

pytestmark = [
    pytest.mark.kernels_ci,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize(
    "smoothing, logit_scale, lse_square_scale, return_z_loss, inplace_backward, precompute_lse",
    [
        (0.0, 1.0, 0.0, False, False, False),
        (0.0, 1.0, 0.0, False, False, True),
        (0.9, 1.0, 0.0, False, True, False),
        (0.0, 0.7, 1e-2, True, False, False),
        (0.9, 0.7, 1e-2, True, True, False),
    ],
)
def test_cross_entropy_loss(
    smoothing,
    logit_scale,
    lse_square_scale,
    return_z_loss,
    inplace_backward,
    precompute_lse,
):
    _cross_entropy.test_cross_entropy_loss(
        50257,
        smoothing,
        logit_scale,
        lse_square_scale,
        return_z_loss,
        inplace_backward,
        precompute_lse,
        torch.float16,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_bias", [True, False])
def test_layer_norm(has_bias, dtype):
    _layer_norm.test_layer_norm(1024, has_bias, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm(dtype):
    _layer_norm.test_rms_norm(1024, dtype)


def test_layer_norm_prenorm_residual():
    _layer_norm.test_layer_norm_prenorm_residual(torch.bfloat16)


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("headdim, rotary_dim", [(64, 64), (128, 64)])
def test_apply_rotary(headdim, rotary_dim, interleaved):
    _rotary.test_apply_rotary(headdim, rotary_dim, interleaved, torch.bfloat16)
