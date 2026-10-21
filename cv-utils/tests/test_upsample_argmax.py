"""Tests for `upsample_argmax`, checked against `F.interpolate` + `argmax`."""

import kernels
import pytest
import torch
import torch.nn.functional as F


cv_utils = kernels.get_kernel("kernels-community/cv-utils", version=2)

pytestmark = pytest.mark.kernels_ci

DEVICE = "cuda"


def smooth_logits(batch, classes, height, width, dtype=torch.float32):
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(batch, classes, height, width, generator=generator)
    return (F.avg_pool2d(logits, 5, stride=1, padding=2) * 5).to(DEVICE, dtype)


def reference(logits, size):
    return F.interpolate(logits[None].float(), size=size, mode="bilinear", align_corners=False)[0].argmax(0)


def mismatch_fraction(result, expected):
    return (result != expected).float().mean().item()


@pytest.mark.parametrize(
    "classes, height, width, size",
    [
        (150, 128, 128, (512, 683)),
        (19, 64, 128, (1024, 2048)),
        (2, 7, 9, (13, 31)),
        (21, 32, 32, (32, 32)),
        (5, 40, 40, (17, 23)),
    ],
)
def test_upsample_argmax_matches_interpolate(classes, height, width, size):
    logits = smooth_logits(2, classes, height, width)
    for result, logit in zip(cv_utils.upsample_argmax(logits, [size, size]), logits):
        assert result.dtype == torch.int64
        assert result.shape == size
        assert mismatch_fraction(result, reference(logit, size)) < 1e-5


def test_upsample_argmax_one_size_per_image():
    logits = smooth_logits(3, 19, 32, 48)
    sizes = [(64, 96), (100, 37), (32, 48)]
    for result, logit, size in zip(cv_utils.upsample_argmax(logits, sizes), logits, sizes):
        assert result.shape == size
        assert mismatch_fraction(result, reference(logit, size)) < 1e-5


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_upsample_argmax_half_precision_is_computed_in_float32(dtype):
    logits = smooth_logits(1, 150, 64, 64, dtype)
    result = cv_utils.upsample_argmax(logits, [(256, 256)])[0]
    assert mismatch_fraction(result, reference(logits[0], (256, 256))) < 1e-5


def test_upsample_argmax_on_other_device():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    logits = smooth_logits(1, 19, 32, 32).to("cuda:1")
    result = cv_utils.upsample_argmax(logits, [(64, 64)])[0]
    assert result.device == logits.device
    assert mismatch_fraction(result, reference(logits[0], (64, 64))) < 1e-5


def test_upsample_argmax_rejects_wrong_sizes():
    with pytest.raises(ValueError):
        cv_utils.upsample_argmax(smooth_logits(2, 3, 8, 8), [(16, 16)])
