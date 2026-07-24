"""Tests for the standalone ``fp8_act_quant`` op."""

import pytest
import torch

from utils import TEST_DEVICE  # type: ignore

import finegrained_fp8  # type: ignore


_FP8_DTYPE = torch.float8_e4m3fn


def _ref_fp8_act_quant(x: torch.Tensor, block_size: int):
    """Pure-PyTorch reference: per-block dynamic FP8 quant.

    ``s = amax / 448`` (returned verbatim, can be 0 for all-zero blocks);
    the divider is floored at 1e-12 so all-zero blocks emit zeros, not NaN.
    """
    *prefix, K = x.shape
    n_blocks = K // block_size
    groups = x.reshape(*prefix, n_blocks, block_size).float()
    s = groups.abs().amax(dim=-1) / 448.0
    y = (groups / s.unsqueeze(-1).clamp(min=1e-12)).to(_FP8_DTYPE)
    return y.reshape(*prefix, K), s


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="Accelerator not available")
@pytest.mark.parametrize(
    "shape",
    [(16, 128), (1, 256), (4, 8, 128)],
    ids=lambda s: "x".join(map(str, s)),
)
@pytest.mark.parametrize("block_size", [32, 64, 128], ids=lambda b: f"bs{b}")
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float16, torch.float32],
    ids=["bf16", "fp16", "fp32"],
)
def test_fp8_act_quant(shape, block_size, dtype):
    """Kernel matches the pure-PyTorch reference at FP8 granularity."""
    if shape[-1] % block_size != 0:
        pytest.skip(f"shape last dim {shape[-1]} not divisible by {block_size}")
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype, device=TEST_DEVICE)

    y, s = finegrained_fp8.fp8_act_quant(x, block_size)
    y_ref, s_ref = _ref_fp8_act_quant(x, block_size)

    assert y.dtype == _FP8_DTYPE
    assert y.shape == x.shape
    assert s.dtype == torch.float32
    assert s.shape == x.shape[:-1] + (x.shape[-1] // block_size,)
    torch.testing.assert_close(s, s_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(y.float(), y_ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="Accelerator not available")
def test_fp8_act_quant_zero_block():
    """All-zero input emits zero quantized output (not NaN) and a zero scale.

    Regression test for the ``tl.maximum(s, 1e-12)`` divider floor: without it,
    a zero block divides by zero and produces NaN.
    """
    x = torch.zeros(2, 128, dtype=torch.bfloat16, device=TEST_DEVICE)
    y, s = finegrained_fp8.fp8_act_quant(x, block_size=128)
    assert not torch.isnan(y.float()).any()
    assert (y.float() == 0).all()
    assert (s == 0).all()
