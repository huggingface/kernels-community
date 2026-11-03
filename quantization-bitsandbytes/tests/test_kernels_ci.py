"""Fast subset of `test_bitsandbytes.py` for the kernels-community CI runner."""

import pytest

from . import test_bitsandbytes as _bnb


@pytest.mark.kernels_ci
@pytest.mark.parametrize("quant_type", [0, 1])
@pytest.mark.parametrize(
    "M, K, N, group_size",
    [
        (1, 2048, 1024, 64),
        (31, 4096, 1024, 64),
    ],
)
def test_bitsandbytes(M, K, N, group_size, quant_type):
    _bnb.test_bitsandbytes(M, K, N, group_size, quant_type)
