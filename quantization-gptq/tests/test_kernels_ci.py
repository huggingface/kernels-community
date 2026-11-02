"""Fast subset of `test_gptq.py` for the kernels-community CI runner."""

import pytest

from . import test_gptq as _gptq


@pytest.mark.kernels_ci
@pytest.mark.parametrize(
    "M, K, N, group_size",
    [
        (1, 2048, 1024, 64),
        (31, 2048, 1024, 128),
        (244, 4096, 1024, 64),
    ],
)
def test_gptq(M, K, N, group_size):
    _gptq.test_gptq(M, K, N, group_size)
