"""Auto-loaded by pytest before any test module imports — adds the package's
``torch-ext`` source dir to ``sys.path`` so the suite runs without an install
step, then gates the whole suite on FP8 hardware support."""

import sys
from pathlib import Path

import pytest  # noqa: E402

from utils import SUPPORTS_FP8  # type: ignore  # noqa: E402


def pytest_collection_modifyitems(config, items):
    """Skip every test when FP8 is unsupported — kernels require SM90+ on CUDA
    (Hopper/Blackwell) or XPU. SM<89 can't even compile ``fp8e4nv``; SM89 (Ada)
    compiles but has numerics drift we don't gate against."""
    if SUPPORTS_FP8:
        return
    skip = pytest.mark.skip(reason="FP8 kernels require SM90+ on CUDA or XPU")
    for item in items:
        item.add_marker(skip)
