"""Auto-loaded by pytest before any test module imports — adds the package's
``torch-ext`` source dir to ``sys.path`` so the suite runs without an install
step, then gates the whole suite on FP8 hardware support."""

import sys
from pathlib import Path

_TORCH_EXT = Path(__file__).resolve().parent.parent / "torch-ext"
if str(_TORCH_EXT) not in sys.path:
    sys.path.insert(0, str(_TORCH_EXT))

import pytest  # noqa: E402

from utils import SUPPORTS_FP8  # type: ignore  # noqa: E402


def pytest_collection_modifyitems(config, items):
    """Skip every test when FP8 is unsupported — Triton's ``fp8e4nv`` fails to
    compile on SM<89, so the entire FP8/FP4 suite is moot."""
    if SUPPORTS_FP8:
        return
    skip = pytest.mark.skip(reason="FP8 kernels require SM89+ on CUDA or XPU")
    for item in items:
        item.add_marker(skip)
