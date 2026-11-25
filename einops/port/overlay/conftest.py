import os

import pytest

# Test the frameworks that are available in CI, while allowing local overrides.
os.environ.setdefault("EINOPS_TEST_BACKENDS", "numpy,torch")

SKIPPED = {
    "test_einops_threading": "starts a subprocess that imports einops by name",
}


def pytest_collection_modifyitems(config, items):
    skip_compile = pytest.mark.skip(reason="torch.compile needs a C++ toolchain and openssl")
    for item in items:
        item.add_marker(pytest.mark.kernels_ci)
        if "torch_compile" in item.name:
            item.add_marker(skip_compile)
        if item.name in SKIPPED:
            item.add_marker(pytest.mark.skip(reason=SKIPPED[item.name]))
