"""Make the built kernel importable as `mlx_quantization_metal_kernels`.

Resolving through `kernels.get_local_kernel` is what a consumer does, so the tests exercise the same
loading path rather than a layout detail. When no kernel loads, the modules that need one are left
uncollected; `test_vendor_drift.py` needs neither a build nor a GPU and always runs.
"""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

NEEDS_KERNEL = ["test_mlx_quantization.py"]

collect_ignore = []

try:
    from kernels import get_local_kernel

    sys.modules["mlx_quantization_metal_kernels"] = get_local_kernel(REPO_ROOT / "build")
except Exception:  # noqa: BLE001
    collect_ignore.extend(NEEDS_KERNEL)
