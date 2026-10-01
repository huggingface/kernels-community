"""Make the built kernel importable as `ggml_quantization`, whichever backend it was built for.

A variant directory is not a package, so `import ggml_quantization` only works if something puts it on
the path. Older CUDA builds happened to ship a `ggml_quantization/` shim that did this; builds from
kernel-builder 0.17 onwards do not, so relying on it would pass on one backend and fail on the other.
Resolving through `kernels.get_local_kernel` is what a consumer does, so the tests exercise the same
loading path rather than a layout detail.

When no kernel loads, the modules that need one are left uncollected rather than skipped from inside
a hook: a skip in `pytest_configure` is an `INTERNALERROR`, and a stub module turns into a confusing
collection error the first time a test touches it. `test_artifacts.py` never needs the kernel and is
always collected, so a build that ships but will not load fails there instead of vanishing into
skips.
"""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# modules that cannot run without a loadable kernel; `test_artifacts.py` deliberately is not one
NEEDS_KERNEL = ["test_ggml_quantization.py", "test_vendor_drift.py"]

collect_ignore = []

try:
    from kernels import get_local_kernel

    for _backend in (None, "cuda", "metal"):
        try:
            sys.modules["ggml_quantization"] = get_local_kernel(REPO_ROOT, _backend)
            break
        except Exception:  # noqa: BLE001, S112
            continue
    else:
        collect_ignore.extend(NEEDS_KERNEL)
except ImportError:
    collect_ignore.extend(NEEDS_KERNEL)
