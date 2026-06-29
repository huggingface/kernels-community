"""Minimal ``get_gfx()`` replacement.

Upstream lives in ``aiter/jit/utils/chip_info.py`` and pulls in the JIT C++
build machinery. We only need the chip-arch string here.
"""

from __future__ import annotations

import functools
import os


@functools.lru_cache(maxsize=1)
def get_gfx() -> str:
    """Return the active GPU arch string (e.g. ``"gfx942"``).

    Resolution order:
    1. ``GFX_ARCH`` env var (explicit override).
    2. Triton's active driver target.
    3. ``torch.cuda.get_device_properties(0).gcnArchName`` (ROCm/HIP only).
    4. ``""`` if none are available — callers using this for fast-path
       selection should treat that as "unknown arch, use safe defaults".
    """

    env = os.environ.get("GFX_ARCH")
    if env:
        return env

    try:
        import triton

        target = triton.runtime.driver.active.get_current_target()
        arch = getattr(target, "arch", None)
        if isinstance(arch, str) and arch.startswith("gfx"):
            return arch
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_properties(0).gcnArchName
            return name.split(":")[0] if name else ""
    except Exception:
        pass

    return ""
