"""Auto-loaded by pytest before any test module imports — adds the package's
``torch-ext`` source dir to ``sys.path`` so the suite runs without an install
step, pins each xdist worker to its own GPU, then gates the whole suite on FP8
hardware support."""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "torch-ext"))


def _visible_gpu_pool() -> list[str]:
    """The GPU ids this run may use: an explicit ``CUDA_VISIBLE_DEVICES`` restriction if
    set, else every GPU ``nvidia-smi`` reports (counted out-of-process so we don't create a
    CUDA context here — that would fix the parent's device before the per-worker pin)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [d for d in visible.split(",") if d != ""]
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
        )
        return [str(i) for i, ln in enumerate(out.stdout.splitlines()) if ln.strip()]
    except Exception:
        return []


def _pin_worker_gpu() -> None:
    """Under ``pytest -n`` (xdist) give each worker its own GPU: worker ``gwN`` takes the
    ``N``-th id of the pool (wrapping if there are more workers than GPUs — they then share,
    still correct, just contended). Must run at import time, before ``utils`` pulls in torch
    and initializes CUDA against whatever ``CUDA_VISIBLE_DEVICES`` currently says. No-op when
    not distributed (single process keeps the caller's device selection)."""
    worker = os.environ.get("PYTEST_XDIST_WORKER")  # "gw0", "gw1", ...; unset when serial
    if not worker or not worker.startswith("gw"):
        return
    try:
        idx = int(worker[2:])
    except ValueError:
        return
    pool = _visible_gpu_pool()
    if pool:
        os.environ["CUDA_VISIBLE_DEVICES"] = pool[idx % len(pool)]


_pin_worker_gpu()

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
