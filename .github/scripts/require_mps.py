"""Pytest plugin for Metal CI: fail before collection if MPS is unavailable."""

import pytest


def pytest_sessionstart(session):
    import torch

    if not torch.backends.mps.is_available():
        pytest.exit(
            "Metal CI requires an available MPS device; check the runner's macOS "
            "version, GPU access, and PyTorch build.",
            returncode=1,
        )
