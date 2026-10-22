"""Host-side helpers shared by the kernel's launch path."""

from contextlib import contextmanager

import torch


def infer_device() -> str:
    """An available accelerator supported by this package, or CPU for fallback tests."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


@contextmanager
def device_context(device: torch.device):
    """Launch on the tensors' device, including when model layers are sharded."""
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "device"):
        with backend.device(device):
            yield
    else:
        yield
