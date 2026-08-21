"""Host-side helpers shared by the kernel's launch path."""

from contextlib import contextmanager

import torch


@contextmanager
def device_context(device: torch.device):
    """Context manager that sets the active device for any backend (cuda, xpu, etc.).

    Every host-side launch has to sit inside one of these. Triton launches on whatever device is
    current, not on the device the tensors live on, so under `device_map="auto"` a shard on `cuda:1`
    would otherwise be launched against `cuda:0`.
    """
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "device"):
        with backend.device(device):
            yield
    else:
        yield
