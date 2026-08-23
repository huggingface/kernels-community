"""Host-side helpers shared by the kernel's launch path."""

from contextlib import contextmanager

import torch


def infer_device() -> str:
    """The accelerator this process should run on, or `"cpu"` if there is none.

    Follows Liger-Kernel's helper of the same name. `torch.cuda` covers AMD as well, since ROCm
    reports itself as cuda; Ascend NPU and Intel XPU each need their own probe. Kept here rather
    than in the kernel so tests and callers agree on what device to use.
    """
    if torch.cuda.is_available():  # Nvidia and AMD both
        return "cuda"
    if _is_npu_available():
        return "npu"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def _is_npu_available() -> bool:
    """Ascend NPU, which registers itself as `torch.npu` when its plugin is installed."""
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


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
