import torch


def device_module(device):
    """Return the torch submodule owning ``device`` (``torch.cuda``, ``torch.xpu``, ...).

    Resolves to ``torch.cuda`` on CUDA, so behaviour there is unchanged.
    """
    if device is None:
        return torch.cuda
    if not isinstance(device, torch.device):
        device = torch.device(device)
    return torch.get_device_module(device)


def device_guard(tensor):
    """Replaces ``with torch.cuda.device(t.device.index):``."""
    return device_module(tensor.device).device(tensor.device.index)


def get_sm_count(device):
    """Number of independently schedulable compute units, used to size grids.

    CUDA and ROCm call this ``multi_processor_count``, Intel ``max_compute_units``.
    """
    props = device_module(device).get_device_properties(device)
    count = getattr(props, "multi_processor_count", None)
    if count is None:
        count = getattr(props, "max_compute_units", None)
    if count is None:
        raise RuntimeError(f"cannot determine compute unit count for device {device}")
    return count
