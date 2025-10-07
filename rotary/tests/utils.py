import torch


def infer_device():
    """
    Get current device name based on available devices
    """
    if torch.cuda.is_available():  # Works for both Nvidia and AMD
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    else:
        return None


def supports_bfloat16():
    device = infer_device()
    if device == "cuda":
        return torch.cuda.get_device_capability() >= (8, 0)  # Ampere and newer
    elif device == "xpu":
        return True
    else:
        return False