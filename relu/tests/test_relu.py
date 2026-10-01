import pytest
import torch
import torch.nn.functional as F

import kernels

relu = kernels.get_kernel("kernels-community/relu", version=1)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.mark.kernels_ci
def test_relu():
    device = get_device()
    x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    torch.testing.assert_close(F.relu(x), relu.relu(x))


@pytest.mark.kernels_ci
def test_relu_layer():
    device = get_device()
    x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    layer = relu.layers.ReLU()
    torch.testing.assert_close(F.relu(x), layer(x))
