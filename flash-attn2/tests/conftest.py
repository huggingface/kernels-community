import pytest
import torch


@pytest.fixture(scope="session")
def device(request):
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    else:
        return pytest.skip("No supported device found (CUDA or XPU)")
        