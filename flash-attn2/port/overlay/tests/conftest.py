import gc

import pytest
import torch


@pytest.fixture(scope="session")
def device(request):
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.xpu.is_available():
        torch.xpu.empty_cache()
