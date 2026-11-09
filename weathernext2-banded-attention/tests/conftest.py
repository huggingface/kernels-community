import pytest
import torch


@pytest.fixture
def reset_dynamo():
    """Drops compiled graphs so one test's guards cannot leak into the next."""
    yield
    torch._dynamo.reset()


@pytest.fixture
def float32_matmul_precision():
    """Restores the global float32 matmul precision a test changes."""
    previous = torch.get_float32_matmul_precision()
    yield torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision(previous)
