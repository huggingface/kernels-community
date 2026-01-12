import torch
import pytest

import tinygrad_rms


def reference_rms_norm(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Reference implementation of RMSNorm."""
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + epsilon)
    return x / rms


@pytest.mark.parametrize(
    "shape",
    [
        (32, 512, 1024),  # 3D tensor (batch=32, seq=512, hidden=1024) -> 16384 rows
        (16, 1024),       # 2D tensor (16 rows)
        (32, 1024),       # 2D tensor (32 rows)
        (64, 1024),       # 2D tensor (64 rows)
        (128, 1024),      # 2D tensor (128 rows)
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_tinygrad_rms_norm(shape, dtype):
    """Test tinygrad_rms_norm against reference implementation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    x = torch.randn(shape, dtype=dtype, device=device)
    epsilon = 1e-6

    # Reference output
    ref_output = reference_rms_norm(x, epsilon)

    # Kernel output
    kernel_output, rms_inv = tinygrad_rms.tinygrad_rms_norm(x, epsilon)

    torch.testing.assert_close(kernel_output, ref_output, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [
        (32, 512, 1024),
        (16, 1024),
        (32, 1024),
    ],
)
def test_tinygrad_rms_norm_simple(shape):
    """Test tinygrad_rms_norm_simple against reference implementation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    x = torch.randn(shape, dtype=torch.float32, device=device)
    epsilon = 1e-6

    # Reference output
    ref_output = reference_rms_norm(x, epsilon)

    # Kernel output
    kernel_output = tinygrad_rms.tinygrad_rms_norm_simple(x, epsilon)

    torch.testing.assert_close(kernel_output, ref_output, rtol=1e-4, atol=1e-4)


def test_tinygrad_rms_norm_with_preallocated_output():
    """Test with pre-allocated output tensor."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    shape = (32, 512, 1024)
    x = torch.randn(shape, dtype=torch.float32, device=device)
    out = torch.empty_like(x)
    epsilon = 1e-6

    # Reference output
    ref_output = reference_rms_norm(x, epsilon)

    # Kernel output with pre-allocated tensor
    kernel_output, _ = tinygrad_rms.tinygrad_rms_norm(x, epsilon, out=out)

    # Verify out was used
    assert kernel_output.data_ptr() == out.data_ptr()
    torch.testing.assert_close(kernel_output, ref_output, rtol=1e-4, atol=1e-4)


def test_rms_inv_values():
    """Test that rms_inv values are correct."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    shape = (16, 1024)  # 16 rows, divisible by 16
    x = torch.randn(shape, dtype=torch.float32, device=device)
    epsilon = 1e-6

    _, rms_inv = tinygrad_rms.tinygrad_rms_norm(x, epsilon)

    # Compute reference rms_inv
    ref_rms_inv = 1.0 / torch.sqrt(torch.mean(x**2, dim=-1) + epsilon)

    torch.testing.assert_close(rms_inv, ref_rms_inv, rtol=1e-4, atol=1e-4)


def test_contiguous_requirement():
    """Test that non-contiguous input raises an error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    x = torch.randn(32, 1024, dtype=torch.float32, device=device)
    x_noncontig = x.t()  # Transpose makes it non-contiguous

    with pytest.raises(RuntimeError, match="contiguous"):
        tinygrad_rms.tinygrad_rms_norm(x_noncontig)
