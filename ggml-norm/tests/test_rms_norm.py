"""Small Metal RMS norm checks against PyTorch's CPU implementation."""

import kernels
import pytest
import torch
import torch.nn.functional as F


pytestmark = [
    pytest.mark.kernels_ci,
    pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS"),
]


@pytest.fixture(scope="module")
def norm():
    return kernels.get_kernel("ggml-org/ggml-norm", version=1)


def reference(x, weight, eps):
    # Use double precision on CPU so the reference does not share the Metal path.
    return F.rms_norm(x.double(), (x.shape[-1],), weight.double(), eps).to(x.dtype)


@pytest.mark.parametrize("shape", [(1, 128), (2, 3, 33)])
def test_rms_norm(norm, shape):
    # Exercise both float4 rows and the scalar path for widths not divisible by 4.
    x = torch.randn(shape, generator=torch.Generator().manual_seed(0))
    weight = torch.linspace(-0.5, 1.5, shape[-1])
    eps = 1e-5

    out = norm.rms_norm(x.to("mps"), weight.to("mps"), eps)

    assert out.device.type == "mps"
    torch.testing.assert_close(out.cpu(), reference(x, weight, eps), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("stride", [1, 2], ids=["storage-offset", "strided"])
def test_rms_norm_views(norm, stride):
    x = torch.randn(4, 128, generator=torch.Generator().manual_seed(1))
    weight = torch.linspace(-0.5, 1.5, 132)
    # Slice after transfer: contiguous views retain a nonzero MTLBuffer offset,
    # while strided views exercise the wrapper's contiguous copies.
    x_mps = x.to("mps")[1:3, ::stride]
    weight_mps = weight.to("mps")[4::stride]
    eps = 1e-5

    out = norm.rms_norm(x_mps, weight_mps, eps)

    expected = reference(x[1:3, ::stride], weight[4::stride], eps)
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=1e-6)


def test_rms_norm_epsilon(norm):
    x = torch.full((2, 32), 1e-4)
    x[0].zero_()
    weight = torch.linspace(-1.0, 1.0, 32)
    eps = 0.25

    out = norm.rms_norm(x.to("mps"), weight.to("mps"), eps)

    torch.testing.assert_close(out.cpu(), reference(x, weight, eps), rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_zero_centered_layer(norm, dtype):
    x = torch.randn(2, 32, generator=torch.Generator().manual_seed(2)).to(dtype)
    weight = torch.linspace(-0.5, 0.5, 32)
    layer = norm.RMSNormZeroCentered()
    layer.weight = torch.nn.Parameter(weight.to("mps"))
    layer.eps = 1e-5

    with torch.no_grad():
        # The second call also exercises reuse of the folded weight.
        for scale in (1.0, 0.5):
            inputs = x * scale
            out = layer(inputs.to("mps"))
            expected = reference(inputs.float(), 1.0 + weight, layer.eps).to(dtype)
            torch.testing.assert_close(out.cpu(), expected, rtol=1e-3, atol=1e-5)
