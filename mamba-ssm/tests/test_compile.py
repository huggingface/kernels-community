import kernels
import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def layers():
    return kernels.get_kernel("kernels-community/mamba-ssm", version=4).layers


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("optionals", [False, True])
def test_selective_scan_compile(layers, dtype, optionals):
    torch.manual_seed(0)

    def rand(*shape, dtype=dtype):
        return torch.randn(*shape, device="cuda", dtype=dtype).requires_grad_()

    # More than one native scan chunk, including an incomplete final chunk.
    args = (
        rand(2, 8, 2051), rand(2, 8, 2051),
        -rand(8, 16, dtype=torch.float32).detach().abs().requires_grad_(),
        rand(2, 16, 2051), rand(2, 16, 2051),
        rand(8, dtype=torch.float32) if optionals else None,
        rand(2, 8, 2051) if optionals else None,
        rand(8, dtype=torch.float32) if optionals else None,
    )

    def scan(*inputs):
        return layers.selective_scan_fn()(
            *inputs, delta_softplus=True, return_last_state=True,
        )

    compiled = torch.compile(scan, fullgraph=True)
    try:
        for _ in range(2):
            results = []
            for fn in (compiled, scan):
                inputs = tuple(t.detach().clone().requires_grad_() if t is not None else None for t in args)
                out, state = fn(*inputs)
                # Last-state gradients are explicitly unsupported by selective_scan_fn.
                out.float().square().mean().backward()
                results.append((out, state, [t.grad for t in inputs if t is not None]))
            torch.testing.assert_close(
                results[0], results[1],
                rtol=3e-2 if dtype == torch.bfloat16 else 2e-3,
                atol=3e-3 if dtype == torch.bfloat16 else 2e-5,
            )
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("projection_width", [64, 68])
def test_causal_conv1d_compile(layers, dtype, projection_width):
    torch.manual_seed(0)
    # Channel-last input triggers the original non-contiguous out= graph break.
    args = [torch.randn(*shape, device="cuda", dtype=dtype)
            for shape in ((2, 33, projection_width), (64, 4), (64,))]

    def conv(x, weight, bias):
        return layers.causal_conv1d_fn()(x[..., :64].transpose(1, 2), weight, bias, activation="silu")

    compiled = torch.compile(conv, fullgraph=True)
    try:
        for _ in range(2):
            results = []
            for fn in (compiled, conv):
                inputs = [t.detach().clone().requires_grad_() for t in args]
                out = fn(*inputs)
                out.float().square().mean().backward()
                results.append((out, [t.grad for t in inputs]))
            torch.testing.assert_close(results[0], results[1], rtol=3e-2, atol=3e-3)
    finally:
        torch._dynamo.reset()


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mamba_inner_output_bias_gradient(dtype):
    kernel = kernels.get_kernel("kernels-community/mamba-ssm", version=4)
    torch.manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=dtype) * 0.1

    # Compare with the independent Torch reference, including a nonzero bias.
    args = (
        rand(2, 32, 17), rand(16, 1, 4), rand(16), rand(18, 16),
        rand(16, 2), rand(8, 16), rand(8).float().requires_grad_(),
        -torch.rand(16, 8, device="cuda"),
    )
    bias = args[6]
    ref_bias = bias.detach().clone().requires_grad_()
    actual = kernel.mamba_inner_fn(*args, delta_softplus=True)
    expected = kernel.ops.selective_scan_interface.mamba_inner_ref(
        *args[:6], ref_bias, args[7], delta_softplus=True,
    )
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-3)
    torch.testing.assert_close(bias.grad, ref_bias.grad)
