"""Fast smoke tests for `nix run .#ci-test`.

Not derived from upstream. The upstream suite is far too large for the ~60 s
budget that the `kernels_ci` marker targets, so this module covers the public
API re-exported by `quack/__init__.py` at a single small shape each. Together
these exercise both CuTe-DSL kernel families in the package (row reductions
and the softmax/cross-entropy online reduction) plus their autograd paths.
"""

import pytest
import torch

from .kernel import quack, submodule

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

M, N = 256, 1024


@pytest.mark.kernels_ci
def test_rmsnorm():
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(N, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    out = quack.rmsnorm(x, weight)
    out_ref = submodule("rmsnorm").rmsnorm_ref(x.detach().float(), weight.detach().float())
    torch.testing.assert_close(out.float(), out_ref, atol=1e-1, rtol=1e-2)

    out.sum().backward()
    assert x.grad is not None and weight.grad is not None


@pytest.mark.kernels_ci
def test_softmax():
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    out = quack.softmax(x)
    out_ref = torch.softmax(x.detach().float(), dim=-1)
    torch.testing.assert_close(out.float(), out_ref, atol=1e-2, rtol=1e-2)

    out.sum().backward()
    assert x.grad is not None


@pytest.mark.kernels_ci
def test_cross_entropy():
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(0, N, (M,), device="cuda")

    loss = quack.cross_entropy(x, target)
    loss_ref = torch.nn.functional.cross_entropy(x.detach().float(), target)
    torch.testing.assert_close(loss.float(), loss_ref, atol=1e-2, rtol=1e-2)

    loss.backward()
    assert x.grad is not None


@pytest.mark.kernels_ci
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="GEMM requires Hopper (SM90) or newer",
)
def test_gemm():
    torch.manual_seed(0)
    gemm = submodule("gemm_interface").gemm
    a = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, M, device="cuda", dtype=torch.bfloat16)

    # tuned=False: the autotuner's config sweep does not fit the CI time budget.
    out = gemm(a, b, tuned=False)
    out_ref = a.float() @ b.float()
    torch.testing.assert_close(out.float(), out_ref, atol=3e-2, rtol=1e-2)
