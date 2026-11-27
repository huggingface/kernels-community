import kernels
import pytest
import torch

from tests.utils import infer_device, supports_bfloat16

rotary = kernels.get_kernel("kernels-community/rotary", version=2)

def apply_rotary_torch(x1: torch.Tensor, x2: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, conj: bool = False):
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"

    if not conj:
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
    else:
        out1 = x1 * cos + x2 * sin
        out2 = -x1 * sin + x2 * cos
    return out1, out2


def apply_rotary_torch_wrapper(q, k, cos, sin, conj: bool = False):
    """the wrapper for apply_rotary_torch"""
    rotary_dim = cos.shape[-1]
    
    # apply rotation encoding to Q
    q1 = q[..., :rotary_dim]
    q2 = q[..., rotary_dim : 2 * rotary_dim]
    q_out_1, q_out_2 = apply_rotary_torch(q1, q2, cos, sin, conj)
    q_out = torch.cat([q_out_1, q_out_2, q[..., 2 * rotary_dim:]], dim=-1)

    # apply rotation encoding to K
    k1 = k[..., :rotary_dim]
    k2 = k[..., rotary_dim : 2 * rotary_dim]
    k_out_1, k_out_2 = apply_rotary_torch(k1, k2, cos, sin, conj)
    k_out = torch.cat([k_out_1, k_out_2, k[..., 2 * rotary_dim:]], dim=-1)

    return q_out, k_out


def apply_rotary_kernel_wrapper(q, k, cos, sin, conj: bool = False):
    """the wrapper for apply_rotary_kernel"""
    rotary_dim = cos.shape[-1]
    
    # apply rotation encoding to Q
    q1 = q[..., :rotary_dim]
    q2 = q[..., rotary_dim : 2 * rotary_dim]
    rotary.apply_rotary(q1, q2, cos, sin, q1, q2, conj)

    # apply rotation encoding to K
    k1 = k[..., :rotary_dim]
    k2 = k[..., rotary_dim : 2 * rotary_dim]
    rotary.apply_rotary(k1, k2, cos, sin, k1, k2, conj)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("nheads", [8, 16])
@pytest.mark.parametrize("seqlen", [128, 256])
@pytest.mark.parametrize("headdim, rotary_dim", [(64, 32), (128, 64), (64, 30)])
@pytest.mark.parametrize("qk_dim", [3, 4])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-1,
            1e-5,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
@pytest.mark.parametrize("conj", [False, True])
@pytest.mark.flaky(max_runs=2, min_passes=1)
def test_rotary_equivalence(batch_size, nheads, seqlen, headdim, rotary_dim, qk_dim, dtype, atol, rtol, conj):
    device = infer_device()
    if device is None:
        pytest.skip("No suitable device found for testing")

    if qk_dim == 4:
        q_shape = (batch_size, seqlen, nheads, headdim)
        cos_sin_shape = (seqlen, 1, rotary_dim)
    elif qk_dim == 3:
        q_shape = (batch_size * seqlen, nheads, headdim)
        cos_sin_shape = (batch_size * seqlen, 1, rotary_dim)

    q_orig = torch.randn(q_shape, device=device, dtype=dtype)
    k_orig = torch.randn(q_shape, device=device, dtype=dtype)
    cos = torch.randn(cos_sin_shape, device=device, dtype=dtype)
    sin = torch.randn(cos_sin_shape, device=device, dtype=dtype)

    q_kernel, k_kernel = q_orig.clone(), k_orig.clone()
    q_torch, k_torch = q_orig.clone(), k_orig.clone()

    q_torch_out, k_torch_out = apply_rotary_torch_wrapper(q_torch, k_torch, cos, sin, conj)
    apply_rotary_kernel_wrapper(q_kernel, k_kernel, cos, sin, conj)

    # verify the rotation results of Q and K are consistent
    try:
        assert torch.allclose(q_torch_out, q_kernel, atol=atol, rtol=rtol), "Rotary transformation results for Q do not match"
    except AssertionError:
        diff_q = torch.abs(q_torch_out - q_kernel)
        max_diff_q = torch.max(diff_q)
        print(f"Max difference for Q: {max_diff_q}")
        raise
    try:
        assert torch.allclose(k_torch_out, k_kernel, atol=atol, rtol=rtol), "Rotary transformation results for K do not match"
    except AssertionError:
        diff_k = torch.abs(k_torch_out - k_kernel)
        max_diff_k = torch.max(diff_k)
        print(f"Max difference for K: {max_diff_k}")
        raise

    # verify the non-rotated part of Q and K remains unchanged
    if (2 * rotary_dim) < headdim:
        assert torch.equal(
            q_kernel[..., 2 * rotary_dim:], q_orig[..., 2 * rotary_dim:]
        ), "Non-rotated part of Q should be unchanged"
        assert torch.equal(
            k_kernel[..., 2 * rotary_dim:], k_orig[..., 2 * rotary_dim:]
        ), "Non-rotated part of K should be unchanged"


@pytest.mark.kernels_ci
@pytest.mark.parametrize("entry_point", ["function", "layer"])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-1,
            1e-5,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_apply_rotary_transformers_backward(entry_point, dtype, atol, rtol):
    device = infer_device()
    if device is None:
        pytest.skip("No suitable device found for testing")

    batch_size, nheads, seqlen, headdim = 2, 8, 128, 64
    rotary_dim = headdim // 2

    q_orig = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_orig = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    cos = torch.randn(seqlen, headdim, device=device, dtype=dtype)
    sin = torch.randn(seqlen, headdim, device=device, dtype=dtype)
    grad_q, grad_k = torch.randn_like(q_orig), torch.randn_like(k_orig)

    if entry_point == "function":
        apply_rotary_kernel = rotary.apply_rotary_transformers
    else:
        apply_rotary_kernel = rotary.layers.apply_rotary_transformers()

    q_torch, k_torch = q_orig.clone().requires_grad_(), k_orig.clone().requires_grad_()
    q_torch_out, k_torch_out = apply_rotary_torch_wrapper(
        q_torch, k_torch, cos[:, None, :rotary_dim], sin[:, None, :rotary_dim]
    )
    torch.autograd.backward((q_torch_out, k_torch_out), (grad_q, grad_k))

    q_kernel, k_kernel = q_orig.clone().requires_grad_(), k_orig.clone().requires_grad_()
    q_kernel_out, k_kernel_out = apply_rotary_kernel(q_kernel, k_kernel, cos, sin, unsqueeze_dim=1)
    torch.autograd.backward((q_kernel_out, k_kernel_out), (grad_q, grad_k))

    torch.testing.assert_close(q_kernel_out, q_torch_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_kernel_out, k_torch_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(q_kernel.grad, q_torch.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_kernel.grad, k_torch.grad, atol=atol, rtol=rtol)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("entry_point", ["function", "layer"])
@pytest.mark.parametrize(
    "dtype, grad_atol",
    [
        (torch.float16, 2e-2),
        pytest.param(
            torch.bfloat16,
            1.5e-1,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_apply_rotary_transformers_mixed_dtypes(entry_point, dtype, grad_atol):
    # Under autocast with float32 weights, q and k come out of a linear layer in
    # half precision while cos and sin are float32.
    device = infer_device()
    if device is None:
        pytest.skip("No suitable device found for testing")

    batch_size, nheads, seqlen, headdim = 2, 8, 128, 64
    rotary_dim = headdim // 2

    q_orig = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k_orig = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    cos = torch.randn(seqlen, headdim, device=device, dtype=torch.float32)
    sin = torch.randn(seqlen, headdim, device=device, dtype=torch.float32)
    grad_q = torch.randn(q_orig.shape, device=device, dtype=torch.float32)
    grad_k = torch.randn(k_orig.shape, device=device, dtype=torch.float32)

    if entry_point == "function":
        apply_rotary_kernel = rotary.apply_rotary_transformers
    else:
        apply_rotary_kernel = rotary.layers.apply_rotary_transformers()

    q_torch, k_torch = q_orig.clone().requires_grad_(), k_orig.clone().requires_grad_()
    q_torch_out, k_torch_out = apply_rotary_torch_wrapper(
        q_torch, k_torch, cos[:, None, :rotary_dim], sin[:, None, :rotary_dim]
    )
    torch.autograd.backward((q_torch_out, k_torch_out), (grad_q, grad_k))

    q_kernel, k_kernel = q_orig.clone().requires_grad_(), k_orig.clone().requires_grad_()
    q_kernel_out, k_kernel_out = apply_rotary_kernel(q_kernel, k_kernel, cos, sin, unsqueeze_dim=1)
    torch.autograd.backward((q_kernel_out, k_kernel_out), (grad_q, grad_k))

    assert q_kernel_out.dtype == q_torch_out.dtype == torch.float32
    # The outputs are float32. The gradients are stored in the dtype of q and k,
    # so they only match up to one rounding step of that dtype.
    torch.testing.assert_close(q_kernel_out, q_torch_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k_kernel_out, k_torch_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(q_kernel.grad, q_torch.grad, atol=grad_atol, rtol=1e-5)
    torch.testing.assert_close(k_kernel.grad, k_torch.grad, atol=grad_atol, rtol=1e-5)
