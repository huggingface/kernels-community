import pytest
import torch

from tests.utils import infer_device, supports_bfloat16
from pathlib import Path

# import rotary
# from transformers.trainer_utils import set_seed
# set_seed(42)

# Set the local repo path, relative path
try:
    import rotary
except ImportError:
    from kernels import get_local_kernel
    repo_path = Path(__file__).parent.parent
    rotary = get_local_kernel(repo_path=repo_path, package_name="rotary")

def apply_rotary_torch(x1: torch.Tensor, x2: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, conj: bool = False):
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"

    dtype = x1.dtype
    x1, x2, cos, sin = (t.float() for t in (x1, x2, cos, sin))
    if not conj:
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
    else:
        out1 = x1 * cos + x2 * sin
        out2 = -x1 * sin + x2 * cos
    return out1.to(dtype), out2.to(dtype)


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
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
@pytest.mark.parametrize("conj", [False, True])
def test_rotary_equivalence(batch_size, nheads, seqlen, headdim, rotary_dim, qk_dim, dtype, conj):
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
    torch.testing.assert_close(q_kernel, q_torch_out)
    torch.testing.assert_close(k_kernel, k_torch_out)

    # verify the non-rotated part of Q and K remains unchanged
    if (2 * rotary_dim) < headdim:
        assert torch.equal(
            q_kernel[..., 2 * rotary_dim:], q_orig[..., 2 * rotary_dim:]
        ), "Non-rotated part of Q should be unchanged"
        assert torch.equal(
            k_kernel[..., 2 * rotary_dim:], k_orig[..., 2 * rotary_dim:]
        ), "Non-rotated part of K should be unchanged"
