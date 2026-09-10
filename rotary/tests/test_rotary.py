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

    if not conj:
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
    else:
        out1 = x1 * cos + x2 * sin
        out2 = -x1 * sin + x2 * cos
    return out1, out2


def apply_rotary_torch_wrapper(
    q, k, cos, sin, conj: bool = False, transformers_style: bool = False, unsqueeze_dim: int = 1
):
    """the wrapper for apply_rotary_torch.

    If transformers_style is True, implements the HuggingFace transformers RoPE convention
    (rotate_half with duplicated frequencies), supporting partial RoPE where cos.shape[-1] <= q.shape[-1].
    """
    if transformers_style:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        rotary_dim = min(cos.shape[-1], q.shape[-1])
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
        return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)

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


def apply_rotary_kernel_wrapper(
    q, k, cos, sin, conj: bool = False, transformers_style: bool = False, unsqueeze_dim: int = 1, as_module: bool = False
):
    """the wrapper for apply_rotary_kernel"""
    if transformers_style:
        if as_module:
            layer = rotary.layers.apply_rotary_transformers()
            return layer(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        return rotary.apply_rotary_transformers(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

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
@pytest.mark.parametrize("headdim, rotary_dim", [(64, 32), (128, 64), (64, 30), (128, 32)])
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


def apply_rotary_transformers_torch_wrapper(q, k, cos, sin, unsqueeze_dim=1):
    """Convenience alias delegating to apply_rotary_torch_wrapper with transformers_style=True."""
    return apply_rotary_torch_wrapper(q, k, cos, sin, transformers_style=True, unsqueeze_dim=unsqueeze_dim)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("nheads", [8, 16])
@pytest.mark.parametrize("seqlen", [128, 256])
@pytest.mark.parametrize("headdim, rotary_dim", [(128, 128), (128, 32), (64, 32), (64, 16), (80, 20)])
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
@pytest.mark.parametrize("as_module", [False, True])
def test_apply_rotary_transformers(batch_size, nheads, seqlen, headdim, rotary_dim, dtype, atol, rtol, as_module):
    device = infer_device()
    if device is None:
        pytest.skip("No suitable device found for testing")

    q = torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype)
    k = torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype)

    freqs = torch.randn(batch_size, seqlen, rotary_dim // 2, device=device, dtype=torch.float32)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)

    ref_q, ref_k = apply_rotary_transformers_torch_wrapper(q, k, cos, sin, unsqueeze_dim=1)
    out_q, out_k = apply_rotary_kernel_wrapper(
        q.clone(), k.clone(), cos, sin, transformers_style=True, unsqueeze_dim=1, as_module=as_module
    )

    assert torch.allclose(ref_q, out_q, atol=atol, rtol=rtol), "Rotary transformation results for Q do not match"
    assert torch.allclose(ref_k, out_k, atol=atol, rtol=rtol), "Rotary transformation results for K do not match"

    # verify the non-rotated part of Q and K remains unchanged
    if rotary_dim < headdim:
        assert torch.equal(
            out_q[..., rotary_dim:], q[..., rotary_dim:]
        ), "Non-rotated part of Q should be unchanged"
        assert torch.equal(
            out_k[..., rotary_dim:], k[..., rotary_dim:]
        ), "Non-rotated part of K should be unchanged"

