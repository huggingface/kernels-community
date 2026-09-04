"""Fast smoke tests for the kernels-community CI runner.

The vendored upstream suite in ``test_flash_attn.py`` is intentionally
exhaustive. These tests cover the packaged fixed-length, variable-length, and
KV-cache APIs with small inputs so ``pytest -m kernels_ci`` stays inexpensive.
"""

import kernels
import pytest
import torch
import torch.nn.functional as F


flash_attn3 = kernels.get_kernel("kernels-community/flash-attn3", version=2)

cuda_major = (
    torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else None
)
cuda_supported = cuda_major in (8, 9)

pytestmark = [
    pytest.mark.kernels_ci,
    pytest.mark.skipif(
        not cuda_supported,
        reason="flash-attn3 requires an sm80 or sm90 CUDA device",
    ),
]


def reference_attention(q, k, v, causal=False):
    """Compute attention in float32 for an independent numerical reference."""
    q_ref = q.transpose(1, 2).float()
    k_ref = k.transpose(1, 2).float()
    v_ref = v.transpose(1, 2).float()
    if q_ref.shape[1] != k_ref.shape[1]:
        groups = q_ref.shape[1] // k_ref.shape[1]
        k_ref = k_ref.repeat_interleave(groups, dim=1)
        v_ref = v_ref.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, is_causal=causal
    ).transpose(1, 2)


def assert_close(actual, expected):
    torch.testing.assert_close(actual.float(), expected.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_kv_heads", [4, 2], ids=["mha", "gqa"])
def test_flash_attn_forward_backward(causal, num_kv_heads):
    """Exercise fixed-length MHA/GQA forward and backward dispatch."""
    torch.manual_seed(0)
    batch, seqlen, num_heads, head_dim = 2, 64, 4, 64

    q = torch.randn(
        batch,
        seqlen,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn(
        batch,
        seqlen,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    v = torch.randn_like(k, requires_grad=True)

    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    v_ref = v.detach().float().requires_grad_(True)

    out = flash_attn3.flash_attn_func(q, k, v, causal=causal)
    out_ref = reference_attention(q_ref, k_ref, v_ref, causal=causal)
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert_close(out, out_ref)

    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout)
    grads_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dout.float())
    for grad, grad_ref in zip(grads, grads_ref):
        assert_close(grad, grad_ref)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_varlen(causal):
    """Exercise the packed variable-length path with unequal sequences."""
    torch.manual_seed(1)
    lengths = (48, 31)
    num_heads, head_dim = 4, 64
    total = sum(lengths)

    q, k, v = [
        torch.randn(
            total,
            num_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(3)
    ]
    cu_seqlens = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)

    out = flash_attn3.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max(lengths),
        max(lengths),
        causal=causal,
    )
    out_ref = torch.cat(
        [
            reference_attention(
                q[start:end].unsqueeze(0),
                k[start:end].unsqueeze(0),
                v[start:end].unsqueeze(0),
                causal=causal,
            ).squeeze(0)
            for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())
        ]
    )

    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert_close(out, out_ref)


def test_flash_attn_with_kvcache():
    """Exercise the decode path against full attention over the cache."""
    torch.manual_seed(2)
    batch, seqlen, num_heads, head_dim = 2, 64, 4, 64
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(
        batch,
        seqlen,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn_like(k_cache)
    cache_seqlens = torch.full((batch,), seqlen, device="cuda", dtype=torch.int32)

    out = flash_attn3.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        causal=True,
    )
    out_ref = reference_attention(q, k_cache, v_cache)

    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert_close(out, out_ref)


def test_ops_fwd():
    """Exercise the raw op through the packaged ``ops`` namespace."""
    torch.manual_seed(3)
    batch, seqlen, num_heads, head_dim = 2, 64, 4, 64

    q, k, v = [
        torch.randn(
            batch,
            seqlen,
            num_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(3)
    ]

    out, softmax_lse, _, _ = flash_attn3.ops.fwd(q, k, v, is_causal=True)
    out_ref = reference_attention(q, k, v, causal=True)

    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert softmax_lse.shape == (batch, num_heads, seqlen)
    assert_close(out, out_ref)
