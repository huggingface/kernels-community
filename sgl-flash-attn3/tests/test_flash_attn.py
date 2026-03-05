"""Minimal smoke tests for sgl_flash_attn3 kernel.

Adapted from sgl-kernel/tests/test_flash_attention.py.
These tests verify that the kernel loads correctly and produces
numerically reasonable results for basic attention configurations.
"""

import math

import pytest
import torch


def is_fa3_supported():
    if not torch.cuda.is_available():
        return False
    return (torch.version.cuda >= "12.3") and (
        torch.cuda.get_device_capability()[0] in (8, 9)
    )


requires_fa3 = pytest.mark.skipif(
    not is_fa3_supported(),
    reason="FA3 requires CUDA >= 12.3 and sm80/sm90",
)


def attention_ref(q, k, v, causal=False, softmax_scale=None):
    """Simple reference attention for correctness checks."""
    seqlen_q, nheads, d = q.shape
    seqlen_k = k.shape[0]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * softmax_scale
    if causal:
        row_idx = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(seqlen_k, device=q.device).unsqueeze(0)
        causal_mask = col_idx <= row_idx + (seqlen_k - seqlen_q)
        scores.masked_fill_(~causal_mask.unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,khd->qhd", attn, v.float())
    return out.to(q.dtype)


@requires_fa3
@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("headdim", [128])
def test_flash_attn_varlen_basic(dtype, causal, headdim):
    """Test flash_attn_varlen_func with a single sequence."""
    from sgl_flash_attn3 import flash_attn_varlen_func

    device = "cuda"
    batch_size = 1
    seqlen_q = 64
    seqlen_k = 64
    nheads = 8
    nheads_k = 8

    q = torch.randn(seqlen_q, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(seqlen_k, nheads_k, headdim, device=device, dtype=dtype)
    v = torch.randn(seqlen_k, nheads_k, headdim, device=device, dtype=dtype)

    cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, seqlen_k], dtype=torch.int32, device=device)

    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        causal=causal,
    )

    ref = attention_ref(q, k, v, causal=causal)
    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)


@requires_fa3
@pytest.mark.kernels_ci
def test_flash_attn_with_kvcache_basic():
    """Test flash_attn_with_kvcache with a basic decode-like setup."""
    from sgl_flash_attn3 import flash_attn_with_kvcache

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 2
    seqlen_k = 128
    nheads = 8
    nheads_k = 8
    headdim = 128

    q = torch.randn(batch_size, 1, nheads, headdim, device=device, dtype=dtype)
    k_cache = torch.randn(batch_size, seqlen_k, nheads_k, headdim, device=device, dtype=dtype)
    v_cache = torch.randn(batch_size, seqlen_k, nheads_k, headdim, device=device, dtype=dtype)
    cache_seqlens = torch.full((batch_size,), seqlen_k, dtype=torch.int32, device=device)

    out = flash_attn_with_kvcache(
        q, k_cache, v_cache,
        cache_seqlens=cache_seqlens,
        causal=True,
    )

    assert out.shape == (batch_size, 1, nheads, headdim)
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"


@requires_fa3
@pytest.mark.kernels_ci
def test_flash_attn_varlen_with_lse():
    """Test that return_softmax_lse works and returns finite values."""
    from sgl_flash_attn3 import flash_attn_varlen_func

    device = "cuda"
    dtype = torch.bfloat16
    seqlen = 32
    nheads = 4
    headdim = 128

    q = torch.randn(seqlen, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(seqlen, nheads, headdim, device=device, dtype=dtype)
    v = torch.randn(seqlen, nheads, headdim, device=device, dtype=dtype)

    cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32, device=device)

    out, lse = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        causal=True,
        return_softmax_lse=True,
    )

    assert out.shape == (seqlen, nheads, headdim)
    assert lse is not None
    assert not torch.isnan(lse).any(), "LSE contains NaN"
