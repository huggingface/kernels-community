# Smoke tests for CI that use a subset of upstream tests. Since the upstream
# tests are parametrized, we call them here with our own (much more limited)
# parametrizations.
#
# The upstream module is imported as a whole rather than importing the test
# functions by name, so that pytest does not collect the full upstream matrix a
# second time from this module.

import pytest
import torch

from . import test_flash_attn as upstream


@pytest.mark.kernels_ci
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_output_ci(mha_type, causal, device):
    upstream.test_flash_attn_output(
        seqlen_q=113,
        seqlen_k=203,
        d=64,
        dropout_p=0.0,
        causal=causal,
        local=False,
        alibi=False,
        deterministic=False,
        mha_type=mha_type,
        dtype=torch.float16,
        kvpacked=False,
        softcap=0.0,
        device=device,
    )


@pytest.mark.kernels_ci
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_varlen_output_ci(mha_type, causal, device):
    upstream.test_flash_attn_varlen_output(
        seqlen_q=113,
        seqlen_k=203,
        d=64,
        dropout_p=0.0,
        causal=causal,
        local=False,
        alibi=False,
        deterministic=False,
        mha_type=mha_type,
        dtype=torch.float16,
        kvpacked=False,
        softcap=0.0,
        device=device,
    )


@pytest.mark.kernels_ci
def test_flash_attn_varlen_lse_ci(device):
    if device == "cpu":
        pytest.skip("CPU implementation does not return softmax_lse")

    torch.manual_seed(0)
    lengths = [2, 4]
    cu_seqlens = torch.tensor([0, 2, 6], dtype=torch.int32, device=device)
    q = torch.randn(6, 2, 64, dtype=torch.float16, device=device, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)

    out, lse, _ = upstream.flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max(lengths), max(lengths), return_attn_probs=True
    )
    expected_lse = []
    expected_out = []
    for start, length in zip((0, 2), lengths):
        scores = torch.einsum("qhd,khd->hqk", q[start : start + length].float(), k[start : start + length].float())
        scores = scores * (64**-0.5)
        expected_lse.append(torch.logsumexp(scores, dim=-1))
        expected_out.append(torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v[start : start + length].float()))

    assert lse.shape == (2, 6)
    torch.testing.assert_close(lse, torch.cat(expected_lse, dim=1), atol=3e-2, rtol=1e-2)
    reference_out = torch.cat(expected_out).to(q.dtype)
    grad = torch.randn_like(out)
    actual_grads = torch.autograd.grad(out, (q, k, v), grad)
    expected_grads = torch.autograd.grad(reference_out, (q, k, v), grad)
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


@pytest.mark.kernels_ci
def test_flash_attn_varlen_padded_query_lse_ci(device):
    cu_seqlens = torch.tensor([0, 5] + [5] * 63, dtype=torch.int32, device=device)
    query = torch.randn(64, 2, 64, dtype=torch.float16, device=device, requires_grad=True)
    key = torch.randn(5, 2, 64, dtype=torch.float16, device=device, requires_grad=True)
    value = torch.randn(5, 2, 64, dtype=torch.float16, device=device, requires_grad=True)

    output, lse, _ = upstream.flash_attn_varlen_func(
        query, key, value, cu_seqlens, cu_seqlens, 64, 64, return_attn_probs=True
    )
    assert output.shape == query.shape
    assert lse.shape == (2, 64)
    scores = torch.einsum("qhd,khd->hqk", query[:5].float(), key.float()) * (64**-0.5)
    torch.testing.assert_close(lse[:, :5], torch.logsumexp(scores, dim=-1), atol=3e-2, rtol=1e-2)
    torch.testing.assert_close(lse[:, 5:], torch.zeros_like(lse[:, 5:]))
    gradients = torch.autograd.grad(output[:5].sum(), (query, key, value))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    torch.testing.assert_close(gradients[0][5:], torch.zeros_like(gradients[0][5:]))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_splitkv_ci(causal, device):
    upstream.test_flash_attn_splitkv(
        seqlen_q=1,
        seqlen_k=339,
        swap_sq_sk=False,
        d=64,
        causal=causal,
        local=False,
        alibi=False,
        deterministic=False,
        dtype=torch.float16,
        device=device,
    )
