"""Numerical smoke tests for the repackaged AITER flash-attention kernel.

Run with `pytest tests/test_flash_attn.py` on a ROCm host.
"""

import math
import pytest
import torch


def _sdpa_reference(q, k, v, causal=False, scale=None):
    """Eager-attention reference matching the FA2 (B, S, H, D) layout."""
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    scale = scale or (1.0 / math.sqrt(D))
    # (B, H, S, D)
    q_ = q.transpose(1, 2).float()
    k_ = k.transpose(1, 2).float()
    v_ = v.transpose(1, 2).float()
    attn = (q_ @ k_.transpose(-1, -2)) * scale  # (B, H, Sq, Sk)
    if causal:
        mask = torch.full((Sq, Sk), float("-inf"), device=q.device).triu(diagonal=1 + Sk - Sq)
        attn = attn + mask
    probs = attn.softmax(dim=-1)
    out = probs @ v_  # (B, H, Sq, D)
    return out.transpose(1, 2).to(q.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm device")
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_func_matches_sdpa(causal):
    from aiter_flash_attn import flash_attn_func

    torch.manual_seed(0)
    B, S, H, D = 2, 64, 4, 64
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)

    out = flash_attn_func(q, k, v, causal=causal)
    ref = _sdpa_reference(q, k, v, causal=causal)

    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm device")
def test_flash_attn_varlen_matches_sdpa():
    from aiter_flash_attn import flash_attn_varlen_func

    torch.manual_seed(0)
    H, D = 4, 64
    seqlens = [32, 48, 24]
    total = sum(seqlens)
    q = torch.randn(total, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(total, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(total, H, D, device="cuda", dtype=torch.float16)
    cu = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    max_s = max(seqlens)

    out = flash_attn_varlen_func(q, k, v, cu, cu, max_s, max_s, causal=True)

    # Reference: run dense attention per sequence and stitch.
    parts = []
    offset = 0
    for s in seqlens:
        qi = q[offset:offset + s].unsqueeze(0)  # (1, s, H, D)
        ki = k[offset:offset + s].unsqueeze(0)
        vi = v[offset:offset + s].unsqueeze(0)
        parts.append(_sdpa_reference(qi, ki, vi, causal=True).squeeze(0))
        offset += s
    ref = torch.cat(parts, dim=0)

    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)
