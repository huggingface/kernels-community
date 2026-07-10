"""Numerical smoke tests for the CK (compiled HIP) AITER flash-attention kernel.

Run with `pytest tests/test_flash_attn.py` on a ROCm host (gfx942 / gfx950).
Only bfloat16 and head dims 64/128 are compiled into this kernel.
"""

import math

import pytest
import torch


def _sdpa_reference(q, k, v, causal=False, scale=None, sink=None):
    """Eager-attention reference matching the FA2 (B, S, H, D) layout.

    If ``sink`` (shape ``(H,)``) is given, an extra per-head logit is appended
    to the softmax denominator (its value vector is zero), matching the
    learnable attention-sink formulation.
    """
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    scale = scale or (1.0 / math.sqrt(D))
    q_ = q.transpose(1, 2).float()
    k_ = k.transpose(1, 2).float()
    v_ = v.transpose(1, 2).float()
    attn = (q_ @ k_.transpose(-1, -2)) * scale  # (B, H, Sq, Sk)
    if causal:
        mask = torch.full((Sq, Sk), float("-inf"), device=q.device).triu(diagonal=1 + Sk - Sq)
        attn = attn + mask
    if sink is not None:
        sink_col = sink.view(1, H, 1, 1).expand(B, H, Sq, 1).float()
        attn = torch.cat([attn, sink_col], dim=-1)  # extra "key"
        probs = attn.softmax(dim=-1)[..., :Sk]  # drop the (zero-value) sink column
    else:
        probs = attn.softmax(dim=-1)
    out = probs @ v_  # (B, H, Sq, D)
    return out.transpose(1, 2).to(q.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs ROCm device")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("D", [64, 128])
def test_flash_attn_func_matches_sdpa(causal, D):
    from aiter_flash_attn_ck import flash_attn_func

    torch.manual_seed(0)
    B, S, H = 2, 128, 4
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

    out = flash_attn_func(q, k, v, causal=causal)
    ref = _sdpa_reference(q, k, v, causal=causal)

    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs ROCm device")
def test_flash_attn_varlen_matches_sdpa():
    from aiter_flash_attn_ck import flash_attn_varlen_func

    torch.manual_seed(0)
    H, D = 4, 128
    seqlens = [32, 48, 24]
    total = sum(seqlens)
    q = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor(
        [0, *torch.tensor(seqlens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    max_s = max(seqlens)

    out = flash_attn_varlen_func(q, k, v, cu, cu, max_s, max_s, causal=True)

    # Compare each sequence against a dense reference.
    start = 0
    for s in seqlens:
        qi = q[start : start + s].unsqueeze(0)
        ki = k[start : start + s].unsqueeze(0)
        vi = v[start : start + s].unsqueeze(0)
        ref = _sdpa_reference(qi, ki, vi, causal=True).squeeze(0)
        torch.testing.assert_close(
            out[start : start + s].float(), ref.float(), atol=2e-2, rtol=2e-2
        )
        start += s


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs ROCm device")
def test_flash_attn_func_with_sink():
    from aiter_flash_attn_ck import flash_attn_func

    torch.manual_seed(0)
    B, S, H, D = 2, 128, 4, 128
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(H, device="cuda", dtype=torch.float32)

    # sink_size > 0 (window_size[2]) enables the sink path.
    out = flash_attn_func(q, k, v, causal=True, window_size=(-1, -1, 1), s_aux=sink)
    ref = _sdpa_reference(q, k, v, causal=True, sink=sink)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


def _decode_reference(q, k_full, v_full):
    """Bottom-right causal attention of q against valid cache (GQA-aware)."""
    B, Sq, Hq, D = q.shape
    L = k_full.shape[1]
    g = Hq // k_full.shape[2]
    kf = k_full.repeat_interleave(g, dim=2).transpose(1, 2).float()
    vf = v_full.repeat_interleave(g, dim=2).transpose(1, 2).float()
    a = (q.transpose(1, 2).float() @ kf.transpose(-1, -2)) / math.sqrt(D)
    a = a + torch.full((Sq, L), float("-inf"), device=q.device).triu(1 + L - Sq)
    return (a.softmax(-1) @ vf).transpose(1, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs ROCm device")
def test_flash_attn_with_kvcache_decode():
    from aiter_flash_attn_ck import flash_attn_with_kvcache

    torch.manual_seed(0)
    B, Hq, Hkv, D, L = 4, 8, 2, 128, 300
    q = torch.randn(B, 1, Hq, D, device="cuda", dtype=torch.bfloat16)
    kc = torch.randn(B, 512, Hkv, D, device="cuda", dtype=torch.bfloat16)
    vc = torch.randn(B, 512, Hkv, D, device="cuda", dtype=torch.bfloat16)

    # uniform length, no append
    out = flash_attn_with_kvcache(q, kc, vc, cache_seqlens=L, causal=True)
    ref = _decode_reference(q, kc[:, :L], vc[:, :L])
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)

    # append a new token, then attend over L+1
    knew = torch.randn(B, 1, Hkv, D, device="cuda", dtype=torch.bfloat16)
    vnew = torch.randn(B, 1, Hkv, D, device="cuda", dtype=torch.bfloat16)
    out2 = flash_attn_with_kvcache(q, kc, vc, k=knew, v=vnew, cache_seqlens=L, causal=True)
    assert torch.equal(kc[:, L:L + 1], knew)  # cache written in place
    ref2 = _decode_reference(q, kc[:, :L + 1], vc[:, :L + 1])
    torch.testing.assert_close(out2.float(), ref2.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs ROCm device")
def test_flash_attn_with_kvcache_ragged():
    from aiter_flash_attn_ck import flash_attn_with_kvcache

    torch.manual_seed(0)
    B, Hq, Hkv, D = 4, 8, 2, 128
    q = torch.randn(B, 1, Hq, D, device="cuda", dtype=torch.bfloat16)
    kc = torch.randn(B, 512, Hkv, D, device="cuda", dtype=torch.bfloat16)
    vc = torch.randn(B, 512, Hkv, D, device="cuda", dtype=torch.bfloat16)
    lens = torch.tensor([100, 250, 300, 512], device="cuda")

    out = flash_attn_with_kvcache(q, kc, vc, cache_seqlens=lens, causal=True)
    for b in range(B):
        ref = _decode_reference(q[b:b + 1], kc[b:b + 1, : int(lens[b])], vc[b:b + 1, : int(lens[b])])
        torch.testing.assert_close(out[b:b + 1].float(), ref.float(), atol=2e-2, rtol=2e-2)
