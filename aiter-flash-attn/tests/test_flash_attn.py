"""Numerical smoke tests for the repackaged AITER flash-attention kernel.

Run with `pytest tests/test_flash_attn.py` on a ROCm host.
"""

import math
import pytest
import torch


def _sdpa_reference(q, k, v, causal=False, scale=None, window_size=(-1, -1), sink=None):
    """Eager-attention reference matching the FA2 (B, S, H, D) layout.

    ``window_size=(left, right)`` mirrors the flash-attn sliding-window
    convention: query i attends to keys in
    [i + Sk - Sq - left, i + Sk - Sq + right] inclusive (a value of -1 removes
    that side's bound).

    ``sink`` (shape ``(H,)``) adds a per-head attention-sink logit that joins the
    softmax denominator but contributes no value (gpt-oss style).
    """
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    scale = scale or (1.0 / math.sqrt(D))
    # (B, H, S, D)
    q_ = q.transpose(1, 2).float()
    k_ = k.transpose(1, 2).float()
    v_ = v.transpose(1, 2).float()
    attn = (q_ @ k_.transpose(-1, -2)) * scale  # (B, H, Sq, Sk)

    i = torch.arange(Sq, device=q.device)[:, None]
    j = torch.arange(Sk, device=q.device)[None, :]
    shift = Sk - Sq
    keep = torch.ones((Sq, Sk), dtype=torch.bool, device=q.device)
    if causal:
        keep &= j <= i + shift
    left, right = int(window_size[0]), int(window_size[1])
    if left != -1:
        keep &= j >= i + shift - left
    if right != -1:
        keep &= j <= i + shift + right
    attn = attn.masked_fill(~keep, float("-inf"))

    if sink is not None:
        sink_col = sink.float().view(1, H, 1, 1).expand(B, H, Sq, 1)
        probs = torch.cat([attn, sink_col], dim=-1).softmax(dim=-1)[..., :Sk]
    else:
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm device")
@pytest.mark.parametrize(
    "causal,window_size",
    [
        (False, (16, 16)),   # symmetric local window
        (False, (32, 0)),    # left-only history + current
        (True, (16, 0)),     # causal sliding window (Gemma3/Mistral/Qwen2 SWA style)
        (True, (8, 0)),
    ],
)
def test_flash_attn_sliding_window_matches_sdpa(causal, window_size):
    # Non-causal right windows are unsupported by the default kernel and must be
    # auto-routed to the dao_ai backend; causal right windows are normalized away
    # (causal subsumes them) and stay on the default kernel. Both must be correct
    # without the caller ever setting mha_set_impl.
    from aiter_flash_attn import flash_attn_func
    from aiter_flash_attn import mha

    assert mha._MHA_IMPL == "default", "impl must stay default (no explicit override)"

    torch.manual_seed(0)
    B, S, H, D = 2, 128, 4, 64
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)

    out = flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    ref = _sdpa_reference(q, k, v, causal=causal, window_size=window_size)

    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm device")
@pytest.mark.parametrize("causal,window_size", [(False, (16, 16)), (True, (16, 0))])
def test_flash_attn_sliding_window_backward_matches_sdpa(causal, window_size):
    # Exercises the re-vendored dao_ai sliding-window backward (ROCm/aiter #3742).
    from aiter_flash_attn import flash_attn_func

    torch.manual_seed(0)
    B, S, H, D = 2, 128, 4, 64
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16, requires_grad=True)
    g = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)

    out = flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    out.backward(g)

    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)
    ref = _sdpa_reference(qr, kr, vr, causal=causal, window_size=window_size)
    ref.backward(g)

    torch.testing.assert_close(q.grad, qr.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k.grad, kr.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v.grad, vr.grad, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm device")
def test_flash_attn_varlen_sliding_window_matches_sdpa():
    from aiter_flash_attn import flash_attn_varlen_func

    torch.manual_seed(0)
    H, D = 4, 64
    window_size = (16, 0)
    seqlens = [32, 48, 24]
    total = sum(seqlens)
    q = torch.randn(total, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(total, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(total, H, D, device="cuda", dtype=torch.float16)
    cu = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    max_s = max(seqlens)

    out = flash_attn_varlen_func(
        q, k, v, cu, cu, max_s, max_s, causal=True, window_size=window_size
    )

    parts = []
    offset = 0
    for s in seqlens:
        qi = q[offset:offset + s].unsqueeze(0)
        ki = k[offset:offset + s].unsqueeze(0)
        vi = v[offset:offset + s].unsqueeze(0)
        parts.append(
            _sdpa_reference(qi, ki, vi, causal=True, window_size=window_size).squeeze(0)
        )
        offset += s
    ref = torch.cat(parts, dim=0)

    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm device")
@pytest.mark.parametrize("window_right", [0, -1])
def test_flash_attn_causal_sliding_window_with_sink(window_right):
    # gpt-oss style: causal sliding window + attention sinks. Under causal
    # masking a right window of 0 is a no-op, so it must be accepted and stay on
    # the default kernel (which supports a left window together with sinks) --
    # dao_ai has no sink support, so this must NOT route there.
    from aiter_flash_attn import flash_attn_func
    from aiter_flash_attn import mha

    assert mha._MHA_IMPL == "default"

    torch.manual_seed(0)
    B, S, H, D = 2, 128, 4, 64
    left = 32
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    sink = torch.randn(H, device="cuda", dtype=torch.float16)

    out = flash_attn_func(
        q, k, v, causal=True, window_size=(left, window_right), sink=sink
    )
    ref = _sdpa_reference(
        q, k, v, causal=True, window_size=(left, -1), sink=sink
    )

    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)
