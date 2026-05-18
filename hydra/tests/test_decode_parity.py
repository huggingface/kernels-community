from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decode_matches_sdpa_full_kv():
    import torch.nn.functional as F
    import hydra

    torch.manual_seed(0)
    q = torch.randn(1, 32, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8, 256, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 8, 256, 128, device="cuda", dtype=torch.bfloat16)

    out = hydra.hydra(q, k, v)
    k_rep = k.repeat_interleave(4, dim=1)
    v_rep = v.repeat_interleave(4, dim=1)
    ref = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False)

    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decode_matches_sdpa_sliding_window():
    import torch.nn.functional as F
    import hydra

    torch.manual_seed(1)
    q = torch.randn(1, 32, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8, 257, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 8, 257, 128, device="cuda", dtype=torch.bfloat16)

    window = 96
    out = hydra.hydra(q, k, v, sliding_window=window)
    k_rep = k[:, :, -window:, :].repeat_interleave(4, dim=1)
    v_rep = v[:, :, -window:, :].repeat_interleave(4, dim=1)
    ref = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False)

    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
