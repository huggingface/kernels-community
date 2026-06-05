"""Numerical parity tests for the AITER Triton RoPE kernel.

Verifies that `apply_rotary_transformers` matches a pure-PyTorch NEOX-style
RoPE reference within fp16 tolerance.
"""

import pytest
import torch


def _apply_rotary_emb_ref(x, cos, sin):
    """Pure-PyTorch NEOX RoPE matching transformers' `_apply_rotary_emb`."""
    first_half, second_half = torch.chunk(x, 2, dim=-1)
    first_ = first_half * cos - second_half * sin
    second_ = second_half * cos + first_half * sin
    return torch.cat((first_, second_), dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm device")
def test_apply_rotary_transformers_matches_reference():
    from aiter_rope import apply_rotary_transformers

    torch.manual_seed(0)
    B, H, S, D = 2, 8, 32, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, 2, S, D, device="cuda", dtype=torch.float16)  # GQA
    cos = torch.randn(B, S, D // 2, device="cuda", dtype=torch.float16)
    sin = torch.randn(B, S, D // 2, device="cuda", dtype=torch.float16)
    # Same positions across the batch — what apply_rotary_pos_emb assumes.
    cos[1] = cos[0]
    sin[1] = sin[0]

    # Reference: pure-PyTorch path (transformers' `apply_rotary_pos_emb`).
    cos_ref = cos.unsqueeze(1)
    sin_ref = sin.unsqueeze(1)
    q_ref = _apply_rotary_emb_ref(q, cos_ref, sin_ref)
    k_ref = _apply_rotary_emb_ref(k, cos_ref, sin_ref)

    # AITER Triton path.
    q_out, k_out = apply_rotary_transformers(q, k, cos, sin)

    torch.testing.assert_close(q_out, q_ref, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(k_out, k_ref, atol=5e-3, rtol=5e-3)
