"""Numerical parity tests for the AITER Triton RoPE kernel.

Verifies that `apply_rotary_transformers` matches a pure-PyTorch NEOX-style
RoPE reference within fp16 tolerance.
"""

import inspect

import kernels
import pytest
import torch


aiter_rope = kernels.get_kernel("amd/aiter-rope", version=3)


def _apply_rotary_emb_ref(x, cos, sin):
    """Pure-PyTorch NEOX RoPE matching transformers' `_apply_rotary_emb`."""
    first_half, second_half = torch.chunk(x, 2, dim=-1)
    first_ = first_half * cos - second_half * sin
    second_ = second_half * cos + first_half * sin
    return torch.cat((first_, second_), dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm device")
def test_apply_rotary_transformers_matches_reference():
    apply_rotary_transformers = aiter_rope.apply_rotary_transformers

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


def _apply_rotary_pos_emb_ref(x, cos, sin):
    """Pure-PyTorch RoPE matching transformers' `apply_rotary_pos_emb` (full head_dim cos / sin)."""
    first_half, second_half = torch.chunk(x, 2, dim=-1)
    return x * cos + torch.cat((-second_half, first_half), dim=-1) * sin


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm device")
def test_apply_rotary_transformers_per_row_positions():
    # Left padding gives every batch row its own positions, so cos / sin differ per row.
    torch.manual_seed(0)
    B, H, S, D = 3, 8, 32, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, 2, S, D, device="cuda", dtype=torch.float16)
    # Full head_dim cos / sin, built like transformers' rotary embeddings.
    inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, device="cuda").float() / D))
    positions = torch.stack(
        [torch.arange(S, device="cuda") + offset for offset in (0, 5, 11)]
    )
    freqs = positions[..., None].float() * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos().half(), emb.sin().half()

    # Reference: transformers' `apply_rotary_pos_emb` on full head_dim cos / sin.
    q_ref, k_ref = (
        _apply_rotary_pos_emb_ref(x, cos.unsqueeze(1), sin.unsqueeze(1)) for x in (q, k)
    )

    for fn in (
        aiter_rope.apply_rotary_transformers,
        aiter_rope.layers.apply_rotary_transformers(),
    ):
        q_out, k_out = fn(q, k, cos, sin)
        torch.testing.assert_close(q_out, q_ref, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(k_out, k_ref, atol=5e-3, rtol=5e-3)


@pytest.mark.kernels_ci
def test_layer_signature_matches_function():
    # `kernels` rejects a layer whose forward signature differs from the function it replaces.
    layer_params = list(
        inspect.signature(
            aiter_rope.layers.apply_rotary_transformers.forward
        ).parameters
    )[1:]
    assert layer_params == list(
        inspect.signature(aiter_rope.apply_rotary_transformers).parameters
    )
