"""Tests for the Triton-backed entry points.

`sageattn_qk_int8_pv_fp16_triton` is what `sageattn` dispatches to on sm86, and
`sageattn_varlen` has no CUDA equivalent — both are backed by the vendored
`sage_attention._triton` kernels rather than the compiled ops.
"""

import kernels
import pytest
import torch
import torch.nn.functional as F

sage_attention = kernels.get_kernel("kernels-community/sage-attention", version=3)

cuda_available = torch.cuda.is_available()

MIN_COS_SIM = 0.99
MAX_RELATIVE_L1 = 0.05


def assert_close_enough(out, ref):
    out, ref = out.to(torch.float32).flatten(), ref.to(torch.float32).flatten()
    cos_sim = torch.dot(out, ref) / (out.norm() * ref.norm())
    relative_l1 = (out - ref).abs().sum() / ref.abs().sum()
    assert torch.isfinite(out).all()
    assert cos_sim > MIN_COS_SIM, f"cosine similarity {cos_sim:.5f}"
    assert relative_l1 < MAX_RELATIVE_L1, f"relative L1 {relative_l1:.5f}"


@pytest.mark.kernels_ci
@pytest.mark.skipif(not cuda_available, reason="CUDA is required")
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
@pytest.mark.parametrize("is_causal", [False, True])
def test_sageattn_qk_int8_pv_fp16_triton(tensor_layout, is_causal):
    torch.manual_seed(0)
    shape = (
        (1, 4, 256, 128) if tensor_layout == "HND" else (1, 256, 4, 128)
    )
    q, k, v = (
        torch.randn(shape, dtype=torch.float16, device="cuda") for _ in range(3)
    )

    out = sage_attention.sageattn_qk_int8_pv_fp16_triton(
        q, k, v, tensor_layout=tensor_layout, is_causal=is_causal
    )

    assert out.shape == q.shape
    assert out.dtype == q.dtype

    q_ref, k_ref, v_ref = (
        (x.transpose(1, 2) if tensor_layout == "NHD" else x) for x in (q, k, v)
    )
    ref = F.scaled_dot_product_attention(
        q_ref.to(torch.float32),
        k_ref.to(torch.float32),
        v_ref.to(torch.float32),
        is_causal=is_causal,
    )
    assert_close_enough(out, ref.transpose(1, 2) if tensor_layout == "NHD" else ref)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not cuda_available, reason="CUDA is required")
def test_sageattn_qk_int8_pv_fp16_triton_cuda_quant_backend():
    """The Triton attention kernel can be fed by the CUDA quantizer."""
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(1, 4, 256, 128, dtype=torch.float16, device="cuda")
        for _ in range(3)
    )

    out = sage_attention.sageattn_qk_int8_pv_fp16_triton(
        q, k, v, tensor_layout="HND", quantization_backend="cuda"
    )

    ref = F.scaled_dot_product_attention(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
    )
    assert_close_enough(out, ref)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not cuda_available, reason="CUDA is required")
@pytest.mark.parametrize("is_causal", [False, True])
def test_sageattn_varlen(is_causal):
    """Packed variable-length layout: [total_tokens, num_heads, head_dim]."""
    torch.manual_seed(0)
    seqlens = [128, 256]
    heads, head_dim = 4, 128
    total = sum(seqlens)

    q, k, v = (
        torch.randn(total, heads, head_dim, dtype=torch.float16, device="cuda")
        for _ in range(3)
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seqlens).cumsum(0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )

    out = sage_attention.sageattn_varlen(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(seqlens),
        max_seqlen_k=max(seqlens),
        is_causal=is_causal,
    )

    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert torch.isfinite(out.to(torch.float32)).all()

    # Compare each packed sequence against a dense reference.
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        qs, ks, vs = (x[start:end].transpose(0, 1).unsqueeze(0) for x in (q, k, v))
        ref = F.scaled_dot_product_attention(
            qs.to(torch.float32),
            ks.to(torch.float32),
            vs.to(torch.float32),
            is_causal=is_causal,
        )
        assert_close_enough(out[start:end], ref.squeeze(0).transpose(0, 1))
