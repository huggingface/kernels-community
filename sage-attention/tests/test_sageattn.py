import kernels
import pytest
import torch
import torch.nn.functional as F

sage_attention = kernels.get_kernel("kernels-community/sage-attention", version=3)

cuda_available = torch.cuda.is_available()

# SageAttention quantizes QK to INT8 and PV to FP8/FP16, so the output only
# matches a full-precision reference approximately. These are the thresholds
# upstream reports in `bench/` for a lossless configuration.
MIN_COS_SIM = 0.99
MAX_RELATIVE_L1 = 0.05


def reference(q, k, v, tensor_layout, is_causal):
    if tensor_layout == "NHD":
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    out = F.scaled_dot_product_attention(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), is_causal=is_causal
    )
    return out.transpose(1, 2) if tensor_layout == "NHD" else out


def assert_close_enough(out, ref):
    out, ref = out.to(torch.float32).flatten(), ref.to(torch.float32).flatten()
    cos_sim = torch.dot(out, ref) / (out.norm() * ref.norm())
    relative_l1 = (out - ref).abs().sum() / ref.abs().sum()
    assert torch.isfinite(out).all()
    assert cos_sim > MIN_COS_SIM, f"cosine similarity {cos_sim:.5f}"
    assert relative_l1 < MAX_RELATIVE_L1, f"relative L1 {relative_l1:.5f}"


def qkv(tensor_layout, dtype, head_dim, seq_len=256, batch=1, heads=4):
    shape = (
        (batch, heads, seq_len, head_dim)
        if tensor_layout == "HND"
        else (batch, seq_len, heads, head_dim)
    )
    return [
        torch.randn(shape, dtype=dtype, device="cuda") for _ in range(3)
    ]


@pytest.mark.kernels_ci
@pytest.mark.skipif(not cuda_available, reason="CUDA is required")
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
@pytest.mark.parametrize("is_causal", [False, True])
def test_sageattn(tensor_layout, is_causal):
    torch.manual_seed(0)
    q, k, v = qkv(tensor_layout, torch.float16, head_dim=128)

    out = sage_attention.sageattn(
        q, k, v, tensor_layout=tensor_layout, is_causal=is_causal
    )

    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert_close_enough(out, reference(q, k, v, tensor_layout, is_causal))


@pytest.mark.kernels_ci
@pytest.mark.skipif(not cuda_available, reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_sageattn_dtypes_and_head_dims(dtype, head_dim):
    torch.manual_seed(0)
    q, k, v = qkv("HND", dtype, head_dim=head_dim)

    out = sage_attention.sageattn(q, k, v, tensor_layout="HND")

    assert out.shape == q.shape
    assert out.dtype == dtype
    assert_close_enough(out, reference(q, k, v, "HND", is_causal=False))


@pytest.mark.kernels_ci
@pytest.mark.skipif(not cuda_available, reason="CUDA is required")
def test_sageattn_return_lse():
    torch.manual_seed(0)
    q, k, v = qkv("HND", torch.float16, head_dim=128)

    out, lse = sage_attention.sageattn(q, k, v, tensor_layout="HND", return_lse=True)

    assert out.shape == q.shape
    assert lse.shape == q.shape[:3]
    assert lse.dtype == torch.float32
    assert torch.isfinite(lse).all()


@pytest.mark.kernels_ci
@pytest.mark.skipif(not cuda_available, reason="CUDA is required")
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
@pytest.mark.parametrize("return_lse", [False, True])
def test_sageattn_gqa(tensor_layout, return_lse):
    """num_qo_heads is a multiple of num_kv_heads: k/v are broadcast over groups.

    Regression test for the head counts being read from the sequence dimension:
    under "HND" that made `q_per_kv_heads` collapse to 1, so the `km` broadcast
    was skipped and the `lse_correction` matmul failed outright.
    """
    torch.manual_seed(0)
    shape = lambda h: (
        (1, h, 256, 128) if tensor_layout == "HND" else (1, 256, h, 128)
    )
    q = torch.randn(shape(8), dtype=torch.float16, device="cuda")
    k = torch.randn(shape(2), dtype=torch.float16, device="cuda")
    v = torch.randn(shape(2), dtype=torch.float16, device="cuda")

    result = sage_attention.sageattn(
        q, k, v, tensor_layout=tensor_layout, return_lse=return_lse
    )
    out, lse = result if return_lse else (result, None)

    assert out.shape == q.shape
    if return_lse:
        assert lse.shape == (1, 8, 256)
        assert lse.dtype == torch.float32
        assert torch.isfinite(lse).all()

    q_ref, k_ref, v_ref = (
        (x.transpose(1, 2) if tensor_layout == "NHD" else x) for x in (q, k, v)
    )
    ref = F.scaled_dot_product_attention(
        q_ref.to(torch.float32),
        k_ref.to(torch.float32),
        v_ref.to(torch.float32),
        enable_gqa=True,
    )
    assert_close_enough(out, ref.transpose(1, 2) if tensor_layout == "NHD" else ref)
