import kernels
import pytest
import torch
import torch.nn.functional as F

sage_blackwell = kernels.get_kernel("kernels-community/sage-blackwell", version=1)

sageattn3_blackwell = sage_blackwell.sageattn3_blackwell

# SageAttention3 only runs on consumer Blackwell (sm_120/sm_121): the FP4
# attention kernel is built from the SM120 blockscaled NVFP4 MMA atom and
# `mha_fwd` rejects every other device outright.
_CAPABILITY = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
BLACKWELL = _CAPABILITY in {(12, 0), (12, 1)}
requires_blackwell = pytest.mark.skipif(
    not BLACKWELL, reason="requires a consumer Blackwell GPU (sm_120/sm_121)"
)

# Q, K and V are all quantized to NVFP4 (e2m1 with a per-16-element e4m3
# scale), so the output only matches a full-precision reference
# approximately. FP4 is far more aggressive than the INT8/FP8 quantization in
# `sage-attention`, hence the looser thresholds.
MIN_COS_SIM = 0.97
MAX_RELATIVE_L1 = 0.12


def reference(q, k, v, is_causal=False):
    return F.scaled_dot_product_attention(
        q.to(torch.float32),
        k.to(torch.float32),
        v.to(torch.float32),
        is_causal=is_causal,
    )


def assert_close_enough(out, ref):
    out, ref = out.to(torch.float32).flatten(), ref.to(torch.float32).flatten()
    cos_sim = torch.dot(out, ref) / (out.norm() * ref.norm())
    relative_l1 = (out - ref).abs().sum() / ref.abs().sum()
    assert torch.isfinite(out).all()
    assert cos_sim > MIN_COS_SIM, f"cosine similarity {cos_sim:.5f}"
    assert relative_l1 < MAX_RELATIVE_L1, f"relative L1 {relative_l1:.5f}"


def qkv(dtype, head_dim, seq_len=256, batch=1, heads=4):
    """HND-layout inputs: `sageattn3_blackwell` only accepts [B, H, L, D]."""
    shape = (batch, heads, seq_len, head_dim)
    return [torch.randn(shape, dtype=dtype, device="cuda") for _ in range(3)]


@pytest.mark.kernels_ci
@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_sageattn3_dtypes_and_head_dims(dtype, head_dim):
    torch.manual_seed(0)
    q, k, v = qkv(dtype, head_dim=head_dim)
    ref = reference(q, k, v)

    # `sageattn3_blackwell` subtracts the key mean in place, so hand it copies.
    out = sageattn3_blackwell(q.clone(), k.clone(), v.clone())

    assert out.shape == q.shape
    assert out.dtype == dtype
    assert_close_enough(out, ref)


@pytest.mark.kernels_ci
@requires_blackwell
@pytest.mark.parametrize("is_causal", [False, True])
def test_sageattn3_causal(is_causal):
    torch.manual_seed(0)
    q, k, v = qkv(torch.float16, head_dim=128)
    ref = reference(q, k, v, is_causal=is_causal)

    out = sageattn3_blackwell(q.clone(), k.clone(), v.clone(), is_causal=is_causal)

    assert out.shape == q.shape
    assert_close_enough(out, ref)


@pytest.mark.kernels_ci
@requires_blackwell
@pytest.mark.parametrize("per_block_mean", [False, True])
def test_sageattn3_per_block_mean(per_block_mean):
    """`per_block_mean` picks between a per-128-block and a whole-sequence mean.

    The block mean goes through the Triton `group_mean_kernel`, the global one
    through plain Torch, and both feed the `delta_s` correction term.
    """
    torch.manual_seed(0)
    q, k, v = qkv(torch.float16, head_dim=128)
    ref = reference(q, k, v)

    out = sageattn3_blackwell(
        q.clone(), k.clone(), v.clone(), per_block_mean=per_block_mean
    )

    assert out.shape == q.shape
    assert_close_enough(out, ref)


@requires_blackwell
# 320 and 1000 are not multiples of 128, so they exercise the zero-padding path
# in `preprocess_qkv` and the `unpadded_k` masking in the kernel.
@pytest.mark.parametrize("seq_len", [128, 320, 1000, 1024])
def test_sageattn3_seq_lens(seq_len):
    torch.manual_seed(0)
    q, k, v = qkv(torch.float16, head_dim=128, seq_len=seq_len)
    ref = reference(q, k, v)

    out = sageattn3_blackwell(q.clone(), k.clone(), v.clone())

    assert out.shape == q.shape
    assert_close_enough(out, ref)


@requires_blackwell
@pytest.mark.parametrize("batch,heads", [(1, 1), (2, 8)])
def test_sageattn3_batch_and_heads(batch, heads):
    torch.manual_seed(0)
    q, k, v = qkv(torch.float16, head_dim=128, batch=batch, heads=heads)
    ref = reference(q, k, v)

    out = sageattn3_blackwell(q.clone(), k.clone(), v.clone())

    assert out.shape == q.shape
    assert_close_enough(out, ref)


@requires_blackwell
def test_sageattn3_unsupported_head_dim_falls_back_to_sdpa():
    """Head dims >= 256 are not dispatched; upstream falls back to SDPA."""
    torch.manual_seed(0)
    q, k, v = qkv(torch.float16, head_dim=256)

    out = sageattn3_blackwell(q.clone(), k.clone(), v.clone())

    assert out.shape == q.shape
    # Upstream returns `sdpa(q, k, v)` verbatim, so this must match exactly.
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v))
