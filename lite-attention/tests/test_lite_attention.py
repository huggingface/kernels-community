import math
import os
import warnings

import pytest
import torch

from lite_attention import LiteAttention


def _requires_hopper():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip("LiteAttention Hopper path requires SM90+")


def _make_qkv(batch, seqlen_q, seqlen_k, heads, head_dim, dtype=torch.bfloat16):
    q = torch.randn(batch, seqlen_q, heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, heads, head_dim, device="cuda", dtype=dtype)
    return q, k, v


def _attention_ref(q, k, v, scale=None):
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    q_ref = q.transpose(1, 2).float()
    k_ref = k.transpose(1, 2).float()
    v_ref = v.transpose(1, 2).float()
    scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_ref).transpose(1, 2)
    lse = torch.logsumexp(scores, dim=-1)
    return out, lse


@pytest.mark.parametrize("head_dim", [64, 96, 128, 192, 256])
@pytest.mark.parametrize("shape", [(1, 64, 64), (2, 97, 113)])
def test_bf16_no_skip_matches_pytorch(head_dim, shape):
    _requires_hopper()
    torch.manual_seed(0)
    batch, seqlen_q, seqlen_k = shape
    q, k, v = _make_qkv(batch, seqlen_q, seqlen_k, 4, head_dim)
    ref, lse_ref = _attention_ref(q, k, v)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        attn = LiteAttention(enable_skipping=False)
    out, lse = attn(q, k, v, return_softmax_lse=True)

    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse.float(), lse_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_int8_skip_path_matches_pytorch_loosely(head_dim):
    _requires_hopper()
    torch.manual_seed(1)
    q, k, v = _make_qkv(1, 64, 64, 4, head_dim)
    ref, _ = _attention_ref(q, k, v)

    attn = LiteAttention(use_int8=True, threshold=float("-inf"))
    out = attn(q, k, v)

    assert out.shape == q.shape
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref, atol=2e-1, rtol=2e-1)


def test_skip_list_state_and_phase():
    _requires_hopper()
    torch.manual_seed(2)
    q, k, v = _make_qkv(2, 128, 128, 4, 128)

    attn = LiteAttention(threshold=float("-inf"))
    out = attn(q, k, v)

    assert out.shape == q.shape
    assert attn._skip_list is not None
    assert attn._skip_list.dtype == torch.int16
    assert attn._skip_list.shape[:3] == (2, attn.max_batch_size, q.shape[2])
    assert attn._phase == 1

    out = attn(q, k, v, must_do_list=[0, 64])
    assert out.shape == q.shape
    assert attn._phase == 0
    assert attn.read_list is not None
    assert attn.write_list is not None


def test_no_skip_matches_flash_attn3_if_available():
    _requires_hopper()
    try:
        from flash_attn3 import flash_attn_func
    except ImportError:
        pytest.skip("flash_attn3 is not installed in this test environment")

    torch.manual_seed(3)
    q, k, v = _make_qkv(1, 128, 128, 4, 128)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        attn = LiteAttention(enable_skipping=False)
    out = attn(q, k, v)
    ref, *_ = flash_attn_func(q, k, v, return_softmax_lse=True)

    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    os.getenv("LITE_ATTENTION_RUN_STRESS", "FALSE") != "TRUE",
    reason="set LITE_ATTENTION_RUN_STRESS=TRUE to run long-sequence stress tests",
)
@pytest.mark.parametrize("use_int8", [False, True])
@pytest.mark.parametrize("seqlen", [4096, 8192, 16384])
def test_long_sequence_stress(seqlen, use_int8):
    _requires_hopper()
    torch.manual_seed(4)
    q, k, v = _make_qkv(1, seqlen, seqlen, 32, 128)

    attn = LiteAttention(use_int8=use_int8, threshold=-10.0)
    out = attn(q, k, v)
    torch.cuda.synchronize()

    assert out.shape == q.shape
    assert torch.isfinite(out).all()
    assert attn.read_list is not None
