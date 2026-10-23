import math

import kernels
import pytest
import torch

flash_attn2_rs = kernels.get_kernel("kernels-community/flash-attn2-rs", version=1)


def fwd_explicit(
    q,
    k,
    v,
    *,
    alibi_slopes=None,
    softmax_scale=None,
    causal=False,
    window_left=-1,
    window_right=-1,
    softcap=0.0,
):
    """Call the positional tvm-ffi forward op with keyword arguments."""
    return flash_attn2_rs.fwd_explicit(
        q,
        k,
        v,
        alibi_slopes,
        softmax_scale,
        causal,
        window_left,
        window_right,
        softcap,
    )


def reference_attention(
    q,
    k,
    v,
    *,
    alibi_slopes=None,
    softmax_scale=None,
    causal=False,
    window_left=-1,
    window_right=-1,
    softcap=0.0,
):
    """Compute attention in float32 with flash-attn2 masking semantics."""
    _batch, seqlen_q, num_heads, head_dim = q.shape
    seqlen_k, num_kv_heads = k.shape[1], k.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    groups = num_heads // num_kv_heads
    q_ref = q.float().transpose(1, 2)
    k_ref = k.float().transpose(1, 2).repeat_interleave(groups, 1)
    v_ref = v.float().transpose(1, 2).repeat_interleave(groups, 1)

    scores = torch.einsum("bhqd,bhkd->bhqk", q_ref, k_ref)
    if softcap > 0:
        scores = softcap * torch.tanh(scores * (softmax_scale / softcap))
    else:
        scores = scores * softmax_scale

    # Queries are aligned to the end of the keys when the lengths differ.
    offset = seqlen_k - seqlen_q
    rows = torch.arange(seqlen_q, device=q.device).view(1, 1, seqlen_q, 1)
    cols = torch.arange(seqlen_k, device=q.device).view(1, 1, 1, seqlen_k)
    if alibi_slopes is not None:
        slopes = alibi_slopes.float().view(-1, num_heads, 1, 1)
        if causal and window_left < 0:
            # Pure causal ALiBi only depends on the key position.
            scores = scores + slopes * cols.float()
        else:
            scores = scores - slopes * (rows + offset - cols).abs().float()

    mask = torch.zeros(1, 1, seqlen_q, seqlen_k, dtype=torch.bool, device=q.device)
    if causal or window_right >= 0:
        mask |= cols > rows + offset + (0 if causal else window_right)
    if window_left >= 0:
        mask |= cols < rows + offset - window_left
    scores = scores.masked_fill(mask, float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", scores.softmax(dim=-1), v_ref)
    return out.transpose(1, 2), lse


def case(name, batch, num_heads, num_kv_heads, seqlen_q, seqlen_k, **options):
    shape = (batch, num_heads, num_kv_heads, seqlen_q, seqlen_k)
    return pytest.param(shape, options, id=name)


PARITY_CASES = [
    case("full-mha", 2, 4, 4, 256, 256),
    case("causal-mha", 2, 4, 4, 256, 256, causal=True),
    case("causal-gqa-h8-hk2", 2, 8, 2, 256, 256, causal=True),
    case("full-mqa-h4-hk1", 1, 4, 1, 192, 192),
    case("causal-sq320-sk256", 1, 4, 4, 320, 256, causal=True),
    case("full-sq128-sk256", 1, 4, 4, 128, 256),
    case("custom-scale", 2, 4, 4, 256, 256, causal=True, softmax_scale=0.05),
    case("local-causal-wl64", 2, 4, 4, 256, 256, causal=True, window_left=64),
    case("window-wl64-wr64", 1, 4, 4, 256, 256, window_left=64, window_right=64),
    case("softcap30-causal", 2, 4, 4, 256, 256, causal=True, softcap=30.0),
    case("softcap50-full", 1, 4, 4, 192, 192, softcap=50.0),
    case("alibi-causal", 2, 4, 4, 256, 256, causal=True, alibi=True),
    case("alibi-full", 1, 4, 4, 192, 192, alibi=True),
    case(
        "alibi-local-wl64", 1, 4, 4, 256, 256, causal=True, window_left=64, alibi=True
    ),
    case(
        "softcap-alibi-gqa-causal",
        2,
        8,
        2,
        256,
        256,
        causal=True,
        softcap=20.0,
        alibi=True,
    ),
    case("odd-causal-s333", 2, 4, 4, 333, 333, causal=True),
    case(
        "odd-window-softcap-s300",
        1,
        4,
        4,
        300,
        300,
        causal=True,
        window_left=100,
        softcap=30.0,
    ),
    case("odd-gqa-causal-s130", 2, 8, 2, 130, 130, causal=True),
]


@pytest.mark.kernels_ci
@pytest.mark.parametrize(("shape", "options"), PARITY_CASES)
def test_forward_parity(shape, options):
    """Forward output and LSE match FP32 attention across masks and GQA."""
    batch, num_heads, num_kv_heads, seqlen_q, seqlen_k = shape
    options = dict(options)
    alibi = options.pop("alibi", False)

    torch.manual_seed(0)
    q = torch.randn(batch, seqlen_q, num_heads, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, seqlen_k, num_kv_heads, 128, device="cuda", dtype=q.dtype)
    v = torch.randn(batch, seqlen_k, num_kv_heads, 128, device="cuda", dtype=q.dtype)
    if alibi:
        heads = torch.arange(num_heads, device="cuda").float()
        options["alibi_slopes"] = 2.0 ** (-8.0 * (heads + 1) / num_heads)

    out, lse = fwd_explicit(q, k, v, **options)
    expected_out, expected_lse = reference_attention(q, k, v, **options)

    out, expected_out = out.float(), expected_out.float()
    assert torch.isfinite(out).all()
    # Fully masked rows are NaN in the reference but zero from the kernel.
    finite = torch.isfinite(expected_out)
    torch.testing.assert_close(
        out[finite], expected_out[finite], atol=2e-3, rtol=2.5e-2
    )
    assert (out[finite] - expected_out[finite]).abs().mean() < 2e-3
    masked = torch.isneginf(expected_lse)
    assert (out.transpose(1, 2)[masked] == 0).all()
    # FlashAttention defines +inf LSE for empty rows; the reference gives -inf.
    expected_lse = expected_lse.masked_fill(masked, float("inf"))
    torch.testing.assert_close(lse.float(), expected_lse, atol=3e-2, rtol=0)


# The head size must be at most 256 and a multiple of 8, and inputs FP16/BF16.
@pytest.mark.kernels_ci
@pytest.mark.parametrize(
    ("head_size", "dtype"),
    [(264, torch.float16), (9, torch.float16), (128, torch.float32)],
    ids=["head-over-256", "head-not-multiple-of-8", "float32"],
)
def test_rejects_unsupported_inputs(head_size, dtype):
    q = torch.randn(1, 128, 4, head_size, device="cuda", dtype=dtype)
    with pytest.raises((NotImplementedError, ValueError, TypeError)):
        fwd_explicit(q, q, q)
