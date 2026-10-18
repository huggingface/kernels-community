import kernels
import pytest
import torch
import torch.nn.functional as F

sage = kernels.get_kernel("kernels-community/aiter-sage-attention", version=1)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1201",
    reason="gfx1201 only",
)


def inputs(batch, rows, heads, scale=1.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        (torch.randn(batch, rows, heads, 128, device="cuda", generator=g) * scale)
        .bfloat16()
        for _ in range(3)
    )


def reference(q, k, v, scale=None):
    out = F.scaled_dot_product_attention(
        *(t.transpose(1, 2).float() for t in (q, k, v)), scale=scale
    )
    return out.transpose(1, 2)


def assert_close_to_reference(out, ref):
    # Lossy by design: INT8 Q/K per 32-row block, FP8 V per channel.
    out, ref = out.float(), ref.float()
    assert torch.isfinite(out).all()
    cosine = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    row_cosine = F.cosine_similarity(out, ref, dim=-1).median().item()
    assert cosine >= 0.998 and row_cosine >= 0.99, (cosine, row_cosine)


@pytest.mark.kernels_ci
@pytest.mark.parametrize(
    "batch,rows,heads,scale",
    [(1, 128, 28, 1.0), (2, 1024, 3, 1.0), (1, 4096, 8, 1.5), (2, 3000, 5, 2.0)],
)
def test_matches_sdpa(batch, rows, heads, scale):
    q, k, v = inputs(batch, rows, heads, scale)
    out = sage.gfx1201_sage_attention(q, k, v)
    assert out.shape == q.shape and out.dtype == q.dtype and out.is_contiguous()
    assert_close_to_reference(out, reference(q, k, v))


def test_softmax_scale():
    q, k, v = inputs(1, 777, 4)
    out = sage.gfx1201_sage_attention(q, k, v, softmax_scale=0.05)
    assert_close_to_reference(out, reference(q, k, v, scale=0.05))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("rows", [1, 15, 16, 17, 31, 32, 33, 97, 513, 1025])
def test_sequence_tails(rows):
    q, k, v = inputs(2, rows, 3, seed=rows)
    assert_close_to_reference(sage.gfx1201_sage_attention(q, k, v), reference(q, k, v))


def rope_tables(rows, rope_dim):
    inv = 1.0 / (10000 ** (torch.arange(0, rope_dim, 2, device="cuda").float() / rope_dim))
    angle = torch.arange(rows, device="cuda").float()[:, None] * inv[None, :]
    angle = torch.cat([angle, angle], dim=-1)
    return angle.cos().contiguous(), angle.sin().contiguous()


def norm_rope(x, weight, cosine, sine, eps):
    x = F.rms_norm(x, (x.shape[-1],), weight, eps)
    rope_dim = cosine.shape[-1]
    head = x[..., :rope_dim].float()
    rotated = torch.cat([-head[..., rope_dim // 2 :], head[..., : rope_dim // 2]], dim=-1)
    head = head * cosine[None, :, None, :] + rotated * sine[None, :, None, :]
    return torch.cat([head.to(x.dtype), x[..., rope_dim:]], dim=-1)


@pytest.mark.parametrize("rows,heads,rope_dim", [(4097, 28, 96), (513, 7, 128), (300, 3, 64)])
def test_fused_norm_rope_is_bitwise_equal_to_the_unfused_chain(rows, heads, rope_dim):
    q, k, v = inputs(1, rows, heads, 3.0, seed=rope_dim)
    qw, kw = (torch.rand(128, device="cuda").add(0.5).bfloat16() for _ in range(2))
    cosine, sine = rope_tables(rows, rope_dim)
    fused = sage.gfx1201_sage_attention(q, k, v, None, qw, kw, cosine, sine, 1e-5)
    q2, k2 = norm_rope(q, qw, cosine, sine, 1e-5), norm_rope(k, kw, cosine, sine, 1e-5)
    assert torch.equal(fused, sage.gfx1201_sage_attention(q2, k2, v))


def test_non_default_stream():
    q, k, v = inputs(2, 97, 3)
    expected = sage.gfx1201_sage_attention(q, k, v)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = sage.gfx1201_sage_attention(q, k, v)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(actual, expected)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_sageattn_entry_point(layout):
    q, k, v = inputs(1, 513, 4)
    expected = sage.gfx1201_sage_attention(q, k, v, softmax_scale=0.07)
    if layout == "HND":
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    out = sage.sageattn(q, k, v, tensor_layout=layout, sm_scale=0.07)
    if layout == "HND":
        out = out.transpose(1, 2)
    assert torch.equal(out, expected)


def test_rejects_unsupported_inputs():
    q, k, v = inputs(1, 64, 2)
    with pytest.raises(ValueError):
        sage.sageattn(q, k, v, tensor_layout="NHD", is_causal=True)
    with pytest.raises(ValueError):
        sage.sageattn(q, k, v, tensor_layout="NHD", return_lse=True)
    with pytest.raises(ValueError):
        sage.gfx1201_sage_attention(q.half(), k.half(), v.half())
    with pytest.raises(ValueError):
        sage.gfx1201_sage_attention(q[..., :64].contiguous(), k[..., :64].contiguous(), v[..., :64].contiguous())
    with pytest.raises(ValueError):
        sage.gfx1201_sage_attention(q, k[:, :32], v[:, :32])
