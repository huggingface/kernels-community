"""The flash-attention kernel against `F.scaled_dot_product_attention`.

The reference is torch's own attention, so a pass means the kernel is a drop-in for it. Two things
these cases exist to catch, because both are silent: grouped-query attention is native here (k and v
keep `n_heads_kv` heads and the kernel maps them itself), and the result comes out with tokens before
heads rather than heads before tokens.
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F


DEV = "mps"
LIB = os.environ.get("GGML_ATTN_LOCAL_LIB")

if LIB:
    torch.ops.load_library(LIB)
    ops = getattr(torch.ops, os.path.basename(LIB).removesuffix(".so"))
else:
    try:
        import kernels

        # resolves to the local build when LOCAL_KERNELS points at it, as in kernels-community CI
        ops = kernels.get_kernel("ggml-org/ggml-attn", version=1)
    except Exception as error:  # pragma: no cover
        pytest.skip(f"no kernel to test ({error})", allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


def reference(q, k, v, mask, scale):
    """torch's attention, with k and v expanded the way it needs them."""
    n_rep = q.shape[1] // k.shape[1]
    if n_rep > 1:
        b, h, s, d = k.shape
        k = k[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)
        b, h, s, d = v.shape
        v = v[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return out.transpose(1, 2)  # (n_seqs, n_q, n_heads, head_dim), as the kernel returns


def inputs(n_seqs=1, n_heads=16, n_heads_kv=4, n_q=1, n_kv=512, head_dim=256, mask=None, seed=0):
    gen = torch.Generator().manual_seed(seed)

    def rnd(*s):
        return torch.randn(*s, generator=gen).to(DEV)

    q = rnd(n_seqs, n_heads, n_q, head_dim)
    k = rnd(n_seqs, n_heads_kv, n_kv, head_dim)
    v = rnd(n_seqs, n_heads_kv, n_kv, head_dim)
    m = None
    if mask == "zeros":  # attend everywhere, but exercise the masked code path
        m = torch.zeros(n_seqs, 1, n_q, n_kv, device=DEV)
    elif mask == "causal":  # the last `n_q` positions of the cache are the queries' own
        m = torch.zeros(n_seqs, 1, n_q, n_kv, device=DEV)
        rows = torch.arange(n_q, device=DEV)[:, None]
        cols = torch.arange(n_kv, device=DEV)[None, :]
        m.masked_fill_(cols > (n_kv - n_q + rows), float("-inf"))
    return q, k, v, m


@pytest.mark.parametrize("n_kv", [32, 512, 141])  # aligned, aligned-large, and unaligned (pads)
@pytest.mark.parametrize("n_heads,n_heads_kv", [(16, 4), (4, 4), (16, 16)])
@pytest.mark.parametrize("mask", [None, "zeros", "causal"])
def test_matches_sdpa(n_kv, n_heads, n_heads_kv, mask):
    head_dim, n_q = 256, 1
    if not ops.supports_flash_attn(n_q, head_dim, head_dim):
        pytest.skip("no kernel for these shapes")
    q, k, v, m = inputs(n_heads=n_heads, n_heads_kv=n_heads_kv, n_q=n_q, n_kv=n_kv, mask=mask)
    scale = head_dim**-0.5
    got = ops.flash_attn(q, k, v, m, scale)
    ref = reference(q, k, v, m, scale)
    torch.mps.synchronize()

    assert got.shape == ref.shape, f"{got.shape} against {ref.shape}"
    denom = max(ref.abs().max().item(), 1e-6)
    # the kernel accumulates in f16 where torch's math path uses f32, so the bar is looser than the
    # delta rule's -- this is the same precision llama.cpp runs its attention at
    assert (got - ref).abs().max().item() / denom < 3e-3


@pytest.mark.parametrize("n_q", [2, 8, 19])
def test_multiple_queries(n_q):
    head_dim = 256
    if not ops.supports_flash_attn(n_q, head_dim, head_dim):
        pytest.skip("no kernel for these shapes")
    q, k, v, m = inputs(n_q=n_q, mask="causal")
    scale = head_dim**-0.5
    got = ops.flash_attn(q, k, v, m, scale)
    ref = reference(q, k, v, m, scale)
    torch.mps.synchronize()
    assert (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6) < 3e-3


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_head_dims(head_dim):
    if not ops.supports_flash_attn(1, head_dim, head_dim):
        pytest.skip(f"no kernel for head_dim {head_dim}")
    q, k, v, m = inputs(n_q=1, head_dim=head_dim, mask="zeros")
    scale = head_dim**-0.5
    got = ops.flash_attn(q, k, v, m, scale)
    ref = reference(q, k, v, m, scale)
    torch.mps.synchronize()
    assert (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6) < 3e-3


def test_shape_coverage():
    """Both of upstream's paths are ported, so what is declined is shapes it has no template for.

    20 is where upstream switches from the vector kernel to the tiled one; both sides of that boundary
    are covered here, which is the point of porting the second path.
    """
    assert ops.supports_flash_attn(19, 256, 256)   # vector
    assert ops.supports_flash_attn(20, 256, 256)   # tiled, upstream's switch point
    assert ops.supports_flash_attn(512, 256, 256)  # tiled, a real prefill
    assert not ops.supports_flash_attn(1, 100, 100)   # no dk100 template either side
    assert not ops.supports_flash_attn(64, 100, 100)
    assert not ops.supports_flash_attn(1, 256, 320)   # "assume K is larger or equal than V"


def test_compiles_without_a_graph_break():
    if LIB:
        pytest.skip("fakes live in the packaged python wrapper, not in the raw library")
    head_dim = 256
    if not ops.supports_flash_attn(1, head_dim, head_dim):
        pytest.skip("no kernel")
    q, k, v, m = inputs(mask="zeros")
    fn = torch.compile(lambda *a: ops.flash_attn(*a), fullgraph=True)
    out = fn(q, k, v, m, head_dim**-0.5)
    torch.mps.synchronize()
    assert out.shape == (q.shape[0], q.shape[2], q.shape[1], head_dim)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# --- `flash_attn_forward`, the `transformers` entry point -------------------------------------------
#
# These need the packaged python wrapper rather than the raw library, since the mask reconstruction and
# the sdpa fallback live there.

forward_only = pytest.mark.skipif(bool(LIB), reason="flash_attn_forward lives in the python wrapper")


def causal_mask(n_q, n_kv):
    """Lower-right aligned: the `n_q` queries are the last `n_q` of the `n_kv` positions."""
    m = torch.zeros(1, 1, n_q, n_kv, device=DEV)
    rows = torch.arange(n_q, device=DEV)[:, None]
    cols = torch.arange(n_kv, device=DEV)[None, :]
    return m.masked_fill_(cols > (n_kv - n_q + rows), float("-inf"))


@forward_only
@pytest.mark.parametrize("n_q,n_kv", [(2, 2), (8, 8), (19, 19), (4, 20), (6, 22), (8, 64),
                                     (20, 20), (32, 32), (64, 64), (37, 128), (20, 512)])
def test_forward_is_causal_without_a_mask(n_q, n_kv):
    """With no mask and `n_q > 1`, the entry point must still be causal.

    `transformers` is entitled to hand an attention function `mask=None` -- `sdpa` carries causality in its
    own `is_causal` flag, so the mask is dropped as an optimisation -- and that flag never reaches the
    function. ggml has no equivalent: a null mask means attend to everything. Without reconstruction this is
    silently bidirectional, and it is easy to miss, because the logits move while greedy text often does not.

    Both directions are asserted: close to causal, *and* far from bidirectional. Checking only the first
    would pass for a kernel that returned either, at short `n_kv` where the two nearly agree.
    """
    head_dim = 256
    if not ops.supports_flash_attn(n_q, head_dim, head_dim):
        pytest.skip("no kernel for these shapes")
    q, k, v, _ = inputs(n_q=n_q, n_kv=n_kv, head_dim=head_dim)
    scale = head_dim**-0.5

    got, _ = ops.flash_attn_forward(None, q, k, v, None, scaling=scale)
    want_causal = reference(q, k, v, causal_mask(n_q, n_kv), scale)
    want_full = reference(q, k, v, None, scale)
    torch.mps.synchronize()

    denom = max(want_causal.abs().max().item(), 1e-6)
    to_causal = (got - want_causal).abs().max().item() / denom
    to_full = (got - want_full).abs().max().item() / denom
    assert to_causal < 3e-3, f"not causal: {to_causal:.2e} from causal, {to_full:.2e} from full"
    assert to_full > to_causal, f"indistinguishable: {to_causal:.2e} vs {to_full:.2e}"


@forward_only
@pytest.mark.parametrize("n_kv", [1, 141, 512])
def test_forward_single_query_needs_no_mask(n_kv):
    """At `n_q == 1` there is no future to mask, so no mask is built and none is needed."""
    head_dim = 256
    q, k, v, _ = inputs(n_q=1, n_kv=n_kv, head_dim=head_dim)
    scale = head_dim**-0.5
    got, _ = ops.flash_attn_forward(None, q, k, v, None, scaling=scale)
    ref = reference(q, k, v, None, scale)
    torch.mps.synchronize()
    assert (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6) < 3e-3


@forward_only
def test_forward_honours_an_explicit_mask():
    """A supplied mask must win -- the reconstruction only fills in for a missing one."""
    n_q, n_kv, head_dim = 8, 64, 256
    q, k, v, _ = inputs(n_q=n_q, n_kv=n_kv, head_dim=head_dim)
    scale = head_dim**-0.5
    m = torch.zeros(1, 1, n_q, n_kv, device=DEV)
    m[..., n_kv // 2:] = float("-inf")  # a pattern the reconstruction would never produce
    got, _ = ops.flash_attn_forward(None, q, k, v, m, scaling=scale)
    ref = reference(q, k, v, m, scale)
    torch.mps.synchronize()
    assert (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6) < 3e-3


@forward_only
def test_forward_handles_wide_queries():
    """Prefill shapes go through the tiled kernel now, not torch, and must still be causal."""
    n_q, head_dim = 64, 256
    assert ops.supports_flash_attn(n_q, head_dim, head_dim)
    q, k, v, _ = inputs(n_q=n_q, n_kv=n_q, head_dim=head_dim)
    scale = head_dim**-0.5
    got, _ = ops.flash_attn_forward(None, q, k, v, None, scaling=scale)
    ref = reference(q, k, v, causal_mask(n_q, n_q), scale)
    torch.mps.synchronize()
    assert got.shape == ref.shape
    assert (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6) < 3e-3




@pytest.mark.parametrize("cache_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n_q", [1, 8, 32])  # the vector path, and the tiled one above 20 queries
@pytest.mark.parametrize("n_kv", [141, 512])  # unaligned (pads, with 2-byte rows) and aligned
@pytest.mark.parametrize("head_dim", [64, 128])
def test_half_cache_is_read_natively(cache_dtype, n_q, n_kv, head_dim):
    """An f16 or bf16 cache goes to ggml's kernels for that type, as llama.cpp binds it, not through f32.

    Same answer as attention over those half values: the cache is not rounded any further.
    """
    q, k, v, m = inputs(n_q=n_q, n_kv=n_kv, head_dim=head_dim, mask="causal" if n_q > 1 else None)
    k16, v16 = k.to(cache_dtype), v.to(cache_dtype)
    scale = head_dim**-0.5
    got = ops.flash_attn(q, k16, v16, m, scale)
    ref = reference(q, k16.float(), v16.float(), m, scale)
    torch.mps.synchronize()
    assert got.dtype == torch.float32 and got.shape == ref.shape
    torch.testing.assert_close(got, ref, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_forward_returns_the_models_dtype(dtype):
    q, k, v, _ = inputs(n_q=1, n_kv=64, head_dim=64)
    out, _ = ops.flash_attn_forward(None, q.to(dtype), k.to(dtype), v.to(dtype), scaling=64**-0.5)
    assert out.dtype == dtype


def eager(q, k, v, mask, scale, sinks=None, softcap=0.0):
    """Attention written out, the way transformers' eager paths add sinks (gpt-oss) and softcapping
    (Gemma 2): the cap goes on the scaled scores before the mask, the sink joins the softmax as one
    more column per head and is dropped before the values."""
    n_rep = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(n_rep, dim=1).float()
    v = v.repeat_interleave(n_rep, dim=1).float()
    scores = q.float() @ k.transpose(-1, -2) * scale
    if softcap:
        scores = softcap * torch.tanh(scores / softcap)
    if mask is not None:
        scores = scores + mask
    if sinks is not None:
        sink = sinks.float().reshape(1, -1, 1, 1).expand(*scores.shape[:-1], 1)
        probs = torch.cat([scores, sink], dim=-1).softmax(-1)[..., :-1]
    else:
        probs = scores.softmax(-1)
    return (probs @ v).transpose(1, 2)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("n_q", [1, 32])  # the vector path and the tiled one
@pytest.mark.parametrize("sinks,softcap", [(True, 0.0), (False, 30.0), (True, 30.0)])
def test_sinks_and_softcap(cache_dtype, n_q, sinks, softcap):
    q, k, v, m = inputs(n_q=n_q, n_kv=141, head_dim=64, mask="causal" if n_q > 1 else None)
    # big enough logits that the sinks and the cap both move the answer
    q = q * 4
    s = torch.randn(q.shape[1], generator=torch.Generator().manual_seed(1)).to(DEV) + 10 if sinks else None
    k, v = k.to(cache_dtype), v.to(cache_dtype)
    scale = 64**-0.5
    got = ops.flash_attn(q, k, v, m, scale, s, softcap)
    ref = eager(q, k, v, m, scale, s, softcap)
    plain = eager(q, k, v, m, scale)
    torch.mps.synchronize()
    torch.testing.assert_close(got, ref, rtol=5e-3, atol=5e-3)
    assert (plain - ref).abs().max() > 0.05  # the case actually exercises them


def test_forward_takes_transformers_sinks_and_softcap():
    """transformers passes gpt-oss's sinks as `s_aux` and Gemma 2's cap as `softcap`."""
    q, k, v, _ = inputs(n_q=1, n_kv=64, head_dim=64)
    s = torch.randn(q.shape[1], device=DEV)
    scale = 64**-0.5
    out, _ = ops.flash_attn_forward(None, q, k, v, None, scaling=scale, s_aux=s, softcap=20.0)
    torch.testing.assert_close(out, eager(q, k, v, None, scale, s, 20.0), rtol=5e-3, atol=5e-3)


def test_sinks_must_have_one_per_head():
    q, k, v, _ = inputs(n_q=1, n_kv=64, head_dim=64)
    with pytest.raises(RuntimeError):
        ops.flash_attn(q, k, v, None, 0.125, torch.zeros(q.shape[1] + 1, device=DEV), 0.0)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("n_q", [8, 40])  # the vector path and the tiled one
def test_one_mask_serves_the_whole_batch(n_q):
    """A (1, 1, n_q, n_kv) mask -- what transformers hands over, and what `flash_attn_forward` builds
    when it gets none -- is broadcast over the batch, as `ggml_flash_attn_ext` allows, not read past."""
    q, k, v, m = inputs(n_seqs=3, n_q=n_q, n_kv=n_q, head_dim=64, mask="causal")
    scale = 64**-0.5
    ref = reference(q, k, v, m, scale)
    got = ops.flash_attn(q, k, v, m[:1], scale)
    fwd, _ = ops.flash_attn_forward(None, q, k, v, None, scaling=scale)
    torch.mps.synchronize()
    torch.testing.assert_close(got, ref, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(fwd, ref, rtol=3e-3, atol=3e-3)
