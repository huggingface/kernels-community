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
    from pathlib import Path

    try:
        from kernels import get_local_kernel

        ops = get_local_kernel(Path(__file__).resolve().parent.parent, "metal")
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


