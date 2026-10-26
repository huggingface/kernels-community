"""`l2_norm` against torch, and against the expression it replaces.

Both exist to cut dispatches rather than to compute anything new, so the test that matters is that
they agree with the torch spelling a model would otherwise use.
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
        ops = kernels.get_kernel("ggml-org/ggml-gated-delta-net", version=1)
    except Exception as error:  # pragma: no cover
        pytest.skip(f"no kernel to test ({error})", allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


def torch_ops(loaded):
    """The registered ops: a packaged kernel keeps them on `.ops`, a bare library is them."""
    return getattr(loaded, "ops", loaded)


@pytest.mark.parametrize("shape", [(1, 1, 32, 128), (1, 1, 32, 256), (7, 64), (1, 1, 1, 96)])
def test_l2_norm_matches_normalize(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device=DEV)
    got = ops.l2_norm(x, 1e-6)
    ref = F.normalize(x, p=2.0, dim=-1, eps=1e-6)
    torch.mps.synchronize()
    assert got.shape == x.shape
    assert (got - ref).abs().max().item() < 1e-6


@pytest.mark.parametrize("scale", [1.0, 1e-20, 0.0])
def test_l2_norm_degenerate_rows(scale):
    """A zero or denormal row must not produce nan: that is what the epsilon is for."""
    x = torch.full((4, 128), scale, device=DEV)
    got = ops.l2_norm(x, 1e-6)
    ref = F.normalize(x, p=2.0, dim=-1, eps=1e-6)
    torch.mps.synchronize()
    assert torch.isfinite(got).all()
    assert (got - ref).abs().max().item() < 1e-6
def test_compile_without_graph_breaks():
    if LIB:
        pytest.skip("fakes live in the packaged python wrapper, not in the raw library")
    x = torch.randn(1, 1, 32, 128, device=DEV)
    normalise = torch.compile(lambda t: ops.l2_norm(t, 1e-6), fullgraph=True)
    assert normalise(x).shape == x.shape
    torch.mps.synchronize()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


@pytest.mark.parametrize("n_seqs", [1, 3, 8])
def test_delta_gates_match_torch(n_seqs):
    """`delta_gates` against the two expressions it fuses, for one sequence and for a batch of them.

    `b` and `a` carry one value per head for every sequence; `A_log` and `dt_bias` one per head, shared
    by all of them -- a batched decode step hands the kernel exactly that.
    """
    n_heads = 32
    torch.manual_seed(0)
    b = torch.randn(n_seqs, 1, n_heads, device=DEV)
    a = torch.randn(n_seqs, 1, n_heads, device=DEV) * 4
    a_log = torch.randn(n_heads, device=DEV)
    dt_bias = torch.randn(n_heads, device=DEV)
    beta, g = torch_ops(ops).delta_gates(b.reshape(-1), a.reshape(-1), a_log, dt_bias)
    ref_beta = b.sigmoid()
    ref_g = -a_log.exp() * F.softplus(a + dt_bias)
    torch.mps.synchronize()
    assert beta.numel() == g.numel() == n_seqs * n_heads
    torch.testing.assert_close(beta.view_as(b), ref_beta, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(g.view_as(a), ref_g, rtol=1e-5, atol=1e-5)


def test_delta_gates_rejects_mismatched_heads():
    b = torch.randn(33, device=DEV)
    with pytest.raises(RuntimeError, match="one value per head"):
        torch_ops(ops).delta_gates(b, b, torch.randn(32, device=DEV), torch.randn(32, device=DEV))
