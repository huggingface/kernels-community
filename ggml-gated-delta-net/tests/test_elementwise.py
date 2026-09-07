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
    from pathlib import Path

    try:
        from kernels import get_local_kernel

        ops = get_local_kernel(Path(__file__).resolve().parent.parent, "metal")
    except Exception as error:  # pragma: no cover
        pytest.skip(f"no kernel to test ({error})", allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


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
