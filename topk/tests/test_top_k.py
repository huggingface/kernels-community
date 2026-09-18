"""`top_k` against `torch.topk`, which is what it replaces.

The kernel exists to cut a sort down to a selection, so the test that matters is that it picks the
same elements, in the same order, as the sort it stands in for.
"""

import os

import pytest
import torch


DEV = "mps"
LIB = os.environ.get("TOPK_LOCAL_LIB")

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

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs mps")


@pytest.mark.parametrize(("rows", "n", "k"), [(1, 256, 8), (1, 128, 4), (7, 512, 8), (1, 64, 1), (3, 320, 16)])
def test_matches_torch_topk(rows, n, k):
    torch.manual_seed(0)
    logits = torch.randn(rows, n, device=DEV)
    values, indices = ops.top_k(logits, k, False)
    ref_values, ref_indices = torch.topk(logits, k, dim=-1)
    assert torch.equal(indices.long(), ref_indices)
    assert torch.equal(values, ref_values)
    assert indices.dtype == torch.int32


def test_softmax_is_over_the_selected_k():
    torch.manual_seed(0)
    logits = torch.randn(4, 256, device=DEV)
    values, _ = ops.top_k(logits, 8, True)
    reference = torch.softmax(torch.topk(logits, 8, dim=-1).values.float(), dim=-1)
    assert torch.allclose(values, reference, atol=1e-6)
    assert torch.allclose(values.sum(-1), torch.ones(4, device=DEV), atol=1e-6)


def test_k_equal_to_the_row():
    """Degenerate but legal: selecting everything is a sort, and must still agree."""
    torch.manual_seed(0)
    logits = torch.randn(2, 64, device=DEV)
    _, indices = ops.top_k(logits, 64, False)
    assert torch.equal(indices.long(), torch.topk(logits, 64, dim=-1).indices)
