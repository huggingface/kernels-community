"""CPU-level tests for the UCCL kernel.

UCCL is repackaged from the upstream release wheels. Its device code needs
Hopper (CUDA) or gfx942/gfx950 (ROCm) and RDMA hardware, so these tests only
check that the kernel is loadable and complete. No GPU work is launched.
"""

import importlib
import os

import kernels
import pytest
import torch

uccl = kernels.get_kernel("kernels-community/uccl", version=1)

pytestmark = pytest.mark.kernels_ci


def import_submodule(name):
    return importlib.import_module(f"{uccl.__name__}.{name}")


def test_version():
    assert isinstance(uccl.__version__, str)
    assert uccl.__version__


def test_plugin_path_exists():
    if torch.version.hip is not None:
        path = uccl.rccl_plugin_path()
    else:
        path = uccl.nccl_plugin_path()

    assert os.path.isabs(path)
    assert os.path.isfile(path)


def test_p2p_library_exists():
    lib_dir = os.path.dirname(uccl.nccl_plugin_path())
    assert os.path.isfile(os.path.join(lib_dir, "libuccl_p2p.so"))


def test_vendored_interval_tree():
    utils = import_submodule("utils")

    # The vendored packages must be used (and importable) instead of the
    # `intervaltree` package.
    assert utils.IntervalTree.__module__.startswith(f"{uccl.__name__}._vendor.")

    tree = utils.ClosedIntervalTree()
    tree.add(0, 9, "a")
    tree.add(5, 14, "b")
    tree.add(20, 29, "c")

    assert list(tree) == [(0, 9, "a"), (5, 14, "b"), (20, 29, "c")]
    assert sorted(tree.query_containing(6, 8)) == [(0, 9, "a"), (5, 14, "b")]
    assert tree.query_containing(0, 12) == []
    assert sorted(tree.query_overlap(8, 21)) == [
        (0, 9, "a"),
        (5, 14, "b"),
        (20, 29, "c"),
    ]
    assert tree.query_exact_match(5, 14) == [(5, 14, "b")]
    assert tree.query_exact_match(5, 14, "x") == []

    assert tree.remove(5, 14, "b") == 1
    assert tree.remove(5, 14, "b") == 0
    assert list(tree) == [(0, 9, "a"), (20, 29, "c")]

    with pytest.raises(ValueError):
        tree.add(10, 5, "bad")

    tree.clear()
    assert list(tree) == []


@pytest.mark.parametrize("name", ["p2p", "ep"])
def test_extension_imports(name):
    # The extensions link against libibverbs and the CUDA/ROCm driver, which can
    # be missing on machines without RDMA support.
    try:
        module = import_submodule(name)
    except ImportError as e:
        pytest.skip(f"Cannot import {name}: {e}")

    assert module is not None
