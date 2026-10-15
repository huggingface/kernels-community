# Smoke tests for the kernels-community CI runner (`nix run .#ci-test`, which
# runs `pytest -m kernels_ci`).
#
# The rest of the suite is vendored from upstream: multi-process scripts that
# need several NVLink-connected Hopper/Blackwell GPUs, NCCL >= 2.32.3 and a
# CUDA >= 13.1 toolkit for the runtime JIT (run them as
# `python tests/ep/test_ep.py --num-processes 2`). The tests here only need one
# CUDA device and cover what does not launch communication kernels: loading the
# extension against PyTorch's NCCL, the constants, events, allocation plans and
# the runtime requirement checks.

import os

import kernels
import pytest
import torch

# `kernels-community/deep-ep` is `[general.hub] repo-id` and 1 is
# `[general] version` in `build.toml`.
deep_ep = kernels.get_kernel("kernels-community/deep-ep", version=1)

pytestmark = [
    pytest.mark.kernels_ci,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device"),
]


def test_uses_torch_nccl():
    # The extension does not ship NCCL: its symbols resolve to the library PyTorch loaded
    loaded = deep_ep._runtime.get_loaded_nccl_libraries()
    assert len(loaded) == 1
    assert deep_ep._nccl_symbols.torch_nccl_library == loaded[0]
    assert deep_ep._C.get_nccl_runtime_version() >= 20000
    assert deep_ep._C.get_nccl_compile_version() >= 23200


def test_constants():
    assert deep_ep.__version__ == "2.5.0"
    assert deep_ep.topk_idx_t == torch.int64
    assert deep_ep.get_num_allocation_alignment() == 2 << 20
    assert deep_ep.get_num_tma_alignment() == 32
    assert deep_ep.get_num_rdma_alignment() == 64


def test_event_overlap():
    x = torch.randn(1024, device="cuda")
    event = deep_ep.EPBuffer.capture()
    overlap = deep_ep.EventOverlap(event)
    overlap.register_hook_after_wait(lambda: x * 2)
    torch.testing.assert_close(overlap.wait(), x * 2)


def test_comm_stream():
    stream = deep_ep.comm.get_comm_stream(None)
    assert isinstance(stream, torch.cuda.Stream)
    assert stream != torch.cuda.current_stream()


def test_allocation_plan():
    plan = deep_ep.BufferAllocator()
    a = plan.allocate((4, 1024), torch.bfloat16)
    b = plan.allocate((8,), torch.float32)
    assert plan.num_bytes % deep_ep.get_num_allocation_alignment() == 0
    storage = torch.zeros(plan.num_bytes, dtype=torch.uint8, device="cuda")
    plan.materialize(storage)
    assert a.shape == (4, 1024) and b.shape == (8,)
    assert a.data_ptr() >= storage.data_ptr() and b.data_ptr() > a.data_ptr()


def test_runtime_checks():
    runtime = deep_ep._runtime
    version = deep_ep._C.get_nccl_runtime_version()
    if version < 23203:
        with pytest.raises(runtime.DeepEPRuntimeError, match="requires NCCL >= 2.32.3"):
            runtime.check_nccl()
    else:
        runtime.check_nccl()

    saved = {name: os.environ.get(name) for name in ("CUDA_HOME", "EP_JIT_NVCC_COMPILER")}
    try:
        os.environ["CUDA_HOME"] = "/nonexistent"
        os.environ.pop("EP_JIT_NVCC_COMPILER", None)
        with pytest.raises(runtime.DeepEPRuntimeError, match="needs a CUDA toolkit"):
            runtime.check_cuda_toolkit()
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
