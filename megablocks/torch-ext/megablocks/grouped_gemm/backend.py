# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# On ROCm there is no CUTLASS grouped GEMM; dispatch to the vendored AITER
# Triton kernels instead. On CUDA we use the compiled CUTLASS `gmm` op.
_IS_ROCM = torch.version.hip is not None

if _IS_ROCM:
    from .._grouped_gemm_triton import adapter as backend
else:
    # We import the backend operations from the megablocks package as
    # grouped_gemm is vendored in megablocks in this repository.
    from .._ops import ops as backend  # type: ignore

def _allocate_output(a, b, batch_sizes, trans_a, trans_b):
    assert not (trans_a and trans_b)
    assert batch_sizes.ndim == 1, "Expected 1d tensor for batch_sizes"
    assert a.ndim == 2, "Expected 2d tensor for 'a'"
    assert b.ndim == (2 if trans_a else 3)

    shape = (
        (batch_sizes.shape[0], a.shape[1], b.shape[1])
        if trans_a else
        (a.shape[0], (b.shape[1] if trans_b else b.shape[2]))
    )
    return torch.empty(*shape, device=a.device, dtype=a.dtype)

def gmm(a, b, batch_sizes, trans_a=False, trans_b=False, c=None):
    if c is None:
        c = _allocate_output(a, b, batch_sizes, trans_a, trans_b)
    backend.gmm(a, b, c, batch_sizes, trans_a, trans_b)
    return c
