import torch
from contextlib import contextmanager


# FP4 (E2M1) packs two 4-bit values per byte; scale factors are UE8M0, one per
# K-group of 32 elements. These are format constants, not tunables.
FP4_VALUES_PER_BYTE = 2
FP4_SCALE_GROUP_K = 32


@contextmanager
def device_context(device: torch.device):
    """Context manager that sets the active device for any backend (cuda, xpu, etc.)."""
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "device"):
        with backend.device(device):
            yield
    else:
        yield


def fp4_resolve_block_k(block_size: list[int] | None, K: int) -> int:
    """Pick the K-block size used to drive activation quantization for FP4 matmul.

    `block_size[1]` is the quant block_k requested by the caller; if K isn't
    divisible by it, fall back to the FP4 scale group (32) only when the caller
    passed `None`. A mismatched explicit `block_size[1]` is a user error.
    """
    block_k = 128 if block_size is None else block_size[1]
    if K % block_k == 0:
        return block_k
    if block_size is None:
        return FP4_SCALE_GROUP_K
    raise AssertionError(f"K (={K}) must be divisible by block_k (={block_k})")


def fp4_expand_activation_scales(As: torch.Tensor, K: int, block_k: int) -> torch.Tensor:
    """Repeat-interleave activation scales from per-block_k granularity to the
    kernel's per-FP4-scale-group granularity, when the two differ. The kernel
    always reads scales at FP4_SCALE_GROUP_K granularity; callers that quantized
    at a larger block_k must expand."""
    expected_groups = K // FP4_SCALE_GROUP_K
    if As.shape[-1] == expected_groups:
        return As
    expected_blocks = K // block_k
    assert As.shape[-1] == expected_blocks, (
        f"As shape {tuple(As.shape)} incompatible with K={K}, block_k={block_k}; "
        f"expected last dim {expected_blocks} or {expected_groups}"
    )
    assert block_k % FP4_SCALE_GROUP_K == 0, (
        f"block_k (={block_k}) must be divisible by {FP4_SCALE_GROUP_K} for FP4 scale expansion"
    )
    return As.repeat_interleave(block_k // FP4_SCALE_GROUP_K, dim=-1)
