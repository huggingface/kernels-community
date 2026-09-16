"""Grouped matmuls must stay inside their logical ``(S, N)`` / ``K`` extents.

The M direction is bounded by ``row_mask``, but a tile also runs past ``N`` whenever
``BLOCK_SIZE_N`` does not divide it — or simply exceeds it, which autotune is free to
pick — and past ``K`` whenever ``BLOCK_SIZE_K`` does not divide it. ``C`` is a
contiguous ``(S, N)`` tensor, so an unmasked store writes the overhang into the FOLLOWING
rows: the corruption lands in another token's output, not in slack.

Rows here hold distinct powers of two, so every product is exact and the expected value
is simply ``K * 2**row``: a failure means the kernel left its extent, never rounding.
"""

import pytest
import torch

from utils import MX_SCALE_GROUP_K, TEST_DEVICE  # type: ignore

import finegrained_fp8  # type: ignore


EXPERTS = 2
ROWS_PER_EXPERT = 2


def _inputs(route, N, K):
    """Activations, weights and scales whose exact product is ``K * 2**row``."""
    rows = EXPERTS * ROWS_PER_EXPERT
    A = (2.0 ** torch.arange(rows, dtype=torch.float32)).unsqueeze(1).expand(rows, K)
    A = A.contiguous().to(TEST_DEVICE, torch.bfloat16)
    counts = torch.full((EXPERTS,), ROWS_PER_EXPERT, device=TEST_DEVICE, dtype=torch.int32)
    offsets = counts.cumsum(0).to(torch.int32)
    if route == "tensor":
        B = torch.ones((EXPERTS, N, K), device=TEST_DEVICE).to(torch.float8_e4m3fn)
        scales = torch.ones((EXPERTS,), device=TEST_DEVICE, dtype=torch.float32)
    elif route == "mxfp8":
        B = torch.ones((EXPERTS, N, K), device=TEST_DEVICE).to(torch.float8_e4m3fn)
        scales = torch.ones((EXPERTS, N, K // MX_SCALE_GROUP_K), device=TEST_DEVICE)
        scales = scales.to(torch.float8_e8m0fnu)
    else:  # packed E2M1: nibble 0b0010 is +1.0, two values per byte
        B = torch.full((EXPERTS, N, K // 2), 0x22, device=TEST_DEVICE, dtype=torch.int8)
        scales = torch.ones((EXPERTS, N, K // MX_SCALE_GROUP_K), device=TEST_DEVICE)
        scales = scales.to(torch.float8_e8m0fnu)
    return A, B, scales, offsets, counts


@pytest.mark.parametrize("route", ["tensor", "mxfp8", "mxfp4"])
@pytest.mark.parametrize(
    ("N", "K"),
    [
        (32, 64),  # N below every autotuned BLOCK_SIZE_N
        (33, 64),  # N not a multiple of any of them
        (65, 64),
        (128, 64),
        (129, 64),
        (128, 160),  # K not a multiple of BLOCK_SIZE_K
    ],
)
def test_grouped_matmul_stays_within_n_and_k(route, N, K):
    A, B, scales, offsets, counts = _inputs(route, N, K)
    expected = (A[:, :1].float() * K).expand(A.shape[0], N)
    # Repeated: the overhang and the legitimate write target the same addresses, so a
    # single clean call proves nothing about which store retires last.
    for _ in range(3):
        C = finegrained_fp8.matmul_grouped(A, B, scales, offsets, counts, None)
        torch.testing.assert_close(C.float(), expected, atol=0, rtol=0)
