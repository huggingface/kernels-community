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
import triton

from utils import MX_SCALE_GROUP_K, TEST_DEVICE  # type: ignore

import finegrained_fp8  # type: ignore
import finegrained_fp8.grouped  # type: ignore


EXPERTS = 2
ROWS_PER_EXPERT = 2

# Pin non-dividing tiles (BN=64, BK=128): the autotuner's choice is timing
# noise, not a contract — left alone it may crown a config that divides N and/or
# K, and then the tail path this file exists to exercise never runs. Pinned,
# every (N, K) below leaves a tail somewhere: N=32/33 wrap inside one tile,
# N=65/129 wrap in a second, K=64 runs the tail block only (BLOCK_SIZE_K > K),
# K=160 runs one full tile then a 32-wide tail.
PINNED_TENSOR = triton.Config(
    {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2
)
PINNED_MX = triton.Config(
    {"COMPUTE_MODE": "dot_scaled", "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128},
    num_warps=4,
    num_stages=2,
)
# The `dot` arm pins BLOCK_SIZE_K to the 32-element scale group, and the wrapper
# already requires K % 32 == 0, so it can never take a K tail — but it shares the
# N wrap and the store mask, which this config covers.
PINNED_MX_DOT = triton.Config(
    {"COMPUTE_MODE": "dot", "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
    num_warps=4,
    num_stages=2,
)


@pytest.fixture(autouse=True)
def _pin_grouped_tiles(monkeypatch):
    grouped = finegrained_fp8.grouped
    monkeypatch.setattr(
        grouped.w8a8_tensor_dynamic_fp8_matmul_grouped_kernel,
        "configs",
        [PINNED_TENSOR],
        raising=False,
    )
    monkeypatch.setattr(
        grouped.mxfp_dynamic_matmul_grouped_kernel,
        "configs",
        [PINNED_MX],
        raising=False,
    )


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
        (33, 128),  # N tail with NO K tail — isolates the wrap + store mask
    ],
)
@pytest.mark.kernels_ci
def test_grouped_matmul_stays_within_n_and_k(route, N, K):
    A, B, scales, offsets, counts = _inputs(route, N, K)
    expected = (A[:, :1].float() * K).expand(A.shape[0], N)
    # Repeated: the overhang and the legitimate write target the same addresses, so a
    # single clean call proves nothing about which store retires last.
    for _ in range(3):
        C = finegrained_fp8.matmul_grouped(A, B, scales, offsets, counts, None)
        torch.testing.assert_close(C.float(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("route", ["mxfp8", "mxfp4"])
@pytest.mark.parametrize("N", [33, 129])
@pytest.mark.kernels_ci
def test_mx_dot_mode_stays_within_n(route, N, monkeypatch):
    """The MX ``dot`` arm reaches the same wrap/store path with a different tile."""
    monkeypatch.setattr(
        finegrained_fp8.grouped.mxfp_dynamic_matmul_grouped_kernel,
        "configs",
        [PINNED_MX_DOT],
        raising=False,
    )
    K = 128
    A, B, scales, offsets, counts = _inputs(route, N, K)
    expected = (A[:, :1].float() * K).expand(A.shape[0], N)
    for _ in range(3):
        C = finegrained_fp8.matmul_grouped(A, B, scales, offsets, counts, None)
        torch.testing.assert_close(C.float(), expected, atol=0, rtol=0)
