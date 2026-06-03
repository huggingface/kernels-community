"""Tests for ``matmul`` (FP8 and FP4 weights, including non-aligned dims)."""

from dataclasses import dataclass
from typing import Optional, Tuple

import pytest
import torch

from utils import (  # type: ignore
    DTYPE_TAG,
    DTYPE_TO_TOL,
    SUPPORTS_FP4,
    TEST_DEVICE,
    make_fp4_weights,
    make_fp8_weights,
    ref_matmul,
)

import finegrained_fp8  # type: ignore


@dataclass(frozen=True)
class Problem:
    M: int
    N: int
    K: int
    block_size: Optional[Tuple[int, int]] = None
    dtype: torch.dtype = torch.bfloat16
    weight_format: str = "fp8"

    @property
    def id(self) -> str:
        head = f"{self.weight_format}_M{self.M}_N{self.N}_K{self.K}"
        # FP4 ignores block_size (tile shape autotuned).
        if self.weight_format == "fp4":
            tail = head
        elif self.block_size is None:
            tail = f"{head}_tensor"
        else:
            tail = f"{head}_b{self.block_size[0]}x{self.block_size[1]}"
        return f"{tail}_{DTYPE_TAG[self.dtype]}"


PROBLEMS = [
    # ── FP8: per-tensor ──
    Problem(M=16, N=512, K=1024, block_size=None),
    # ── FP8: non-aligned N (MLA kv_a_proj style) ──
    Problem(M=16, N=320, K=1024, block_size=(128, 128)),
    # ── FP8: aligned block-wise ──
    Problem(M=16, N=1024, K=2048, block_size=(128, 128)),
    # fp16 / fp32 dtype coverage
    Problem(M=16, N=320, K=1024, block_size=(128, 128), dtype=torch.float16),
    Problem(M=16, N=320, K=1024, block_size=(128, 128), dtype=torch.float32),
]
if SUPPORTS_FP4:
    # ``block_size`` is ignored on the FP4 path (tile shape autotuned).
    PROBLEMS += [
        Problem(M=32, N=256, K=512, weight_format="fp4"),
        Problem(M=32, N=256, K=512, weight_format="fp4", dtype=torch.float16),
        Problem(M=32, N=256, K=512, weight_format="fp4", dtype=torch.float32),
    ]


COMPILE_PROBLEMS = [
    Problem(M=16, N=320, K=1024, block_size=(128, 128)),
]
if SUPPORTS_FP4:
    COMPILE_PROBLEMS += [Problem(M=32, N=256, K=512, weight_format="fp4")]


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="Accelerator not available")
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_matmul(problem: Problem):
    """``matmul`` matches the pure-PyTorch dequant+matmul reference."""
    torch.manual_seed(42)
    device = TEST_DEVICE
    A = torch.randn(problem.M, problem.K, dtype=torch.bfloat16, device=device)

    if problem.weight_format == "fp4":
        B, Bs = make_fp4_weights(problem.N, problem.K, device)
    else:
        B, Bs = make_fp8_weights(problem.N, problem.K, device, problem.block_size)

    out = finegrained_fp8.matmul(A, B, Bs, problem.block_size, problem.dtype)
    ref = ref_matmul(A, B, Bs, problem.block_size, problem.dtype)
    assert out.dtype == problem.dtype
    assert out.shape == (problem.M, problem.N)
    atol, rtol = DTYPE_TO_TOL[problem.dtype]
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


# ── torch.compile compatibility ────────────────────────────────────────────────
@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="Accelerator not available")
@pytest.mark.parametrize("problem", COMPILE_PROBLEMS, ids=lambda p: p.id)
def test_matmul_compile(problem: Problem):
    torch.manual_seed(0)
    torch.compiler.reset()
    A = torch.randn(problem.M, problem.K, dtype=torch.bfloat16, device=TEST_DEVICE)
    if problem.weight_format == "fp4":
        B, Bs = make_fp4_weights(problem.N, problem.K, TEST_DEVICE)
    else:
        B, Bs = make_fp8_weights(problem.N, problem.K, TEST_DEVICE, problem.block_size)

    def fn(A, B, Bs):
        return finegrained_fp8.matmul(A, B, Bs, problem.block_size, torch.bfloat16)

    compiled = torch.compile(fn, mode="max-autotune", fullgraph=True)
    out_compiled = compiled(A, B, Bs)
    out_ref = fn(A, B, Bs)
    torch.testing.assert_close(out_compiled, out_ref)
