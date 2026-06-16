"""Tests for ``matmul`` (FP8, MXFP4, and MXFP8 weights, including non-aligned dims)."""

from dataclasses import dataclass
from typing import Optional, Tuple

import pytest
import torch

from utils import (  # type: ignore
    DTYPE_TAG,
    DTYPE_TO_TOL,
    MX_SCALE_GROUP_K,
    TEST_DEVICE,
    accelerator_module,
    make_fp8_weights,
    make_fp4_weights,
    make_static_activation_scale,
    ref_matmul,
)

import finegrained_fp8  # type: ignore


@dataclass(frozen=True)
class Problem:
    M: int
    N: int
    K: int
    block_size: Optional[Tuple[int, int]] = None
    weight_scale_dtype: torch.dtype = torch.float32
    static_activation_scale: bool = False
    dtype: torch.dtype = torch.bfloat16
    weight_format: str = "fp8"
    compile: bool = False

    @property
    def id(self) -> str:
        head = f"{self.weight_format}_M{self.M}_N{self.N}_K{self.K}"
        is_mx = self.weight_format in ("mxfp4", "mxfp8")
        # MX recipes ignore block_size (MX group is fixed at 32; tile autotuned).
        if is_mx:
            tail = head
        elif self.block_size is None:
            tail = f"{head}_tensor"
        else:
            tail = f"{head}_b{self.block_size[0]}x{self.block_size[1]}"
        tail = f"{tail}_{DTYPE_TAG[self.dtype]}"
        # UE8M0 weight scales — implied by the MX recipe name, tagged only for FP8.
        if self.weight_scale_dtype is torch.float8_e8m0fnu and not is_mx:
            tail = f"{tail}_ue8m0"
        if self.static_activation_scale:
            tail = f"{tail}_static"
        if self.compile:
            tail = f"{tail}_compile"
        return tail


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
    # UE8M0 weight scales (DSv4-Flash style)
    Problem(
        M=16,
        N=320,
        K=1024,
        block_size=(128, 128),
        weight_scale_dtype=torch.float8_e8m0fnu,
    ),
    # Static activation scale (calibration-time per-tensor)
    Problem(
        M=16,
        N=320,
        K=1024,
        block_size=(128, 128),
        static_activation_scale=True,
    ),
    Problem(
        M=16,
        N=1024,
        K=2048,
        block_size=(128, 128),
        static_activation_scale=True,
    ),
    # torch.compile compatibility
    Problem(M=16, N=320, K=1024, block_size=(128, 128), compile=True),
    # ── MX recipes: block_size is the MX 1x32 scale group (the kernel fixes the
    # group at 32 and autotunes its compute tile). ``tl.dot_scaled`` runs natively
    # on Blackwell, emulated elsewhere. ──
    Problem(
        M=32, N=256, K=512, block_size=(1, MX_SCALE_GROUP_K), weight_format="mxfp4"
    ),
    Problem(
        M=32,
        N=256,
        K=512,
        block_size=(1, MX_SCALE_GROUP_K),
        weight_format="mxfp4",
        dtype=torch.float16,
    ),
    Problem(
        M=32,
        N=256,
        K=512,
        block_size=(1, MX_SCALE_GROUP_K),
        weight_format="mxfp4",
        dtype=torch.float32,
    ),
    Problem(
        M=32,
        N=256,
        K=512,
        block_size=(1, MX_SCALE_GROUP_K),
        weight_format="mxfp4",
        compile=True,
    ),
    # ── MXFP8 (E4M3 weights + E4M3 act, UE8M0 group-32) ──
    Problem(
        M=32,
        N=256,
        K=512,
        block_size=(1, MX_SCALE_GROUP_K),
        weight_scale_dtype=torch.float8_e8m0fnu,
        weight_format="mxfp8",
    ),
    Problem(  # non-aligned N
        M=16,
        N=320,
        K=1024,
        block_size=(1, MX_SCALE_GROUP_K),
        weight_scale_dtype=torch.float8_e8m0fnu,
        weight_format="mxfp8",
    ),
    Problem(
        M=32,
        N=256,
        K=512,
        block_size=(1, MX_SCALE_GROUP_K),
        weight_scale_dtype=torch.float8_e8m0fnu,
        weight_format="mxfp8",
        dtype=torch.float16,
    ),
    Problem(
        M=32,
        N=256,
        K=512,
        block_size=(1, MX_SCALE_GROUP_K),
        weight_scale_dtype=torch.float8_e8m0fnu,
        weight_format="mxfp8",
        dtype=torch.float32,
    ),
    Problem(
        M=32,
        N=256,
        K=512,
        block_size=(1, MX_SCALE_GROUP_K),
        weight_scale_dtype=torch.float8_e8m0fnu,
        weight_format="mxfp8",
        compile=True,
    ),
]


def _setup_problem(problem: Problem):
    torch.manual_seed(42)
    A = torch.randn(problem.M, problem.K, dtype=problem.dtype, device=TEST_DEVICE)
    if problem.weight_format.endswith("fp4"):
        B, Bs = make_fp4_weights(
            problem.N,
            problem.K,
            TEST_DEVICE,
            problem.block_size,
        )
    else:
        B, Bs = make_fp8_weights(
            problem.N,
            problem.K,
            TEST_DEVICE,
            problem.block_size,
            scale_dtype=problem.weight_scale_dtype,
        )
    a_scale = (
        make_static_activation_scale(A) if problem.static_activation_scale else None
    )
    return A, B, Bs, a_scale


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="Accelerator not available")
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_matmul(problem: Problem):
    """``matmul`` matches the pure-PyTorch dequant+matmul reference."""
    A, B, Bs, a_scale = _setup_problem(problem)
    matmul = finegrained_fp8.matmul
    if problem.compile:
        torch.compiler.reset()
        accelerator_module().empty_cache()
        matmul = torch.compile(matmul, mode="max-autotune", fullgraph=True)
    out = matmul(
        A,
        B,
        Bs,
        problem.block_size,
        problem.dtype,
        activation_scale=a_scale,
    )
    ref = ref_matmul(
        A,
        B,
        Bs,
        problem.block_size,
        problem.dtype,
        activation_scale=a_scale,
    )
    assert out.dtype == problem.dtype
    assert out.shape == (problem.M, problem.N)
    atol, rtol = DTYPE_TO_TOL[problem.dtype]
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
