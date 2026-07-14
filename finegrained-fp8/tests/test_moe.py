"""Tests for MoE expert dispatch kernels: batched and grouped."""

import statistics
from dataclasses import dataclass
from typing import Optional, Tuple

import pytest
import torch

import triton

from utils import (  # type: ignore
    DTYPE_TAG,
    DTYPE_TO_TOL,
    IS_SM90,
    MX_SCALE_GROUP_K,
    TEST_DEVICE,
    accelerator_module,
    make_weights,
    maybe_compile,
)

import finegrained_fp8  # type: ignore
from finegrained_fp8.utils import (  # type: ignore
    fp8_act_quant_block_dynamic,
    fp8_act_quant_tensor_wide,
    mxfp_act_quant,
)
from finegrained_fp8 import moe  # type: ignore


BENCH_REPEATS = 10


@dataclass(frozen=True)
class Expectations:
    batched_ms: float
    grouped_ms: float


@dataclass(frozen=True)
class Problem:
    S: int
    E: int
    N: int
    K: int
    TOP_K: int
    scale_layout: str
    weight_dtype: torch.dtype = torch.float8_e4m3fn
    weight_scale_dtype: torch.dtype = torch.float32
    block_size: Optional[Tuple[int, int]] = None
    expectation: Optional[Expectations] = None
    dtype: torch.dtype = torch.bfloat16
    sentinel_fraction: float = 0.0
    contiguous: bool = True
    compile: bool = False

    @property
    def is_mxfp(self):
        return self.block_size == (1, MX_SCALE_GROUP_K)

    @property
    def id(self):
        # Recipe label derived from the stored dtype + scale group: packed E2M1 is
        # MXFP4; E4M3 with a 1x32 group is MXFP8; otherwise plain FP8.
        if self.weight_dtype == torch.int8:
            fmt = "mxfp4"
        elif self.is_mxfp:
            fmt = "mxfp8"
        else:
            fmt = "fp8"
        head = f"{fmt}_S{self.S}_E{self.E}_N{self.N}_K{self.K}_top{self.TOP_K}"
        # MX recipes pin block_size to the 1x32 scale group, so it isn't part of
        # the id (the kernel fixes the group at 32 and autotunes its compute tile).
        if self.is_mxfp:
            tail = head
        # FP8 with block_size carries block dims; without it, scale_layout names
        # the tensor-mode variant.
        elif self.block_size is None:
            tail = f"{head}_{self.scale_layout}"
        else:
            tail = f"{head}_b{self.block_size[0]}x{self.block_size[1]}"
        contig_tag = "contiguous" if self.contiguous else "noncontiguous"
        tail = f"{tail}_{contig_tag}_{DTYPE_TAG[self.dtype]}"
        # UE8M0 weight scales — implied by the MX recipe name, tagged only for FP8.
        if self.weight_scale_dtype is torch.float8_e8m0fnu and not self.is_mxfp:
            tail = f"{tail}_ue8m0"
        if self.sentinel_fraction > 0:
            tail = f"{tail}_sentinel"
        if self.compile:
            tail = f"{tail}_compile"
        return tail


PROBLEMS = [
    # ── Small problems (correctness only, no speedup expectations) ──
    Problem(
        S=8,
        E=4,
        N=256,
        K=256,
        TOP_K=1,
        scale_layout="block",
        block_size=(128, 128),
    ),
    Problem(
        S=32,
        E=4,
        N=256,
        K=512,
        TOP_K=2,
        scale_layout="block",
        block_size=(64, 128),
    ),
    Problem(
        S=32,
        E=4,
        N=256,
        K=512,
        TOP_K=2,
        scale_layout="block",
        block_size=(128, 128),
    ),
    Problem(
        S=128,
        E=16,
        N=1024,
        K=2048,
        TOP_K=2,
        scale_layout="per_tensor_1d",
        block_size=None,
    ),
    Problem(
        S=64,
        E=8,
        N=512,
        K=1024,
        TOP_K=4,
        scale_layout="per_tensor_111",
        block_size=None,
    ),
    # Non-contig expert_ids — exercises stride-aware loads in the batched kernel and
    # the contiguous-normalization inside the grouped scheduling op.
    Problem(
        S=32,
        E=4,
        N=256,
        K=512,
        TOP_K=2,
        scale_layout="block",
        block_size=(128, 128),
        contiguous=False,
    ),
    # ── Qwen3-30B-A3B (E=128, H=2048, I=768, top_k=8) ──
    # gate_up: N=1536, K=2048 — down: N=2048, K=768
    Problem(
        S=256,
        E=128,
        N=1536,
        K=2048,
        TOP_K=8,
        scale_layout="block",
        block_size=(128, 128),
        expectation=Expectations(batched_ms=0.1641, grouped_ms=0.1596),
    ),
    Problem(
        S=256,
        E=128,
        N=2048,
        K=768,
        TOP_K=8,
        scale_layout="block",
        block_size=(128, 128),
        expectation=Expectations(batched_ms=0.0956, grouped_ms=0.1582),
    ),
    Problem(
        S=1024,
        E=128,
        N=1536,
        K=2048,
        TOP_K=8,
        scale_layout="block",
        block_size=(128, 128),
        expectation=Expectations(batched_ms=0.5731, grouped_ms=0.1904),
    ),
    Problem(
        S=1024,
        E=128,
        N=2048,
        K=768,
        TOP_K=8,
        scale_layout="block",
        block_size=(128, 128),
        expectation=Expectations(batched_ms=0.3151, grouped_ms=0.1571),
    ),
    # EP sentinel coverage: most rows routed to non-local experts, kernel skips tail
    Problem(
        S=64,
        E=8,
        N=256,
        K=512,
        TOP_K=4,
        scale_layout="block",
        block_size=(128, 128),
        sentinel_fraction=0.875,
    ),
    # torch.compile compatibility
    Problem(
        S=8,
        E=4,
        N=256,
        K=256,
        TOP_K=1,
        scale_layout="block",
        block_size=(128, 128),
        compile=True,
    ),
    # UE8M0 weight scales (DSv4-Flash style)
    Problem(
        S=32,
        E=4,
        N=256,
        K=512,
        TOP_K=2,
        scale_layout="block",
        block_size=(128, 128),
        weight_scale_dtype=torch.float8_e8m0fnu,
    ),
    # fp16 / fp32 dtype coverage on the smallest FP8 shape
    Problem(
        S=8,
        E=4,
        N=256,
        K=256,
        TOP_K=1,
        scale_layout="block",
        block_size=(128, 128),
        dtype=torch.float16,
    ),
    Problem(
        S=8,
        E=4,
        N=256,
        K=256,
        TOP_K=1,
        scale_layout="block",
        block_size=(128, 128),
        dtype=torch.float32,
    ),
    # MX recipes (MXFP4 here; MXFP8 below). DeepSeek-V4-Flash MXFP4 shapes
    # (E=256, top_k=6); first entry kept small so CI doesn't dwell on the big
    # ones. block_size is the MX 1x32 scale group used to build the weights
    # (the kernel fixes the group at 32 and autotunes its compute tile).
    Problem(
        S=32,
        E=4,
        N=256,
        K=512,
        TOP_K=2,
        scale_layout="block",
        block_size=(1, MX_SCALE_GROUP_K),
        weight_dtype=torch.int8,
    ),
    Problem(
        S=32,
        E=4,
        N=256,
        K=512,
        TOP_K=2,
        scale_layout="block",
        block_size=(1, MX_SCALE_GROUP_K),
        weight_dtype=torch.int8,
        contiguous=False,
    ),
    Problem(
        S=192,
        E=256,
        N=4096,
        K=4096,
        TOP_K=6,
        scale_layout="block",
        block_size=(1, MX_SCALE_GROUP_K),
        weight_dtype=torch.int8,
    ),
    Problem(
        S=192,
        E=256,
        N=4096,
        K=2048,
        TOP_K=6,
        scale_layout="block",
        block_size=(1, MX_SCALE_GROUP_K),
        weight_dtype=torch.int8,
    ),
    Problem(
        S=1536,
        E=256,
        N=4096,
        K=4096,
        TOP_K=6,
        scale_layout="block",
        block_size=(1, MX_SCALE_GROUP_K),
        weight_dtype=torch.int8,
    ),
    # EP sentinel coverage on MXFP4
    Problem(
        S=64,
        E=8,
        N=256,
        K=512,
        TOP_K=4,
        scale_layout="block",
        block_size=(1, MX_SCALE_GROUP_K),
        weight_dtype=torch.int8,
        sentinel_fraction=0.875,
    ),
    # torch.compile compatibility on MXFP4
    Problem(
        S=32,
        E=4,
        N=256,
        K=512,
        TOP_K=2,
        scale_layout="block",
        block_size=(1, MX_SCALE_GROUP_K),
        weight_dtype=torch.int8,
        compile=True,
    ),
    # fp16 / fp32 dtype coverage on the smallest MXFP4 shape
    Problem(
        S=32,
        E=4,
        N=256,
        K=512,
        TOP_K=2,
        scale_layout="block",
        block_size=(1, MX_SCALE_GROUP_K),
        weight_dtype=torch.int8,
        dtype=torch.float16,
    ),
    Problem(
        S=32,
        E=4,
        N=256,
        K=512,
        TOP_K=2,
        scale_layout="block",
        block_size=(1, MX_SCALE_GROUP_K),
        weight_dtype=torch.int8,
        dtype=torch.float32,
    ),
    # ── MXFP8 (E4M3 weights + E4M3 act, UE8M0 group-32; block_size ignored) ──
    Problem(
        S=32,
        E=4,
        N=256,
        K=512,
        TOP_K=2,
        scale_layout="block",
        block_size=(1, MX_SCALE_GROUP_K),
        weight_scale_dtype=torch.float8_e8m0fnu,
    ),
    Problem(
        S=64,
        E=8,
        N=512,
        K=1024,
        TOP_K=4,
        scale_layout="block",
        block_size=(1, MX_SCALE_GROUP_K),
        weight_scale_dtype=torch.float8_e8m0fnu,
    ),
    # non-contiguous indexing tensors on the MXFP8 path
    Problem(
        S=32,
        E=4,
        N=256,
        K=512,
        TOP_K=2,
        scale_layout="block",
        block_size=(1, MX_SCALE_GROUP_K),
        weight_scale_dtype=torch.float8_e8m0fnu,
        contiguous=False,
    ),
    # int32 pointer-offset overflow guard: the last experts' weight offsets exceed 2^31
    # elements (127 * 4096 * 4224 = 2.197e9) — a regressed int64 expert-offset cast wraps
    # them negative and corrupts every token routed high (S=512 random routing makes
    # missing the top experts a ~1e-7 event under the fixed seed). Covers the batched
    # expert_setup cast and the grouped tile-search cast through the standard suites.
    Problem(
        S=512,
        E=128,
        N=4096,
        K=4224,
        TOP_K=4,
        scale_layout="block",
        block_size=(128, 128),
    ),
]


def _make_routed_inputs(S, E, K, dtype, device, top_k, sentinel_fraction=0.0):
    """Build flattened routed inputs like FP8Experts paths.

    ``sentinel_fraction`` marks a random subset of ``expert_ids`` as out-of-range
    (== E), simulating EP routing where some tokens are owned by other ranks.
    """
    assert top_k > 0
    assert S % top_k == 0, f"S ({S}) must be divisible by top_k ({top_k})"
    num_tokens = S // top_k

    hidden_states = torch.randn(num_tokens, K, dtype=dtype, device=device)
    top_k_index = torch.randint(0, E, (num_tokens, top_k), device=device)

    token_idx = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
    )
    expert_ids = top_k_index.reshape(-1).to(torch.int32)
    selected_hidden_states = hidden_states[token_idx]
    if sentinel_fraction > 0:
        n_sentinel = int(round(S * sentinel_fraction))
        idx = torch.randperm(S, device=device)[:n_sentinel]
        expert_ids[idx] = E
    return selected_hidden_states, expert_ids


def _make_noncontig_1d(x: torch.Tensor) -> torch.Tensor:
    # Stride-2 view with duplicated gaps: a stride-1 read returns
    # [x[0], x[0], x[1], x[1], ...] instead of x, catching wrong-stride loads.
    base = torch.empty((x.numel(), 2), dtype=x.dtype, device=x.device)
    base[:, 0] = x
    base[:, 1] = x
    return base[:, 0]


def _setup_problem(problem: Problem):
    A, expert_ids = _make_routed_inputs(
        problem.S,
        problem.E,
        problem.K,
        dtype=problem.dtype,
        device=TEST_DEVICE,
        top_k=problem.TOP_K,
        sentinel_fraction=problem.sentinel_fraction,
    )
    B, Bs = make_weights(
        problem.N,
        problem.K,
        TEST_DEVICE,
        problem.block_size,
        weight_dtype=problem.weight_dtype,
        scale_dtype=problem.weight_scale_dtype,
        scale_layout=problem.scale_layout,
        num_experts=problem.E,
    )
    if not problem.contiguous:
        expert_ids = _make_noncontig_1d(expert_ids)
    return A, expert_ids, B, Bs


def _routed_matmul_ref(A, B, Bs, expert_ids, block_size):
    """Per-routed-row reference that re-uses the neutral ``matmul_2d`` dispatcher
    (it routes by weight dtype / scale layout: MXFP4, MXFP8, or block/tensor FP8).
    Output dtype matches the batched/grouped kernels (both follow ``A.dtype``). Rows
    with ``expert_ids[i] >= num_experts`` are EP sentinels — skipped, output undefined."""
    S = A.shape[0]
    N = B.shape[1]
    E = B.shape[0]
    out = torch.empty(S, N, dtype=A.dtype, device=A.device)
    for i in range(S):
        e = int(expert_ids[i])
        if e >= E:
            continue
        out[i] = finegrained_fp8.matmul_2d(A[i : i + 1], B[e], Bs[e], block_size)
    return out


# ── unified wrapper correctness/shape ──────────────────────────────────────────
def _assert_correctness(out, ref, expert_ids, problem):
    """Shape, dtype, and per-local-row value checks. Sentinel rows
    (``expert_ids >= problem.E``) are uninit by kernel design and excluded."""
    assert out.shape == (problem.S, problem.N)
    assert out.dtype == problem.dtype
    local_mask = expert_ids.to(torch.int64) < problem.E
    atol, rtol = DTYPE_TO_TOL[problem.dtype]
    torch.testing.assert_close(out[local_mask], ref[local_mask], atol=atol, rtol=rtol)


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="Accelerator not available")
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_batched(problem):
    torch.manual_seed(0)
    A, expert_ids, B, Bs = _setup_problem(problem)
    matmul_batched = maybe_compile(finegrained_fp8.matmul_batched, problem.compile)
    out = matmul_batched(A, B, Bs=Bs, expert_ids=expert_ids)
    ref = _routed_matmul_ref(A, B, Bs, expert_ids, problem.block_size)
    _assert_correctness(out, ref, expert_ids, problem)


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="Accelerator not available")
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_grouped(problem):
    torch.manual_seed(0)
    A, expert_ids, B, Bs = _setup_problem(problem)
    expert_start, gather_idx, scatter_idx = finegrained_fp8.compute_grouped_scheduling(
        expert_ids, problem.E, 1
    )
    A_q, As = _quant_act(A, problem, problem.block_size)
    matmul_grouped = maybe_compile(finegrained_fp8.matmul_grouped, problem.compile)
    out = matmul_grouped(
        A_q,
        B,
        As=As,
        Bs=Bs,
        expert_start=expert_start,
        output_dtype=problem.dtype,
        gather_idx=gather_idx,
        scatter_idx=scatter_idx,
    )
    ref = _routed_matmul_ref(A, B, Bs, expert_ids, problem.block_size)
    _assert_correctness(out, ref, expert_ids, problem)


# ── Fused batched MoE (two-kernel gate_up + down) vs the unfused path ────────────
@dataclass(frozen=True)
class MoEProblem:
    """End-to-end fused-MoE shape: ``num_tokens`` routed ``num_top_k`` ways through
    ``num_experts`` experts, hidden ``hidden_dim``, per-gate ``intermediate_dim``. The
    recipe is the weight dtype + scale group: ``int8`` (packed E2M1) is MXFP4 and ``e4m3``
    with a 1x32 group is MXFP8 (both UE8M0); ``e4m3`` with a square ``block_size`` is
    block-dynamic FP8."""

    num_tokens: int
    num_experts: int
    hidden_dim: int
    intermediate_dim: int
    num_top_k: int
    sentinel_fraction: float = 0.0
    dtype: torch.dtype = torch.bfloat16
    weight_dtype: torch.dtype = torch.float8_e4m3fn
    weight_scale_dtype: torch.dtype = torch.float32
    block_size: Optional[Tuple[int, int]] = None
    swiglu_alpha: Optional[float] = None
    swiglu_limit: Optional[float] = None
    act_fn: str = "silu"

    @property
    def is_mxfp(self):
        return self.block_size == (1, MX_SCALE_GROUP_K)

    @property
    def id(self):
        if self.weight_dtype == torch.int8:
            fmt = "mxfp4"
        elif self.is_mxfp:
            fmt = "mxfp8_" + (
                "u8scale" if self.weight_scale_dtype == torch.uint8 else "e8m0scale"
            )
        else:
            fmt = f"fp8_b{self.block_size[0]}x{self.block_size[1]}"
        if self.swiglu_alpha is not None and self.swiglu_limit is not None:
            act = "_swiglu"
        elif self.swiglu_alpha is not None:
            act = "_swiglu_alpha"
        elif self.swiglu_limit is not None:
            act = "_swiglu_limit"
        elif self.act_fn != "silu":
            act = f"_{self.act_fn}"
        else:
            act = ""
        return (
            f"{fmt}_T{self.num_tokens}_E{self.num_experts}_H{self.hidden_dim}"
            f"_I{self.intermediate_dim}_top{self.num_top_k}_{DTYPE_TAG[self.dtype]}{act}"
            f"{'_sentinel' if self.sentinel_fraction > 0 else ''}"
        )


MOE_PROBLEMS = [
    # ── MXFP4 (packed E2M1 + UE8M0 group-32) ──
    MoEProblem(
        num_tokens=1,
        num_experts=8,
        hidden_dim=512,
        intermediate_dim=256,
        num_top_k=8,
        weight_dtype=torch.int8,
        block_size=(1, MX_SCALE_GROUP_K),
    ),
    MoEProblem(
        num_tokens=4,
        num_experts=8,
        hidden_dim=512,
        intermediate_dim=256,
        num_top_k=8,
        weight_dtype=torch.int8,
        block_size=(1, MX_SCALE_GROUP_K),
    ),
    # fp16 activations on the smallest MXFP4 shape
    MoEProblem(
        num_tokens=4,
        num_experts=8,
        hidden_dim=512,
        intermediate_dim=256,
        num_top_k=8,
        weight_dtype=torch.int8,
        block_size=(1, MX_SCALE_GROUP_K),
        dtype=torch.float16,
    ),
    # ── MXFP8 (E4M3 + UE8M0 group-32) ──
    MoEProblem(
        num_tokens=1,
        num_experts=8,
        hidden_dim=512,
        intermediate_dim=256,
        num_top_k=8,
        block_size=(1, MX_SCALE_GROUP_K),
        weight_scale_dtype=torch.float8_e8m0fnu,
    ),
    MoEProblem(
        num_tokens=4,
        num_experts=8,
        hidden_dim=512,
        intermediate_dim=256,
        num_top_k=8,
        block_size=(1, MX_SCALE_GROUP_K),
        weight_scale_dtype=torch.float8_e8m0fnu,
    ),
    # UE8M0 scales stored as raw uint8 (e.g. MiniMax-M3-MXFP8 checkpoints) — must still
    # detect as MXFP8 and route to the MX path, not fall back to block-dynamic.
    MoEProblem(
        num_tokens=4,
        num_experts=8,
        hidden_dim=512,
        intermediate_dim=256,
        num_top_k=8,
        block_size=(1, MX_SCALE_GROUP_K),
        weight_scale_dtype=torch.uint8,
    ),
    # ── Block-dynamic FP8 (E4M3 + fp32 128x128 block scales) ──
    MoEProblem(
        num_tokens=1,
        num_experts=8,
        hidden_dim=512,
        intermediate_dim=256,
        num_top_k=8,
        block_size=(128, 128),
    ),
    MoEProblem(
        num_tokens=4,
        num_experts=8,
        hidden_dim=512,
        intermediate_dim=256,
        num_top_k=8,
        block_size=(128, 128),
    ),
    # ── Clamped/scaled SwiGLU (GPT-OSS / MiniMax-M3), MXFP8 (glu is recipe-independent) ──
    MoEProblem(
        num_tokens=4,
        num_experts=8,
        hidden_dim=512,
        intermediate_dim=256,
        num_top_k=8,
        block_size=(1, MX_SCALE_GROUP_K),
        weight_scale_dtype=torch.float8_e8m0fnu,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
    ),
    # alpha / limit are independent glu branches — cover each alone
    MoEProblem(
        num_tokens=4,
        num_experts=8,
        hidden_dim=512,
        intermediate_dim=256,
        num_top_k=8,
        block_size=(1, MX_SCALE_GROUP_K),
        weight_scale_dtype=torch.float8_e8m0fnu,
        swiglu_alpha=1.702,
    ),
    MoEProblem(
        num_tokens=4,
        num_experts=8,
        hidden_dim=512,
        intermediate_dim=256,
        num_top_k=8,
        block_size=(1, MX_SCALE_GROUP_K),
        weight_scale_dtype=torch.float8_e8m0fnu,
        swiglu_limit=7.0,
    ),
    # ── GeGLU / ReGLU coverage (activation is orthogonal to recipe, so one MXFP8 shape each) ──
    MoEProblem(
        num_tokens=4,
        num_experts=8,
        hidden_dim=512,
        intermediate_dim=256,
        num_top_k=8,
        block_size=(1, MX_SCALE_GROUP_K),
        weight_scale_dtype=torch.float8_e8m0fnu,
        act_fn="gelu",
    ),
    MoEProblem(
        num_tokens=4,
        num_experts=8,
        hidden_dim=512,
        intermediate_dim=256,
        num_top_k=8,
        block_size=(1, MX_SCALE_GROUP_K),
        weight_scale_dtype=torch.float8_e8m0fnu,
        act_fn="relu",
    ),
    # ── Expert parallelism: non-local experts sentinel-masked (routing is orthogonal to recipe) ──
    # MXFP8 (MiniMax-M3) + block-dynamic FP8, exercised on both the grouped and batched fused paths.
    MoEProblem(
        num_tokens=8,
        num_experts=8,
        hidden_dim=512,
        intermediate_dim=256,
        num_top_k=8,
        block_size=(1, MX_SCALE_GROUP_K),
        weight_scale_dtype=torch.float8_e8m0fnu,
        sentinel_fraction=0.875,
    ),
    MoEProblem(
        num_tokens=8,
        num_experts=8,
        hidden_dim=512,
        intermediate_dim=256,
        num_top_k=8,
        block_size=(128, 128),
        sentinel_fraction=0.875,
    ),
    # int32 pointer-offset overflow guard for the fused paths: the last experts'
    # gate_up offsets exceed 2^31 elements (127 * 2*2048 * 6144 = 3.196e9); a regressed
    # int64 cast corrupts the high-routed tokens vs the torch reference. E is a power of
    # two (the fused-grouped scheduling kernels require it).
    MoEProblem(
        num_tokens=512,
        num_experts=128,
        hidden_dim=6144,
        intermediate_dim=2048,
        num_top_k=4,
        block_size=(128, 128),
    ),
]


def _make_moe_weights(problem: MoEProblem):
    """gate_up ``(E, 2I, H)`` and down ``(E, H, I)`` weights + inv-scales for the recipe."""
    bs = list(problem.block_size)
    kw = dict(
        weight_dtype=problem.weight_dtype,
        scale_dtype=problem.weight_scale_dtype,  # forced to UE8M0 internally for int8
        num_experts=problem.num_experts,
    )
    gate_up, gate_up_s = make_weights(
        2 * problem.intermediate_dim, problem.hidden_dim, TEST_DEVICE, bs, **kw
    )
    down, down_s = make_weights(
        problem.hidden_dim, problem.intermediate_dim, TEST_DEVICE, bs, **kw
    )
    return gate_up, gate_up_s, down, down_s, bs


def _make_moe_inputs(problem: MoEProblem):
    """Random ``(hidden, top_k_index, top_k_weights)`` for the fused-MoE problem shape."""
    hidden = torch.randn(
        problem.num_tokens, problem.hidden_dim, device=TEST_DEVICE, dtype=problem.dtype
    )
    top_k_index = torch.randint(
        0,
        problem.num_experts,
        (problem.num_tokens, problem.num_top_k),
        device=TEST_DEVICE,
        dtype=torch.int32,
    )
    if problem.sentinel_fraction > 0:
        # EP: mark a random subset of routed slots non-local with an out-of-range id (== num_experts),
        # which the fused path must skip. Mirrors _make_routed_inputs.
        flat = top_k_index.reshape(-1)
        n_sentinel = int(round(flat.numel() * problem.sentinel_fraction))
        idx = torch.randperm(flat.numel(), device=flat.device)[:n_sentinel]
        flat[idx] = problem.num_experts
    top_k_weights = torch.rand(
        problem.num_tokens, problem.num_top_k, device=TEST_DEVICE, dtype=problem.dtype
    )
    return hidden, top_k_index, top_k_weights


def _quant_act(x, problem, block_size):
    """Activation quant matching the recipe the grouped GEMM will consume — the same quant
    ``matmul_grouped`` used to do internally, now caller-side. MX → UE8M0 group-32; tensor-wide
    (no block_size) → per-token; block-dynamic → per-``block_k`` blocks."""
    if problem.is_mxfp:
        return mxfp_act_quant(x)
    if block_size is None:
        return fp8_act_quant_tensor_wide(x, x.shape[-1])
    return fp8_act_quant_block_dynamic(x, block_size[1])


def _assert_fused_correctness(out, ref, problem: MoEProblem):
    """Shape, dtype, and value checks against the unfused reference."""
    assert out.shape == (problem.num_tokens, problem.hidden_dim)
    assert out.dtype == problem.dtype
    atol, rtol = DTYPE_TO_TOL[problem.dtype]
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="Accelerator not available")
@pytest.mark.parametrize("problem", MOE_PROBLEMS, ids=lambda p: p.id)
def test_fused_batched(problem):
    """Fused two-kernel MoE (gate_up + activation + FP8 requant + down + top-k reduce) via the
    ``moe_fused_batched`` dispatcher vs the unfused reference. ``simulate_unfused`` rounds each
    fused step through the activation dtype so the two agree to reduce order. The activation
    (``act_fn`` / clamped SwiGLU) is a per-problem field, not a separate axis."""
    torch.manual_seed(0)
    gate_up, gate_up_s, down, down_s, block_size = _make_moe_weights(problem)
    hidden, top_k_index, top_k_weights = _make_moe_inputs(problem)
    ref = moe.moe_unfused_batched(
        hidden,
        top_k_index,
        top_k_weights,
        gate_up,
        down,
        gate_up_s,
        down_s,
        act_fn=problem.act_fn,
        swiglu_alpha=problem.swiglu_alpha,
        swiglu_limit=problem.swiglu_limit,
    )
    out = moe.moe_fused_batched(
        hidden,
        top_k_index,
        top_k_weights,
        gate_up,
        down,
        gate_up_s,
        down_s,
        act_fn=problem.act_fn,
        swiglu_alpha=problem.swiglu_alpha,
        swiglu_limit=problem.swiglu_limit,
        simulate_unfused=True,
    )
    _assert_fused_correctness(out, ref, problem)


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="Accelerator not available")
@pytest.mark.parametrize("problem", MOE_PROBLEMS, ids=lambda p: p.id)
def test_fused_grouped(problem):
    """Fused grouped MoE (gather gate_up + activation + FP8 requant + grouped down + top-k
    reduce) via the ``moe_fused_grouped`` dispatcher vs the same unfused reference, with
    ``simulate_unfused`` rounding each fused step through the activation dtype. The activation
    (``act_fn`` / clamped SwiGLU) is a per-problem field, not a separate axis."""
    torch.manual_seed(0)
    gate_up, gate_up_s, down, down_s, block_size = _make_moe_weights(problem)
    hidden, top_k_index, top_k_weights = _make_moe_inputs(problem)
    ref = moe.moe_unfused_grouped(
        hidden,
        top_k_index,
        top_k_weights,
        gate_up,
        down,
        gate_up_s,
        down_s,
        act_fn=problem.act_fn,
        swiglu_alpha=problem.swiglu_alpha,
        swiglu_limit=problem.swiglu_limit,
    )
    out = moe.moe_fused_grouped(
        hidden,
        top_k_index,
        top_k_weights,
        gate_up,
        down,
        gate_up_s,
        down_s,
        act_fn=problem.act_fn,
        swiglu_alpha=problem.swiglu_alpha,
        swiglu_limit=problem.swiglu_limit,
        simulate_unfused=True,
    )
    _assert_fused_correctness(out, ref, problem)


def _bench_setup(problem: Problem):
    accelerator_module().empty_cache()
    torch.compiler.reset()
    torch.manual_seed(0)
    A, expert_ids, B, Bs = _setup_problem(problem)
    return A, B, Bs, expert_ids


# ── Benchmarks ────────────────────────────────────────────────────────────────
def _assert_latency_with_tolerance(measured_ms: float, expected_ms: float):
    lower = expected_ms * 0.85
    upper = expected_ms * 1.15
    if measured_ms < lower:
        raise AssertionError(
            "latency "
            f"{measured_ms:.4f}ms is faster than expected range [{lower:.4f}ms, {upper:.4f}ms]. "
            f"Update baseline expected_ms from {expected_ms:.4f}ms to {measured_ms:.4f}ms if this speedup is intended."
        )
    if measured_ms > upper:
        raise AssertionError(
            f"latency {measured_ms:.4f}ms is slower than expected range [{lower:.4f}ms, {upper:.4f}ms]."
        )


def _run_speedup(problem, label, call, expected_ms):
    """Median-of-``BENCH_REPEATS`` ``do_bench`` latency for ``call``, printed with
    spread, then asserted within the ±15% SM90 baseline band of ``expected_ms``."""
    runs_ms = [triton.testing.do_bench(call) for _ in range(BENCH_REPEATS)]
    median_ms = statistics.median(runs_ms)
    print(
        f"\n[{label}] S={problem.S:4d} E={problem.E:4d} N={problem.N:5d} K={problem.K:5d} | "
        f"{label} median={median_ms:.4f}ms mean={statistics.mean(runs_ms):.4f}ms "
        f"min={min(runs_ms):.4f}ms max={max(runs_ms):.4f}ms "
        f"(expected {expected_ms:.4f}ms ±15%) repeats={BENCH_REPEATS}"
    )
    _assert_latency_with_tolerance(median_ms, expected_ms)


@pytest.mark.benchmark
@pytest.mark.skipif(
    not IS_SM90, reason="Latency baselines are calibrated for SM90 (H100) only"
)
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_batched_speedup(problem):
    if problem.expectation is None:
        pytest.skip("No expected benchmark latency for this problem")
    A, B, Bs, expert_ids = _bench_setup(problem)
    _run_speedup(
        problem,
        "batched",
        lambda: finegrained_fp8.matmul_batched(
            A, B, Bs=Bs, expert_ids=expert_ids
        ),
        problem.expectation.batched_ms,
    )


@pytest.mark.benchmark
@pytest.mark.skipif(
    not IS_SM90, reason="Latency baselines are calibrated for SM90 (H100) only"
)
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_grouped_speedup(problem):
    if problem.expectation is None:
        pytest.skip("No expected benchmark latency for this problem")
    A, B, Bs, expert_ids = _bench_setup(problem)
    expert_start, gather_idx, scatter_idx = finegrained_fp8.compute_grouped_scheduling(
        expert_ids, problem.E, 1
    )
    A_q, As = _quant_act(A, problem, problem.block_size)
    _run_speedup(
        problem,
        "grouped",
        lambda: finegrained_fp8.matmul_grouped(
            A_q,
            B,
            As=As,
            Bs=Bs,
            expert_start=expert_start,
            output_dtype=problem.dtype,
            gather_idx=gather_idx,
            scatter_idx=scatter_idx,
        ),
        problem.expectation.grouped_ms,
    )
