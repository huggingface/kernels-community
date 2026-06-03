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
    SUPPORTS_FP4,
    TEST_DEVICE,
    accelerator_module,
    make_fp4_weights,
    make_fp8_weights,
)

import finegrained_fp8  # type: ignore


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
    block_size: Optional[Tuple[int, int]] = None
    expectation: Optional[Expectations] = None
    dtype: torch.dtype = torch.bfloat16
    sentinel_fraction: float = 0.0
    weight_format: str = "fp8"
    contiguous: bool = True

    @property
    def id(self):
        head = (
            f"{self.weight_format}_S{self.S}_E{self.E}_N{self.N}_K{self.K}"
            f"_top{self.TOP_K}"
        )
        # FP4 ignores block_size (tile shape autotuned); always block-mode.
        if self.weight_format == "fp4":
            tail = head
        # FP8 with block_size carries block dims; without it, scale_layout names
        # the tensor-mode variant.
        elif self.block_size is None:
            tail = f"{head}_{self.scale_layout}"
        else:
            tail = f"{head}_b{self.block_size[0]}x{self.block_size[1]}"
        contig_tag = "contiguous" if self.contiguous else "noncontiguous"
        tail = f"{tail}_{contig_tag}_{DTYPE_TAG[self.dtype]}"
        if self.sentinel_fraction > 0:
            tail = f"{tail}_sentinel"
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
    # Non-contig indexing tensors — exercises stride-aware loads for
    # expert_ids (batched) and offsets / tokens_per_expert (grouped).
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
]
if SUPPORTS_FP4:
    # DeepSeek-V4-Flash FP4 shapes (E=256, top_k=6); first entry kept small so
    # CI doesn't spend minutes on the larger problems. ``block_size`` is ignored
    # by the FP4 path (tile shape autotuned).
    PROBLEMS += [
        Problem(
            S=32,
            E=4,
            N=256,
            K=512,
            TOP_K=2,
            scale_layout="block",
            block_size=None,
            weight_format="fp4",
        ),
        Problem(
            S=32,
            E=4,
            N=256,
            K=512,
            TOP_K=2,
            scale_layout="block",
            block_size=None,
            weight_format="fp4",
            contiguous=False,
        ),
        Problem(
            S=192,
            E=256,
            N=4096,
            K=4096,
            TOP_K=6,
            scale_layout="block",
            block_size=None,
            weight_format="fp4",
        ),
        Problem(
            S=192,
            E=256,
            N=4096,
            K=2048,
            TOP_K=6,
            scale_layout="block",
            block_size=None,
            weight_format="fp4",
        ),
        Problem(
            S=1536,
            E=256,
            N=4096,
            K=4096,
            TOP_K=6,
            scale_layout="block",
            block_size=None,
            weight_format="fp4",
        ),
        # EP sentinel coverage on FP4
        Problem(
            S=64,
            E=8,
            N=256,
            K=512,
            TOP_K=4,
            scale_layout="block",
            block_size=None,
            weight_format="fp4",
            sentinel_fraction=0.875,
        ),
        # fp16 / fp32 dtype coverage on the smallest FP4 shape
        Problem(
            S=32,
            E=4,
            N=256,
            K=512,
            TOP_K=2,
            scale_layout="block",
            block_size=None,
            weight_format="fp4",
            dtype=torch.float16,
        ),
        Problem(
            S=32,
            E=4,
            N=256,
            K=512,
            TOP_K=2,
            scale_layout="block",
            block_size=None,
            weight_format="fp4",
            dtype=torch.float32,
        ),
    ]

COMPILE_PROBLEMS = [
    Problem(
        S=8,
        E=4,
        N=256,
        K=256,
        TOP_K=1,
        scale_layout="block",
        block_size=(128, 128),
    ),
]
if SUPPORTS_FP4:
    COMPILE_PROBLEMS += [
        Problem(
            S=32,
            E=4,
            N=256,
            K=512,
            TOP_K=2,
            scale_layout="block",
            block_size=None,
            weight_format="fp4",
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


def _convert_scale_layout(Bs, layout: str):
    if layout == "block":
        assert Bs.ndim == 3, "block scale layout expects Bs with shape [E, nb, kb]"
        return Bs

    if Bs.ndim == 1:
        per_tensor = Bs.contiguous()
    else:
        per_tensor = Bs[:, 0, 0].contiguous()

    if layout == "per_tensor_1d":
        return per_tensor
    if layout == "per_tensor_111":
        return per_tensor.view(-1, 1, 1).contiguous()
    raise ValueError(f"Unsupported scale layout: {layout}")


def _make_noncontig_1d(x: torch.Tensor) -> torch.Tensor:
    # Stride-2 view with duplicated gaps: a stride-1 read returns
    # [x[0], x[0], x[1], x[1], ...] instead of x, catching wrong-stride loads.
    base = torch.empty((x.numel(), 2), dtype=x.dtype, device=x.device)
    base[:, 0] = x
    base[:, 1] = x
    return base[:, 0]


def _setup_problem(problem: Problem, dtype):
    A, expert_ids = _make_routed_inputs(
        problem.S,
        problem.E,
        problem.K,
        dtype=dtype,
        device=TEST_DEVICE,
        top_k=problem.TOP_K,
        sentinel_fraction=problem.sentinel_fraction,
    )
    if problem.weight_format == "fp4":
        B, Bs = make_fp4_weights(
            problem.N, problem.K, TEST_DEVICE, num_experts=problem.E
        )
    else:
        B_fp8, Bs_block = make_fp8_weights(
            problem.N, problem.K, TEST_DEVICE, problem.block_size, num_experts=problem.E
        )
        Bs = _convert_scale_layout(Bs_block, problem.scale_layout)
        B = B_fp8
    if not problem.contiguous:
        expert_ids = _make_noncontig_1d(expert_ids)
    return A, expert_ids, B, Bs


def _prepare_grouped(A, expert_ids, num_experts, contiguous: bool = True):
    perm = torch.argsort(expert_ids)
    A_sorted = A[perm].contiguous()
    expert_ids_sorted = expert_ids[perm]
    tokens_per_expert = torch.histc(
        expert_ids_sorted.float(), bins=num_experts, min=0, max=num_experts - 1
    ).to(torch.int32)
    offsets = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)
    if not contiguous:
        offsets = _make_noncontig_1d(offsets)
        tokens_per_expert = _make_noncontig_1d(tokens_per_expert)
    return A_sorted, expert_ids_sorted, offsets, tokens_per_expert


def _routed_matmul_ref(A, B, Bs, expert_ids, block_size):
    """Per-routed-row reference that re-uses the neutral ``matmul`` dispatcher
    (routes to FP4 when ``B.dtype == int8``, else block/tensor FP8). Output dtype
    matches the batched/grouped kernels (both follow ``A.dtype``). Rows with
    ``expert_ids[i] >= num_experts`` are EP sentinels — skipped, output undefined."""
    S = A.shape[0]
    N = B.shape[1]
    E = B.shape[0]
    out = torch.empty(S, N, dtype=A.dtype, device=A.device)
    for i in range(S):
        e = int(expert_ids[i])
        if e >= E:
            continue
        out[i] = finegrained_fp8.matmul(A[i : i + 1], B[e], Bs[e], block_size)
    return out


# ── unified wrapper correctness/shape ──────────────────────────────────────────
def _assert_correctness(out, ref, expert_ids, problem):
    """Shape, dtype, and per-local-row value checks. Sentinel rows
    (``expert_ids >= problem.E``) are uninit by kernel design and excluded."""
    assert out.shape == (problem.S, problem.N)
    assert out.dtype == problem.dtype
    local_mask = expert_ids.to(torch.int64) < problem.E
    atol, rtol = DTYPE_TO_TOL[problem.dtype]
    torch.testing.assert_close(
        out[local_mask], ref[local_mask], atol=atol, rtol=rtol
    )


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="Accelerator not available")
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_batched(problem):
    torch.manual_seed(0)
    A, expert_ids, B, Bs = _setup_problem(problem, dtype=problem.dtype)
    out = finegrained_fp8.matmul_batched(A, B, Bs, expert_ids, problem.block_size)
    ref = _routed_matmul_ref(A, B, Bs, expert_ids, problem.block_size)
    _assert_correctness(out, ref, expert_ids, problem)


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="Accelerator not available")
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_grouped(problem):
    torch.manual_seed(0)
    A, expert_ids, B, Bs = _setup_problem(problem, dtype=problem.dtype)
    A_sorted, expert_ids_sorted, offsets, tokens_per_expert = _prepare_grouped(
        A, expert_ids, problem.E, contiguous=problem.contiguous
    )
    out = finegrained_fp8.matmul_grouped(
        A_sorted, B, Bs, offsets, tokens_per_expert, problem.block_size
    )
    ref = _routed_matmul_ref(A_sorted, B, Bs, expert_ids_sorted, problem.block_size)
    _assert_correctness(out, ref, expert_ids_sorted, problem)


# ── torch.compile compatibility ────────────────────────────────────────────────
@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="Accelerator not available")
@pytest.mark.parametrize("problem", COMPILE_PROBLEMS, ids=lambda p: p.id)
def test_batched_compile(problem):
    torch.manual_seed(0)
    torch.compiler.reset()
    accelerator_module().empty_cache()
    A, expert_ids, B, Bs = _setup_problem(problem, dtype=torch.bfloat16)

    def fn(A, B, Bs, expert_ids):
        return finegrained_fp8.matmul_batched(A, B, Bs, expert_ids, problem.block_size)

    compiled = torch.compile(fn, mode="max-autotune", fullgraph=True)
    out_compiled = compiled(A, B, Bs, expert_ids)
    out_ref = fn(A, B, Bs, expert_ids)
    torch.testing.assert_close(out_compiled, out_ref)


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="Accelerator not available")
@pytest.mark.parametrize("problem", COMPILE_PROBLEMS, ids=lambda p: p.id)
def test_grouped_compile(problem):
    torch.manual_seed(0)
    torch.compiler.reset()
    accelerator_module().empty_cache()
    A, expert_ids, B, Bs = _setup_problem(problem, dtype=torch.bfloat16)
    A_sorted, _, offsets, tokens_per_expert = _prepare_grouped(A, expert_ids, problem.E)

    def fn(A_sorted, B, Bs, offsets, tokens_per_expert):
        return finegrained_fp8.matmul_grouped(
            A_sorted,
            B,
            Bs,
            offsets,
            tokens_per_expert,
            problem.block_size,
        )

    compiled = torch.compile(fn, mode="max-autotune", fullgraph=True)
    out_compiled = compiled(A_sorted, B, Bs, offsets, tokens_per_expert)
    out_ref = fn(A_sorted, B, Bs, offsets, tokens_per_expert)
    torch.testing.assert_close(out_compiled, out_ref)


def _bench_setup(problem: Problem, device=TEST_DEVICE):
    accelerator_module().empty_cache()
    torch.compiler.reset()
    torch.manual_seed(0)
    A, expert_ids = _make_routed_inputs(
        problem.S,
        problem.E,
        problem.K,
        dtype=torch.bfloat16,
        device=device,
        top_k=problem.TOP_K,
    )
    B_fp8, Bs = make_fp8_weights(
        problem.N, problem.K, device, problem.block_size, num_experts=problem.E
    )
    A_sorted, expert_ids_sorted, offsets, tokens_per_expert = _prepare_grouped(
        A, expert_ids, problem.E
    )
    return A_sorted, B_fp8, Bs, expert_ids_sorted, offsets, tokens_per_expert


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


def _measure_latency_over_repeats(fn, repeats: int = BENCH_REPEATS):
    runs_ms = [triton.testing.do_bench(fn) for _ in range(repeats)]
    median_ms = statistics.median(runs_ms)
    mean_ms = statistics.mean(runs_ms)
    min_ms = min(runs_ms)
    max_ms = max(runs_ms)
    return median_ms, mean_ms, min_ms, max_ms


@pytest.mark.benchmark
@pytest.mark.skipif(
    not IS_SM90, reason="Latency baselines are calibrated for SM90 (H100) only"
)
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_batched_speedup(problem):
    if problem.expectation is None:
        pytest.skip("No expected benchmark latency for this problem")

    A, B_fp8, Bs, expert_ids, _, _ = _bench_setup(problem)

    batched_ms, batched_mean_ms, batched_min_ms, batched_max_ms = (
        _measure_latency_over_repeats(
            lambda: finegrained_fp8.matmul_batched(
                A, B_fp8, Bs, expert_ids, problem.block_size
            )
        )
    )
    expected_ms = problem.expectation.batched_ms
    print(
        f"\n[batched] S={problem.S:4d} E={problem.E:4d} N={problem.N:5d} K={problem.K:5d} | "
        f"batched median={batched_ms:.4f}ms mean={batched_mean_ms:.4f}ms "
        f"min={batched_min_ms:.4f}ms max={batched_max_ms:.4f}ms "
        f"(expected {expected_ms:.4f}ms ±10%) repeats={BENCH_REPEATS}"
    )
    _assert_latency_with_tolerance(batched_ms, expected_ms)


@pytest.mark.benchmark
@pytest.mark.skipif(
    not IS_SM90, reason="Latency baselines are calibrated for SM90 (H100) only"
)
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_grouped_speedup(problem):
    if problem.expectation is None:
        pytest.skip("No expected benchmark latency for this problem")

    A, B_fp8, Bs, _, offsets, tokens_per_expert = _bench_setup(problem)

    grouped_ms, grouped_mean_ms, grouped_min_ms, grouped_max_ms = (
        _measure_latency_over_repeats(
            lambda: finegrained_fp8.matmul_grouped(
                A, B_fp8, Bs, offsets, tokens_per_expert, problem.block_size
            )
        )
    )
    expected_ms = problem.expectation.grouped_ms
    print(
        f"\n[grouped] S={problem.S:4d} E={problem.E:4d} N={problem.N:5d} K={problem.K:5d} | "
        f"grouped median={grouped_ms:.4f}ms mean={grouped_mean_ms:.4f}ms "
        f"min={grouped_min_ms:.4f}ms max={grouped_max_ms:.4f}ms "
        f"(expected {expected_ms:.4f}ms ±10%) repeats={BENCH_REPEATS}"
    )
    _assert_latency_with_tolerance(grouped_ms, expected_ms)
