"""Tests for MoE expert dispatch kernels: batched and grouped."""

import statistics
from dataclasses import dataclass
from typing import Optional, Tuple

import pytest
import torch
import triton

import finegrained_fp8  # type: ignore
import finegrained_fp8.fp4 as finegrained_fp4  # type: ignore


FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
FP8_DTYPE = torch.float8_e4m3fn
BENCH_REPEATS = 10
TEST_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else None
)


def _accelerator_available():
    return TEST_DEVICE is not None


def _accelerator_module():
    if TEST_DEVICE == "cuda":
        return torch.cuda
    if TEST_DEVICE == "xpu":
        return torch.xpu
    raise RuntimeError("No supported accelerator available for finegrained-fp8 tests")


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

    @property
    def id(self):
        bsz = (
            "NxK"
            if self.block_size is None
            else f"{self.block_size[0]}x{self.block_size[1]}"
        )
        return (
            f"S{self.S}_E{self.E}_N{self.N}_K{self.K}_T{self.TOP_K}"
            f"__{self.scale_layout}__bsz{bsz}"
        )


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
]

COMPILE_PROBLEM = PROBLEMS[0]


def _make_experts_weights(num_experts, out_features, in_features, block_size, device):
    """Create FP8 expert weights/scales with FP8Experts-compatible layouts.

    Returns:
        weights_fp8: [E, N, K] where E=num_experts, N=out_features, K=in_features
        scales_inv: [E, N // block_n, K // block_k] for block mode,
                    [E, 1, 1] for per-tensor mode (block_size=None)
    """
    W = torch.randn(
        num_experts, out_features, in_features, dtype=torch.float32, device=device
    )
    E, N, K = W.shape

    if block_size is None:
        block_n, block_k = N, K
    else:
        block_n, block_k = block_size

    assert N % block_n == 0, f"N ({N}) must be divisible by block_n ({block_n})"
    assert K % block_k == 0, f"K ({K}) must be divisible by block_k ({block_k})"

    rt, ct = N // block_n, K // block_k
    R = W.reshape(E, rt, block_n, ct, block_k)

    max_abs = R.abs().amax(dim=(-3, -1))
    safe = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    scale = FP8_MAX / safe

    Wq = (R * scale.unsqueeze(-1).unsqueeze(-3)).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
    Wq = Wq.reshape(E, N, K).contiguous()
    inv_scales = (1.0 / scale).to(torch.float32)

    return Wq, inv_scales


def _make_experts_weights_fp4(num_experts, out_features, in_features, device):
    assert in_features % 32 == 0, f"K ({in_features}) must be divisible by 32 for FP4"
    packed_k = in_features // 2
    weights = torch.randint(
        -8, 8, (num_experts, out_features, packed_k), dtype=torch.int8, device=device
    )
    scales = torch.ones(
        (num_experts, out_features, in_features // 32),
        dtype=torch.float8_e8m0fnu,
        device=device,
    )
    return weights.contiguous(), scales.contiguous()


def _make_routed_inputs(S, E, K, dtype, device, top_k):
    """Build flattened routed inputs like FP8Experts paths."""
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


def _setup_problem(problem: Problem, dtype):
    A, expert_ids = _make_routed_inputs(
        problem.S,
        problem.E,
        problem.K,
        dtype=dtype,
        device=TEST_DEVICE,
        top_k=problem.TOP_K,
    )
    B_fp8, Bs_block = _make_experts_weights(
        problem.E,
        problem.N,
        problem.K,
        problem.block_size,
        TEST_DEVICE,
    )
    Bs = _convert_scale_layout(Bs_block, problem.scale_layout)
    return A, expert_ids, B_fp8, Bs


def _make_noncontiguous_1d(x: torch.Tensor) -> torch.Tensor:
    base = torch.empty((x.numel(), 2), dtype=x.dtype, device=x.device)
    base[:, 0] = x
    base[:, 1] = x
    return base[:, 0]


def _make_noncontiguous_bs(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return _make_noncontiguous_1d(x)

    assert x.ndim == 3
    base = torch.empty(
        (*x.shape[:-1], x.shape[-1] * 2),
        dtype=x.dtype,
        device=x.device,
    )
    base[..., ::2] = x
    base[..., 1::2] = x
    return base[..., ::2]


def _prepare_grouped(A, expert_ids, num_experts):
    perm = torch.argsort(expert_ids)
    A_sorted = A[perm].contiguous()
    expert_ids_sorted = expert_ids[perm]
    tokens_per_expert = torch.histc(
        expert_ids_sorted.float(), bins=num_experts, min=0, max=num_experts - 1
    ).to(torch.int32)
    offsets = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)
    return A_sorted, expert_ids_sorted, offsets, tokens_per_expert


def _ref_fp4_grouped(qA, As, B, Bs, offsets, output_dtype=torch.float32):
    S = qA.shape[0]
    out = torch.empty(S, B.shape[1], dtype=output_dtype, device=qA.device)
    start = 0
    for expert_id in range(B.shape[0]):
        end = int(offsets[expert_id].item())
        if end > start:
            out[start:end] = finegrained_fp4.w4a8_block_fp8_matmul(
                qA[start:end],
                As[start:end],
                B[expert_id],
                Bs[expert_id],
                [B.shape[1], qA.shape[1]],
                output_dtype,
            )
        start = end
    return out


def _ref_fp4_grouped_block(qA, As, B, Bs, offsets, block_size, output_dtype=torch.float32):
    S = qA.shape[0]
    out = torch.empty(S, B.shape[1], dtype=output_dtype, device=qA.device)
    start = 0
    for expert_id in range(B.shape[0]):
        end = int(offsets[expert_id].item())
        if end > start:
            out[start:end] = finegrained_fp4.w4a8_block_fp8_matmul(
                qA[start:end],
                As[start:end],
                B[expert_id],
                Bs[expert_id],
                block_size,
                output_dtype,
            )
        start = end
    return out


def _ref_fp4_batched(qA, As, B, Bs, expert_ids, block_size, output_dtype=torch.float32):
    out = torch.empty(qA.shape[0], B.shape[1], dtype=output_dtype, device=qA.device)
    for idx in range(qA.shape[0]):
        expert_id = int(expert_ids[idx].item())
        out[idx] = finegrained_fp4.w4a8_block_fp8_matmul(
            qA[idx : idx + 1],
            As[idx : idx + 1],
            B[expert_id],
            Bs[expert_id],
            block_size,
            output_dtype,
        )
    return out


def _ref(A, B_fp8, Bs, expert_ids, block_size):
    S = A.shape[0]
    N = B_fp8.shape[1]
    if block_size is None:
        bi = A.shape[-1]
        matmul_block_size = None
    else:
        bi = block_size[1]
        matmul_block_size = block_size

    out = torch.empty(S, N, dtype=torch.float32, device=A.device)
    for i in range(S):
        e = expert_ids[i]
        qA_i, sA_i = finegrained_fp8.fp8_act_quant(A[i : i + 1], bi)
        out[i] = finegrained_fp8.w8a8_fp8_matmul(
            qA_i, B_fp8[e], sA_i, Bs[e], matmul_block_size
        )
    return out


# ── unified wrapper correctness/shape ──────────────────────────────────────────
@pytest.mark.kernels_ci
@pytest.mark.skipif(not _accelerator_available(), reason="Accelerator not available")
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_batched_vs_ref(problem):
    torch.manual_seed(0)
    A, expert_ids, B_fp8, Bs = _setup_problem(problem, dtype=torch.float32)
    out = finegrained_fp8.w8a8_fp8_matmul_batched(
        A, B_fp8, Bs, expert_ids, problem.block_size
    )
    ref = _ref(A, B_fp8, Bs, expert_ids, problem.block_size)
    torch.testing.assert_close(out, ref)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not _accelerator_available(), reason="Accelerator not available")
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_batched_output_shape(problem):
    A, expert_ids, B_fp8, Bs = _setup_problem(problem, dtype=torch.bfloat16)
    out = finegrained_fp8.w8a8_fp8_matmul_batched(
        A, B_fp8, Bs, expert_ids, problem.block_size
    )
    assert out.shape == (problem.S, problem.N)
    assert out.dtype == torch.bfloat16


@pytest.mark.kernels_ci
@pytest.mark.skipif(not _accelerator_available(), reason="Accelerator not available")
def test_w4a8_batched_noncontiguous_bs_and_expert_ids():
    torch.manual_seed(0)
    S, E, N, K = 8, 4, 256, 256
    A = torch.randn(S, K, dtype=torch.bfloat16, device=TEST_DEVICE)
    B = torch.randint(-8, 8, (E, N, K // 2), dtype=torch.int8, device=TEST_DEVICE)
    Bs = torch.ones((E, N, K // 32), dtype=torch.float8_e8m0fnu, device=TEST_DEVICE)
    expert_ids = torch.randint(0, E, (S,), dtype=torch.int32, device=TEST_DEVICE)

    out = finegrained_fp8.fp8_matmul_batched(
        A,
        B,
        _make_noncontiguous_bs(Bs),
        _make_noncontiguous_1d(expert_ids),
        [128, 128],
    )
    ref = finegrained_fp8.fp8_matmul_batched(A, B, Bs, expert_ids, [128, 128])
    torch.testing.assert_close(out, ref)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not _accelerator_available(), reason="Accelerator not available")
def test_w4a8_batched_tensor_vs_ref():
    torch.manual_seed(0)
    S, E, N, K = 32, 4, 256, 512
    A, expert_ids = _make_routed_inputs(
        S,
        E,
        K,
        dtype=torch.bfloat16,
        device=TEST_DEVICE,
        top_k=1,
    )
    B, Bs = _make_experts_weights_fp4(E, N, K, TEST_DEVICE)
    qA, As = finegrained_fp8.fp8_act_quant(A, K)

    out = finegrained_fp4.w4a8_block_fp8_matmul_batched(
        qA, As, B, Bs, expert_ids, [N, K], torch.float32
    )
    ref = _ref_fp4_batched(qA, As, B, Bs, expert_ids, [N, K], torch.float32)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not _accelerator_available(), reason="Accelerator not available")
def test_w4a8_batched_block_vs_ref():
    torch.manual_seed(0)
    S, E, N, K = 32, 4, 256, 512
    A, expert_ids = _make_routed_inputs(
        S,
        E,
        K,
        dtype=torch.bfloat16,
        device=TEST_DEVICE,
        top_k=1,
    )
    B, Bs = _make_experts_weights_fp4(E, N, K, TEST_DEVICE)
    qA, As = finegrained_fp8.fp8_act_quant(A, 128)

    out = finegrained_fp4.w4a8_block_fp8_matmul_batched(
        qA, As, B, Bs, expert_ids, [128, 128], torch.float32
    )
    ref = _ref_fp4_batched(qA, As, B, Bs, expert_ids, [128, 128], torch.float32)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not _accelerator_available(), reason="Accelerator not available")
def test_w4a8_grouped_tensor_vs_ref():
    torch.manual_seed(0)
    S, E, N, K = 32, 4, 256, 512
    A, expert_ids = _make_routed_inputs(
        S,
        E,
        K,
        dtype=torch.bfloat16,
        device=TEST_DEVICE,
        top_k=1,
    )
    B, Bs = _make_experts_weights_fp4(E, N, K, TEST_DEVICE)
    A_sorted, _, offsets, tokens_per_expert = _prepare_grouped(A, expert_ids, E)
    qA, As = finegrained_fp8.fp8_act_quant(A_sorted, K)

    out = finegrained_fp4.w4a8_block_fp8_matmul_grouped(
        qA,
        As,
        B,
        Bs,
        offsets,
        tokens_per_expert,
        [N, K],
        torch.float32,
    )
    ref = _ref_fp4_grouped(qA, As, B, Bs, offsets)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not _accelerator_available(), reason="Accelerator not available")
def test_w4a8_grouped_block_vs_ref():
    torch.manual_seed(0)
    S, E, N, K = 32, 4, 256, 512
    A, expert_ids = _make_routed_inputs(
        S,
        E,
        K,
        dtype=torch.bfloat16,
        device=TEST_DEVICE,
        top_k=1,
    )
    B, Bs = _make_experts_weights_fp4(E, N, K, TEST_DEVICE)
    A_sorted, _, offsets, tokens_per_expert = _prepare_grouped(A, expert_ids, E)
    qA, As = finegrained_fp8.fp8_act_quant(A_sorted, 128)

    out = finegrained_fp4.w4a8_block_fp8_matmul_grouped(
        qA,
        As,
        B,
        Bs,
        offsets,
        tokens_per_expert,
        [128, 128],
        torch.float32,
    )
    ref = _ref_fp4_grouped_block(qA, As, B, Bs, offsets, [128, 128])
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not _accelerator_available(), reason="Accelerator not available")
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_grouped_vs_ref(problem):
    torch.manual_seed(0)
    A, expert_ids, B_fp8, Bs = _setup_problem(problem, dtype=torch.float32)
    A_sorted, expert_ids_sorted, offsets, tokens_per_expert = _prepare_grouped(
        A, expert_ids, problem.E
    )
    out = finegrained_fp8.w8a8_fp8_matmul_grouped(
        A_sorted,
        B_fp8,
        Bs,
        offsets,
        tokens_per_expert,
        problem.block_size,
    )
    ref = _ref(A_sorted, B_fp8, Bs, expert_ids_sorted, problem.block_size)
    torch.testing.assert_close(out, ref)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not _accelerator_available(), reason="Accelerator not available")
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_grouped_output_shape(problem):
    A, expert_ids, B_fp8, Bs = _setup_problem(problem, dtype=torch.bfloat16)
    A_sorted, _, offsets, tokens_per_expert = _prepare_grouped(A, expert_ids, problem.E)
    out = finegrained_fp8.w8a8_fp8_matmul_grouped(
        A_sorted,
        B_fp8,
        Bs,
        offsets,
        tokens_per_expert,
        problem.block_size,
    )
    assert out.shape == (problem.S, problem.N)
    assert out.dtype == torch.bfloat16


# ── torch.compile compatibility ────────────────────────────────────────────────
@pytest.mark.kernels_ci
@pytest.mark.skipif(not _accelerator_available(), reason="Accelerator not available")
def test_batched_compile():
    torch.manual_seed(0)
    torch.compiler.reset()
    _accelerator_module().empty_cache()

    A, expert_ids, B_fp8, Bs = _setup_problem(COMPILE_PROBLEM, dtype=torch.bfloat16)

    def fn(A, B_fp8, Bs, expert_ids):
        return finegrained_fp8.w8a8_fp8_matmul_batched(
            A, B_fp8, Bs, expert_ids, COMPILE_PROBLEM.block_size
        )

    compiled = torch.compile(fn, mode="max-autotune", fullgraph=True)
    out_compiled = compiled(A, B_fp8, Bs, expert_ids)
    out_ref = fn(A, B_fp8, Bs, expert_ids)
    torch.testing.assert_close(out_compiled, out_ref)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not _accelerator_available(), reason="Accelerator not available")
def test_grouped_compile():
    torch.manual_seed(0)
    torch.compiler.reset()
    _accelerator_module().empty_cache()

    A, expert_ids, B_fp8, Bs = _setup_problem(COMPILE_PROBLEM, dtype=torch.bfloat16)
    A_sorted, _, offsets, tokens_per_expert = _prepare_grouped(
        A, expert_ids, COMPILE_PROBLEM.E
    )

    def fn(A_sorted, B_fp8, Bs, offsets, tokens_per_expert):
        return finegrained_fp8.w8a8_fp8_matmul_grouped(
            A_sorted,
            B_fp8,
            Bs,
            offsets,
            tokens_per_expert,
            COMPILE_PROBLEM.block_size,
        )

    compiled = torch.compile(fn, mode="max-autotune", fullgraph=True)
    out_compiled = compiled(A_sorted, B_fp8, Bs, offsets, tokens_per_expert)
    out_ref = fn(A_sorted, B_fp8, Bs, offsets, tokens_per_expert)
    torch.testing.assert_close(out_compiled, out_ref)


def _bench_setup(problem: Problem, device=TEST_DEVICE):
    _accelerator_module().empty_cache()
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
    B_fp8, Bs = _make_experts_weights(
        problem.E,
        problem.N,
        problem.K,
        problem.block_size,
        device,
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
@pytest.mark.skipif(not _accelerator_available(), reason="Accelerator not available")
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="Latency baselines are calibrated for CUDA only")
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_batched_speedup(problem):
    if problem.expectation is None:
        pytest.skip("No expected benchmark latency for this problem")

    A, B_fp8, Bs, expert_ids, _, _ = _bench_setup(problem)

    batched_ms, batched_mean_ms, batched_min_ms, batched_max_ms = (
        _measure_latency_over_repeats(
            lambda: finegrained_fp8.w8a8_fp8_matmul_batched(
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
@pytest.mark.skipif(not _accelerator_available(), reason="Accelerator not available")
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="Latency baselines are calibrated for CUDA only")
@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_grouped_speedup(problem):
    if problem.expectation is None:
        pytest.skip("No expected benchmark latency for this problem")

    A, B_fp8, Bs, _, offsets, tokens_per_expert = _bench_setup(problem)

    grouped_ms, grouped_mean_ms, grouped_min_ms, grouped_max_ms = (
        _measure_latency_over_repeats(
            lambda: finegrained_fp8.w8a8_fp8_matmul_grouped(
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
