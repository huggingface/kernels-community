"""Tests for MoE expert dispatch kernels: batched and grouped."""

import pytest
import torch
import triton

import finegrained_fp8


FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_DTYPE = torch.float8_e4m3fn
BLOCK_SIZE = [128, 128]
PROBLEM_SIZES = [
    (8, 4, 256, 256, 1),
    (32, 4, 256, 512, 2),
    (64, 8, 512, 1024, 4),
    (128, 16, 1024, 2048, 2),
    # Edge case with large expert ids * strides to trigger potential int32 overflow bugs
    (256, 256, 4096, 4096, 1)
    if torch.cuda.get_device_properties(0).total_memory >= 40 * 1024**3
    else (256, 256, 1024, 1024, 1),
]
SCALE_LAYOUTS = ["block", "per_tensor_1d", "per_tensor_111"]


def _make_experts_weights(num_experts, out_features, in_features, block_size, device):
    """Create FP8 expert weights/scales with FP8Experts-compatible layouts.

    Returns:
        weights_fp8: [E, N, K] where E=num_experts, N=out_features, K=in_features
        scales_inv: [E, N // block_n, K // block_k]
    """
    block_n, block_k = block_size
    W = torch.randn(
        num_experts, out_features, in_features, dtype=torch.float32, device=device
    )
    E, N, K = W.shape
    assert N % block_n == 0, f"N ({N}) must be divisible by block_n ({block_n})"
    assert K % block_k == 0, f"K ({K}) must be divisible by block_k ({block_k})"

    rt, ct = N // block_n, K // block_k
    R = W.reshape(E, rt, block_n, ct, block_k)

    max_abs = R.abs().amax(dim=(-3, -1))
    safe = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    scale = FP8_MAX / safe

    Wq = (R * scale.unsqueeze(-1).unsqueeze(-3)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    Wq = Wq.reshape(E, N, K).contiguous()
    inv_scales = (1.0 / scale).to(torch.float32)
    return Wq, inv_scales


def _make_routed_inputs(S, E, K, dtype, device, top_k):
    """Build flattened routed inputs like FP8Experts paths.

    FP8Experts builds token-expert pairs from `top_k_index` by flattening
    `[num_tokens, num_top_k] -> [S]`, then gathers hidden states with token_idx.
    This helper reproduces that pattern while keeping total routed pairs `S`.
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
    return selected_hidden_states, expert_ids


def _ref(A, B_fp8, Bs, expert_ids, block_size):
    S = A.shape[0]
    N = B_fp8.shape[1]
    bi = block_size[1]
    out = torch.empty(S, N, dtype=torch.float32, device=A.device)
    for i in range(S):
        e = expert_ids[i]
        qA_i, sA_i = finegrained_fp8.fp8_act_quant(A[i : i + 1], bi)
        out[i] = finegrained_fp8.w8a8_block_fp8_matmul(
            qA_i, B_fp8[e], sA_i, Bs[e], block_size
        )
    return out


def _convert_scale_layout(Bs, layout: str):
    if layout == "block":
        return Bs
    per_tensor = Bs[:, 0, 0].contiguous()
    if layout == "per_tensor_1d":
        return per_tensor
    if layout == "per_tensor_111":
        return per_tensor.view(-1, 1, 1).contiguous()
    raise ValueError(f"Unsupported scale layout: {layout}")


# ── w8a8_block_fp8_matmul_batched ─────────────────────────────────────────────
@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("S,E,N,K,TOP_K", PROBLEM_SIZES)
@pytest.mark.parametrize("scale_layout", SCALE_LAYOUTS)
def test_batched_vs_ref(S, E, N, K, TOP_K, scale_layout):
    """Batched output should match the per-token reference (both in float32)."""
    torch.manual_seed(0)
    A, expert_ids = _make_routed_inputs(
        S, E, K, dtype=torch.float32, device="cuda", top_k=TOP_K
    )
    B_fp8, Bs = _make_experts_weights(E, N, K, BLOCK_SIZE, "cuda")
    Bs = _convert_scale_layout(Bs, scale_layout)
    out = finegrained_fp8.w8a8_block_fp8_matmul_batched(
        A, B_fp8, Bs, expert_ids, BLOCK_SIZE
    )
    ref = _ref(A, B_fp8, Bs, expert_ids, BLOCK_SIZE)
    torch.testing.assert_close(out, ref)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("S,E,N,K,TOP_K", PROBLEM_SIZES)
@pytest.mark.parametrize("scale_layout", SCALE_LAYOUTS)
def test_batched_output_shape(S, E, N, K, TOP_K, scale_layout):
    A, expert_ids = _make_routed_inputs(
        S, E, K, dtype=torch.bfloat16, device="cuda", top_k=TOP_K
    )
    B_fp8, Bs = _make_experts_weights(E, N, K, BLOCK_SIZE, "cuda")
    Bs = _convert_scale_layout(Bs, scale_layout)
    out = finegrained_fp8.w8a8_block_fp8_matmul_batched(
        A, B_fp8, Bs, expert_ids, BLOCK_SIZE
    )
    assert out.shape == (S, N)
    assert out.dtype == torch.bfloat16


# ── w8a8_block_fp8_matmul_grouped ─────────────────────────────────────────────
@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("S,E,N,K,TOP_K", PROBLEM_SIZES)
@pytest.mark.parametrize("scale_layout", SCALE_LAYOUTS)
def test_grouped_vs_ref(S, E, N, K, TOP_K, scale_layout):
    """Grouped output (on sorted tokens) should match the per-token reference (both in float32)."""
    torch.manual_seed(0)
    A, expert_ids = _make_routed_inputs(
        S, E, K, dtype=torch.float32, device="cuda", top_k=TOP_K
    )
    B_fp8, Bs = _make_experts_weights(E, N, K, BLOCK_SIZE, "cuda")
    Bs = _convert_scale_layout(Bs, scale_layout)
    perm = torch.argsort(expert_ids)
    A_sorted = A[perm].contiguous()
    expert_ids_sorted = expert_ids[perm]
    tokens_per_expert = torch.histc(
        expert_ids_sorted.float(), bins=E, min=0, max=E - 1
    ).to(torch.int32)
    offsets = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)
    out = finegrained_fp8.w8a8_block_fp8_matmul_grouped(
        A_sorted, B_fp8, Bs, offsets, tokens_per_expert, BLOCK_SIZE
    )
    ref = _ref(A_sorted, B_fp8, Bs, expert_ids_sorted, BLOCK_SIZE)
    torch.testing.assert_close(out, ref)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("S,E,N,K,TOP_K", PROBLEM_SIZES)
@pytest.mark.parametrize("scale_layout", SCALE_LAYOUTS)
def test_grouped_output_shape(S, E, N, K, TOP_K, scale_layout):
    A, expert_ids = _make_routed_inputs(
        S, E, K, dtype=torch.bfloat16, device="cuda", top_k=TOP_K
    )
    B_fp8, Bs = _make_experts_weights(E, N, K, BLOCK_SIZE, "cuda")
    Bs = _convert_scale_layout(Bs, scale_layout)
    perm = torch.argsort(expert_ids)
    A_sorted = A[perm].contiguous()
    expert_ids_sorted = expert_ids[perm]
    tokens_per_expert = torch.histc(
        expert_ids_sorted.float(), bins=E, min=0, max=E - 1
    ).to(torch.int32)
    offsets = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)
    out = finegrained_fp8.w8a8_block_fp8_matmul_grouped(
        A_sorted, B_fp8, Bs, offsets, tokens_per_expert, BLOCK_SIZE
    )
    assert out.shape == (S, N)
    assert out.dtype == torch.bfloat16


# ── torch.compile compatibility ────────────────────────────────────────────────
@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batched_compile():
    """Batched kernel output must match eager under torch.compile(max-autotune, fullgraph)."""
    S, E, N, K, TOP_K = PROBLEM_SIZES[0]
    torch.manual_seed(0)
    torch.compiler.reset()
    torch.cuda.empty_cache()

    A, expert_ids = _make_routed_inputs(
        S, E, K, dtype=torch.bfloat16, device="cuda", top_k=TOP_K
    )
    B_fp8, Bs = _make_experts_weights(E, N, K, BLOCK_SIZE, "cuda")

    def fn(A, B_fp8, Bs, expert_ids):
        return finegrained_fp8.w8a8_block_fp8_matmul_batched(
            A, B_fp8, Bs, expert_ids, BLOCK_SIZE
        )

    compiled = torch.compile(fn, mode="max-autotune", fullgraph=True)
    out_compiled = compiled(A, B_fp8, Bs, expert_ids)
    out_ref = fn(A, B_fp8, Bs, expert_ids)
    torch.testing.assert_close(out_compiled, out_ref)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_grouped_compile():
    """Grouped kernel output must match eager under torch.compile(max-autotune, fullgraph)."""
    S, E, N, K, TOP_K = PROBLEM_SIZES[0]
    torch.manual_seed(0)
    torch.compiler.reset()
    torch.cuda.empty_cache()

    A, expert_ids = _make_routed_inputs(
        S, E, K, dtype=torch.bfloat16, device="cuda", top_k=TOP_K
    )
    B_fp8, Bs = _make_experts_weights(E, N, K, BLOCK_SIZE, "cuda")
    perm = torch.argsort(expert_ids)
    A_sorted = A[perm].contiguous()
    expert_ids_sorted = expert_ids[perm]
    tokens_per_expert = torch.histc(
        expert_ids_sorted.float(), bins=E, min=0, max=E - 1
    ).to(torch.int32)
    offsets = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)

    def fn(A_sorted, B_fp8, Bs, offsets, tokens_per_expert):
        return finegrained_fp8.w8a8_block_fp8_matmul_grouped(
            A_sorted, B_fp8, Bs, offsets, tokens_per_expert, BLOCK_SIZE
        )

    compiled = torch.compile(fn, mode="max-autotune", fullgraph=True)
    out_compiled = compiled(A_sorted, B_fp8, Bs, offsets, tokens_per_expert)
    out_ref = fn(A_sorted, B_fp8, Bs, offsets, tokens_per_expert)
    torch.testing.assert_close(out_compiled, out_ref)


# ── Benchmarks ────────────────────────────────────────────────────────────────
EXPECTED_MS_BATCHED = {
    (8, 4, 256, 256, 1): 0.0296,
    (32, 4, 256, 512, 2): 0.0269,
    (64, 8, 512, 1024, 4): 0.0276,
    (128, 16, 1024, 2048, 2): 0.0558,
    (256, 256, 4096, 4096, 1): 1.0040,
}
EXPECTED_MS_GROUPED = {
    (8, 4, 256, 256, 1): 0.1291,
    (32, 4, 256, 512, 2): 0.1283,
    (64, 8, 512, 1024, 4): 0.1272,
    (128, 16, 1024, 2048, 2): 0.1306,
    (256, 256, 4096, 4096, 1): 0.9558,
}


def _bench_setup(S, E, N, K, top_k, device="cuda"):
    """Create pre-sorted inputs shared by both batched and grouped benchmarks."""
    torch.cuda.empty_cache()
    torch.compiler.reset()
    torch.manual_seed(0)
    A, expert_ids = _make_routed_inputs(
        S, E, K, dtype=torch.bfloat16, device=device, top_k=top_k
    )
    B_fp8, Bs = _make_experts_weights(E, N, K, BLOCK_SIZE, device)
    perm = torch.argsort(expert_ids)
    A_sorted = A[perm].contiguous()
    expert_ids_sorted = expert_ids[perm].to(torch.int32)
    tokens_per_expert = torch.histc(
        expert_ids_sorted.float(), bins=E, min=0, max=E - 1
    ).to(torch.int32)
    offsets = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)
    return A_sorted, B_fp8, Bs, expert_ids_sorted, offsets, tokens_per_expert


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


@pytest.mark.benchmark
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("S,E,N,K,TOP_K", EXPECTED_MS_BATCHED.keys())
def test_batched_speedup(S, E, N, K, TOP_K):
    """Batched kernel median latency stays within ±15% of baseline."""
    A, B_fp8, Bs, expert_ids, _, _ = _bench_setup(S, E, N, K, TOP_K)

    batched_ms = triton.testing.do_bench(
        lambda: finegrained_fp8.w8a8_block_fp8_matmul_batched(
            A, B_fp8, Bs, expert_ids, BLOCK_SIZE
        ),
        quantiles=[0.5],
    )
    expected_ms = EXPECTED_MS_BATCHED[(S, E, N, K, TOP_K)]
    print(
        f"\n[batched] S={S:4d} E={E:4d} N={N:5d} K={K:5d} | "
        f"batched={batched_ms:.4f}ms  (expected {expected_ms:.4f}ms ±15%)"
    )
    _assert_latency_with_tolerance(batched_ms, expected_ms)


@pytest.mark.benchmark
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("S,E,N,K,TOP_K", EXPECTED_MS_GROUPED.keys())
def test_grouped_speedup(S, E, N, K, TOP_K):
    """Grouped kernel median latency stays within ±15% of baseline."""
    A, B_fp8, Bs, _, offsets, tokens_per_expert = _bench_setup(S, E, N, K, TOP_K)

    grouped_ms = triton.testing.do_bench(
        lambda: finegrained_fp8.w8a8_block_fp8_matmul_grouped(
            A, B_fp8, Bs, offsets, tokens_per_expert, BLOCK_SIZE
        ),
        quantiles=[0.5],
    )
    expected_ms = EXPECTED_MS_GROUPED[(S, E, N, K, TOP_K)]
    print(
        f"\n[grouped] S={S:4d} E={E:4d} N={N:5d} K={K:5d} | "
        f"grouped={grouped_ms:.4f}ms  (expected {expected_ms:.4f}ms ±15%)"
    )
    _assert_latency_with_tolerance(grouped_ms, expected_ms)
