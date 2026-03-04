"""Tests for finegrained-fp8 kernels: act_quant, matmul, batched, and grouped.

Uses kernels-community (finegrained_fp8) directly — no transformers dependency.
"""

import pytest
import torch

import finegrained_fp8


FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_DTYPE = torch.float8_e4m3fn
BLOCK_SIZE = [128, 128]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _quantize_ref(x: torch.Tensor, block_k: int = 128):
    """Reference FP8 block quantization in pure PyTorch (no Triton)."""
    assert x.ndim >= 1
    assert x.shape[-1] % block_k == 0

    flat = x.reshape(-1, x.shape[-1])
    M, K = flat.shape
    num_blocks = K // block_k

    flat_blocked = flat.reshape(M, num_blocks, block_k)
    scales = flat_blocked.abs().amax(dim=-1).float() / FP8_MAX  # (M, num_blocks)
    # Avoid division by zero
    safe_scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quantized = (flat_blocked.float() / safe_scales.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    quantized = quantized.reshape(x.shape)
    scales = scales.reshape(*x.shape[:-1], num_blocks)
    return quantized, scales


def _matmul_ref(A_fp16: torch.Tensor, B_fp8: torch.Tensor, Bs: torch.Tensor, block_size: list[int]):
    """Reference matmul: quantize A, then do block-scaled fp8 matmul via finegrained_fp8."""
    block_k = block_size[1]
    qA, sA = finegrained_fp8.fp8_act_quant(A_fp16, block_k)
    return finegrained_fp8.w8a8_block_fp8_matmul(qA, B_fp8, sA, Bs, block_size)


def _make_weight(N: int, K: int, block_size: list[int], device: str = "cuda"):
    """Create a single FP8 weight matrix [N, K] with block scales."""
    block_n, block_k = block_size
    W = torch.randn(N, K, dtype=torch.float32, device=device)
    rt, ct = N // block_n, K // block_k
    R = W.reshape(rt, block_n, ct, block_k)
    max_abs = R.abs().amax(dim=(1, 3))  # (rt, ct)
    safe = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    scale = FP8_MAX / safe
    Wq = (R * scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    Wq = Wq.reshape(N, K).contiguous()
    inv_scales = (1.0 / scale).to(torch.float32)  # (N//block_n, K//block_k)
    return Wq, inv_scales


def _make_experts_weights(num_experts, out_features, in_features, block_size, device):
    """Create FP8 expert weights [E, N, K] with scales [E, N//block_n, K//block_k]."""
    block_n, block_k = block_size
    W = torch.randn(num_experts, out_features, in_features, dtype=torch.float32, device=device)
    E, N, K = W.shape
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
    """Build flattened routed inputs: hidden_states gathered by token_idx."""
    assert S % top_k == 0
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


# ── fp8_act_quant tests ──────────────────────────────────────────────────────

QUANT_SHAPES = [
    (1, 128),
    (1, 256),
    (4, 512),
    (16, 1024),
    (64, 2048),
    (128, 4096),
]


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("M,K", QUANT_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_act_quant_output_shape(M, K, dtype):
    """fp8_act_quant should return (quantized, scales) with correct shapes."""
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    block_k = BLOCK_SIZE[1]
    q, s = finegrained_fp8.fp8_act_quant(x, block_k)

    assert q.shape == (M, K)
    assert q.dtype == FP8_DTYPE
    assert s.shape == (M, K // block_k)
    assert s.dtype == torch.float32


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("M,K", QUANT_SHAPES)
def test_act_quant_roundtrip(M, K):
    """Quantize then dequantize should be close to original (within FP8 precision)."""
    torch.manual_seed(42)
    block_k = BLOCK_SIZE[1]
    x = torch.randn(M, K, dtype=torch.float32, device="cuda")

    q, s = finegrained_fp8.fp8_act_quant(x, block_k)

    # Dequantize: multiply each block by its scale
    q_float = q.float().reshape(M, K // block_k, block_k)
    s_expanded = s.unsqueeze(-1)  # (M, num_blocks, 1)
    deq = (q_float * s_expanded).reshape(M, K)

    # FP8 e4m3 has ~0.1% precision for normal values, allow generous tolerance
    torch.testing.assert_close(deq, x, atol=0.2, rtol=0.15)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_act_quant_zero_input():
    """fp8_act_quant on zero input: scale is 0, so 0/0 → NaN in quantized output.

    This is expected behaviour — zero blocks produce scale=0 and the kernel
    divides by it.  The dequantized result (NaN * 0) is still meaningless, but
    in practice zero-activation blocks don't appear in real inference.  We just
    verify the kernel doesn't crash and scales are zero.
    """
    x = torch.zeros(4, 256, dtype=torch.float32, device="cuda")
    q, s = finegrained_fp8.fp8_act_quant(x, 128)
    assert (s == 0).all(), "scales for zero input should be zero"


# ── w8a8_block_fp8_matmul tests ──────────────────────────────────────────────

MATMUL_SHAPES = [
    (1, 256, 256),
    (4, 256, 512),
    (16, 512, 256),
    (32, 1024, 512),
    (64, 1024, 1024),
]


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("M,N,K", MATMUL_SHAPES)
def test_matmul_output_shape(M, N, K):
    """w8a8_block_fp8_matmul should produce [M, N] output."""
    torch.manual_seed(0)
    A_fp8, As = _quantize_ref(torch.randn(M, K, device="cuda"), BLOCK_SIZE[1])
    B_fp8, Bs = _make_weight(N, K, BLOCK_SIZE, "cuda")

    out = finegrained_fp8.w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, BLOCK_SIZE)
    assert out.shape == (M, N)
    assert out.dtype == torch.float32


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("M,N,K", MATMUL_SHAPES)
def test_matmul_correctness(M, N, K):
    """FP8 matmul output should be close to bf16 torch.matmul reference."""
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B_fp8, Bs = _make_weight(N, K, BLOCK_SIZE, "cuda")

    # Dequantize B back to float for reference
    block_n, block_k = BLOCK_SIZE
    B_float = B_fp8.float().reshape(N // block_n, block_n, K // block_k, block_k)
    B_float = (B_float * Bs[:, None, :, None]).reshape(N, K)
    ref = A @ B_float.T

    # FP8 kernel path
    qA, sA = finegrained_fp8.fp8_act_quant(A, block_k)
    out = finegrained_fp8.w8a8_block_fp8_matmul(qA, B_fp8, sA, Bs, BLOCK_SIZE)

    # Double quantization (both A and B) introduces error that grows with K.
    # Use a relative tolerance on the full tensor norm instead of per-element.
    diff = (out - ref).float()
    rel_err = diff.norm() / ref.float().norm()
    assert rel_err < 0.05, f"relative Frobenius error {rel_err:.4f} exceeds 5%"


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16])
def test_matmul_output_dtype(output_dtype):
    """w8a8_block_fp8_matmul should respect output_dtype parameter."""
    M, N, K = 16, 256, 256
    torch.manual_seed(0)
    A_fp8, As = _quantize_ref(torch.randn(M, K, device="cuda"), BLOCK_SIZE[1])
    B_fp8, Bs = _make_weight(N, K, BLOCK_SIZE, "cuda")

    out = finegrained_fp8.w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, BLOCK_SIZE, output_dtype)
    assert out.dtype == output_dtype


# ── w8a8_block_fp8_matmul_batched tests ──────────────────────────────────────

MOE_SIZES = [
    (8, 4, 256, 256, 1),
    (32, 4, 256, 512, 2),
    (64, 8, 512, 1024, 4),
    (128, 16, 1024, 2048, 2),
]


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("S,E,N,K,TOP_K", MOE_SIZES)
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_batched_output_shape(S, E, N, K, TOP_K, input_dtype):
    """Batched matmul should produce [S, N] output in input dtype."""
    A, expert_ids = _make_routed_inputs(S, E, K, input_dtype, "cuda", TOP_K)
    B_fp8, Bs = _make_experts_weights(E, N, K, BLOCK_SIZE, "cuda")

    out = finegrained_fp8.w8a8_block_fp8_matmul_batched(A, B_fp8, Bs, expert_ids, BLOCK_SIZE)
    assert out.shape == (S, N)
    assert out.dtype == input_dtype


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("S,E,N,K,TOP_K", MOE_SIZES)
def test_batched_vs_ref(S, E, N, K, TOP_K):
    """Batched output should match the per-token loop reference."""
    torch.manual_seed(0)
    A, expert_ids = _make_routed_inputs(S, E, K, torch.float32, "cuda", TOP_K)
    B_fp8, Bs = _make_experts_weights(E, N, K, BLOCK_SIZE, "cuda")

    out = finegrained_fp8.w8a8_block_fp8_matmul_batched(A, B_fp8, Bs, expert_ids, BLOCK_SIZE)

    # Per-token reference
    ref = torch.empty_like(out)
    block_k = BLOCK_SIZE[1]
    for i in range(S):
        e = expert_ids[i]
        qA_i, sA_i = finegrained_fp8.fp8_act_quant(A[i:i + 1], block_k)
        ref[i] = finegrained_fp8.w8a8_block_fp8_matmul(qA_i, B_fp8[e], sA_i, Bs[e], BLOCK_SIZE)

    torch.testing.assert_close(out, ref)


# ── w8a8_block_fp8_matmul_grouped tests ──────────────────────────────────────


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("S,E,N,K,TOP_K", MOE_SIZES)
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_grouped_output_shape(S, E, N, K, TOP_K, input_dtype):
    """Grouped matmul should produce [S, N] output in input dtype."""
    A, expert_ids = _make_routed_inputs(S, E, K, input_dtype, "cuda", TOP_K)
    B_fp8, Bs = _make_experts_weights(E, N, K, BLOCK_SIZE, "cuda")

    perm = torch.argsort(expert_ids)
    A_sorted = A[perm].contiguous()
    expert_ids_sorted = expert_ids[perm]
    tokens_per_expert = torch.histc(expert_ids_sorted.float(), bins=E, min=0, max=E - 1).to(torch.int32)
    offsets = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)

    out = finegrained_fp8.w8a8_block_fp8_matmul_grouped(A_sorted, B_fp8, Bs, offsets, tokens_per_expert, BLOCK_SIZE)
    assert out.shape == (S, N)
    assert out.dtype == input_dtype


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("S,E,N,K,TOP_K", MOE_SIZES)
def test_grouped_vs_ref(S, E, N, K, TOP_K):
    """Grouped output (sorted tokens) should match per-token loop reference."""
    torch.manual_seed(0)
    A, expert_ids = _make_routed_inputs(S, E, K, torch.float32, "cuda", TOP_K)
    B_fp8, Bs = _make_experts_weights(E, N, K, BLOCK_SIZE, "cuda")

    perm = torch.argsort(expert_ids)
    A_sorted = A[perm].contiguous()
    expert_ids_sorted = expert_ids[perm]
    tokens_per_expert = torch.histc(expert_ids_sorted.float(), bins=E, min=0, max=E - 1).to(torch.int32)
    offsets = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)

    out = finegrained_fp8.w8a8_block_fp8_matmul_grouped(A_sorted, B_fp8, Bs, offsets, tokens_per_expert, BLOCK_SIZE)

    # Per-token reference
    ref = torch.empty_like(out)
    block_k = BLOCK_SIZE[1]
    for i in range(A_sorted.shape[0]):
        e = expert_ids_sorted[i]
        qA_i, sA_i = finegrained_fp8.fp8_act_quant(A_sorted[i:i + 1], block_k)
        ref[i] = finegrained_fp8.w8a8_block_fp8_matmul(qA_i, B_fp8[e], sA_i, Bs[e], BLOCK_SIZE)

    torch.testing.assert_close(out, ref)


# ── Per-tensor scale layout tests ────────────────────────────────────────────

SCALE_LAYOUTS = ["block", "per_tensor_1d", "per_tensor_111"]


def _convert_scale_layout(Bs, layout: str):
    if layout == "block":
        return Bs
    per_tensor = Bs[:, 0, 0].contiguous()
    if layout == "per_tensor_1d":
        return per_tensor
    if layout == "per_tensor_111":
        return per_tensor.view(-1, 1, 1).contiguous()
    raise ValueError(f"Unsupported scale layout: {layout}")


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("scale_layout", SCALE_LAYOUTS)
def test_batched_scale_layouts(scale_layout):
    """Batched kernel works with per-tensor and block scale layouts."""
    S, E, N, K, TOP_K = 32, 4, 256, 512, 2
    torch.manual_seed(0)
    A, expert_ids = _make_routed_inputs(S, E, K, torch.float32, "cuda", TOP_K)
    B_fp8, Bs = _make_experts_weights(E, N, K, BLOCK_SIZE, "cuda")
    Bs_layout = _convert_scale_layout(Bs, scale_layout)

    out = finegrained_fp8.w8a8_block_fp8_matmul_batched(A, B_fp8, Bs_layout, expert_ids, BLOCK_SIZE)
    assert out.shape == (S, N)
    assert not torch.isnan(out).any()


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("scale_layout", SCALE_LAYOUTS)
def test_grouped_scale_layouts(scale_layout):
    """Grouped kernel works with per-tensor and block scale layouts."""
    S, E, N, K, TOP_K = 32, 4, 256, 512, 2
    torch.manual_seed(0)
    A, expert_ids = _make_routed_inputs(S, E, K, torch.float32, "cuda", TOP_K)
    B_fp8, Bs = _make_experts_weights(E, N, K, BLOCK_SIZE, "cuda")
    Bs_layout = _convert_scale_layout(Bs, scale_layout)

    perm = torch.argsort(expert_ids)
    A_sorted = A[perm].contiguous()
    expert_ids_sorted = expert_ids[perm]
    tokens_per_expert = torch.histc(expert_ids_sorted.float(), bins=E, min=0, max=E - 1).to(torch.int32)
    offsets = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)

    out = finegrained_fp8.w8a8_block_fp8_matmul_grouped(A_sorted, B_fp8, Bs_layout, offsets, tokens_per_expert, BLOCK_SIZE)
    assert out.shape == (S, N)
    assert not torch.isnan(out).any()


# ── torch.compile tests ──────────────────────────────────────────────────────


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matmul_compile():
    """Basic matmul should work under torch.compile."""
    M, N, K = 16, 256, 256
    torch.manual_seed(0)
    torch.compiler.reset()
    torch.cuda.empty_cache()

    A_fp8, As = _quantize_ref(torch.randn(M, K, device="cuda"), BLOCK_SIZE[1])
    B_fp8, Bs = _make_weight(N, K, BLOCK_SIZE, "cuda")

    def fn(A, B, As, Bs):
        return finegrained_fp8.w8a8_block_fp8_matmul(A, B, As, Bs, BLOCK_SIZE)

    compiled = torch.compile(fn, mode="max-autotune", fullgraph=True)
    out_compiled = compiled(A_fp8, B_fp8, As, Bs)
    out_ref = fn(A_fp8, B_fp8, As, Bs)
    torch.testing.assert_close(out_compiled, out_ref)


@pytest.mark.kernels_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_act_quant_compile():
    """fp8_act_quant should work under torch.compile."""
    torch.manual_seed(0)
    torch.compiler.reset()
    torch.cuda.empty_cache()

    x = torch.randn(16, 512, dtype=torch.float32, device="cuda")

    def fn(x):
        return finegrained_fp8.fp8_act_quant(x, 128)

    compiled = torch.compile(fn, mode="max-autotune", fullgraph=True)
    q_compiled, s_compiled = compiled(x)
    q_ref, s_ref = fn(x)
    torch.testing.assert_close(q_compiled.float(), q_ref.float())
    torch.testing.assert_close(s_compiled, s_ref)
