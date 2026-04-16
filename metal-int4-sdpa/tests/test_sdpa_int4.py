"""
Correctness tests for sdpa_int4.

Tests the PyTorch reference implementations against each other and,
when available, against the Metal kernel. Validates:
  1. Quantization round-trip (int4 quantize → dequantize)
  2. Fused vs materialized SDPA match
  3. GQA (grouped-query attention) correctness
  4. Sliding window masking
  5. Numerical stability with large/small values
  6. Edge cases: N=1, N=large, uniform inputs
"""

import math
import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reference import (
    quantize_int4,
    dequantize_int4,
    sdpa_int4_reference,
    sdpa_int4_fused,
    sdpa_int4_fused_vectorized,
)


# --- Helpers ---

def make_test_data(
    num_heads=8, num_kv_heads=4, D=256, N=64, group_size=64, seed=42
):
    torch.manual_seed(seed)
    Q = torch.randn(num_heads, D)
    K = torch.randn(num_kv_heads, N, D)
    V = torch.randn(num_kv_heads, N, D)
    scale = 1.0 / math.sqrt(D)
    gqa_factor = num_heads // num_kv_heads

    k_packed, k_scales, k_biases = quantize_int4(K, group_size)
    v_packed, v_scales, v_biases = quantize_int4(V, group_size)

    return {
        "Q": Q, "K": K, "V": V, "scale": scale,
        "gqa_factor": gqa_factor,
        "k_packed": k_packed, "k_scales": k_scales, "k_biases": k_biases,
        "v_packed": v_packed, "v_scales": v_scales, "v_biases": v_biases,
    }


# --- Quantization tests ---

class TestQuantization:
    def test_roundtrip_shape(self):
        x = torch.randn(4, 128, 256)
        packed, scales, biases = quantize_int4(x, group_size=64)
        assert packed.shape == (4, 128, 32)   # 256/8 = 32
        assert scales.shape == (4, 128, 4)    # 256/64 = 4
        assert biases.shape == (4, 128, 4)

    def test_roundtrip_accuracy(self):
        x = torch.randn(2, 64, 128)
        packed, scales, biases = quantize_int4(x, group_size=64)
        x_recon = dequantize_int4(packed, scales, biases, group_size=64)
        # int4 quantization error should be bounded
        rel_error = (x - x_recon).abs().mean() / x.abs().mean()
        assert rel_error < 0.15, f"Relative error too high: {rel_error:.4f}"

    def test_nibble_range(self):
        """All packed nibbles should be in [0, 15]."""
        x = torch.randn(1, 1, 64)
        packed, _, _ = quantize_int4(x, group_size=64)
        for i in range(8):
            nibbles = (packed >> (4 * i)) & 0xF
            assert nibbles.max() <= 15
            assert nibbles.min() >= 0

    def test_constant_input(self):
        """Constant input should quantize to same nibble value everywhere."""
        x = torch.ones(1, 1, 64) * 3.14
        packed, scales, biases = quantize_int4(x, group_size=64)
        x_recon = dequantize_int4(packed, scales, biases, group_size=64)
        # All values should be identical after dequant
        assert (x_recon - x_recon[0, 0, 0]).abs().max() < 1e-6


# --- SDPA correctness tests ---

class TestSDPACorrectness:
    def test_reference_vs_fused_vectorized(self):
        d = make_test_data()
        out_ref = sdpa_int4_reference(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"],
        )
        out_vec = sdpa_int4_fused_vectorized(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"],
        )
        diff = (out_ref - out_vec).abs().max().item()
        assert diff < 1e-5, f"ref vs vectorized: {diff}"

    @pytest.mark.slow
    def test_reference_vs_fused_loop(self):
        """Slow: tests the per-token loop implementation."""
        d = make_test_data(num_heads=2, num_kv_heads=1, D=128, N=16)
        out_ref = sdpa_int4_reference(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"],
        )
        out_fused = sdpa_int4_fused(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"],
        )
        diff = (out_ref - out_fused).abs().max().item()
        assert diff < 1e-4, f"ref vs fused loop: {diff}"


class TestGQA:
    @pytest.mark.parametrize("gqa_factor", [1, 2, 4, 8])
    def test_gqa_factors(self, gqa_factor):
        num_kv_heads = 2
        num_heads = num_kv_heads * gqa_factor
        d = make_test_data(num_heads=num_heads, num_kv_heads=num_kv_heads, D=128, N=32)
        out_ref = sdpa_int4_reference(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"],
        )
        out_vec = sdpa_int4_fused_vectorized(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"],
        )
        diff = (out_ref - out_vec).abs().max().item()
        assert diff < 1e-5, f"GQA factor {gqa_factor}: {diff}"

    def test_shared_kv_heads_produce_same_output(self):
        """Query heads sharing a KV head should differ only due to different Q."""
        d = make_test_data(num_heads=4, num_kv_heads=1, D=128, N=32)
        # Use identical queries for heads 0 and 1 (same KV head)
        d["Q"][1] = d["Q"][0].clone()
        out = sdpa_int4_fused_vectorized(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"],
        )
        diff = (out[0] - out[1]).abs().max().item()
        assert diff < 1e-6, f"Identical queries should give identical output: {diff}"


class TestSlidingWindow:
    def test_sliding_window_basic(self):
        d = make_test_data(D=128, N=64)
        out_full = sdpa_int4_reference(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"], sliding_window=0,
        )
        out_window = sdpa_int4_reference(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"], sliding_window=32,
        )
        # With a window, output should differ from full attention
        diff = (out_full - out_window).abs().max().item()
        assert diff > 1e-3, "Sliding window should change output"

    def test_window_equals_N_matches_full(self):
        """Window >= N should give same result as no window."""
        d = make_test_data(D=128, N=32)
        out_full = sdpa_int4_reference(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"], sliding_window=0,
        )
        out_window = sdpa_int4_reference(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"], sliding_window=32,
        )
        diff = (out_full - out_window).abs().max().item()
        assert diff < 1e-6


class TestEdgeCases:
    def test_single_token(self):
        """N=1: output should be V[0] regardless of Q."""
        d = make_test_data(D=128, N=1)
        out = sdpa_int4_fused_vectorized(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"],
        )
        V_deq = dequantize_int4(d["v_packed"], d["v_scales"], d["v_biases"])
        # Each head should output the single V token (expanded via GQA)
        for h in range(out.shape[0]):
            kv_h = h // d["gqa_factor"]
            diff = (out[h] - V_deq[kv_h, 0]).abs().max().item()
            assert diff < 1e-5, f"N=1 head {h}: {diff}"

    def test_large_values(self):
        """Should handle large input magnitudes without NaN."""
        d = make_test_data(D=128, N=32)
        d["Q"] *= 100
        out = sdpa_int4_fused_vectorized(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"],
        )
        assert not torch.isnan(out).any(), "NaN in output with large Q"
        assert not torch.isinf(out).any(), "Inf in output with large Q"

    @pytest.mark.parametrize("D", [128, 256, 512])
    def test_head_dimensions(self, D):
        d = make_test_data(D=D, N=32)
        out_ref = sdpa_int4_reference(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"],
        )
        out_vec = sdpa_int4_fused_vectorized(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"],
        )
        diff = (out_ref - out_vec).abs().max().item()
        assert diff < 1e-5, f"D={D}: {diff}"


class TestNumericalStability:
    def test_uniform_scores(self):
        """When all K are identical, attention should be uniform → output = mean(V)."""
        torch.manual_seed(0)
        num_heads, num_kv_heads, D, N = 2, 1, 128, 16
        Q = torch.randn(num_heads, D)
        # All K identical
        k_row = torch.randn(1, 1, D).expand(num_kv_heads, N, D).contiguous()
        V = torch.randn(num_kv_heads, N, D)
        scale = 1.0 / math.sqrt(D)

        k_packed, k_scales, k_biases = quantize_int4(k_row)
        v_packed, v_scales, v_biases = quantize_int4(V)

        out = sdpa_int4_fused_vectorized(
            Q, k_packed, k_scales, k_biases,
            v_packed, v_scales, v_biases,
            scale, num_heads // num_kv_heads,
        )

        V_deq = dequantize_int4(v_packed, v_scales, v_biases)
        expected = V_deq[0].mean(dim=0)  # uniform attention = mean

        for h in range(num_heads):
            diff = (out[h] - expected).abs().max().item()
            assert diff < 0.05, f"Uniform K, head {h}: diff={diff}"

    def test_output_normalized(self):
        """Output should be a convex combination of V rows."""
        d = make_test_data(D=128, N=32)
        out = sdpa_int4_fused_vectorized(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"],
        )
        V_deq = dequantize_int4(d["v_packed"], d["v_scales"], d["v_biases"])
        # Output should be bounded by min/max of V values per KV head
        for h in range(out.shape[0]):
            kv_h = h // d["gqa_factor"]
            v_min = V_deq[kv_h].min(dim=0).values
            v_max = V_deq[kv_h].max(dim=0).values
            assert (out[h] >= v_min - 1e-4).all(), "Output below V range"
            assert (out[h] <= v_max + 1e-4).all(), "Output above V range"


# --- Metal kernel tests (requires MPS + compiled kernel) ---

def _load_metal_ext():
    """Try loading the compiled Metal kernel extension."""
    try:
        if not torch.backends.mps.is_available():
            return None
        from torch.utils.cpp_extension import load
        from pathlib import Path
        root = Path(__file__).parent.parent
        build_dir = root / "build"
        if not (build_dir / "sdpa_int4_bridge.mm").exists():
            return None
        return load(
            name="sdpa_int4_ext",
            sources=[str(build_dir / "sdpa_int4_bridge.mm"), str(build_dir / "binding.mm")],
            extra_include_paths=[str(root / "torch-ext"), str(build_dir)],
            extra_cflags=["-std=c++17"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation",
                           "-framework", "MetalPerformanceShaders"],
            is_python_module=True,
        )
    except Exception:
        return None


_metal_ext = _load_metal_ext()
requires_metal = pytest.mark.skipif(_metal_ext is None, reason="Metal kernel not compiled")


def _run_metal(ext, d, sliding_window=0):
    """Run Metal kernel with test data dict."""
    Q_m = d["Q"].to("mps")
    kp = d["k_packed"].to("mps")
    ks = d["k_scales"].to("mps")
    kb = d["k_biases"].to("mps")
    vp = d["v_packed"].to("mps")
    vs = d["v_scales"].to("mps")
    vb = d["v_biases"].to("mps")
    N = d["k_packed"].shape[1]
    return ext.sdpa_int4(Q_m, kp, ks, kb, vp, vs, vb,
                         d["gqa_factor"], N, d["scale"], sliding_window).cpu()


class TestMetalKernel:
    @requires_metal
    @pytest.mark.parametrize("D", [128, 256, 512])
    def test_metal_vs_reference(self, D):
        d = make_test_data(D=D, N=128)
        out_ref = sdpa_int4_reference(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"],
        )
        out_metal = _run_metal(_metal_ext, d)
        diff = (out_ref - out_metal).abs().max().item()
        assert diff < 1e-3, f"Metal vs ref D={D}: {diff}"

    @requires_metal
    @pytest.mark.parametrize("N", [1, 32, 512, 2048, 8192])
    def test_metal_sequence_lengths(self, N):
        d = make_test_data(D=256, N=N)
        out_ref = sdpa_int4_reference(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"],
        )
        out_metal = _run_metal(_metal_ext, d)
        diff = (out_ref - out_metal).abs().max().item()
        assert diff < 1e-3, f"Metal vs ref N={N}: {diff}"

    @requires_metal
    def test_metal_sliding_window(self):
        d = make_test_data(D=256, N=256)
        out_ref = sdpa_int4_reference(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"], sliding_window=64,
        )
        out_metal = _run_metal(_metal_ext, d, sliding_window=64)
        diff = (out_ref - out_metal).abs().max().item()
        assert diff < 1e-3, f"Metal sliding window: {diff}"

    @requires_metal
    @pytest.mark.parametrize("gqa_factor", [1, 2, 4, 8])
    def test_metal_gqa(self, gqa_factor):
        num_kv_heads = 2
        num_heads = num_kv_heads * gqa_factor
        d = make_test_data(num_heads=num_heads, num_kv_heads=num_kv_heads, D=256, N=128)
        out_ref = sdpa_int4_reference(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"],
        )
        out_metal = _run_metal(_metal_ext, d)
        diff = (out_ref - out_metal).abs().max().item()
        assert diff < 1e-3, f"Metal GQA factor {gqa_factor}: {diff}"

    @requires_metal
    def test_metal_no_nan_inf(self):
        d = make_test_data(D=256, N=512)
        d["Q"] *= 100  # Large values
        out = _run_metal(_metal_ext, d)
        assert not torch.isnan(out).any(), "NaN in Metal output"
        assert not torch.isinf(out).any(), "Inf in Metal output"


    @requires_metal
    @pytest.mark.parametrize("batch", [2, 4, 8])
    def test_metal_batched_decode(self, batch):
        """Batched decode: flatten Q as (batch*num_heads, D) for speculative decoding."""
        d = make_test_data(num_heads=8, num_kv_heads=4, D=256, N=256)
        Q_batch = torch.randn(batch, 8, 256)
        ref = torch.stack([
            sdpa_int4_reference(
                Q_batch[b], d["k_packed"], d["k_scales"], d["k_biases"],
                d["v_packed"], d["v_scales"], d["v_biases"],
                d["scale"], d["gqa_factor"],
            )
            for b in range(batch)
        ])
        Q_flat = Q_batch.reshape(batch * 8, 256).to("mps")
        kp = d["k_packed"].to("mps"); ks = d["k_scales"].to("mps"); kb = d["k_biases"].to("mps")
        vp = d["v_packed"].to("mps"); vs = d["v_scales"].to("mps"); vb = d["v_biases"].to("mps")
        out = _metal_ext.sdpa_int4(Q_flat, kp, ks, kb, vp, vs, vb,
                                    d["gqa_factor"], 256, d["scale"], 0)
        out = out.cpu().reshape(batch, 8, 256)
        diff = (ref - out).abs().max().item()
        assert diff < 1e-3, f"Batched decode batch={batch}: {diff}"


class TestMetalArchitectures:
    """Validate kernel against real model architectures."""

    @requires_metal
    def test_llama31_8b(self):
        """Llama 3.1 8B: 32 heads, 8 KV heads, D=128, GQA=4."""
        d = make_test_data(num_heads=32, num_kv_heads=8, D=128, N=512)
        out_ref = sdpa_int4_reference(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"],
        )
        out_metal = _run_metal(_metal_ext, d)
        diff = (out_ref - out_metal).abs().max().item()
        assert diff < 1e-3, f"Llama 3.1 8B: {diff}"

    @requires_metal
    def test_gemma4_sliding(self):
        """Gemma 4 31B sliding layer: 32 heads, 16 KV heads, D=256, sw=1024."""
        d = make_test_data(num_heads=32, num_kv_heads=16, D=256, N=2048)
        out_ref = sdpa_int4_reference(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"], sliding_window=1024,
        )
        out_metal = _run_metal(_metal_ext, d, sliding_window=1024)
        diff = (out_ref - out_metal).abs().max().item()
        assert diff < 1e-3, f"Gemma 4 sliding: {diff}"

    @requires_metal
    def test_gemma4_full(self):
        """Gemma 4 31B full attention layer: 32 heads, 16 KV heads, D=512."""
        d = make_test_data(num_heads=32, num_kv_heads=16, D=512, N=1024)
        out_ref = sdpa_int4_reference(
            d["Q"], d["k_packed"], d["k_scales"], d["k_biases"],
            d["v_packed"], d["v_scales"], d["v_biases"],
            d["scale"], d["gqa_factor"],
        )
        out_metal = _run_metal(_metal_ext, d)
        diff = (out_ref - out_metal).abs().max().item()
        assert diff < 1e-3, f"Gemma 4 full: {diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
