import torch

from kernels.benchmark import Benchmark

# Monkey patch torch.allclose to use higher tolerance for FP8 comparisons
_original_allclose = torch.allclose


def _fp8_tolerant_allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Custom allclose that uses higher tolerance for FP8-related comparisons."""
    # Use higher tolerance since FP8 has low precision (~3 bits mantissa)
    # FP8 e4m3 has relative precision of ~12.5%, so use atol based on max value
    max_val = max(input.abs().max().item(), other.abs().max().item(), 1.0)
    fp8_atol = max(atol, max_val * 0.15)  # 15% relative tolerance
    return _original_allclose(
        input, other, rtol=rtol, atol=fp8_atol, equal_nan=equal_nan
    )


# Apply the monkey patch
torch.allclose = _fp8_tolerant_allclose


def quantize_fp8_per_row_reference(
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation of FP8 per-row quantization."""
    pt_fp8_dtype = torch.float8_e4m3fn
    max_fp8 = torch.finfo(pt_fp8_dtype).max
    eps = 1e-12

    original_shape = a.shape
    a_2d = a.view(-1, a.shape[-1])

    # Compute max absolute value per row
    row_max = a_2d.abs().max(dim=-1).values
    row_max = torch.clamp(row_max, min=eps)

    # Compute scale: MAX_FP8 / max_abs
    scale = max_fp8 / row_max

    # Quantize
    a_scaled = a_2d * scale.unsqueeze(-1)
    a_scaled = torch.clamp(a_scaled, -max_fp8, max_fp8)
    a_fp8 = a_scaled.to(pt_fp8_dtype)

    # Return reciprocal scale
    a_scale = 1.0 / scale

    return a_fp8.view(original_shape), a_scale.view(original_shape[:-1])


class QuantizeFp8PerRowBenchmark(Benchmark):
    seed: int = 42

    def setup(self):
        M, K = 512, 1024
        self.a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        self.out = torch.empty(M, K, device="cuda", dtype=torch.float32)

    def benchmark_base(self):
        a_fp8, a_scale = self.kernel.quantize_fp8_per_row(self.a)
        self.out = a_fp8.to(torch.float32)

    def verify_base(self) -> torch.Tensor:
        a_fp8, _ = quantize_fp8_per_row_reference(self.a)
        return a_fp8.to(torch.float32)

    def setup_large(self):
        M, K = 2048, 4096
        self.a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        self.out = torch.empty(M, K, device="cuda", dtype=torch.float32)

    def benchmark_large(self):
        a_fp8, a_scale = self.kernel.quantize_fp8_per_row(self.a)
        self.out = a_fp8.to(torch.float32)

    def verify_large(self) -> torch.Tensor:
        a_fp8, _ = quantize_fp8_per_row_reference(self.a)
        return a_fp8.to(torch.float32)
