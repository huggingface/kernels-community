import torch

from kernels.benchmark import Benchmark


def rmsnorm_reference(x: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return x / rms


class TinygradRmsBenchmark(Benchmark):
    seed: int = 42

    def setup(self):
        batch_size = 32
        seq_len = 512
        hidden_size = 1024
        self.eps = 1e-6

        self.x = torch.randn(
            batch_size, seq_len, hidden_size, device=self.device, dtype=torch.float32
        )
        self.out = torch.empty_like(self.x)

    def benchmark_base(self):
        self.out = self.kernel.tinygrad_rms_norm_simple(self.x, self.eps)

    def verify_base(self) -> torch.Tensor:
        return rmsnorm_reference(self.x, self.eps)

    def setup_large(self):
        # Note: hidden_size must be 1024 (kernel constraint)
        batch_size = 64
        seq_len = 1024
        hidden_size = 1024
        self.eps = 1e-6

        self.x = torch.randn(
            batch_size, seq_len, hidden_size, device=self.device, dtype=torch.float32
        )
        self.out = torch.empty_like(self.x)

    def benchmark_large(self):
        self.out = self.kernel.tinygrad_rms_norm_simple(self.x, self.eps)

    def verify_large(self) -> torch.Tensor:
        return rmsnorm_reference(self.x, self.eps)
