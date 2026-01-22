import torch
import torch.nn.functional as F

from kernels.benchmark import Benchmark


class ReluBenchmark(Benchmark):
    seed: int = 42

    def setup(self):
        self.x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
        self.out = torch.empty_like(self.x)

    def benchmark_base(self):
        self.out = self.kernel.relu(self.x)

    def verify_base(self) -> torch.Tensor:
        return F.relu(self.x)

    def setup_large(self):
        self.x = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
        self.out = torch.empty_like(self.x)

    def benchmark_large(self):
        self.out = self.kernel.relu(self.x)

    def verify_large(self) -> torch.Tensor:
        return F.relu(self.x)
