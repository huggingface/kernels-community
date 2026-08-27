import torch
import torch.nn.functional as F

from kernels.benchmark import Benchmark


class CausalConv1dBenchmark(Benchmark):
    seed: int = 42

    def setup(self):
        batch_size, dim, seqlen, width = 2, 64, 128, 4
        self.x = torch.randn(
            batch_size, dim, seqlen, device=self.device, dtype=torch.float16
        )
        self.weight = torch.randn(dim, width, device=self.device, dtype=torch.float32)
        self.bias = torch.randn(dim, device=self.device, dtype=torch.float32)
        self.out = torch.empty(
            batch_size, dim, seqlen, device=self.device, dtype=torch.float16
        )
        self.dim = dim
        self.width = width
        self.seqlen = seqlen

    def benchmark_base(self):
        self.out = self.kernel.causal_conv1d_fn(self.x, self.weight, self.bias)

    def verify_base(self) -> torch.Tensor:
        x_fp32 = self.x.to(self.weight.dtype)
        out = F.conv1d(
            x_fp32,
            self.weight.unsqueeze(1),
            self.bias,
            padding=self.width - 1,
            groups=self.dim,
        )
        return out[..., : self.seqlen].to(self.x.dtype)

    def setup_large(self):
        batch_size, dim, seqlen, width = 8, 256, 512, 4
        self.x = torch.randn(
            batch_size, dim, seqlen, device=self.device, dtype=torch.float16
        )
        self.weight = torch.randn(dim, width, device=self.device, dtype=torch.float32)
        self.bias = torch.randn(dim, device=self.device, dtype=torch.float32)
        self.out = torch.empty(
            batch_size, dim, seqlen, device=self.device, dtype=torch.float16
        )
        self.dim = dim
        self.width = width
        self.seqlen = seqlen

    def benchmark_large(self):
        self.out = self.kernel.causal_conv1d_fn(self.x, self.weight, self.bias)

    def verify_large(self) -> torch.Tensor:
        x_fp32 = self.x.to(self.weight.dtype)
        out = F.conv1d(
            x_fp32,
            self.weight.unsqueeze(1),
            self.bias,
            padding=self.width - 1,
            groups=self.dim,
        )
        return out[..., : self.seqlen].to(self.x.dtype)

    def setup_xlarge(self):
        batch_size, dim, seqlen, width = 16, 512, 1024, 4
        self.x = torch.randn(
            batch_size, dim, seqlen, device=self.device, dtype=torch.float16
        )
        self.weight = torch.randn(dim, width, device=self.device, dtype=torch.float32)
        self.bias = torch.randn(dim, device=self.device, dtype=torch.float32)
        self.out = torch.empty(
            batch_size, dim, seqlen, device=self.device, dtype=torch.float16
        )
        self.dim = dim
        self.width = width
        self.seqlen = seqlen

    def benchmark_xlarge(self):
        self.out = self.kernel.causal_conv1d_fn(self.x, self.weight, self.bias)

    def verify_xlarge(self) -> torch.Tensor:
        x_fp32 = self.x.to(self.weight.dtype)
        out = F.conv1d(
            x_fp32,
            self.weight.unsqueeze(1),
            self.bias,
            padding=self.width - 1,
            groups=self.dim,
        )
        return out[..., : self.seqlen].to(self.x.dtype)
