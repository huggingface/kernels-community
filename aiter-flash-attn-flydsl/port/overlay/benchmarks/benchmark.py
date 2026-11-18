"""Hub benchmark entry point; the provider comparison lives in comparison.py."""

import torch
import torch.nn.functional as F

from kernels.benchmark import Benchmark


class FlydslAttentionBenchmark(Benchmark):
    seed = 42

    def setup(self):
        self.q = torch.randn(1, 4096, 24, 128, device=self.device, dtype=torch.bfloat16)
        self.k = torch.randn_like(self.q)
        self.v = torch.randn_like(self.q)

    def benchmark_bf16(self):
        self.out = self.kernel.attention(self.q, self.k, self.v)

    def verify_bf16(self):
        return F.scaled_dot_product_attention(
            self.q.transpose(1, 2), self.k.transpose(1, 2), self.v.transpose(1, 2)
        ).transpose(1, 2)

    def benchmark_fp8_total(self):
        self.out = self.kernel.attention(self.q, self.k, self.v, fp8=True)

    # The generic runner fixes allclose(atol=1e-2). FP8 accuracy uses the
    # explicit relative-L2/cosine gates in tests and the provider comparison.
