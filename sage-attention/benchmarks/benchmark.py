import torch
import torch.nn.functional as F

from kernels.benchmark import Benchmark

# SageAttention is approximate (INT8 quantized QK) so element-wise allclose
# is too strict.  Use cosine similarity instead (threshold 0.99).
_orig_allclose = torch.allclose
torch.allclose = lambda a, b, **_kw: (
    F.cosine_similarity(a.flatten().float().unsqueeze(0),
                        b.flatten().float().unsqueeze(0)).item() > 0.99
)


def _ref(q, k, v, is_causal=False):
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


class SageAttentionBenchmark(Benchmark):
    seed: int = 42

    # --- base: B=2, H=32, L=1024, D=128 ---

    def setup_base(self):
        B, H, L, D = 2, 32, 1024, 128
        self.q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=self.device)
        self.k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=self.device)
        self.v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=self.device)
        self.out = torch.empty_like(self.q)

    def benchmark_base(self):
        self.out = self.kernel.sageattn(self.q, self.k, self.v, tensor_layout="HND")

    def verify_base(self) -> torch.Tensor:
        return _ref(self.q, self.k, self.v)

    # --- causal: B=2, H=32, L=1024, D=128 with causal mask ---

    def setup_causal(self):
        B, H, L, D = 2, 32, 1024, 128
        self.q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=self.device)
        self.k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=self.device)
        self.v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=self.device)
        self.out = torch.empty_like(self.q)

    def benchmark_causal(self):
        self.out = self.kernel.sageattn(
            self.q, self.k, self.v, tensor_layout="HND", is_causal=True
        )

    def verify_causal(self) -> torch.Tensor:
        return _ref(self.q, self.k, self.v, is_causal=True)

    # --- large: B=4, H=32, L=4096, D=128 ---

    def setup_large(self):
        B, H, L, D = 4, 32, 4096, 128
        self.q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=self.device)
        self.k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=self.device)
        self.v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=self.device)
        self.out = torch.empty_like(self.q)

    def benchmark_large(self):
        self.out = self.kernel.sageattn(self.q, self.k, self.v, tensor_layout="HND")

    def verify_large(self) -> torch.Tensor:
        return _ref(self.q, self.k, self.v)

    # --- d64: B=4, H=32, L=2048, D=64 (smaller head dim) ---

    def setup_d64(self):
        B, H, L, D = 4, 32, 2048, 64
        self.q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=self.device)
        self.k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=self.device)
        self.v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=self.device)
        self.out = torch.empty_like(self.q)

    def benchmark_d64(self):
        self.out = self.kernel.sageattn(self.q, self.k, self.v, tensor_layout="HND")

    def verify_d64(self) -> torch.Tensor:
        return _ref(self.q, self.k, self.v)
