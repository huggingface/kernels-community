import torch

from kernels.benchmark import Benchmark


class CountCumsumBenchmark(Benchmark):
    seed: int = 42

    def setup_small(self):
        self.num_experts = 8
        self.x = torch.randint(
            0, self.num_experts, (1024,), device=self.device, dtype=torch.int32
        )

    def benchmark_small(self):
        self.out, _ = self.kernel.count_cumsum(
            x=self.x, E=self.num_experts, do_cumsum=True
        )

    def verify_small(self) -> torch.Tensor:
        return self.x.bincount(minlength=self.num_experts).to(torch.int32)

    def setup_large(self):
        self.num_experts = 256
        self.x = torch.randint(
            0, self.num_experts, (2097152,), device=self.device, dtype=torch.int32
        )

    def benchmark_large(self):
        self.out, _ = self.kernel.count_cumsum(
            x=self.x, E=self.num_experts, do_cumsum=True
        )

    def verify_large(self) -> torch.Tensor:
        return self.x.bincount(minlength=self.num_experts).to(torch.int32)


try:
    import quack  # noqa: F401

    _HAS_QUACK = True
except ImportError:
    _HAS_QUACK = False

if _HAS_QUACK:

    class MoEBenchmark(Benchmark):
        """Full MoE forward pass. Requires quack-kernels and Hopper+ GPU."""

        seed: int = 42

        def _make_moe(self, T, H, I, E, K):
            MoE = self.kernel.MoE
            ActivationType = self.kernel.enums.ActivationType
            self.T, self.H, self.E, self.K = T, H, E, K
            self.backend = self.kernel.KernelBackendMoE
            with torch.device(self.device):
                self.moe = MoE(
                    num_experts=E,
                    num_experts_per_tok=K,
                    hidden_size=H,
                    intermediate_size=I,
                    activation_function=ActivationType.SWIGLU,
                    add_bias=False,
                    std=0.02,
                ).to(dtype=torch.bfloat16)
            self.x = 0.02 * torch.randn(
                T, H, device=self.device, dtype=torch.bfloat16
            )

        def setup_small(self):
            self._make_moe(T=8192, H=768, I=256, E=128, K=8)

        def benchmark_small(self):
            with torch.no_grad():
                self.out, _ = self.moe(
                    self.x, kernel_backend_moe=self.backend.sonicmoe
                )

        def verify_small(self) -> torch.Tensor:
            with torch.no_grad():
                ref, _ = self.moe(
                    self.x, kernel_backend_moe=self.backend.torch
                )
            return ref

        def setup_large(self):
            self._make_moe(T=8192, H=4096, I=1024, E=64, K=4)

        def benchmark_large(self):
            with torch.no_grad():
                self.out, _ = self.moe(
                    self.x, kernel_backend_moe=self.backend.sonicmoe
                )

        def verify_large(self) -> torch.Tensor:
            with torch.no_grad():
                ref, _ = self.moe(
                    self.x, kernel_backend_moe=self.backend.torch
                )
            return ref
