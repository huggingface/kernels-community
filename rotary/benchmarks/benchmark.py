import torch

from kernels.benchmark import Benchmark


def apply_rotary_reference(
    x1: torch.Tensor, x2: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, conj: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if not conj:
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
    else:
        out1 = x1 * cos + x2 * sin
        out2 = -x1 * sin + x2 * cos
    return out1, out2


class RotaryBenchmark(Benchmark):
    seed: int = 42

    def setup(self):
        batch_size = 2
        seqlen = 128
        num_heads = 8
        head_dim = 64
        rotary_dim = 32

        # Query tensor split into rotary parts
        self.x1 = torch.randn(
            batch_size,
            seqlen,
            num_heads,
            rotary_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.x2 = torch.randn(
            batch_size,
            seqlen,
            num_heads,
            rotary_dim,
            device=self.device,
            dtype=torch.float32,
        )

        # Rotary position embeddings
        self.cos = torch.randn(
            seqlen, 1, rotary_dim, device=self.device, dtype=torch.float32
        )
        self.sin = torch.randn(
            seqlen, 1, rotary_dim, device=self.device, dtype=torch.float32
        )

        # Output tensors (in-place, so clone inputs)
        self.out1 = self.x1.clone()
        self.out2 = self.x2.clone()

    def benchmark_base(self):
        # Reset outputs to input values for in-place operation
        self.out1.copy_(self.x1)
        self.out2.copy_(self.x2)
        self.kernel.apply_rotary(
            self.out1, self.out2, self.cos, self.sin, self.out1, self.out2, False
        )

    def verify_base(self) -> torch.Tensor:
        ref_out1, ref_out2 = apply_rotary_reference(
            self.x1, self.x2, self.cos, self.sin, False
        )
        # Concatenate for comparison (benchmark compares self.out with returned tensor)
        self.out = torch.cat([self.out1, self.out2], dim=-1)
        return torch.cat([ref_out1, ref_out2], dim=-1)

    def setup_large(self):
        batch_size = 8
        seqlen = 512
        num_heads = 32
        rotary_dim = 64

        self.x1 = torch.randn(
            batch_size,
            seqlen,
            num_heads,
            rotary_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.x2 = torch.randn(
            batch_size,
            seqlen,
            num_heads,
            rotary_dim,
            device=self.device,
            dtype=torch.float32,
        )

        self.cos = torch.randn(
            seqlen, 1, rotary_dim, device=self.device, dtype=torch.float32
        )
        self.sin = torch.randn(
            seqlen, 1, rotary_dim, device=self.device, dtype=torch.float32
        )

        self.out1 = self.x1.clone()
        self.out2 = self.x2.clone()

    def benchmark_large(self):
        self.out1.copy_(self.x1)
        self.out2.copy_(self.x2)
        self.kernel.apply_rotary(
            self.out1, self.out2, self.cos, self.sin, self.out1, self.out2, False
        )

    def verify_large(self) -> torch.Tensor:
        ref_out1, ref_out2 = apply_rotary_reference(
            self.x1, self.x2, self.cos, self.sin, False
        )
        self.out = torch.cat([self.out1, self.out2], dim=-1)
        return torch.cat([ref_out1, ref_out2], dim=-1)
