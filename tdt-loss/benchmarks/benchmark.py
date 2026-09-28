import sys
from pathlib import Path

import torch

from kernels.benchmark import Benchmark

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.reference import tdt_loss_reference  # noqa: E402

DURATIONS = [0, 1, 2, 3, 4]


class TDTLossBenchmark(Benchmark):
    seed: int = 42

    def _make_inputs(self, B, T, U, V, dtype):
        D = len(DURATIONS)
        # Slices of a joint output, as produced by ParakeetForTDT.
        self.logits = torch.randn(B, T, U + 1, V + D, device=self.device, dtype=dtype, requires_grad=True)
        self.targets = torch.randint(1, V, (B, U), device=self.device, dtype=torch.int32)
        self.logit_lengths = torch.full((B,), T, device=self.device, dtype=torch.int32)
        self.target_lengths = torch.full((B,), U, device=self.device, dtype=torch.int32)
        self.V = V

    def _run(self):
        self.logits.grad = None
        losses = self.kernel.tdt_loss(
            self.logits[..., : self.V],
            self.logits[..., self.V :],
            self.targets,
            self.logit_lengths,
            self.target_lengths,
            DURATIONS,
            0,
            reduction="none",
        )
        losses.sum().backward()
        self.out = losses.detach()

    def _reference(self):
        with torch.no_grad():
            return tdt_loss_reference(
                self.logits[..., : self.V],
                self.logits[..., self.V :],
                self.targets,
                self.logit_lengths,
                self.target_lengths,
                0,
                DURATIONS,
            )

    # Small vocabulary, short utterances.
    def setup(self):
        self._make_inputs(B=8, T=100, U=20, V=1025, dtype=torch.float32)

    def benchmark_base(self):
        self._run()

    def verify_base(self) -> torch.Tensor:
        return self._reference()

    # Parakeet TDT v3 sizes (8192 tokens + blank), float32.
    def setup_large(self):
        self._make_inputs(B=4, T=200, U=40, V=8193, dtype=torch.float32)

    def benchmark_large(self):
        self._run()

    def verify_large(self) -> torch.Tensor:
        return self._reference()

    # Same sizes with bfloat16 logits.
    def setup_large_bf16(self):
        self._make_inputs(B=4, T=200, U=40, V=8193, dtype=torch.bfloat16)

    def benchmark_large_bf16(self):
        self._run()

    def verify_large_bf16(self) -> torch.Tensor:
        return self._reference()
