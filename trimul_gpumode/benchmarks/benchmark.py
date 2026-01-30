import torch
from torch import nn, einsum

from kernels.benchmark import Benchmark


class TriMulReference(nn.Module):
    """Reference implementation of Triangle Multiplicative Module."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.left_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.left_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum("... i k d, ... j k d -> ... i j d", left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


class TrimulGpumodeBenchmark(Benchmark):
    seed: int = 42

    def setup(self):
        # Note: hidden_dim must be 128 (kernel constraint)
        batch_size = 1
        seq_len = 128
        dim = 128
        hidden_dim = 128

        self.config = {"dim": dim, "hidden_dim": hidden_dim}

        self.input_tensor = torch.randn(
            batch_size, seq_len, seq_len, dim, device="cuda", dtype=torch.float32
        )
        self.mask = torch.ones(
            batch_size, seq_len, seq_len, device="cuda", dtype=torch.float32
        )

        self.weights = {
            "norm.weight": torch.ones(dim, device="cuda", dtype=torch.float32),
            "norm.bias": torch.zeros(dim, device="cuda", dtype=torch.float32),
            "left_proj.weight": torch.randn(
                hidden_dim, dim, device="cuda", dtype=torch.float32
            )
            * 0.02,
            "right_proj.weight": torch.randn(
                hidden_dim, dim, device="cuda", dtype=torch.float32
            )
            * 0.02,
            "left_gate.weight": torch.randn(
                hidden_dim, dim, device="cuda", dtype=torch.float32
            )
            * 0.02,
            "right_gate.weight": torch.randn(
                hidden_dim, dim, device="cuda", dtype=torch.float32
            )
            * 0.02,
            "out_gate.weight": torch.randn(
                hidden_dim, dim, device="cuda", dtype=torch.float32
            )
            * 0.02,
            "to_out_norm.weight": torch.ones(
                hidden_dim, device="cuda", dtype=torch.float32
            ),
            "to_out_norm.bias": torch.zeros(
                hidden_dim, device="cuda", dtype=torch.float32
            ),
            "to_out.weight": torch.randn(
                dim, hidden_dim, device="cuda", dtype=torch.float32
            )
            * 0.02,
        }

        self.out = torch.empty(
            batch_size, seq_len, seq_len, dim, device="cuda", dtype=torch.float32
        )

    def benchmark_base(self):
        data = (self.input_tensor, self.mask, self.weights, self.config)
        self.out = self.kernel.kernel_global(data)

    def verify_base(self) -> torch.Tensor:
        ref = TriMulReference(
            dim=self.config["dim"], hidden_dim=self.config["hidden_dim"]
        ).cuda()

        ref.norm.weight = nn.Parameter(self.weights["norm.weight"])
        ref.norm.bias = nn.Parameter(self.weights["norm.bias"])
        ref.left_proj.weight = nn.Parameter(self.weights["left_proj.weight"])
        ref.right_proj.weight = nn.Parameter(self.weights["right_proj.weight"])
        ref.left_gate.weight = nn.Parameter(self.weights["left_gate.weight"])
        ref.right_gate.weight = nn.Parameter(self.weights["right_gate.weight"])
        ref.out_gate.weight = nn.Parameter(self.weights["out_gate.weight"])
        ref.to_out_norm.weight = nn.Parameter(self.weights["to_out_norm.weight"])
        ref.to_out_norm.bias = nn.Parameter(self.weights["to_out_norm.bias"])
        ref.to_out.weight = nn.Parameter(self.weights["to_out.weight"])

        with torch.no_grad():
            return ref(self.input_tensor, self.mask)
