import torch

from kernels.benchmark import Benchmark


def rwkv_wkv_reference(
    w: torch.Tensor, u: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    B, T, C = k.shape
    device = k.device
    dtype = k.dtype

    y = torch.zeros(B, T, C, device=device, dtype=dtype)

    # State: accumulated numerator, denominator, and max exponent
    aa = torch.zeros(B, C, device=device, dtype=torch.float32)
    bb = torch.zeros(B, C, device=device, dtype=torch.float32)
    pp = torch.full((B, C), -1e38, device=device, dtype=torch.float32)

    w = w.float()
    u = u.float()

    for t in range(T):
        kt = k[:, t, :].float()  # [B, C]
        vt = v[:, t, :].float()  # [B, C]

        # Output computation
        ww = u + kt
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        y[:, t, :] = ((e1 * aa + e2 * vt) / (e1 * bb + e2)).to(dtype)

        # State update (note: w + pp, not pp - w)
        ww = w + pp
        p = torch.maximum(ww, kt)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kt - p)
        aa = e1 * aa + e2 * vt
        bb = e1 * bb + e2
        pp = p

    return y


class RwkvBenchmark(Benchmark):
    seed: int = 42

    def setup(self):
        B, T, C = 2, 64, 256

        self.w = torch.randn(
            C, device=self.device, dtype=torch.float32
        ).abs()  # Decay should be positive
        self.u = torch.randn(C, device=self.device, dtype=torch.float32)
        self.k = torch.randn(B, T, C, device=self.device, dtype=torch.float32) * 0.1
        self.v = torch.randn(B, T, C, device=self.device, dtype=torch.float32) * 0.1
        self.out = torch.zeros(B, T, C, device=self.device, dtype=torch.float32)

    def benchmark_base(self):
        self.out.zero_()
        self.kernel.forward(self.w, self.u, self.k, self.v, self.out)

    def verify_base(self) -> torch.Tensor:
        return rwkv_wkv_reference(self.w, self.u, self.k, self.v)

    def setup_large(self):
        B, T, C = 8, 256, 512

        self.w = torch.randn(C, device=self.device, dtype=torch.float32).abs()
        self.u = torch.randn(C, device=self.device, dtype=torch.float32)
        self.k = torch.randn(B, T, C, device=self.device, dtype=torch.float32) * 0.1
        self.v = torch.randn(B, T, C, device=self.device, dtype=torch.float32) * 0.1
        self.out = torch.zeros(B, T, C, device=self.device, dtype=torch.float32)

    def benchmark_large(self):
        self.out.zero_()
        self.kernel.forward(self.w, self.u, self.k, self.v, self.out)

    def verify_large(self) -> torch.Tensor:
        return rwkv_wkv_reference(self.w, self.u, self.k, self.v)
