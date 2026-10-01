import torch

from kernels.benchmark import Benchmark

SHAPES = {
    "qkv": (8704, 6656),
    "o": (6656, 4096),
    "mlp": (39936, 6656),
    "down": (6656, 19968),
    "lm_head": (202048, 6656),
}


def _dequant(pw):
    grid = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=pw.qweight.device
    )
    gs = pw.global_scale.float()
    out = torch.empty(pw.n, pw.k, dtype=torch.float16, device=pw.qweight.device)
    step = 16384
    for i in range(0, pw.n, step):
        qw = pw.qweight[i : i + step]
        nib = torch.stack([qw & 0xF, qw >> 4], dim=-1)
        nib = nib.reshape(qw.shape[0], pw.k).long()
        sign = torch.where((nib & 0x8) != 0, -1.0, 1.0)
        vals = grid[nib & 0x7] * sign
        sf = pw.sf_rowmajor[i : i + step].view(torch.float8_e4m3fn).float()
        scale = sf.repeat_interleave(16, dim=1)
        out[i : i + step] = ((vals * scale) / gs).to(torch.float16)
    return out


class NvFP4GemmBenchmark(Benchmark):
    seed: int = 42

    def _setup_shape(self, name, m):
        n, k = SHAPES[name]
        torch.manual_seed(self.seed)
        w = torch.randn(n, k, dtype=torch.bfloat16) * 0.02
        self.pw = self.kernel.pack(w.to(self.device))
        self.x = torch.randn(m, k, dtype=torch.bfloat16, device=self.device) * 0.125
        self.ref_w = _dequant(self.pw)
        self.xs = self.x.to(torch.float16)

    def setup_decode_qkv(self):
        self._setup_shape("qkv", 1)

    def benchmark_decode_qkv(self):
        self.out = self.kernel.gemm(self.pw, self.x)

    def verify_decode_qkv(self):
        return (self.xs @ self.ref_w.T).to(torch.bfloat16)

    def setup_decode_o(self):
        self._setup_shape("o", 1)

    def benchmark_decode_o(self):
        self.out = self.kernel.gemm(self.pw, self.x)

    def verify_decode_o(self):
        return (self.xs @ self.ref_w.T).to(torch.bfloat16)

    def setup_decode_mlp(self):
        self._setup_shape("mlp", 1)

    def benchmark_decode_mlp(self):
        self.out = self.kernel.gemm(self.pw, self.x)

    def verify_decode_mlp(self):
        return (self.xs @ self.ref_w.T).to(torch.bfloat16)

    def setup_decode_down(self):
        self._setup_shape("down", 1)

    def benchmark_decode_down(self):
        self.out = self.kernel.gemm(self.pw, self.x)

    def verify_decode_down(self):
        return (self.xs @ self.ref_w.T).to(torch.bfloat16)

    def setup_decode_lm_head(self):
        self._setup_shape("lm_head", 1)

    def benchmark_decode_lm_head(self):
        self.out = self.kernel.gemm(self.pw, self.x)

    def verify_decode_lm_head(self):
        return (self.xs @ self.ref_w.T).to(torch.bfloat16)

    def setup_prefill_mlp(self):
        self._setup_shape("mlp", 512)
        self.x = self.x * 0.2
        xq = self.kernel.quantize_reference(self.x.cpu()).to(self.device)
        self.xs = xq.to(torch.float16)

    def benchmark_prefill_mlp(self):
        self.out = self.kernel.gemm(self.pw, self.x)

    def verify_prefill_mlp(self):
        return (self.xs @ self.ref_w.T).to(torch.bfloat16)


class NvFP4RawGemvBenchmark(Benchmark):
    seed: int = 42

    def _setup_shape(self, name):
        n, k = SHAPES[name]
        torch.manual_seed(self.seed)
        w = torch.randn(n, k, dtype=torch.bfloat16) * 0.02
        pw = self.kernel.pack(w.to(self.device))
        self.qw = pw.qweight
        self.sf = pw.sf_rowmajor
        self.alpha = (1.0 / pw.global_scale).float().reshape(1)
        self.x = torch.randn(1, k, dtype=torch.bfloat16, device=self.device) * 0.125
        self.ref_w = _dequant(pw)
        self.xs = self.x.to(torch.float16)

    def setup_gemv_qkv(self):
        self._setup_shape("qkv")

    def benchmark_gemv_qkv(self):
        self.out = self.kernel._ops.ops.nvfp4_gemv(self.x, self.qw, self.sf, self.alpha)

    def verify_gemv_qkv(self):
        return (self.xs @ self.ref_w.T).to(torch.bfloat16)

    def setup_gemv_o(self):
        self._setup_shape("o")

    def benchmark_gemv_o(self):
        self.out = self.kernel._ops.ops.nvfp4_gemv(self.x, self.qw, self.sf, self.alpha)

    def verify_gemv_o(self):
        return (self.xs @ self.ref_w.T).to(torch.bfloat16)

    def setup_gemv_mlp(self):
        self._setup_shape("mlp")

    def benchmark_gemv_mlp(self):
        self.out = self.kernel._ops.ops.nvfp4_gemv(self.x, self.qw, self.sf, self.alpha)

    def verify_gemv_mlp(self):
        return (self.xs @ self.ref_w.T).to(torch.bfloat16)

    def setup_gemv_down(self):
        self._setup_shape("down")

    def benchmark_gemv_down(self):
        self.out = self.kernel._ops.ops.nvfp4_gemv(self.x, self.qw, self.sf, self.alpha)

    def verify_gemv_down(self):
        return (self.xs @ self.ref_w.T).to(torch.bfloat16)

    def setup_gemv_lm_head(self):
        self._setup_shape("lm_head")

    def benchmark_gemv_lm_head(self):
        self.out = self.kernel._ops.ops.nvfp4_gemv(self.x, self.qw, self.sf, self.alpha)

    def verify_gemv_lm_head(self):
        return (self.xs @ self.ref_w.T).to(torch.bfloat16)
