# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "kernels"
# ]
# ///
"""
DeepGEMM example via kernel-builder.

Demonstrates cuBLASLt GEMM (any GPU) and FP8 GEMM (SM90+ only).
Computes D = A @ B.T and compares with a torch reference.
"""
import torch
from kernels import get_local_kernel
from pathlib import Path

deep_gemm = get_local_kernel(Path("build"), "deep_gemm")

m, n, k = 256, 1024, 512
device = "cuda"

a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
b = torch.randn((n, k), device=device, dtype=torch.bfloat16)
ref = a @ b.T


def compare(name, result, ref):
    cos = torch.nn.functional.cosine_similarity(
        result.float().flatten(), ref.float().flatten(), dim=0
    )
    diff = (result.float() - ref.float()).abs().max().item()
    print(f"[{name}] shape: {m}x{n}x{k}, cosine_sim: {cos.item():.6f}, max_diff: {diff:.4f}")


# --- cuBLASLt GEMM (works on any GPU) ---
d = torch.empty((m, n), device=device, dtype=torch.bfloat16)
deep_gemm.cublaslt_gemm_nt(a, b, d)
compare("cuBLASLt BF16", d, ref)

# --- FP8 GEMM (requires SM90+ / Hopper+) ---
arch = torch.cuda.get_device_capability()[0]
if arch >= 9:
    def per_token_cast_to_fp8(x, gran_k=128):
        m, n = x.shape
        padded_n = (n + gran_k - 1) // gran_k * gran_k
        x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
        x_padded[:, :n] = x
        x_view = x_padded.view(m, -1, gran_k)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        sf = x_amax / 448.0
        data = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn)
        return data.view(m, padded_n)[:, :n].contiguous(), sf

    a_fp8 = per_token_cast_to_fp8(a)
    b_fp8 = per_token_cast_to_fp8(b)
    d_fp8 = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt(a_fp8, b_fp8, d_fp8)
    compare("FP8 1D2D", d_fp8, ref)
else:
    print(f"[FP8 GEMM] Skipped: requires SM90+ (Hopper), detected SM{arch}x")
