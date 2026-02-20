# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "kernels"
# ]
# ///


# CUDA_HOME=/usr/local/cuda-12.9 uv run scripts/readme_example.py
import torch
from kernels import get_local_kernel, get_kernel
from pathlib import Path

# deep_gemm = get_local_kernel(Path("build"), "deep_gemm")
deep_gemm = get_kernel("drbh/deep-gemm", version=1)

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
    # SFA: per-row (1, 128), SFB: per-block (128, 128) — SM90 recipe
    a_fp8 = deep_gemm.utils.per_token_cast_to_fp8(a, use_ue8m0=False)
    b_fp8 = deep_gemm.utils.per_block_cast_to_fp8(b, use_ue8m0=False)
    d_fp8 = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt(a_fp8, b_fp8, d_fp8)
    compare("FP8 1D2D", d_fp8, ref)
else:
    print(f"[FP8 GEMM] Skipped: requires SM90+ (Hopper), detected SM{arch}x")
