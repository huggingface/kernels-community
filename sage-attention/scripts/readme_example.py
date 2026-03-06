# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "kernels",
#   "triton",
# ]
# ///
import torch
from kernels import get_kernel, get_local_kernel
from pathlib import Path

# Setup
torch.manual_seed(42)
sage_attention = get_local_kernel(Path("build"), "sage_attention")

print(sage_attention)

# Try calling sageattn to verify no duplicate registration error
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    B, H, L, D = 1, 8, 256, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device=device)
    out = sage_attention.sageattn(q, k, v)
    print(f"sageattn output shape: {out.shape}")
else:
    print("No CUDA device available - but kernel loaded without registration errors")
