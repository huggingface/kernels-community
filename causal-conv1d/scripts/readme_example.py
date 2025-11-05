# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "kernels"
# ]
# ///
import torch
import torch.nn.functional as F
from kernels import get_kernel

# Setup
torch.manual_seed(42)
causal_conv1d = get_kernel("kernels-community/causal-conv1d")
device = torch.device("cuda")

# Create test tensor for causal conv1d
# Input shape: (batch, dim, seqlen)
batch_size, dim, seqlen, width = 2, 64, 128, 4
input_tensor = torch.randn(batch_size, dim, seqlen, device=device, dtype=torch.float16)
weight = torch.randn(dim, width, device=device, dtype=torch.float32)
bias = torch.randn(dim, device=device, dtype=torch.float32)


# Reference implementation using PyTorch conv1d
def reference_causal_conv1d(x, weight, bias):
    x_fp32 = x.to(weight.dtype)
    out = F.conv1d(x_fp32, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    return out[..., :seqlen].to(x.dtype)


out_ref = reference_causal_conv1d(input_tensor, weight, bias)

# Custom kernel causal conv1d
out_kernel = causal_conv1d.causal_conv1d_fn(input_tensor, weight, bias)

print(f"Reference output: {out_ref.shape}")
print(f"Kernel output: {out_kernel.shape}")
print(f"Outputs close: {torch.allclose(out_kernel, out_ref, atol=1e-2, rtol=1e-3)}")
# Reference output: torch.Size([2, 64, 128])
# Kernel output: torch.Size([2, 64, 128])
# Outputs close: True
