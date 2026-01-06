# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "kernels"
# ]
# ///
import torch
from kernels import get_kernel

# Setup
torch.manual_seed(42)
rmsnorm = get_kernel("kernels-community/rmsnorm")
device = torch.device("cpu")  # RMSNorm CPU kernel

# RMS Layer Normalization
# Used in models like LLaMA, Mistral, and other modern transformers
batch_size, seq_len, hidden_dim = 2, 64, 1024

# Input tensor
input_tensor = torch.randn(
    batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16
)

# Learnable weight parameter [hidden_dim]
weight = torch.ones(hidden_dim, device=device, dtype=torch.bfloat16)

# Epsilon for numerical stability
eps = 1e-6

# Apply RMS normalization
output = rmsnorm.apply_rms_norm(input_tensor, weight, eps)


# Reference implementation
def reference_rmsnorm(x, weight, eps):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(x.dtype)


ref_output = reference_rmsnorm(input_tensor, weight, eps)

print(f"Input shape: {input_tensor.shape}")
print(f"Input dtype: {input_tensor.dtype}")
print(f"Weight shape: {weight.shape}")
print(f"Output shape: {output.shape}")
print(f"Output dtype: {output.dtype}")
print(f"Outputs close: {torch.allclose(output.float(), ref_output.float(), atol=1e-2)}")
# Input shape: torch.Size([2, 64, 1024])
# Input dtype: torch.bfloat16
# Weight shape: torch.Size([1024])
# Output shape: torch.Size([2, 64, 1024])
# Output dtype: torch.bfloat16
# Outputs close: True
