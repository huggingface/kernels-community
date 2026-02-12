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
tinygrad_rms = get_kernel("kernels-community/tinygrad-rms")
device = torch.device("cuda")

# Create test tensor for RMSNorm
# Shape: (batch, seq_len, hidden_size) where hidden_size=1024 and num_rows divisible by 16
batch_size, seq_len, hidden_size = 32, 512, 1024
input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
epsilon = 1e-6


# Reference implementation using PyTorch
def reference_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return x / rms


out_ref = reference_rms_norm(input_tensor, epsilon)

# Custom kernel RMSNorm
out_kernel = tinygrad_rms.tinygrad_rms_norm_simple(input_tensor, epsilon)

print(f"Reference output: {out_ref.shape}")
print(f"Kernel output: {out_kernel.shape}")
print(f"Outputs close: {torch.allclose(out_kernel, out_ref, atol=1e-4, rtol=1e-4)}")
# Reference output: torch.Size([32, 512, 1024])
# Kernel output: torch.Size([32, 512, 1024])
# Outputs close: True
