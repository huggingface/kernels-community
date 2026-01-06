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
relu_kernel = get_kernel("kernels-community/relu")
device = torch.device("cuda")

# ReLU activation kernel
# Note: For production use, prefer the 'activation' kernel instead.
# This kernel is primarily for testing purposes.
batch_size, seq_len, hidden_dim = 4, 128, 512

# Input tensor
input_tensor = torch.randn(
    batch_size, seq_len, hidden_dim, device=device, dtype=torch.float16
)

# Allocate output tensor
output = torch.empty_like(input_tensor)

# Apply ReLU using custom kernel (in-place to output)
relu_kernel.relu(output, input_tensor)

# Reference implementation
ref_output = torch.nn.functional.relu(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Input dtype: {input_tensor.dtype}")
print(f"Output shape: {output.shape}")
print(f"Output dtype: {output.dtype}")
print(f"Outputs match: {torch.allclose(output, ref_output)}")
print(f"Min output: {output.min():.4f}")
print(f"Max output: {output.max():.4f}")
# Input shape: torch.Size([4, 128, 512])
# Input dtype: torch.float16
# Output shape: torch.Size([4, 128, 512])
# Output dtype: torch.float16
# Outputs match: True
# Min output: 0.0000
# Max output: ~4.0 (varies)
