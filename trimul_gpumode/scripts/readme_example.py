# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "kernels",
#   "triton"
# ]
# ///
import torch
from kernels import get_kernel

# Setup
torch.manual_seed(42)
trimul = get_kernel("kernels-community/trimul-gpumode")
device = torch.device("cuda")

# TriMul: Triangle Multiplicative Update from AlphaFold2
# Optimized Triton kernels from the GPUMODE competition
# Available kernels: kernel_a100, kernel_h100, kernel_b200, kernel_mi300, kernel_global
batch_size = 1
seq_len = 128
dim = 128
hidden_dim = 128

# Input tensor [batch_size, seq_len, seq_len, dim]
input_tensor = torch.randn(
    batch_size, seq_len, seq_len, dim, device=device, dtype=torch.float16
)

# Mask tensor [batch_size, seq_len, seq_len]
mask = torch.ones(batch_size, seq_len, seq_len, device=device, dtype=torch.bool)

# Model weights (simulating a TriMul module)
weights = {
    "norm.weight": torch.ones(dim, device=device, dtype=torch.float32),
    "norm.bias": torch.zeros(dim, device=device, dtype=torch.float32),
    "left_proj.weight": torch.randn(hidden_dim, dim, device=device, dtype=torch.float32) * 0.02,
    "right_proj.weight": torch.randn(hidden_dim, dim, device=device, dtype=torch.float32) * 0.02,
    "left_gate.weight": torch.randn(hidden_dim, dim, device=device, dtype=torch.float32) * 0.02,
    "right_gate.weight": torch.randn(hidden_dim, dim, device=device, dtype=torch.float32) * 0.02,
    "out_gate.weight": torch.randn(hidden_dim, dim, device=device, dtype=torch.float32) * 0.02,
    "to_out_norm.weight": torch.ones(hidden_dim, device=device, dtype=torch.float32),
    "to_out_norm.bias": torch.zeros(hidden_dim, device=device, dtype=torch.float32),
    "to_out.weight": torch.randn(dim, hidden_dim, device=device, dtype=torch.float32) * 0.02,
}

config = {"dim": dim, "hidden_dim": hidden_dim}

# Pack data tuple as expected by kernel
data = (input_tensor, mask.flatten(), weights, config)

# Run the kernel (kernel_global works across architectures)
output = trimul.kernel_global(data)

print(f"Input shape: {input_tensor.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Output shape: {output.shape}")
print(f"Output dtype: {output.dtype}")
# Input shape: torch.Size([1, 128, 128, 128])
# Mask shape: torch.Size([1, 128, 128])
# Output shape: torch.Size([1, 128, 128, 128])
# Output dtype: torch.float32
