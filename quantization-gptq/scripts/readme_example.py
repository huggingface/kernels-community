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
gptq = get_kernel("kernels-community/quantization-gptq")
device = torch.device("cpu")  # GPTQ CPU kernel

# GPTQ 4-bit quantized GEMM forward pass
# Popular quantization method for LLMs
batch_size = 2
seq_len = 16
in_features = 64
out_features = 128
blocksize = 64  # GPTQ block size (typically 128)

# Input activation tensor [B * seq_len, in_features]
input_tensor = torch.randn(
    batch_size * seq_len, in_features, device=device, dtype=torch.bfloat16
)

# Simulate GPTQ quantized weights
# weight: packed 4-bit weights [out_features, in_features // 2]
weight = torch.randint(
    0, 255, (out_features, in_features // 2), device=device, dtype=torch.uint8
)

# zeros: zero points for asymmetric quantization [num_blocks]
num_blocks = (out_features * in_features) // blocksize
zeros = torch.zeros(num_blocks, device=device, dtype=torch.float32)

# absmax: scaling factors per block [num_blocks]
absmax = torch.rand(num_blocks, device=device, dtype=torch.float32) * 2

# Run GPTQ 4-bit GEMM forward
output = gptq.gemm_int4_forward(input_tensor, weight, zeros, absmax, blocksize)

print(f"Input shape: {input_tensor.shape}")
print(f"Input dtype: {input_tensor.dtype}")
print(f"Weight shape (packed 4-bit): {weight.shape}")
print(f"Zeros shape: {zeros.shape}")
print(f"Absmax shape: {absmax.shape}")
print(f"Block size: {blocksize}")
print(f"Output shape: {output.shape}")
print(f"Output dtype: {output.dtype}")
# Input shape: torch.Size([32, 64])
# Input dtype: torch.bfloat16
# Weight shape (packed 4-bit): torch.Size([128, 32])
# Zeros shape: torch.Size([128])
# Absmax shape: torch.Size([128])
# Block size: 64
# Output shape: torch.Size([32, 128])
# Output dtype: torch.bfloat16
