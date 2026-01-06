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
eetq = get_kernel("kernels-community/quantization-eetq")
device = torch.device("cuda")

# EETQ: 8-bit weight, 16-bit activation GEMM
# Efficient for inference with INT8 quantized weights
batch_size = 2
seq_len = 64
in_features = 512
out_features = 1024

# Input activation tensor [B * seq_len, in_features] in fp16
input_tensor = torch.randn(
    batch_size * seq_len, in_features, device=device, dtype=torch.float16
)

# Quantized weights [out_features, in_features] as int8
weight = torch.randint(
    -128, 127, (out_features, in_features), device=device, dtype=torch.int8
)

# Per-channel scale factors [out_features]
scale = torch.rand(out_features, device=device, dtype=torch.float16) * 0.1

# Run w8_a16 GEMM (8-bit weights, 16-bit activations)
output = eetq.w8_a16_gemm(input_tensor, weight, scale)

print(f"Input shape: {input_tensor.shape}")
print(f"Input dtype: {input_tensor.dtype}")
print(f"Weight shape (int8): {weight.shape}")
print(f"Weight dtype: {weight.dtype}")
print(f"Scale shape: {scale.shape}")
print(f"Output shape: {output.shape}")
print(f"Output dtype: {output.dtype}")
# Input shape: torch.Size([128, 512])
# Input dtype: torch.float16
# Weight shape (int8): torch.Size([1024, 512])
# Weight dtype: torch.int8
# Scale shape: torch.Size([1024])
# Output shape: torch.Size([128, 1024])
# Output dtype: torch.float16
