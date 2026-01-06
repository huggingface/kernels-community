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
fp8_fbgemm = get_kernel("kernels-community/fp8-fbgemm")
device = torch.device("cuda")

# FP8 row-wise quantization
# Useful for efficient inference with FP8 GEMM operations
batch_size, seq_len, hidden_dim = 4, 128, 512

# Input tensor in higher precision (bf16 or fp16)
input_tensor = torch.randn(
    batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16
)

# Quantize to FP8 with per-row scaling
# Returns (quantized_tensor, scale_per_row)
quantized, scale = fp8_fbgemm.quantize_fp8_per_row(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Input dtype: {input_tensor.dtype}")
print(f"Quantized shape: {quantized.shape}")
print(f"Quantized dtype: {quantized.dtype}")
print(f"Scale shape: {scale.shape}")
print(f"Scale dtype: {scale.dtype}")

# Verify reconstruction
reconstructed = quantized.to(torch.float32) * scale.unsqueeze(-1)
max_error = (input_tensor.float() - reconstructed).abs().max()
print(f"Max reconstruction error: {max_error:.6f}")
# Input shape: torch.Size([4, 128, 512])
# Input dtype: torch.bfloat16
# Quantized shape: torch.Size([4, 128, 512])
# Quantized dtype: torch.float8_e4m3fn
# Scale shape: torch.Size([4, 128])
# Scale dtype: torch.float32
# Max reconstruction error: ~0.03 (varies)
