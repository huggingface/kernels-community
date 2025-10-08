# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "triton",
#     "numpy",
#     "kernels",
# ]
# ///

import torch
from kernels import get_kernel

# Load vllm-flash-attn3 via kernels library
vllm_flash_attn3 = get_kernel("kernels-community/vllm-flash-attn3")

# Access Flash Attention function
flash_attn_func = vllm_flash_attn3.flash_attn_func

# Set device and seed for reproducibility
device = "cuda"
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Parameters
batch_size = 2
seqlen_q = 128  # Query sequence length
seqlen_k = 256  # Key sequence length  
nheads = 8      # Number of attention heads
d = 64          # Head dimension

# Create input tensors (Q, K, V)
q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=torch.bfloat16)
k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=torch.bfloat16)
v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=torch.bfloat16)

print(f"Query shape: {q.shape}")
print(f"Key shape: {k.shape}")
print(f"Value shape: {v.shape}")

# Run Flash Attention 3
output, lse = flash_attn_func(q, k, v, causal=True)

print(f"\nOutput shape: {output.shape}")
print(f"LSE (log-sum-exp) shape: {lse.shape}")
print(f"\nAttention computation successful!")
print(f"Output tensor stats - Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")