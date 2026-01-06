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
rwkv = get_kernel("kernels-community/rwkv")
device = torch.device("cuda")

# RWKV WKV (Weighted Key-Value) operation
# Core operation for RWKV attention-free language models
batch_size = 2
seq_len = 128
channels = 512  # hidden dimension

# Input tensors (all must be on CUDA)
# w: time decay weights [channels] (negative values, log scale)
w = -torch.rand(channels, device=device, dtype=torch.float32) * 5
# u: bonus term [channels]
u = torch.randn(channels, device=device, dtype=torch.float32)
# k: key tensor [B, T, C]
k = torch.randn(batch_size, seq_len, channels, device=device, dtype=torch.float32)
# v: value tensor [B, T, C]
v = torch.randn(batch_size, seq_len, channels, device=device, dtype=torch.float32)
# y: output tensor [B, T, C] (pre-allocated, written in-place)
y = torch.empty(batch_size, seq_len, channels, device=device, dtype=torch.float32)

# Run RWKV WKV forward pass (writes to y in-place)
rwkv.forward(w, u, k, v, y)

print(f"W (time decay) shape: {w.shape}")
print(f"U (bonus) shape: {u.shape}")
print(f"K (key) shape: {k.shape}")
print(f"V (value) shape: {v.shape}")
print(f"Y (output) shape: {y.shape}")
print(f"Output dtype: {y.dtype}")
print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")

# Forward with state tracking (useful for generation)
y_state = torch.empty(batch_size, seq_len, channels, device=device, dtype=torch.float32)
s = torch.zeros(batch_size, channels, 3, device=device, dtype=torch.float32)  # state [B, C, 3]
rwkv.forward_with_state(w, u, k, v, y_state, s)

print(f"State shape: {s.shape}")
# W (time decay) shape: torch.Size([512])
# U (bonus) shape: torch.Size([512])
# K (key) shape: torch.Size([2, 128, 512])
# V (value) shape: torch.Size([2, 128, 512])
# Y (output) shape: torch.Size([2, 128, 512])
# Output dtype: torch.float32
# Output range: [-x.xxx, x.xxx] (varies)
# State shape: torch.Size([2, 512, 3])
