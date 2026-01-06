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
mamba_ssm = get_kernel("kernels-community/mamba-ssm")
device = torch.device("cuda")

# Mamba Selective Scan parameters
# Used in Mamba state space models for efficient sequence modeling
batch_size = 2
seq_len = 128
d_inner = 64  # inner dimension (d_model * expand)
d_state = 16  # SSM state dimension

# Input tensors for selective scan
# u: input sequence [B, L, D]
u = torch.randn(batch_size, d_inner, seq_len, device=device, dtype=torch.float32)
# delta: time step / discretization [B, L, D]
delta = torch.randn(batch_size, d_inner, seq_len, device=device, dtype=torch.float32).softplus()
# A: state matrix [D, N] (negative, typically learned)
A = -torch.rand(d_inner, d_state, device=device, dtype=torch.float32)
# B: input matrix [B, N, L]
B = torch.randn(batch_size, d_state, seq_len, device=device, dtype=torch.float32)
# C: output matrix [B, N, L]
C = torch.randn(batch_size, d_state, seq_len, device=device, dtype=torch.float32)
# D: skip connection (optional) [D]
D = torch.randn(d_inner, device=device, dtype=torch.float32)
# z: gate (optional) [B, D, L]
z = torch.randn(batch_size, d_inner, seq_len, device=device, dtype=torch.float32)
# delta_bias (optional) [D]
delta_bias = torch.randn(d_inner, device=device, dtype=torch.float32)

# Run selective scan forward pass
outputs = mamba_ssm.selective_scan_fwd(
    u, delta, A, B, C,
    D, z, delta_bias,
    delta_softplus=True
)

out, x, *rest = outputs

print(f"Input u shape: {u.shape}")
print(f"Delta shape: {delta.shape}")
print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")
print(f"C shape: {C.shape}")
print(f"Output shape: {out.shape}")
print(f"State shape: {x.shape}")
# Input u shape: torch.Size([2, 64, 128])
# Delta shape: torch.Size([2, 64, 128])
# A shape: torch.Size([64, 16])
# B shape: torch.Size([2, 16, 128])
# C shape: torch.Size([2, 16, 128])
# Output shape: torch.Size([2, 64, 128])
# State shape: torch.Size([2, 64, 16])
