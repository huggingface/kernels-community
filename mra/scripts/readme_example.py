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
mra = get_kernel("kernels-community/mra")
device = torch.device("cuda")

# MRA parameters
batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
block_size = 16
num_blocks_per_seq = seq_len // block_size
total_blocks = batch_size * num_heads * num_blocks_per_seq

# Create dense matrices for attention computation
dense_a = torch.randn(
    batch_size * num_heads, seq_len, head_dim, device=device, dtype=torch.float32
)
dense_b = torch.randn(
    batch_size * num_heads, head_dim, seq_len, device=device, dtype=torch.float32
)

# Create block indices for sparse attention pattern
# Each block attends to a subset of other blocks
indices_per_block = 4
indices = torch.randint(
    0,
    total_blocks,
    (batch_size * num_heads * num_blocks_per_seq, indices_per_block),
    device=device,
    dtype=torch.int32,
)

# Compute sparse attention scores using mm_to_sparse
# This computes dense_a @ dense_b but only keeps values at specified block indices
sparse_scores = mra.mm_to_sparse(dense_a, dense_b, indices)

print(f"Dense A shape: {dense_a.shape}")
print(f"Dense B shape: {dense_b.shape}")
print(f"Indices shape: {indices.shape}")
print(f"Sparse scores shape: {sparse_scores.shape}")

# Now use sparse_dense_mm to compute output from sparse scores
dense_values = torch.randn(
    batch_size * num_heads, seq_len, head_dim, device=device, dtype=torch.float32
)

output = mra.sparse_dense_mm(
    sparse_scores, indices, dense_values, batch_size * num_heads * num_blocks_per_seq
)

print(f"Output shape: {output.shape}")
# Dense A shape: torch.Size([16, 128, 64])
# Dense B shape: torch.Size([16, 64, 128])
# Indices shape: torch.Size([128, 4])
# Sparse scores shape: torch.Size([16, 4, 32, 32])
# Output shape: torch.Size([16, 128, 64, 32])
