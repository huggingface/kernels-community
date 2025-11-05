---
tags:
  - kernel
---

# MRA (Multi-Resolution Attention)

MRA kernels for transformers implementing efficient sparse attention operations.

## Usage

```python
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
indices_per_block = 4
indices = torch.randint(
    0,
    total_blocks,
    (batch_size * num_heads * num_blocks_per_seq, indices_per_block),
    device=device,
    dtype=torch.int32,
)

# Compute sparse attention scores using mm_to_sparse
sparse_scores = mra.mm_to_sparse(dense_a, dense_b, indices)

# Use sparse_dense_mm to compute output from sparse scores
dense_values = torch.randn(
    batch_size * num_heads, seq_len, head_dim, device=device, dtype=torch.float32
)

output = mra.sparse_dense_mm(
    sparse_scores, indices, dense_values, batch_size * num_heads * num_blocks_per_seq
)

print(f"Output shape: {output.shape}")
# Output shape: torch.Size([16, 128, 64, 32])
```

See [scripts/readme_example.py](scripts/readme_example.py) for a complete example.