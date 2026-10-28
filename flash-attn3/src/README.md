---
license: bsd-3-clause
tags:
  - kernel
---

# Flash Attention 3

Flash Attention is a fast and memory-efficient implementation of the
attention mechanism, designed to work with large models and long sequences.
This is a Hugging Face compliant kernel build of Flash Attention.

Original code here [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention).

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
flash_attn3 = get_kernel("kernels-community/flash-attn3")
device = torch.device("cuda")

# Create test tensors
batch_size, seqlen, num_heads, head_dim = 2, 512, 8, 64
query = torch.randn(
    batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16
)
key = torch.randn(
    batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16
)
value = torch.randn(
    batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16
)

# Flash Attention 3
out_flash, softmax_lse = flash_attn3.flash_attn_func(
    query, key, value, softmax_scale=None, causal=False
)

print(f"Output: {out_flash.shape}")
# Output: torch.Size([2, 512, 8, 64])
```

See [scripts/readme_example.py](scripts/readme_example.py) for a complete example.
