---
license: bsd-3-clause
tags:
- kernel
---

![Status](https://hubwebhook.dholtz.com/shield?repo=kernels-community/rotary)

## rotary

rotary embedding kernel from [Flash Attention](https://github.com/Dao-AILab/flash-attention/tree/main/csrc/rotary).

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
rotary = get_kernel("kernels-community/rotary")
device = torch.device("cuda")

# Create test tensors for rotary position embeddings
batch_size, seqlen, num_heads, head_dim = 2, 128, 8, 64
rotary_dim = 32

query = torch.randn(
    batch_size, seqlen, num_heads, head_dim, device=device, dtype=torch.float32
)
cos = torch.randn(seqlen, 1, rotary_dim, device=device, dtype=torch.float32)
sin = torch.randn(seqlen, 1, rotary_dim, device=device, dtype=torch.float32)

# Split query into rotary parts
q1 = query[..., :rotary_dim]
q2 = query[..., rotary_dim : 2 * rotary_dim]

# Apply rotary embeddings (in-place operation)
rotary.apply_rotary(q1, q2, cos, sin, q1, q2, conj=False)

print(f"Output shape: {q1.shape}")
# Output shape: torch.Size([2, 128, 8, 32])
```

See [scripts/readme_example.py](scripts/readme_example.py) for a complete example.