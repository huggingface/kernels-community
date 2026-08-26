---
license: bsd-3-clause
tags:
  - kernel
---

## causal-conv1d

Causal depthwise conv1d kernel by Tri Dao. Source: https://github.com/Dao-AILab/causal-conv1d/

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
causal_conv1d = get_kernel("kernels-community/causal-conv1d")
device = torch.device("cuda")

# Create test tensor for causal conv1d
# Input shape: (batch, dim, seqlen)
batch_size, dim, seqlen, width = 2, 64, 128, 4
input_tensor = torch.randn(batch_size, dim, seqlen, device=device, dtype=torch.float16)
weight = torch.randn(dim, width, device=device, dtype=torch.float32)
bias = torch.randn(dim, device=device, dtype=torch.float32)

# Custom kernel causal conv1d
out_kernel = causal_conv1d.causal_conv1d_fn(input_tensor, weight, bias)

print(f"Output: {out_kernel.shape}")
# Output: torch.Size([2, 64, 128])
```

See [scripts/readme_example.py](scripts/readme_example.py) for a complete example.
