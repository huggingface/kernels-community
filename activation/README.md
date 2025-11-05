---
license: apache-2.0
tags:
  - kernel
---

## Activation

Activation kernels from [vLLM](https://github.com/vllm-project/vllm/blob/main/csrc/activation_kernels.cu).

Copyright 2023-2024, the vLLM team.

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
activation = get_kernel("kernels-community/activation")
device = torch.device("cuda")

# Create test tensor for SwiGLU (SiLU and Mul)
# Input has shape (num_tokens, 2 * hidden_dim)
num_tokens, hidden_dim = 128, 512
input_tensor = torch.randn(
    num_tokens, 2 * hidden_dim, device=device, dtype=torch.float16
)

# Custom kernel SwiGLU
out_shape = input_tensor.shape[:-1] + (hidden_dim,)
out_kernel = torch.empty(out_shape, dtype=input_tensor.dtype, device=device)
out_kernel = activation.silu_and_mul(out_kernel, input_tensor)

print(f"Output: {out_kernel.shape}")
# Output: torch.Size([128, 512])
```

See [scripts/readme_example.py](scripts/readme_example.py) for a complete example.
