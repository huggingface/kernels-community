---
tags:
- kernel
---
This CUDA extension implements fused dropout + residual + LayerNorm from the [flash-attention](https://github.com/Dao-AILab/flash-attention/tree/main/csrc/layer_norm) repo.

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
layer_norm = get_kernel("kernels-community/layer-norm")
device = torch.device("cuda")

# Create test tensor
batch_size, seq_len, hidden_dim = 2, 5, 768
input_tensor = torch.randn(
    batch_size, seq_len, hidden_dim, device=device, dtype=torch.float16
)
weight = torch.ones(hidden_dim, device=device, dtype=torch.float16)
epsilon = 1e-5

# Custom kernel LayerNorm
out_kernel = layer_norm.dropout_add_ln_fwd(
    input=input_tensor.view(-1, hidden_dim),
    gamma=weight,
    beta=None,
    rowscale=None,
    colscale=None,
    x0_subset=None,
    z_subset=None,
    dropout_p=0.0,
    epsilon=epsilon,
    rowscale_const=1.0,
    z_numrows=seq_len,
    gen=None,
    residual_in_fp32=False,
    is_rms_norm=False,
)[0].view(batch_size, seq_len, hidden_dim)

print(f"Output: {out_kernel.shape}")
# Output: torch.Size([2, 5, 768])
```

See [scripts/readme_example.py](scripts/readme_example.py) for a complete example.