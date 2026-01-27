---
license: mit
tags:
- kernels
---
# triton-kernels

triton-kernels is a set of kernels that enable fast moe on different architectures. These kernels are compatible with different precision (e.g bf16, mxfp4)

Original code here https://github.com/triton-lang/triton/tree/main/python/triton_kernels

The current version is the following commit 7d0efaa7231661299284a603512fce4fa255e62c

Note that we can't update those kernels as we wish as some commits might rely on triton main. We need to wait for a new release unfortunately. 
See releated issue https://github.com/triton-lang/triton/issues/7818


## Quickstart

```bash
uv run https://huggingface.co/kernels-community/gpt-oss-triton-kernels/raw/main/readme_example.py
```

```python
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
import sys
from kernels import get_kernel

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Load triton_kernels module via kernels library
triton_kernels = get_kernel("kernels-community/gpt-oss-triton-kernels")

# Access modules directly from the loaded kernel
swiglu = triton_kernels.swiglu
routing = triton_kernels.routing

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# SwiGLU example
x = torch.randn(512, 1024, device=device, dtype=torch.bfloat16)
y = swiglu.swiglu_torch(x, 0.5, swiglu.PrecisionConfig(limit=1.0))
print(f"SwiGLU: {x.shape} -> {y.shape}")

# Routing example
logits = torch.randn(128, 8, device=device, dtype=torch.float16)
routing_data, gather_idx, scatter_idx = routing.routing_torch(logits, n_expts_act=2)
print(f"Routing: {routing_data.expt_hist.sum()} tokens routed")

# MoE integrated
n_tokens = routing_data.expt_hist.sum().item()
x_moe = torch.randn(n_tokens, 512, device=device, dtype=torch.bfloat16)
y_moe = swiglu.swiglu_torch(x_moe, 0.5, swiglu.PrecisionConfig(limit=1.0))
print(f"MoE SwiGLU: {x_moe.shape} -> {y_moe.shape}")
```