---
tags:
- kernels
- cuda
---
# ThriftAttention

ThriftAttention: Selective Mixed Precision for Long-Context FP4 Attention

[arxiv.org/abs/2605.23081](https://arxiv.org/pdf/2605.23081)

This is the Kernel Hub package for [ThriftAttention](https://github.com/joesharratt1229/ThriftAttention).

The goal is simple: make ThriftAttention load through Transformers with the Kernel Hub attention path.

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="kernels-community/thrift-attention",
)
```

This package exposes FlashAttention-style entry points for the Transformers loader:

```python
flash_attn_func
flash_attn_varlen_func
```

ThriftAttention currently targets CUDA 12.8 or newer and SM120 GPUs.
