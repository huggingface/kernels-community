---
license: apache-2.0
tags:
  - kernel
---

# vllm-flash-attn3

This is an implementation of Flash Attention 3 CUDA kernels with support for attention sinks. The attention sinks implementation was contributed to Flash Attention by the [vLLM team](https://huggingface.co/vllm-project). The [transformers team](https://huggingface.co/transformers-community) packaged the implementation and pre-built it for use with the [kernels library](https://github.com/huggingface/kernels).

## Quickstart

```bash
uv run https://raw.githubusercontent.com/huggingface/kernels-community/refs/heads/main/vllm-flash-attn3/readme_example.py
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
from kernels import get_kernel

# Load vllm-flash-attn3 via kernels library
vllm_flash_attn3 = get_kernel("kernels-community/vllm-flash-attn3")

# Access Flash Attention function
flash_attn_func = vllm_flash_attn3.flash_attn_func

# Set device and seed for reproducibility
device = "cuda"
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Parameters
batch_size = 2
seqlen_q = 128  # Query sequence length
seqlen_k = 256  # Key sequence length
nheads = 8      # Number of attention heads
d = 64          # Head dimension

# Create input tensors (Q, K, V)
q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=torch.bfloat16)
k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=torch.bfloat16)
v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=torch.bfloat16)

print(f"Query shape: {q.shape}")
print(f"Key shape: {k.shape}")
print(f"Value shape: {v.shape}")

# Run Flash Attention 3
output, lse = flash_attn_func(q, k, v, causal=True)

print(f"\nOutput shape: {output.shape}")
print(f"LSE (log-sum-exp) shape: {lse.shape}")
print(f"\nAttention computation successful!")
print(f"Output tensor stats - Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
```

## How to Use

When loading your model with transformers, provide this repository id as the source of the attention implementation:

```diff
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "<your model id on the Hub>"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
+    # Flash Attention with Sinks
+    attn_implementation="kernels-community/vllm-flash-attn3”,
)
```

This will automatically resolve and download the appropriate code for your architecture. See more details in [this post](https://huggingface.co/blog/hello-hf-kernels).

## Credits

- [Tri Dao](https://huggingface.co/tridao) and team for Flash Attention and [Flash Attention 3](https://tridao.me/blog/2024/flash3/).
- The [vLLM team](https://huggingface.co/vllm-project) for their implementation and their contribution of attention sinks.
- The [transformers team](https://huggingface.co/transformers-community) for packaging, testing, building and making it available for use with the [kernels library](https://github.com/huggingface/kernels).

