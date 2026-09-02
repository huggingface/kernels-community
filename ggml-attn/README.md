---
library_name: kernels
license: mit
tags:
  - kernel
---

## ggml-attn

Flash attention from [llama.cpp](https://github.com/ggml-org/llama.cpp), as a torch op (`flash_attn`) and
as a `transformers` attention implementation (`flash_attn_forward`). Grouped-query attention is native, so
k and v are passed unexpanded.

Both of upstream's paths are ported — the vector kernel for `n_q < 20`, the tiled one above it — so
decode and prefill both run on ggml's kernels. A head-dim pair neither has a template for raises rather
than quietly falling back to torch; ask `supports_flash_attn`.

## Usage

```python
import torch
from kernels import get_kernel

attn = get_kernel("kernels-community/ggml-attn", version=1)

q = torch.randn(1, 16, 1, 128, device="mps")    # (n_seqs, n_heads, n_q, head_dim)
k = torch.randn(1, 4, 512, 128, device="mps")   # 4 kv heads, left unexpanded
v = torch.randn(1, 4, 512, 128, device="mps")

out = attn.flash_attn(q, k, v)                  # (1, 1, 16, 128) — tokens before heads
```

Or as a model's attention implementation:

```python
model = AutoModelForCausalLM.from_pretrained(
    ..., attn_implementation="kernels-community/ggml-attn"
)
```
