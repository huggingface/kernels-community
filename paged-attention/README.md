---
license: apache-2.0
tags:
  - kernel
---

![Status](https://hubwebhook.dholtz.com/shield?repo=kernels-community/paged-attention)

## attention

Paged attention kernels from [vLLM](https://github.com/vllm-project/) and [mistral.rs](https://github.com/EricLBuehler/mistral.rs).

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
paged_attention = get_kernel("kernels-community/paged-attention")
device = torch.device("cuda")

# Paged Attention parameters
num_seqs, num_heads, head_size, block_size = 2, 8, 64, 16
max_seq_len, num_blocks = 128, 32

query = torch.randn(num_seqs, num_heads, head_size, device=device, dtype=torch.float16)
key_cache = torch.randn(
    num_blocks, num_heads, head_size, block_size, device=device, dtype=torch.float16
)
value_cache = torch.randn(
    num_blocks, num_heads, head_size, block_size, device=device, dtype=torch.float16
)
block_tables = torch.randint(
    0,
    num_blocks,
    (num_seqs, (max_seq_len + block_size - 1) // block_size),
    device=device,
    dtype=torch.int32,
)
seq_lens = torch.tensor([64, 96], device=device, dtype=torch.int32)
scale = 1.0 / (head_size**0.5)
output = torch.empty_like(query)

k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

# Run paged attention
paged_attention.paged_attention_v1(
    output,
    query,
    key_cache,
    value_cache,
    num_kv_heads=num_heads,
    scale=scale,
    block_tables=block_tables,
    seq_lens=seq_lens,
    block_size=block_size,
    max_seq_len=max_seq_len,
    alibi_slopes=None,
    kv_cache_dtype="auto",
    k_scale=k_scale,
    v_scale=v_scale,
)

print(f"Output shape: {output.shape}")
# Output shape: torch.Size([2, 8, 64])
```

See [scripts/readme_example.py](scripts/readme_example.py) for a complete example.
