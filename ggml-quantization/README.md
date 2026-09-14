---
license: mit
tags:
  - kernel
---

## ggml-quantization

GGUF quantization kernels from [llama.cpp](https://github.com/ggml-org/llama.cpp), computing directly on
the packed blocks of a quantized checkpoint rather than on a dense copy of its weights.

- `mul_mat_vec` — fused dequantize + gemv, for up to `MAX_GEMV_ROWS` rows
- `dequantize` — blocks to values
- `get_rows` — gathers rows, unpacking as it goes
- `mul_mat_id` — one dispatch for a bank of routed experts, given the router's choices

`GEMV_TYPES` lists the quantization types this build has a gemv for.

## Usage

```python
import torch
from kernels import get_kernel

k = get_kernel("kernels-community/ggml-quantization", version=1)

Q4_K = 12                          # ggml type id; `k.GEMV_TYPES` lists what this build covers
out_features = in_features = 4096
# a GGUF weight as stored: one row per output feature, 144 bytes per 256-element Q4_K block
blocks = torch.randint(0, 256, (out_features, in_features // 256 * 144), dtype=torch.uint8, device="mps")
x = torch.randn(1, in_features, device="mps")

y = k.mul_mat_vec(blocks, x, Q4_K, out_features)                          # (1, 4096) f32
w = k.dequantize(blocks, Q4_K, out_features, in_features, torch.bfloat16)  # (4096, 4096)
rows = k.get_rows(blocks, torch.tensor([3, 7], device="mps"), Q4_K, in_features, torch.bfloat16)
```
