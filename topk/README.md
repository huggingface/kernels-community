---
library_name: kernels
license: mit
tags:
  - kernel
---

## topk

Top-k over a small row, for a MoE router. One threadgroup per row, one reduction pass per output:
`k*n` comparisons, but `k` and `n` are small and the launch dominates either way. Optionally softmaxes
the `k` it selected, which is what a router wants and saves a second dispatch.

Not a port. ggml has no top-k of its own — `GGML_OP_TOP_K` dispatches `kernel_argsort_f32_i32_desc`,
a full bitonic sort — and torch's MPS `topk` is a full sort too. Selecting the largest 8 of 256 logits
does not need the row ordered: 26 us here against 71 us for either sort, once per layer per token.

`indices` comes back as int32, which is what an expert-routed matmul wants, so routing them onward
costs no cast.

## Usage

```python
import torch
from kernels import get_kernel

topk = get_kernel("kernels-community/topk", version=1)

logits = torch.randn(1, 256, device="mps")          # one row of router logits

values, indices = topk.top_k(logits, 8)             # (1, 8) f32, (1, 8) int32
weights, experts = topk.top_k(logits, 8, True)      # values softmaxed over the selected 8
```
