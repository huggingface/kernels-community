---
library_name: kernels
license: mit
tags:
  - kernel
---

## ggml-gated-delta-net

The gated delta rule from [llama.cpp](https://github.com/ggml-org/llama.cpp) as a single kernel
(`gated_delta_net`) — the linear-attention recurrence behind Qwen3-Next and Qwen3.5, which eager torch
spells out in ~200 ops per layer. `l2_norm` is exposed alongside it.

`q` and `k` carry one head per value head rather than being pre-expanded, and the recurrent `state` is
indexed `[value][key]` — store what the op returns rather than transposing it. Ask
`supports_gated_delta_net` for the head dims it covers.

## Usage

```python
import torch
from kernels import get_kernel

gdn = get_kernel("kernels-community/ggml-gated-delta-net", version=1)

n_seqs, n_tokens, n_heads, head_dim = 1, 1, 32, 128
q = torch.randn(n_seqs, n_tokens, n_heads, head_dim, device="mps")
k = torch.randn_like(q)                                    # one head per value head, not expanded
v = torch.randn_like(q)
g = torch.randn(n_seqs, n_tokens, n_heads, device="mps")   # log-domain gate
beta = torch.rand(n_seqs, n_tokens, n_heads, device="mps")
state = torch.zeros(n_seqs, n_heads, head_dim, head_dim, device="mps")

out, state = gdn.gated_delta_net(q, k, v, g, beta, state)  # (1, 1, 32, 128), (1, 32, 128, 128)
```
