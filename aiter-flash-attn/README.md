---
license: mit
tags:
- kernels
---
# aiter-flash-attn

Self-contained repackaging of the Triton FlashAttention MHA kernels from the
[ROCm/aiter](https://github.com/ROCm/aiter) project, exposed as a Hugging Face
Hub kernel. Provides FlashAttention on AMD ROCm GPUs (MI300X / gfx942,
gfx950, gfx1250) without taking on `aiter` as a pip dependency.

Original code: https://github.com/ROCm/aiter (MIT, © Advanced Micro Devices, Inc.).

The exported API matches the `flash-attn` v2 surface used by `transformers`'s
flash-attention fallback path, so this kernel can be loaded as the ROCm entry
in `FLASH_ATTN_KERNEL_FALLBACK`.

## Functions

### `flash_attn_func(q, k, v, ...)`

Dense FlashAttention forward (and backward for training).
`q, k, v` shape: `(batch, seqlen, nheads, headdim)`.

### `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ...)`

Variable-length / packed FlashAttention. `q` shape: `(total_q, nheads, headdim)`;
`cu_seqlens_*` are int32 cumulative offsets.

Both entry points accept the standard FA2 kwargs (`dropout_p`, `softmax_scale`,
`causal`, `window_size`, `alibi_slopes`, ...) plus a `sink` argument for
learnable attention sinks (e.g. gpt-oss).

## Supported hardware

- gfx942 (MI300X)
- gfx950 (MI355X)
- gfx1250

Tuning configs for these architectures ship under `torch-ext/aiter_flash_attn/configs/`.

## Quickstart

```python
import torch
from kernels import get_kernel

flash_attn = get_kernel("kernels-community/aiter-flash-attn")

q = torch.randn(2, 32, 8, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 32, 8, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 32, 8, 64, device="cuda", dtype=torch.float16)

out = flash_attn.flash_attn_func(q, k, v, causal=True)
```

## Origin

Code is taken from `aiter/ops/triton/attention/mha.py` and its transitive
imports, with the `dao_ai` impl path stripped (it depended on a separate
`flash_attn_triton_amd` subpackage we don't need here). All `from aiter.*`
absolute imports have been rewritten to package-relative form per the
[Hub kernel requirements](https://huggingface.co/docs/kernels/kernel-requirements).

## License

MIT — see `LICENSE`. Upstream copyright: Advanced Micro Devices, Inc.
