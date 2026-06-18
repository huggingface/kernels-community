---
license: mit
tags:
- kernels
---
# aiter-rope

Triton RoPE (Rotary Position Embedding) kernels for AMD ROCm, repackaged from
the [ROCm/aiter](https://github.com/ROCm/aiter) project as a Hugging Face Hub
kernel.

Designed as a drop-in ROCm replacement for `kernels-community/rotary`:
exports `apply_rotary_transformers` with the same signature, so transformers
can route to it under the `"rocm"` entry of
`_KERNEL_MAPPING["rotary_pos_emb"]` with zero model-side changes.

Original code: https://github.com/ROCm/aiter (MIT, © Advanced Micro Devices, Inc.).

## Functions

### `apply_rotary_transformers(q, k, cos, sin, unsqueeze_dim=1)`

Apply NEOX-style RoPE to query and key tensors.

- `q`, `k`: `(batch, num_heads, seq, head_dim)`.
- `cos`, `sin`: `(batch, seq, head_dim // 2)` — the pre-`unsqueeze` form
  transformers produces from `position_ids`.
- Returns `(q_embed, k_embed)` in the same shape as `q`, `k`.

Also exposes the lower-level entry points from AITER for direct use:
`rope_cached_fwd`, `rope_cached_fwd_inplace`, and the `RotateStyle` enum.

## Supported hardware

Triton-portable across AMD ROCm GPUs. Verified on:

- gfx942 (MI300X)
- gfx950 (MI355X)

## Origin

Code is vendored from `aiter/ops/triton/rope/rope.py` and its transitive
imports (the underlying Triton kernels in `_triton_kernels/rope/rope.py` and
the small `AiterTritonLogger` utility). All `from aiter.*` absolute imports
have been rewritten to package-relative form per the
[Hub kernel requirements](https://huggingface.co/docs/kernels/kernel-requirements).

## Quickstart

```python
import torch
from kernels import get_kernel

rope = get_kernel("kernels-community/aiter-rope")

B, H, S, D = 2, 8, 32, 64
q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
cos = torch.randn(B, S, D // 2, device="cuda", dtype=torch.float16)
sin = torch.randn(B, S, D // 2, device="cuda", dtype=torch.float16)

q_embed, k_embed = rope.apply_rotary_transformers(q, k, cos, sin)
```
