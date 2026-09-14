---
license: mit
tags:
- kernels
---
# aiter-flash-attn

Self-contained repackaging of the Triton FlashAttention MHA kernels from the
[ROCm/aiter](https://github.com/ROCm/aiter) project, exposed as a Hugging Face
Hub kernel. Provides FlashAttention on AMD ROCm GPUs (MI300X / gfx942,
gfx950, gfx1250, gfx1150, gfx1151) without taking on `aiter` as a pip dependency.

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
- gfx1150 (Strix Point / RDNA3.5)
- gfx1151 (Strix Halo / RDNA3.5)

Tuning configs for these architectures ship under `torch-ext/aiter_flash_attn/configs/`. The gfx1150 config
reuses the gfx1151 tuning, since both are RDNA3.5.

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
imports (including the `flash_attn_triton_amd` / `dao_ai` backend), vendored at
[`ROCm/aiter@8a9186d`](https://github.com/ROCm/aiter/commit/8a9186d983ed34ece2f68fe039e07f1b0abe147b).
All `from aiter.*` absolute imports have been rewritten to package-relative form
per the [Hub kernel requirements](https://huggingface.co/docs/kernels/kernel-requirements).

Two intentional local deviations from upstream:

- `utils/_triton/arch_info.py` detects the GPU arch lazily (on first `get_arch()`
  call) so the module imports in the kernel-builder Nix sandbox, which has no
  active GPU driver.
- `mha.py` resolves sliding-window attention per call so it works through the
  plain `flash_attn_func` / `flash_attn_varlen_func` entry points without an
  explicit `mha_set_impl`:
    - Under `causal=True` a right window (`window_size[1] >= 0`) is a no-op — the
      causal edge already bounds the right side — so it is normalized to `-1` and
      the call stays on the default kernel, which supports a left window together
      with attention sinks. This keeps causal sliding-window models on the tuned
      default path, **including sink models such as gpt-oss** (Gemma3, Mistral,
      Qwen2 SWA also land here).
    - A non-causal right window (which the default kernel cannot express) auto-
      selects the `dao_ai` backend, but only when the workload is compatible with
      it (no attention sink, no positional-encoding head split, not FP8).
      Sink/PE/FP8 workloads stay on the default kernel.

## License

MIT — see `LICENSE`. Upstream copyright: Advanced Micro Devices, Inc.
