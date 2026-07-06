---
license: mit
tags:
- kernels
---
# aiter-flash-attn-ck

Self-contained repackaging of the **Composable Kernel (CK) FlashAttention**
forward kernels from the [ROCm/aiter](https://github.com/ROCm/aiter) project,
exposed as a Hugging Face Hub kernel. This is the compiled HIP counterpart to
the Triton-based [`aiter-flash-attn`](https://huggingface.co/kernels-community/aiter-flash-attn)
kernel: instead of Triton, it builds the `ck_tile` FMHA kernels ahead of time
for AMD ROCm GPUs (MI300X / gfx942, MI355X / gfx950).

Original code: https://github.com/ROCm/aiter (MIT, © Advanced Micro Devices, Inc.).
Composable Kernel: https://github.com/ROCm/composable_kernel (pinned at
`83566edb0fded5e1c618c2c19110adbb74532762`).

The exported API matches the `flash-attn` v2 forward surface used by
`transformers`'s flash-attention fallback path, so this kernel can serve as the
ROCm entry in `FLASH_ATTN_KERNEL_FALLBACK`.

## Functions

### `flash_attn_func(q, k, v, ...)`

Dense FlashAttention **forward**. `q, k, v` shape:
`(batch, seqlen, nheads, headdim)`.

### `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ...)`

Variable-length / packed FlashAttention **forward**. `q` shape:
`(total_q, nheads, headdim)`; `cu_seqlens_*` are int32 cumulative offsets.

### `flash_attn_with_kvcache(q, k_cache, v_cache, ...)`

Decode against a KV cache, built on the CK split-KV kernel. Optionally appends
the new `k`/`v` into a contiguous cache in place (`cache_seqlens` gives the
per-batch position), then attends with bottom-right causal alignment. Uniform
cache lengths take the dense kernel; ragged lengths and paged caches
(`block_table`) take the split-KV varlen path. Rotary is not applied here
(rotate `q`/`k` beforehand).

Both dense entry points accept the standard FA2 forward kwargs (`dropout_p`,
`softmax_scale`, `causal`, `window_size`, `alibi_slopes`, ...). The
`window_size` argument is a 3-tuple `(left, right, sink_size)` and an optional
`s_aux` (`(nheads,)` fp32) provides learnable attention sinks (e.g.
gpt-oss). Because only the CK kernels are shipped, the sink path is always
taken.

> **Scope:** forward only (inference). The backward pass is intentionally not
> built. Use the Triton `aiter-flash-attn` kernel if you need training.

## Supported configuration

This kernel is **precompiled**, so the supported configuration matrix is fixed
at build time:

- dtype: `bfloat16`
- head dim: 64 and 128
- architectures: gfx942 (MI300X), gfx950 (MI355X)
- forward (dense) + variable-length / split-KV decode

Calls outside this matrix (e.g. fp16, head dim 256) will not find a compiled
instance.

## Quickstart

```python
import torch
from kernels import get_kernel

flash_attn = get_kernel("kernels-community/aiter-flash-attn-ck")

q = torch.randn(2, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(2, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)

out = flash_attn.flash_attn_func(q, k, v, causal=True)

# With learnable attention sinks (gpt-oss style):
sink = torch.randn(8, device="cuda", dtype=torch.float32)
out = flash_attn.flash_attn_func(q, k, v, causal=True, s_aux=sink)
```

## Use with `transformers`

This kernel implements the flash-attention interface that `transformers`
expects from a Hub attention kernel (`flash_attn_func` / `flash_attn_varlen_func`
with the FA2 calling convention), so it can be selected by repo id:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype="bfloat16",
    attn_implementation="kernels-community/aiter-flash-attn-ck",
).to("cuda")
```

Learnable attention sinks are wired through automatically: when `transformers`
passes per-head sink logits as `s_aux`, the kernel enables the sink path even
though `window_size` arrives as a 2-tuple (e.g. gpt-oss on ROCm).

## Origin

Code is taken from `aiter/csrc` (the `cpp_itfs` / `py_itfs_ck` CK FlashAttention
C++ API and Torch interfaces) plus the `ck_tile/01_fmha` instance sources
generated from Composable Kernel's `generate.py`. The pybind module entry
points were rewritten as `torch.library` ops (`mha_fwd`, `mha_varlen_fwd`) per
the [Hub kernel requirements](https://huggingface.co/docs/kernels/kernel-requirements);
the ASM "v3" forward path (which does not support sinks) is compiled out
(`FAV2_ON=1`).

The generated instances were produced with:

```bash
generate.py -d fwd         --receipt 600 --filter '*_bf16*'        --optdim 64,128
generate.py -d fwd_splitkv --receipt 600 --filter '*_bf16*@*_bf16*' --optdim 64,128
```

## License

MIT — see `LICENSE`. Upstream copyright: Advanced Micro Devices, Inc.
