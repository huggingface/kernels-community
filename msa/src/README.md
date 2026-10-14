---
license: mit
tags:
  - kernel
---

## MSA — MiniMax Sparse Attention (SM100)

CuTe-DSL sparse attention kernels from [MiniMax MSA](https://github.com/MiniMax-AI/MSA)
(`python/fmha_sm100/cute`), packaged as a Hub kernel.

The package provides:

- `sparse_atten_func` — block-sparse prefill attention.
- `sparse_atten_nvfp4_kv_func` — block-sparse prefill with NVFP4 K/V.
- `sparse_decode_atten_func` / `SparseDecodePagedAttentionWrapper` — block-sparse
  paged FP8 decode.
- `fp4_indexer_block_scores` — FP4 block-score indexer (top-k selection is
  caller-owned).
- `build_k2q_csr` / `SparseK2qCsrBuilderSm100` — q2k indices → CSR + schedule
  construction.
- NVFP4 quantization helpers (`quantize_bf16_to_nvfp4_128x4`, ...).

The host-side helper kernels (the k2q CSR builder and the paged decode
split-KV scheduler) are precompiled Torch ops. The attention kernels are
CuTe DSL Python and are compiled at runtime through `nvidia-cutlass-dsl`.

Only NVIDIA SM100 (Blackwell) GPUs are supported.

Note: the dense FMHA stack of upstream MSA (`fmha_sm100`'s `csrc/` runtime-JIT
kernels) is not part of this package; it relies on runtime `nvcc` compilation,
which Hub kernels do not support.

## Usage

```python
import torch
from kernels import get_kernel

msa = get_kernel("kernels-community/msa")

out = msa.sparse_atten_func(...)
```

See the [upstream documentation](https://github.com/MiniMax-AI/MSA) for the
full API and benchmarks.
