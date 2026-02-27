---
license: bsd-3-clause
tags:
  - kernel
---

# sgl-flash-attn3

SGLang's Flash Attention 3 kernel (forward-only, with attention sinks support). This is a port of the [sgl-attn fork](https://github.com/sgl-project/sgl-flash-attn) used by [SGLang](https://github.com/sgl-project/sglang), packaged for use with the [kernels library](https://github.com/huggingface/kernels).

Compared to upstream flash-attn3, this kernel includes sglang-specific features:
- Forward-only (no backward pass) for inference workloads
- Attention sinks (`sinks` parameter)
- `attention_chunk` parameter for chunked attention
- `q_v` tensor support for absorbed multi-latent attention
- Varlen-only mode for serving

## Usage

```python
import torch
from kernels import get_kernel

sgl_fa3 = get_kernel("kernels-community/sgl-flash-attn3")

device = "cuda"
dtype = torch.bfloat16
seqlen, nheads, headdim = 128, 8, 128

q = torch.randn(seqlen, nheads, headdim, device=device, dtype=dtype)
k = torch.randn(seqlen, nheads, headdim, device=device, dtype=dtype)
v = torch.randn(seqlen, nheads, headdim, device=device, dtype=dtype)
cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32, device=device)

out = sgl_fa3.flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=seqlen,
    max_seqlen_k=seqlen,
    causal=True,
)
print(f"Output shape: {out.shape}")
```

## Available Functions

- `flash_attn_varlen_func` -- variable-length attention (prefill)
- `flash_attn_with_kvcache` -- paged KV cache attention (decode + prefill)
- `is_fa3_supported` -- check if FA3 is supported on current device

## Supported Backends

- CUDA (sm80: A100, sm90a: H100)

## CUDA Requirements

- CUDA >= 12.4
- PyTorch >= 2.6

## Credits

- [Tri Dao](https://github.com/tridao) and team for Flash Attention 3
- The [SGLang team](https://github.com/sgl-project) for the sgl-attn fork with sinks and inference optimizations
- The [transformers team](https://huggingface.co/transformers-community) for packaging, testing, building and making it available for use with the [kernels library](https://github.com/huggingface/kernels).
