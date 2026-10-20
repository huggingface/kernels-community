---
license: mit
tags:
- kernels
- attention
- triton
- cuda
- rocm
---

# TokenSpeed Attention

Portable Triton attention kernels from
[lightseekorg/tokenspeed](https://github.com/lightseekorg/tokenspeed), repackaged
for the Hugging Face `kernels` library.

This package exposes the Tokenspeed portable MHA and MLA attention paths without
requiring the full Tokenspeed dispatcher:

- `mha_prefill`
- `mha_extend_with_kvcache`
- `mha_decode_with_kvcache`
- `mla_prefill`
- `mla_decode_with_kvcache`
- `attn_merge_state`

The source was ported from `tokenspeed-kernel` commit
`1492030a2a02d32bc7011645a74d2d691e99c2e6`.

## Requirements

- PyTorch with CUDA or ROCm support
- Triton
- NVIDIA Ampere or newer for the CUDA path, or AMD CDNA-class ROCm GPUs for the
  ROCm path

## Usage

```python
import torch
from kernels import get_kernel

attention = get_kernel("kernels-community/tokenspeed-attention")

seqlens = [128, 256]
cu_seqlens_cpu = [0]
for seqlen in seqlens:
    cu_seqlens_cpu.append(cu_seqlens_cpu[-1] + seqlen)

cu_seqlens = torch.tensor(cu_seqlens_cpu, device="cuda", dtype=torch.int32)
total = cu_seqlens_cpu[-1]

q = torch.randn(total, 16, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(total, 4, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(total, 4, 128, device="cuda", dtype=torch.bfloat16)

out = attention.mha_prefill(
    q,
    k,
    v,
    cu_seqlens,
    cu_seqlens_cpu,
    max(seqlens),
)
```

## Notes

This is a direct kernel-level port. It intentionally omits Tokenspeed's runtime
registry, backend selection, profiling, and plugin system. Call the exported
functions directly.

## License

MIT. See `LICENSE`.
