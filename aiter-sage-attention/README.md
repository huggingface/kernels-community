---
license: mit
tags:
- kernels
---
# aiter-sage-attention

SageAttention for AMD Radeon RDNA4 GPUs (gfx1201: Radeon AI PRO R9700 / R9600D,
RX 9070 series), repackaged from [ROCm/aiter](https://github.com/ROCm/aiter) as a
Hugging Face Hub kernel. Q and K are quantized to INT8 per 32-row block, V to FP8
per channel, and both matmuls accumulate in FP32 (INT8 QK^T, FP8 PV).

Original code: https://github.com/ROCm/aiter (MIT, © Advanced Micro Devices, Inc.),
`csrc/kernels/gfx1201_sage_{prepare,attention}.cu` and
`aiter/ops/gfx1201_sage_{prepare,attention}.py` from
[ROCm/aiter#6390](https://github.com/ROCm/aiter/pull/6390) at
`62019a492a433943abf6618fda38ced32d3aef52`. The kernel sources are vendored unchanged;
`aiter/include/aiter_tensor.h` is cut down to the fields the kernels use.

## Functions

### `sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, return_lse=False)`

Drop-in for `sageattention.sageattn` within what the kernel supports. `tensor_layout`
is `"HND"` (`[batch, heads, seq, 128]`) or `"NHD"` (`[batch, seq, heads, 128]`).
Causal attention and `return_lse=True` raise instead of falling back.

### `gfx1201_sage_attention(query, key, value, softmax_scale=None, query_weight=None, key_weight=None, cosine=None, sine=None, eps=1e-6)`

Non-causal attention over contiguous BF16 `[batch, seq, heads, 128]` Q/K/V of equal
shape; returns the same shape. With `query_weight`/`key_weight` (BF16 `[128]`) and
`cosine`/`sine` (FP32 `[seq, rope_dim]`), Q and K first go through RMSNorm and
rotate-half RoPE inside the quantization pass.

### `gfx1201_sage_prepare(...)` / `gfx1201_sage_attention_fwd(...)`

The two stages separately: quantize Q/K/V, then attend over the quantized tensors.

## Constraints

gfx1201 only; BF16; head dim 128; Q, K and V of equal shape; no mask; non-causal;
inference only.

## Benchmarks

Radeon AI PRO R9700 (gfx1201), batch 1, 28 heads, head dim 128, BF16 inputs,
PyTorch 2.11 + ROCm 7.15. `cos` is the cosine similarity of the output to SDPA.

| Sequence | SDPA | aiter-sage-attention | Speedup | cos |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 3.4 ms | 1.7 ms | 1.98x | 0.9992 |
| 16384 | 54.8 ms | 23.6 ms | 2.33x | 0.9992 |
| 57330 | 676.3 ms | 275.2 ms | 2.46x | 0.9992 |
| 114660 | 2775.2 ms | 1109.0 ms | 2.50x | 0.9992 |

Outputs are bit-identical to `aiter.gfx1201_sage_attention` built from the commit above.
