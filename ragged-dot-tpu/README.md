---
license: apache-2.0
tags:
  - kernels
---

# ragged-dot-tpu

MoE experts forward for [torch_tpu](https://github.com/google-pytorch/torch_tpu),
built on [tokamax](https://github.com/openxla/tokamax)'s Pallas grouped matmul
(`tokamax.ragged_dot_general`, `implementation="mosaic_tpu_v2"`) and wrapped
with torch_tpu's `jax_op`. Packaged as a
[`kernels`](https://github.com/huggingface/kernels) Hub kernel.

Written for this repo; the grouped matmul itself comes from tokamax.

## What it does

`experts_forward` is a drop-in transformers experts implementation for gated
experts without bias, with a SiLU (e.g. Qwen3-MoE) or GELU-tanh (e.g. Gemma 4)
gate:

```python
from kernels import get_kernel
from transformers import AutoModelForCausalLM
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

kernel = get_kernel("kernels-community/ragged-dot-tpu", version=1)
ALL_EXPERTS_FUNCTIONS.register("tokamax_ragged_dot", kernel.experts_forward)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    experts_implementation="tokamax_ragged_dot",
)
```

The whole experts forward (sort by expert, both grouped matmuls, gate
activation, routing weights, un-sort and sum) is one JAX function, so
torch_tpu runs a single compiled op per layer instead of dispatching ~20 small
ones. Expert parallelism is supported: picks routed to another rank carry the
sentinel expert id `num_experts`, sort last and are never computed.

`ragged_dot(lhs, rhs, group_sizes)` is also exported: `lhs[M, K] @
rhs[G, N, K].mT -> [M, N]`, with the rows of `lhs` sorted by group and
`group_sizes` int32. Rows past `group_sizes.sum()` are left uninitialized.

## Requirements

torch_tpu, jax and tokamax must be installed. They are not in the `kernels`
library's list of allowed Python dependencies, so `build.toml` does not
declare them.

## Benchmarks

Plain transformers `generate` (greedy, StaticCache), TP=8 with expert
parallelism on a TPU v6e-8, prefill 256, max tokens 512, against
transformers' default `grouped_mm` experts. torch_tpu 0.1.1+release.2026.9.22,
libtpu 0.0.49, transformers 5.17.0, tokamax 0.0.15.dev20260929.

`Qwen/Qwen3-30B-A3B-Instruct-2507`, decode step:

| batch | grouped_mm | ragged-dot-tpu | speedup | tokens/s: grouped_mm | ragged-dot-tpu |
|---|---|---|---|---|---|
| 1 | 30.0 ms | 26.5 ms | 1.13x | 31 | 35 |
| 4 | 64.6 ms | 26.3 ms | 2.45x | 61 | 146 |
| 8 | 126.6 ms | 27.5 ms | 4.60x | 62 | 279 |
| 16 | 297.2 ms | 27.4 ms | 10.8x | 53 | 548 |
| 32 | 599.6 ms | 36.8 ms | 16.3x | 53 | 795 |
| 64 | 1215.9 ms | 56.6 ms | 21.5x | 52 | 988 |

`google/gemma-4-26B-A4B-it` (decode runs op by op, not compiled), decode step:

| batch | grouped_mm | ragged-dot-tpu | speedup | tokens/s: grouped_mm | ragged-dot-tpu |
|---|---|---|---|---|---|
| 1 | 344.0 ms | 322.6 ms | 1.07x | 2.9 | 3.1 |
| 16 | 569.0 ms | 329.5 ms | 1.73x | 28.1 | 48.5 |
| 32 | 866.3 ms | 330.5 ms | 2.62x | 36.9 | 96.3 |
| 64 | 4858.3 ms | 379.4 ms | 12.8x | 13.2 | 162.2 |

One Qwen3 experts layer on one EP8 rank (16 local experts), op by op:

| tokens | grouped_mm | ragged-dot-tpu | speedup |
|---|---|---|---|
| 1 | 1.022 ms | 0.096 ms | 10.7x |
| 256 | 1.116 ms | 0.257 ms | 4.3x |
| 512 | 1.735 ms | 0.362 ms | 4.8x |

## Validation

`tests/` checks `ragged_dot` against a per-group dense reference and
`experts_forward` against the transformers `eager` experts forward (fp32 on
CPU) for Qwen3-MoE and Gemma 4 shapes, with expert-parallel routing. The tests
need a TPU and are skipped otherwise.
