---
license: apache-2.0
tags:
  - kernels
---

# WeatherNext 2 banded mesh-attention kernel

Inference Triton kernel for **WeatherNext 2's mesh attention**, the dominant cost of a
forecast step. Packaged as a [`kernels`](https://github.com/huggingface/kernels) Hub
kernel so `transformers` can load it on demand, with the pure-PyTorch path in
`transformers` as the fallback.

## What it does

WeatherNext 2 processes the atmosphere on an icosahedral mesh whose nodes are ordered by
reverse Cuthill-McKee, which makes the k-hop adjacency **banded**: a block of
`attention_bandwidth` consecutive nodes can only reach itself and its two neighbours.

The in-tree path spells that out by materializing the three neighbouring key/value blocks
with `gather_neighbouring_blocks`, then handing a `[blocks, 1, block, 3 * block]` mask to
`scaled_dot_product_attention`. This kernel instead walks the three neighbours straight
out of the ungathered tensors, so:

- the 3x key/value copy never happens,
- the mask is streamed a tile at a time rather than expanded by `masking_utils`,
- tiles the band never reaches are skipped before their two matmuls, not after.

Forward only. Backward stays on the PyTorch path, so `transformers` registers it
inference-only, and training is unaffected.

## Precision

Upstream runs this attention in float32: `sparse_transformer.py` casts q, k and v before
the call, and all three released configs set `upcast_attn_to_fp32 = True`. Float32 inputs
only reach tensor cores through tf32, so `precision` selects how `tl.dot` treats them:

| `precision` | what it does | measured |
|---|---|---|
| `"tf32"` (default) | tensor cores, 10 explicit mantissa bits | 1.64x the op, 1.73x end to end |
| `"tf32x3"` | three tf32 passes, close to float32 accuracy | not yet measured |
| `"ieee"` | true float32, no tensor-core path | correct, but ~50x slower than the fallback |

`"ieee"` is for checking numerics, not for running: in strict float32 Triton loses to the
cutlass kernel PyTorch already picks. Override with
`WEATHERNEXT2_BANDED_ATTENTION_PRECISION`.

## Benchmarks

`kashif/weathernext2-mini` (1 degree: 4 blocks of 2577 nodes, 7731 keys, 4 heads,
head_dim 128, 13% band density) on an H100, real initial conditions:

| path | ms/step | attention peak GiB |
|---|---|---|
| shipped (gather + sdpa, fp32) | 223.7 | 0.90 |
| this kernel (tf32) | 129.1 | 0.37 |

**1.73x end to end.** Whole-model peak memory is unchanged at this resolution: the 2.4x
saving is real but local to attention, and the global peak is set elsewhere. It has not
been measured at 0.25 degrees, where attention dominates the footprint.

Accuracy, in physical units after `postprocess`, worst variable of each comparison, as a
fraction of that field's own spatial spread:

| comparison | worst mean/std |
|---|---|
| shipped sdpa vs the JAX reference | 7.0e-04 |
| this kernel (tf32) vs shipped sdpa | 1.8e-03 |
| this kernel (tf32) vs the JAX reference | 2.4e-03 |

In `"ieee"` the kernel matches sdpa to `1.97e-06`, so the gap above is tf32 rounding
rather than a difference in what is computed.

## Why a kernel at all

Profiling a forecast step on an H100 puts attention at 21% of GPU time and the graph
scatter at **0.5%**, so the mesh attention is the only part worth a kernel. The fp32
kernel PyTorch falls back to is `fmha_cutlassF_f32_aligned`, which is CUDA-only; on ROCm
there is no fp32 FMHA and the fallback materializes the full score matrix, which at 0.25
degrees is 1.7 GB of mask alone. That is why this builds for both backends.

**ROCm is untested.** The kernel uses only portable Triton and declares
`backends = ["cuda", "rocm"]`, but no AMD GPU was available to run it on.

## How transformers uses it

`WeatherNext2Attention` would be decorated
`@use_kernel_forward_from_hub("WeatherNext2Attention")` and mapped to this repo in
`integrations/hub_kernels.py` (cuda + rocm, `Mode.INFERENCE`). Kernels are opt-in:

```python
import torch
from transformers import WeatherNext2ForWeatherForecasting

model = WeatherNext2ForWeatherForecasting.from_pretrained(
    "kashif/weathernext2-mini", device_map="cuda", use_kernels=True
).eval()
```

`layers.WeatherNext2Attention` reimplements that module's
`forward(hidden_states, attention_mask)` and reads its parameters directly (`q_proj`,
`k_proj`, `v_proj`, `o_proj`, `head_dim`, `scaling`). **Those attribute names and the
forward signature are the contract** and must stay in sync with the in-tree module.

The fast path needs the geometry's own banded mask, `[blocks, block, 3 * block]` bool,
rather than the one `masking_utils` expands. Given a different tensor mask the layer falls
back to gather-plus-sdpa, so it is never wrong, only unaccelerated. Reaching the fast path
therefore needs a companion change in `transformers` to pass the raw mask through.

`attn_implementation="flex_attention"` is not supported: that hands the layer a `BlockMask`,
which neither the kernel nor the fallback can read, so it raises rather than quietly
dropping the mask. Use the default `"sdpa"`.
