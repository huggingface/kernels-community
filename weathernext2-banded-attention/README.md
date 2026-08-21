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

Written for this repo. Builds for CUDA and ROCm from one Triton source.

## What it does

WeatherNext 2 runs on an icosahedral mesh whose nodes are ordered by reverse
Cuthill-McKee, which makes the k-hop adjacency **banded**: a block of consecutive nodes
reaches only itself and its two neighbours. The in-tree path spells that out by
materializing the three neighbouring key/value blocks and handing a
`[blocks, 1, block, 3 * block]` mask to `scaled_dot_product_attention`.

This kernel walks the three neighbours straight out of the ungathered tensors, so:

- the 3x key/value copy never happens,
- the mask is streamed a tile at a time rather than expanded,
- tiles the band never reaches are skipped before their two matmuls, not after.

Forward only. The layer refuses the fast path whenever autograd is live, because the
kernel's output carries no `grad_fn`: a `loss.backward()` would still succeed while every
parameter upstream of attention silently received nothing. Fine-tuning therefore takes
the differentiable fallback. A real backward is what accelerating training would need.

## Precision

The reference implementation runs this attention in float32, and float32 only reaches
tensor cores through tf32, so `precision` selects how `tl.dot` treats it:

| `precision` | what it does |
|---|---|
| `"tf32"` (default) | tensor cores, 10 explicit mantissa bits |
| `"tf32x3"` | three tf32 passes, close to float32 accuracy, about a third of the throughput |
| `"ieee"` | true float32, no tensor-core path, slower than the fallback it replaces |

`"ieee"` is for checking numerics rather than for running. Override with
`WEATHERNEXT2_BANDED_ATTENTION_PRECISION`. On RDNA there is no tf32, so `"tf32"` is a
no-op there and the kernel computes in full float32.

## How transformers uses it

`WeatherNext2Attention` is decorated `@use_kernel_forward_from_hub("WeatherNext2Attention")`
and mapped to this repo in `integrations/hub_kernels.py`, inference-only. Kernels are
opt-in:

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

## Supported shapes

Queries are `[batch, blocks, heads, block_size, head_dim]` float32, keys and values the
same and **not** gathered over neighbours, and the mask is `[blocks, block_size,
3 * block_size]` bool. `head_dim` must be a power of two: it is a `tl.constexpr` tile
width and the loads are not masked along it.

Every mesh node must reach at least itself, or that row softmaxes over nothing. The real
geometry guarantees this; the empty rows past the last mesh node are handled.

Reaching the fast path needs the geometry's own banded mask rather than the one
`masking_utils` expands, which takes a companion change in `transformers` to pass through.
Given a different tensor mask the layer falls back to gather-plus-sdpa, so it is never
wrong, only unaccelerated. `attn_implementation="flex_attention"` is unsupported: that
hands the layer a `BlockMask`, which neither path can read, so it raises rather than
quietly dropping the mask.

## Validation

`tests/` checks the kernel against an fp32 sdpa reference across block counts and band
densities, checks that a node reaching only itself returns its own value vector rather
than NaN, and checks that a backward reaches all four projections.

**CUDA (H100, `kashif/weathernext2-mini`, 1 degree: 4 blocks of 2577 nodes, 7731 keys,
4 heads, head_dim 128, 13% band density), real initial conditions:**

| path | ms/step | attention peak GiB |
|---|---|---|
| shipped (gather + sdpa, fp32) | 223.7 | 0.90 |
| this kernel (tf32) | 129.1 | 0.37 |

**1.73x end to end.** Whole-model peak memory is unchanged at this resolution: the
attention saving is real but the global peak is set elsewhere. Not measured at 0.25
degrees, where attention dominates the footprint.

Accuracy in physical units, worst variable, as a fraction of that field's spatial spread:
the shipped path sits 7.0e-04 from the JAX reference, this kernel 2.4e-03. In `"ieee"` it
matches sdpa to 2e-06, so the gap is tf32 rounding rather than a different computation.

**ROCm (gfx1150, torch 2.13.0+rocm7.2, triton 3.7.1):** all tests pass, matching sdpa to
6.9e-07. Speed and memory against sdpa, sweeping block size:

| block_size | sdpa ms / GiB | kernel ms / GiB | speed | memory |
|---|---|---|---|---|
| 256 | 2.56 / 0.07 | 2.83 / 0.05 | 0.90x | 0.71x |
| 512 | 9.11 / 0.18 | 12.22 / 0.07 | 0.75x | 0.39x |
| 1024 | 34.59 / 0.56 | 48.45 / 0.10 | 0.71x | 0.18x |

The kernel is slower per step on this device and its memory advantage widens sharply with
block size, because sdpa's footprint grows with `block_size**2` while the kernel's stays
close to flat. Torch also reports that memory-efficient attention on AMD is still
experimental, so sdpa is falling back to the math path. gfx1150 is integrated RDNA 3.5
with no tf32 and the autotune configs were chosen on an H100, so **these timings should
not be read as representative of CDNA**.

## Follow-ups

- **Measure on CDNA (MI300).** The ROCm numbers above come from an integrated RDNA 3.5
  GPU at a tenth of the model's real block size. The memory trend is the interesting part
  and wants confirming where it matters.
- **Skip the `.contiguous()` copies.** Queries, keys and values arrive transposed and are
  copied on entry. Their strides are uniform enough to pass straight to the kernel, which
  would remove three full copies per layer.
- **A backward**, if training is ever to use this rather than fall back.
- **Autotune properly.** The current config sweep is fixed and was chosen on one device.
