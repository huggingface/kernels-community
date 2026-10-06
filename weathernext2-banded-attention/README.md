---
license: apache-2.0
tags:
  - kernels
---

# WeatherNext 2 banded mesh-attention kernel

Shared mask preparation and fused Triton attention for
**WeatherNext 2's mesh attention**. Packaged as a
[`kernels`](https://github.com/huggingface/kernels) Hub kernel.

Written for this repo. One Triton source, no backend-specific attention code, so
it builds for anything Triton targets.

The attention kernel uses the same fixed 64 x 32 tiles, 4 warps and 1 stage as
[Faster-WeatherNext](https://github.com/Raymondlol/Faster-WeatherNext), so the first forward
only compiles, without benchmarking alternative configurations. Its QK scaling and online
softmax accumulator update follow their tiled attention implementation.

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

The mask layer builds one packed active-tile representation per forward, shared by all
attention layers: CSR offsets, compact tile indices and one uint32 word per query row
per active 32-key tile, following Faster-WeatherNext. Temporary scan counts and padded
tile lists are discarded after preparation. The model-facing attention consumes those words directly, without
unpacking the mask or storing a score matrix. Boolean masks use the same preparation
when the attention replacement is used without its mask layer.

The fused Triton API is forward only. The model-facing layer uses the differentiable
fallback when gradients are required.

## Precision

The reference implementation runs this attention in float32. `precision` controls how
`tl.dot` treats those inputs:

| `precision` | what it does |
|---|---|
| `"default"` (default) | Triton's backend default |
| `"ieee"` | full float32 multiplication |

The direct `banded_attention` API accepts either mode. Model-facing attention uses IEEE
float32 multiplication. CPU execution and execution requiring gradients use the reference
implementation instead.

IEEE dot products can be substantially slower than accelerated reduced-precision dots,
and can be slower than SDPA. Requesting IEEE precision alone does not reproduce SDPA's
reduction order or guarantee full-forecast parity.

## Supported shapes

Queries are `[batch, blocks, heads, block_size, head_dim]` float32, keys and values the
same and **not** gathered over neighbours, and the mask is `[blocks, block_size,
3 * block_size]` bool, or an already prepared packed mask.

`head_dim` **must be a power of two and at least 16**. It is a `tl.constexpr` tile width and
the loads along it are not masked, so anything else reads past the end of every row, and
`tl.dot` needs 16 along its reduction dimension. Unsupported values raise `ValueError`
rather than returning quietly wrong numbers. WeatherNext 2's released checkpoints use 128.

Every mesh node must reach at least itself, or that row softmaxes over nothing. The real
geometry guarantees this; the empty rows past the last mesh node are handled.

The attention layer accepts boolean geometry masks, with an optional singleton head axis,
and masks whose block axis is folded into the batch axis. Other tensor layouts take
the gather-plus-SDPA fallback. `attn_implementation="flex_attention"` is unsupported: that
hands the layer a `BlockMask`, which neither path can read, so it raises rather than
quietly dropping the mask.

## Validation

`tests/` checks the kernel against an fp32 sdpa reference across block counts and band
densities, checks that a node reaching only itself returns its own value vector rather
than NaN, checks that unsupported head dimensions raise, and checks that a backward
reaches all four projections. Model-facing tests cover
batch-specific masks, shared packed geometry and CPU execution. Tests use the implementation's
fixed launch defaults, without overriding tuning configurations.

Tests pick the device with `infer_device()`, so they run on whichever accelerator is present
rather than assuming CUDA.
