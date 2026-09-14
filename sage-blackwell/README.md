---
tags:
- kernels
- cuda
---
# SageAttention3 (Blackwell)

This is a build of the [SageAttention3](https://github.com/thu-ml/SageAttention) Blackwell kernels
(upstream `sageattention3_blackwell/`) compatible with the kernels library.

SageAttention3 quantizes `Q`, `K` and `V` to microscaling **NVFP4** (`e2m1` values with a per-16-element
`e4m3` scale) and runs the attention with the SM120 blockscaled NVFP4 MMA. The SageAttention,
SageAttention2 and SageAttention2++ kernels (INT8 `QK^T` with FP16/FP8 `PV`) live in a separate build,
[`kernels-community/sage-attention`](https://huggingface.co/kernels-community/sage-attention).

## Hardware support

This kernel is built for **consumer Blackwell only** — compute capability `12.0a` (sm_120, e.g. RTX 50
series). Upstream's `setup.py` also accepts `sm_100a` and `sm_121a`, but:

- the FP4 attention kernel is instantiated from `cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4`,
  which CUTLASS only enables under `CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED` (sm_120a/sm_121a), so
  datacenter Blackwell (sm_100a) does not have the instruction;
- `mha_fwd` rejects at runtime anything that is not sm_120 or sm_121;
- `12.1` is not in kernel-builder's supported capability list, so sm_121 cannot be targeted separately
  (sm_121 devices are not covered by this build).

CUDA >= 12.8 is required.

## Usage

```python
import torch
from kernels import get_kernel

sage_blackwell = get_kernel("kernels-community/sage-blackwell", version=1)

q, k, v = (
    torch.randn(1, 4, 1024, 128, dtype=torch.bfloat16, device="cuda") for _ in range(3)
)

out = sage_blackwell.sageattn3_blackwell(q, k, v, is_causal=False)
```

`sageattn3_blackwell(q, k, v, attn_mask=None, is_causal=False, per_block_mean=True)` takes `[batch,
heads, seq_len, head_dim]` (HND) tensors in `float16` or `bfloat16` and returns a tensor of the same
shape and dtype.

- `head_dim` must be 64 or 128. Head dims `>= 256` fall back to `torch.nn.functional.scaled_dot_product_attention`.
- The number of query and key/value heads must be equal — MQA/GQA is not supported upstream.
- `attn_mask` is accepted for signature compatibility with SDPA but ignored.
- Sequence lengths are padded up to a multiple of 128 internally and the output is sliced back.

> [!WARNING]
> `sageattn3_blackwell` subtracts the key mean **in place** (`k -= k.mean(dim=-2, keepdim=True)`), so
> the `k` tensor you pass in is modified. This is upstream behaviour and is preserved here; pass
> `k.clone()` if you need to keep the original.

## Testing

The tests load the kernel through `get_kernel`, so they exercise the same path
users take. They need an sm_120 GPU; everything is skipped elsewhere.

```bash
nix run .#ci-test        # the `kernels_ci` subset
```

## Accuracy

Upstream reports lossless acceleration for video generation models (CogVideoX-2B, HunyuanVideo, Mochi)
and for most image generation models (Flux, Stable Diffusion 3.5). FP4 attention does not guarantee
lossless acceleration for all models; upstream recommends selectively falling back to SageAttention2++
for the first and last denoising timesteps of other video models.
