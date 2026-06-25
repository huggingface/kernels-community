---
license: mit
tags:
- kernels
---
# aiter-kernels

Triton kernels for AMD ROCm, repackaged from the
[ROCm/aiter](https://github.com/ROCm/aiter) project as a Hugging Face Hub
kernel.

This bundle mirrors the Triton subset of upstream `aiter/ops/triton/**` as a
single self-contained package — the same exercise we apply to
[`kernels-community/liger-kernels`](https://huggingface.co/kernels-community/liger-kernels)
for the Liger-Kernel project.

Flash Attention is intentionally **not** included here — it lives in
[`kernels-community/aiter-flash-attn`](https://huggingface.co/kernels-community/aiter-flash-attn)
and is synced separately.

## What's included

The package mirrors the upstream Triton tree (minus `aiter/ops/triton/attention/**`)
under `aiter_kernels.*`:

- `aiter_kernels.activation` — fused activation + quant
- `aiter_kernels.causal_conv1d` / `causal_conv1d_update_single_token`
- `aiter_kernels.comms` — Triton-based collective ops (all-gather, reduce-scatter)
- `aiter_kernels.fusions` — fused composite kernels
- `aiter_kernels.gated_delta_net` — gated delta-rule recurrent ops
- `aiter_kernels.gather_kv_b_proj`
- `aiter_kernels.gemm` — Triton GEMM variants (basic/batched/fused/feed-forward)
- `aiter_kernels.gluon` — Gluon-family GEMM/quant variants (non-attention)
- `aiter_kernels.gmm`
- `aiter_kernels.kv_cache`
- `aiter_kernels.moe` — fused MoE Triton ops
- `aiter_kernels.normalization` — rmsnorm and friends
- `aiter_kernels.quant` — fp8 / mxfp4 / mxfp8 quant
- `aiter_kernels.rope` — Rotary Position Embedding (supersedes `aiter-rope`)
- `aiter_kernels.softmax`
- `aiter_kernels.topk`
- `aiter_kernels.utils`

The package also ships per-arch autotuner JSON under
`aiter_kernels/configs/**` (data only, not a Python subpackage) — read at
runtime by the gemm, moe, gmm, and fusions ops.

Original code: https://github.com/ROCm/aiter (MIT, © Advanced Micro Devices, Inc.).

## Supported hardware

Triton-portable across AMD ROCm GPUs. Verified arches:

- gfx942 (MI300X)
- gfx950 (MI355X)

Some ops also carry per-arch fast paths for `gfx1250` (Gluon) and `gfx1151`.

## Quickstart

```python
import torch
from kernels import get_kernel

aiter = get_kernel("kernels-community/aiter-kernels")

# Example: rotary
B, H, S, D = 2, 8, 32, 64
q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
cos = torch.randn(B, S, D // 2, device="cuda", dtype=torch.float16)
sin = torch.randn(B, S, D // 2, device="cuda", dtype=torch.float16)
q_embed, k_embed = aiter.rope.apply_rotary_transformers(q, k, cos, sin)
```

## Origin & rewrites

Code is vendored from `aiter/ops/triton/**` and its in-tree transitive
imports. All `from aiter.ops.triton.*` absolute imports are rewritten to use
the local `aiter_kernels` root. The few cross-tree dependencies (`aiter.dtypes`,
`aiter.jit.utils.chip_info.get_gfx`, `aiter.jit.utils.torch_guard.torch_compile_guard`,
`aiter.utility.triton.triton_metadata_redirect.AOTMetadataContext`) are mirrored
into the local `aiter_kernels._aiter_compat` submodule.

See `AGENTS.md` in the source repo for the sync workflow.
