---
license: bsd-3-clause
tags:
  - kernel
---

# Flash Attention Ops

A `kernels`-compliant package of the general-purpose **Triton ops** that ship
inside [Flash Attention](https://github.com/Dao-AILab/flash-attention) under
`flash_attn/ops/triton` (and the `flash_attn.losses` wrapper on top of them).

These ops — cross-entropy, rotary embeddings, and (RMS/Layer) normalization —
are widely imported across training codebases. Packaging them independently lets
projects depend on just the kernels they need without pulling in the full
flash-attn build. See [issue #900](https://github.com/huggingface/kernels-community/issues/900).

The ops are **pure Triton + PyTorch**: there is nothing to compile ahead of time,
and the package is fully self-contained — it does **not** import `flash_attn` at
runtime (everything it needs is vendored in).

Original code: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
(vendored from commit `b02b07e1a10238fe12831b80a8937ed59b1353a5`).

## Available functions

| Symbol | Source | Description |
|--------|--------|-------------|
| `cross_entropy_loss` | `flash_attn/ops/triton/cross_entropy.py` | Fused cross-entropy (functional), with label smoothing, logit scaling, z-loss, tensor-parallel support. |
| `CrossEntropyLoss` | `flash_attn/losses/cross_entropy.py` | `nn.Module` wrapper around `cross_entropy_loss` (the import used by downstream training code). |
| `apply_rotary` | `flash_attn/ops/triton/rotary.py` | Rotary positional embedding (varlen / seqlen-offset aware). |
| `layer_norm_fn`, `rms_norm_fn` | `flash_attn/ops/triton/layer_norm.py` | Fused LayerNorm / RMSNorm with optional residual add + dropout. |
| `RMSNorm` | `flash_attn/ops/triton/layer_norm.py` | `nn.Module` RMSNorm. |
| `layer_norm_linear_fn`, `LayerNormFn` | `flash_attn/ops/triton/layer_norm.py` | Fused norm (+ linear) autograd functions. |
| `layer_norm_ref`, `rms_norm_ref` | `flash_attn/ops/triton/layer_norm.py` | Pure-PyTorch reference implementations (for testing). |

> **Not yet included:** `flash_attn/ops/triton/linear.py` and `mlp.py`. `linear.py`
> relies on `triton.ops.matmul_perf_model` (removed in Triton ≥ 3.0) and `mlp.py`
> relies on the compiled `fused_dense_lib` extension; both require more than an
> import rewrite and are tracked as a follow-up.

## How to use

```python
# make sure `kernels` is installed: `pip install -U kernels`
from kernels import get_kernel

flash_attn_ops = get_kernel("kernels-community/flash-attn-ops")

loss = flash_attn_ops.CrossEntropyLoss(reduction="mean")
out = loss(logits, labels)            # logits: (batch, vocab), labels: (batch,)
```

## Related kernels

These are independent kernels that cover similar ground with different
implementations / APIs — pick whichever fits your use case:

- [`kernels-community/rotary`](https://hf.co/kernels-community/rotary) — compiled CUDA/XPU `apply_rotary`.
- [`kernels-community/liger-kernels`](https://hf.co/kernels-community/liger-kernels) — Triton norms / rope / cross-entropy from Liger.
- [`kernels-community/layer-norm`](https://hf.co/kernels-community/layer-norm) — compiled CUDA build of flash-attn's `csrc/layer_norm`.
