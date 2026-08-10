---
license: apache-2.0
tags:
  - kernels
---

# ESMFold2 TriMul kernel

Fused inference Triton kernel for **ESMFold2's triangle multiplication** — the O(N³)
hotspot of the folding trunk, hit at 76 sites in the full model. Packaged as a
[`kernels`](https://github.com/huggingface/kernels) Hub kernel so `transformers` can
load it on demand, with the pure-PyTorch block in `transformers` as the fallback.

Source: [`Rocketknight1/esmfold2-trimul-kernel`](https://huggingface.co/kernels/Rocketknight1/esmfold2-trimul-kernel).
The Triton implementation is independently written, inspired by
[cuequivariance](https://docs.nvidia.com/cuda/cuequivariance/index.html)'s
`triangle_multiplicative_update`.

## What it fuses

One kernel chain for the whole `EsmFold2TriangleMultiplicativeUpdate`:

```
norm_start → gated dual-GEMM (sigmoid(x@Wg) · (x@Wp)) → triangular einsum
(bikd,bjkd→bijd) → norm_mix → proj_emit → output gate
```

The `delta` intermediate is never written to HBM: the dropout-mask multiply and the
residual add happen in-register in the final gated GEMM, saving one full pair-tensor
write plus one read versus the `TriMul → FusedDropoutResidual` baseline. bf16 in/out
with fp32 accumulation. Forward and backward are both implemented; `transformers`
registers it inference-only.

## How transformers uses it

`EsmFold2TriangleMultiplicativeUpdate` is decorated
`@use_kernel_forward_from_hub("ESMFold2TriangleMultiplication")` and mapped to this
repo in `integrations/hub_kernels.py` (cuda, `Mode.INFERENCE`).

```python
import torch
from transformers import EsmFold2Model

# use_kernels=True swaps in this kernel for the 76 trimul sites (CUDA + inference).
model = EsmFold2Model.from_pretrained(
    "biohub/ESMFold2", dtype=torch.bfloat16, device_map="cuda", use_kernels=True
).eval()
out = model.infer_protein(seq)
```

`layers.ESMFold2TriangleMultiplication` reimplements that module's
`forward(pair_grid, visibility)` and reads its parameters directly
(`norm_start`/`norm_mix`/`proj_bundle`/`proj_emit`/`proj_gate`, plus `dim`/`flow`).
**Those attribute names and the forward signature are the contract** — they must stay in
sync with the in-tree module.

The layer runs everything in bf16, including the norm weights, which the in-tree module
deliberately keeps in fp32 (`_keep_in_fp32_modules_strict`). Outputs therefore differ
from the pure-PyTorch path by ~6e-3 relative — bf16 rounding, flat in sequence length
rather than accumulating.

## Supported shapes

`c_z` (the channel dim, `dim` on the in-tree module) **must be a power of two and at
least 64**. ESMFold2 uses 128. The gated GEMMs tile the channel dim with `TILE_N=64`
and mask only the row dim, so a partial trailing tile reads the weights and writes the
output past their ends; the LayerNorm kernels additionally index channels with
`tl.arange(0, c_z)`, which Triton requires to be a power of two. Unsupported values now
raise `ValueError` instead of silently returning nondeterministic garbage — see
"Follow-ups" for the real fix.

Sequence length `L` is unconstrained (the row dim is masked everywhere), except that
`B == L == 1` makes `M == 1`, which Triton specializes to a constexpr and the LayerNorm
kernel's `M.to(tl.int64)` then rejects.

## Validation

Swapped into all 76 `EsmFold2TriangleMultiplicativeUpdate` instances of the real model
(`biohub/ESMFold2`, bf16, GPU), folds match the pure-PyTorch fallback within the model's
own non-determinism: ubiquitin 0.801 vs 0.799 pLDDT (Δ +0.002), GB1 0.849 vs 0.849, pTM
identical.

Standalone microbenchmark (dim=128, B=1): 5–37× over the chunked fp32 fallback, the gap
growing with N (`torch.compile` of the fallback only reaches ~1–7×).

`tests/` checks the fused path against an fp32 PyTorch reference across both flow
directions, supported channel counts, mask/dropout-mask combinations and a range of
sequence lengths, and asserts repeated calls are bit-identical (drift would indicate an
out-of-bounds read).

## Follow-ups

- **Mask the N dimension in the gated GEMMs.** Guarding the weight loads, the
  residual/dropout loads and the output store with `offs_n < N` would lift the
  power-of-two-≥64 `c_z` restriction and remove the out-of-bounds write on a partial
  trailing tile. It touches the hot inner loop, so it needs a perf check.
- **Residual-optional entry.** The in-tree boundary is delta-only, so the layer passes
  `residual=zeros_like(pair)`, costing one `[B,N,N,C]` alloc+read per call. A
  `residual=None` fast path that skips the in-kernel residual add recovers that.
- **cuequivariance** provides the same op (`triangle_multiplicative_update`) as an
  alternative backend if a vendored-Triton kernel is undesirable.
