# Hydra Kernel Card

## Summary

Hydra provides a bounded-residency decode attention path for long-context
inference. The implementation is Python plus Triton and is packaged here as a
Hugging Face universal CUDA kernel source directory.

## Intended Use

Use Hydra for experiments where full decode attention over a long KV cache is
memory-bound and a bounded resident set is acceptable for evaluation.

Hydra is not intended as a drop-in universal FlashAttention replacement. Users
should keep an exact-model fallback path and validate quality for their prompt
set.

## Kernel Interface

The exported call is:

```python
hydra.hydra(q, k, v, is_causal=True, sliding_window=None)
```

Inputs are bf16 tensors shaped `(B, H, T, D)` with `D=128`. The decode path
supports `Tq == 1`; the prefill path requires sequence length to be a multiple
of the compile-time block size.

## Evidence

This card separates the kernel contribution from benchmark appendices:

- kernel/package validation: import, CSR, CUDA decode parity, builder, example,
  and isolated decode benchmark gates
- broad Hydra campaign context: multi-GPU bounded-residency testing, comparison
  lanes, capacity/OOM boundaries, and diagnostics in the staging repo
- exact-Qwen proof-of-concept: summary-backed demo rows for:

- RTX PRO 6000 WS with `Qwen/Qwen3.6-35B-A3B-FP8`
- RTX 3090 with `Qwen/Qwen3.6-35B-A3B-FP8`

Use `results/reports/QWEN3P6_FP8_EVIDENCE_TABLE.md` in the staging repo as the
claim ledger for the exact-Qwen proof-of-concept only. The table is generated
from raw summary JSON, answer artifacts, and logs. It intentionally excludes
incomplete scopes from completed benchmark rows.

## Non-Claims

- no universal speedup claim
- no production-readiness claim
- no broad quality-preservation claim without scorer or inspection evidence
- no proxy/profile/loader-only benchmark claims
- no results from non-Qwen or non-FP8 runs in the exact-Qwen proof-of-concept table
- no framing that treats the exact-Qwen proof-of-concept as the full Hydra campaign

## Required Validation

Before merge, run:

- import and CSR tests
- CUDA decode parity against PyTorch SDPA on small tensors
- kernel-builder `ci-test`
- one isolated decode benchmark
- one exact-model reproduction on a named GPU/config
