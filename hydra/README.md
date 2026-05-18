---
license: mit
tags:
  - kernels
  - triton
  - attention
  - long-context
---

# Hydra

Hydra is an experimental bounded-residency attention kernel for long-context
decode. It keeps sink tokens, recent tokens, and selected older pages resident
instead of forcing each decode step to attend over the full KV cache.

This submission is intentionally narrow. It is not a general replacement for
full attention, and it does not claim universal speedups or broad quality
preservation. The current target is fit and usability for specific
long-context inference workloads where the full-attention path is memory-bound.

## Usage

After the kernel is published:

```python
import torch
from kernels import get_kernel

hydra = get_kernel("kernels-community/hydra")

q = torch.randn(1, 32, 1, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 8, 8192, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 8, 8192, 128, device="cuda", dtype=torch.bfloat16)

out = hydra.hydra(q, k, v)
print(out.shape)
```

For local development inside the `kernels-community` checkout:

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("hydra") / "torch-ext"))
import hydra
```

`readme_example.py` uses the local source packet by default so it can run before
publication. Set `HYDRA_USE_HUB=1` after publication to exercise the Hub-loaded
path.

## API

```python
hydra.hydra(
    q,
    k,
    v,
    *,
    is_causal=True,
    sliding_window=None,
    policy_layer_idx=None,
    precision="high",
)
```

Current constraints:

- CUDA tensors only
- bf16 `q`, `k`, and `v`
- shape `(B, H, T, D)` with `D=128`
- causal attention only
- decode path supports `Tq == 1` with arbitrary `Tkv`
- prefill path requires `T % BLOCK_SIZE == 0`

## Evidence Boundary

Submission-facing evidence must come from checked artifacts, not prose notes.
Treat evidence in three separate scopes:

- kernel/package validation: tests, CUDA parity logs, `kernel-builder` logs, and
  isolated decode benchmarks for this source packet
- broad Hydra research campaign: capacity, quality, sparse-attention comparison,
  edge/OOM, diagnostic, and model-family reports from the staging repo
- exact-model proof-of-concept: checked `Qwen/Qwen3.6-35B-A3B-FP8` rows for
  named GPUs only

The exact-Qwen proof-of-concept appendix in the staging repo is under:

```text
results/raw/qwen3p6_35b_a3b_fp8/
results/reports/QWEN3P6_FP8_EVIDENCE_TABLE.md
```

Each cited row must include all three:

- fit/headroom: GPU, context length, memory allocated/reserved, and OOM state
- quality/correctness: prompt/task ID and generated answer artifact
- speed/usability: wall time, generated tokens, tokens/sec, and comparison target

Do not cite proxy models, loader-only probes, failed dependency checks, or
non-matching model runs as Hydra benchmark results. Do not describe the
exact-Qwen proof-of-concept subset as the full Hydra validation campaign.

## Current Proof-Of-Concept Scope

The current exact-Qwen artifact-backed proof-of-concept scope is:

| GPU | Model | Scope |
| --- | --- | --- |
| RTX PRO 6000 WS | `Qwen/Qwen3.6-35B-A3B-FP8` | 32k/80k/160k repeat packet, 160k c96 warm packet, and frontier/headroom sweeps |
| RTX 3090 | `Qwen/Qwen3.6-35B-A3B-FP8` | 2k/3k/4k/6k/8k fit probes and completed 10k/12k/14k edge sweep |

The 3090 result should be framed as fit/usability evidence, not a speedup
claim. Token rates are slow in the long-context edge rows. The broader Hydra
campaign includes additional GPUs, tasks, and comparison lanes outside this
exact-model appendix.

## Validation Required Before Merge

Minimum gates for an upstream PR:

```bash
python3 -m pytest -q hydra/tests
nix run ./hydra#ci-test
python3 hydra/benchmarks/benchmark_hydra_decode.py
python3 hydra/readme_example.py
```

Run the CUDA tests on real GPUs. Local syntax checks are not enough for a
kernel submission.
