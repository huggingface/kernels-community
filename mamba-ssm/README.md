---
license: apache-2.0
tags:
- kernel
---

## Mamba

Mamba state space kernels + models from [state-spaces/mamba](https://github.com/state-spaces/mamba).

## Warning

Some functionality is dependent on `einops` and `transformers`, however we
currently don't have any way of defining these dependencies yet. The scope
of the Hub kernel is probably too large (should maybe only contain the
selective-scan and Triton kernels).