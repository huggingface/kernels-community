---
tags:
- kernels
- cuda
---
# SageAttention

This is a build of [SageAttention](https://github.com/thu-ml/SageAttention) compatible with the
kernels library.

It covers the SageAttention, SageAttention2 and SageAttention2++ kernels: INT8 `QK^T` with FP16/FP8
`PV` on Ampere (sm80), Ada (sm89), Hopper (sm90a) and Blackwell consumer (sm120/sm121) GPUs, the
Triton fallback used on sm86, and variable-length attention via `sageattn_varlen`.

The SageAttention3 microscaling FP4 kernels for datacenter Blackwell (sm100a) are not part of this
build.
