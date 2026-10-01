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

The SageAttention3 microscaling FP4 kernels are not part of this build. Despite the name,
SageAttention3 targets *consumer* Blackwell (sm120/sm121), not datacenter Blackwell: its FP4
mainloop is built on the warp-level `mma.sync ... kind::mxf4nvf4` instruction, which exists only on
sm120/sm121. Datacenter Blackwell (sm100) reaches FP4 through the unrelated CTA-level `tcgen05.mma`
instruction family, so it cannot run them.
