# LiteAttention

Hopper-only LiteAttention package for the Hugging Face `kernels` library.

This vendors the NVIDIA Hopper path from [moonmath-ai/LiteAttention](https://github.com/moonmath-ai/LiteAttention) and exposes the upstream `LiteAttention` Python API.

Initial scope:

- CUDA SM90a / H100 / H200
- BF16 forward
- INT8 Q/K quantized forward
- LiteAttention skip-list state
- no ROCm, SM80, backward, paged KV, split KV, local attention, softcap, FP8, or training code
