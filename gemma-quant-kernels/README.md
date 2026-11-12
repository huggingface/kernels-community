---
license: apache-2.0
tags:
  - kernels
---

# Gemma Quant Kernels (`gemma-quant-kernels`)

Fused Triton GPU kernels for low-bit (`int2`, `int4`, `int8`) quantized Gemma 4 models, supporting both dense linear projections (`QuantizedLinear`) and routed Mixture-of-Experts layers (`QuantizedGemma4TextExperts`) in [`huggingface/transformers`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/gemma_quant.py).

## Features

- **`lowbit_gemm(x, weight, weight_scale, bias=None, num_bits=4)`**:
  - Single-token autoregressive decoding (`M == 1`) via fused low-bit GEMV.
  - Batched decoding and prompt prefill (`M > 1`) via tiled Tensor Core (`tl.dot`) GEMM.
  - Streams packed integer weights (`uint8` for offset-binary INT2/INT4, `int8` for INT8) directly from GPU VRAM into registers without materializing intermediate float weight tensors.
- **`grouped_lowbit_gemm(permuted_x, packed_w, scales, offsets, num_bits=4)`**:
  - Fused grouped GEMM for Mixture-of-Experts (MoE) layers across all routed tokens and experts in a single kernel dispatch.
- **`pack_int2` / `pack_int4` / `unpack_int2` / `unpack_int4`**:
  - Offset-binary (`+2` for INT2, `+8` for INT4) packing/unpacking helpers for 2D (`[N, K]`) and 3D (`[E, N, K]`) weight tensors.
- **`torch.compile` & Multi-GPU Compatible**:
  - Registered via `@torch.library.custom_op` with `add_op_namespace_prefix` and `@register_fake`.

## Usage

```python
from kernels import get_kernel

gemma_quant = get_kernel("kernels-community/gemma-quant-kernels", version=1)

# Dense projection (INT2 / INT4 / INT8)
out = gemma_quant.lowbit_gemm(x, packed_weight, weight_scale, bias, num_bits=4)

# MoE grouped expert projection (INT2 / INT4 / INT8)
expert_out = gemma_quant.grouped_lowbit_gemm(permuted_x, expert_packed_w, expert_scales, offsets, num_bits=4)
```

## Benchmarks (NVIDIA H100 80GB HBM3, `M = 1` Decode)

| Layer Configuration | Dimensions (`K -> N`) | Eager PyTorch (Dequant + `F.linear`) | Fused Triton 4-bit | Fused Triton 2-bit | Speedup vs Eager |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gemma 4 E2B/E4B (QKV)** | `2560 x 2560` | `0.219 ms` | `0.0184 ms` | `0.0175 ms` | **11.9x** |
| **Gemma 4 E2B/E4B (MLP)** | `2560 x 10240` | `0.377 ms` | `0.0248 ms` | `0.0239 ms` | **15.2x** |
| **Gemma 4 26B-A4B (Attn)** | `4096 x 4096` | `0.259 ms` | `0.0165 ms` | `0.0158 ms` | **15.7x** |
| **Gemma 4 26B-A4B (Expert FFN)** | `4096 x 14336` | `0.724 ms` | `0.0433 ms` | `0.0414 ms` | **16.7x** |
| **Gemma 4 31B Dense (MLP)** | `5376 x 21504` | `1.356 ms` | `0.0792 ms` | `0.0761 ms` | **17.1x** |
