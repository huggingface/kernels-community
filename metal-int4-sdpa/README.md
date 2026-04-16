---
tags:
  - metal
  - attention
  - quantization
  - int4
  - apple-silicon
  - sdpa
  - kv-cache
library_name: kernels
---

# Fused int4 SDPA for Apple Silicon

Fused int4 Scaled Dot-Product Attention Metal kernel for Apple Silicon (M1/M2/M3/M4).

Computes `softmax(Q @ dequant(K_int4)^T * scale) @ dequant(V_int4)` in a single
kernel dispatch with online softmax — never materializing full K/V or score matrices.

## Performance

Benchmarked on M1 Max (32 GPU cores, 64GB unified memory).

### vs MLX native FP32 SDPA

| Config | N | Fused int4 | MLX FP32 | Speedup |
|---|---|---|---|---|
| Gemma 4 sliding (sw=1024) | 4096 | **0.34ms** | 2.71ms | **7.9x** |
| Gemma 4 sliding (sw=1024) | 8192 | **0.40ms** | 4.64ms | **11.5x** |
| Gemma 4 full (D=512) | 8192 | **0.98ms** | 7.87ms | **8.1x** |
| Llama 3.1 (D=128) | 8192 | **0.89ms** | 2.56ms | **2.9x** |

### vs PyTorch MPS dequant+SDPA

| N | Fused int4 | MPS Deq+SDPA | Speedup |
|---|---|---|---|
| 2048 | 0.36ms | 1.05ms | **2.9x** |
| 4096 | 0.54ms | 1.56ms | **2.9x** |
| 8192 | 0.67ms | 2.26ms | **3.4x** |

**6.4x KV cache compression** across all configurations (int4 vs FP32).

## Features

- **Online softmax** — O(1) memory in sequence length
- **qdot pattern** — eliminates per-nibble shift operations in K dot product
- **Adaptive split-K** — doubles GPU utilization for models with few attention heads
- **Sliding window** — fast-skip for sliding attention layers (e.g., Gemma 4)
- **GQA support** — grouped-query attention with arbitrary factors
- **D = 128, 256, 512** — common head dimensions

## int4 Format

- uint32 packs 8 x 4-bit nibbles
- Per-group (64 elements) asymmetric quantization: `value = scale * nibble + bias`
- 6.4x compression vs FP32

## Usage

```python
import kernels

sdpa_int4 = kernels.get_kernel("zyan1deOG/metal-int4-sdpa", "sdpa_int4")

# Single-token decode
output = sdpa_int4.sdpa_int4(
    queries,       # (num_heads, D) float32, MPS
    k_quant,       # (num_kv_heads, N, D//8) uint32, MPS
    k_scales,      # (num_kv_heads, N, D//64) float32, MPS
    k_biases,      # (num_kv_heads, N, D//64) float32, MPS
    v_quant,       # (num_kv_heads, N, D//8) uint32, MPS
    v_scales,      # (num_kv_heads, N, D//64) float32, MPS
    v_biases,      # (num_kv_heads, N, D//64) float32, MPS
    gqa_factor=4,  # num_heads // num_kv_heads
    N=2048,        # sequence length
    scale=0.0625,  # 1/sqrt(D)
    sliding_window=0,  # 0 = full attention
)
```

## Tested Models

- Gemma 4 31B (32 heads, 16 KV heads, D=256/512, sliding window)
- Llama 3.1 architecture (32 heads, 8 KV heads, D=128)

## Origin

Ported from TurboQuant's `sdpa_int4_vector` kernel.

Paper: [TurboQuant (arxiv.org/abs/2504.19874)](https://arxiv.org/abs/2504.19874), ICLR 2026.
