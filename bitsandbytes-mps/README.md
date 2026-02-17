# bitsandbytes-mps

Metal (MPS) kernels for bitsandbytes 4-bit quantization on Apple Silicon.

Provides NF4 and FP4 blockwise quantization, dequantization, and **fused GEMV/GEMM** operations for efficient inference with 4-bit quantized models on macOS.

## Operations

| Operation | Description |
|-----------|-------------|
| `quantize_4bit` | Blockwise 4-bit quantization (NF4/FP4) with per-block absmax |
| `dequantize_4bit` | Blockwise 4-bit dequantization using codebook lookup |
| `gemv_4bit` | Fused dequantize + matrix-vector multiply (batch_size=1 inference) |
| `gemm_4bit` | Fused dequantize + matrix-matrix multiply (larger batch inference) |
| `linear_4bit` | Auto-selecting linear layer (GEMV for vectors, GEMM for matrices) |

## Quantization Format

Uses the bitsandbytes blockwise quantization scheme:
- **Packing**: 2 values per byte (high nibble = first element, low nibble = second)
- **Scaling**: One `absmax` (float32) per block of `blocksize` elements
- **Codebook**: NF4 (16 values optimized for normal distributions) or FP4 (sign-magnitude floating point)
- **Dequantization**: `value = codebook[4bit_index] * absmax`

## Usage

```python
import torch
from bitsandbytes_mps import quantize_4bit, dequantize_4bit, gemv_4bit, gemm_4bit, NF4

# Quantize a weight matrix
weight = torch.randn(4096, 4096, dtype=torch.float16, device="mps")
packed, absmax = quantize_4bit(weight.flatten(), blocksize=64, quant_type=NF4)

# Dequantize
weight_deq = dequantize_4bit(packed, absmax, blocksize=64, quant_type=NF4,
                              numel=weight.numel(), output_dtype=torch.float16)

# Fused GEMV (single vector)
x = torch.randn(4096, dtype=torch.float16, device="mps")
packed_w = packed.view(4096, -1)  # [N, K/2]
absmax_w = absmax.view(4096, -1)  # [N, K_groups]
y = gemv_4bit(x, packed_w, absmax_w, output_features=4096, blocksize=64, quant_type=NF4)

# Fused GEMM (batch of vectors)
X = torch.randn(8, 4096, dtype=torch.float16, device="mps")
Y = gemm_4bit(X, packed_w, absmax_w, output_features=4096, blocksize=64, quant_type=NF4)
```

## Supported Configurations

- **Scalar types**: float16, bfloat16, float32
- **Block sizes**: 64, 128
- **Quant types**: FP4, NF4

## Architecture

The kernels are adapted from [MLX quantization Metal kernels](https://github.com/ml-explore/mlx) with the following modifications:

1. **Codebook-based dequantization** replaces MLX's affine `scale * q + bias` with `codebook[q] * absmax`
2. **BnB packing format**: high nibble first (vs MLX's low nibble first)
3. **`BnBQuantizedBlockLoader`**: Custom block loader for tiled GEMM that dequantizes on-the-fly using codebook lookup
4. **Binary search quantization**: Efficient NF4/FP4 quantization using decision trees (matching CUDA kernels)

## Building

```bash
pip install kernel-builder
kernel-builder build .
```
