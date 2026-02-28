# quantization-mlx

Metal quantization kernels for Apple Silicon, ported from [MLX](https://github.com/ml-explore/mlx) and wrapped as a PyTorch extension via [kernel-builder](https://github.com/huggingface/kernel-builder).

## Features

- **Affine quantized** matmul/matvec (2/3/4/5/6/8-bit with per-group scales + biases)
- **FP-quantized** matmul/matvec (MXFP4/MXFP8 with FP8 scales)
- **NAX GEMM** variants using MetalPerformancePrimitives for M-series GPUs
- Supports `float32`, `float16`, and `bfloat16`

## Building

```bash
nix build
```

Or with kernel-builder directly:

```bash
pip install kernel-builder
kernel-builder build .
```

## Operations

### FP-quantized (MXFP4)

| Function | Description |
|---|---|
| `mxfp4_qmm_n(x, w, scales, output_features)` | Matrix-matrix, non-transposed weight |
| `mxfp4_qmv(x, w, scales, output_features)` | Matrix-vector |

### Affine quantized (scales + biases)

| Function | Description |
|---|---|
| `affine_qmm_t(x, w, scales, biases, group_size, bits)` | Matrix-matrix, transposed weight |
| `affine_qmm_n(x, w, scales, biases, output_features, group_size, bits)` | Matrix-matrix, non-transposed weight |
| `affine_qmv(x, w, scales, biases, output_features, group_size, bits)` | Matrix-vector |

### NAX variants (MetalPerformancePrimitives)

| Function | Description |
|---|---|
| `affine_qmm_t_nax(x, w, scales, biases, group_size, bits)` | NAX transposed matmul |
| `affine_qmm_n_nax(x, w, scales, biases, output_features, group_size, bits)` | NAX non-transposed matmul |
| `affine_gather_qmm_rhs_nax(x, w, scales, biases, indices, output_features, ...)` | NAX gather + matmul |

## Usage

```python
import torch
import quantization_mlx

# Affine 4-bit quantized matmul (transposed weight)
x = torch.randn(1, 32, 4096, dtype=torch.float16, device="mps")
w = torch.randint(0, 255, (4096, 512), dtype=torch.int32, device="mps")  # [N, K_packed]
scales = torch.randn(4096, 4096 // 128, dtype=torch.float16, device="mps")
biases = torch.zeros(4096, 4096 // 128, dtype=torch.float16, device="mps")

y = quantization_mlx.affine_qmm_t(x, w, scales, biases, group_size=128, bits=4)
# y shape: [1, 32, 4096]
```

## Weight layout conventions

- **Transposed (`qmm_t`)**: `w = [N, K_packed]` — N (output features) is the first dimension, K is packed
- **Non-transposed (`qmm_n`)**: `w = [K_packed, N_packed]` — both dims may be packed; pass `output_features` explicitly
- **Matvec (`qmv`)**: `w = [N, K_packed]` — same as transposed layout

Packing for affine quantized (uint32 storage):
- 4-bit: `pack_factor = 8`, so `K_packed = K // 8`
- 8-bit: `pack_factor = 4`, so `K_packed = K // 4`
- 2-bit: `pack_factor = 16`, so `K_packed = K // 16`
