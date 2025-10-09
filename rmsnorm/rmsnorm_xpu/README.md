# RMSNorm XPU Extension

This repository provides an optimized RMSNorm (Root Mean Square Normalization) kernel for Intel GPUs using Intel's XPU backend. The kernel implementation is adapted from [intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch).

# Benchmark
## Benchmark Hardware
Benchmarks and tests are performed on:
- [Intel Data Center GPU Max 1550](https://www.intel.com/content/www/us/en/products/sku/232873/intel-data-center-gpu-max-1550/specifications.html?wapkw=1550)

## Sample Results

| Input Shape   | Data Type      | Implementation    | Speedup vs. Naive |
|---------------|---------------|------------------|-------------------|
| (1024, 1024)  | torch.bfloat16 | Optimized kernel | 4.3×              |
| (32, 4096)    | torch.bfloat16 | Optimized kernel | 4.3×              |
| (1024, 1024)  | torch.float32  | Optimized kernel | 3.4×              |
| (32, 4096)    | torch.float32  | Optimized kernel | 3.4×              |

> For input shapes `(1024, 1024)` and `(32, 4096)`, the optimized kernel is **4.3× faster** than the naive PyTorch ops implementation with `torch.bfloat16`, and **3.4× faster** with `torch.float32`, on Intel GPU Max 1550.