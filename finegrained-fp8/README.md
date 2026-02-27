---
license: apache-2.0
tags:
  - kernels
---

# finegrained-fp8

Triton kernels for fine-grained block-wise FP8 quantization and expert dispatch, developed as part of the HuggingFace Transformers FP8 + MoE optimization effort.

## Kernels

### `fp8_act_quant`

Dynamic per-block activation quantization from BF16/FP16 to `float8_e4m3fn`. Splits the input into contiguous blocks of `block_size` elements, computes the per-block max-abs scale, and stores the quantized tensor alongside the float32 scales. Used as the activation quantization step before every FP8 matmul in the `eager` path.

### `w8a8_block_fp8_matmul`

Block-wise W8A8 FP8 matrix multiplication. Takes a pre-quantized activation matrix `A` (float8, `[M, K]`) with per-token-group scales `As`, and a pre-quantized weight matrix `B` (float8, `[N, K]`) with per-block scales `Bs`, and returns the result in the requested output dtype. The tile shape adapts to M so that small decode batches use the smallest valid FP8 WGMMA tile (16), while larger prefill batches use 128. Used by `FP8Linear` and by the `eager` per-expert loop in `FP8Expert`.

### `w8a8_block_fp8_matmul_batched`

Batched expert matmul for MoE decode, designed for the case where each token routes to a single expert (S tokens × 1 expert each, or small top-k). Activation quantization is **fused** into the inner loop — no separate `fp8_act_quant` call or weight-gather is needed. Each program handles one token: it reads `expert_ids[batch_id]` to stride directly into the correct expert slice of the `[E, N, K]` weight tensor, quantizes its single activation row on the fly (per-block scale, M=1), and writes one output row. Eliminates the `[S, N, K]` gather copy required by non-FP8 batched implementations.

### `w8a8_block_fp8_matmul_grouped`

Grouped GEMM expert matmul for MoE prefill, designed for the case where many tokens share each expert. Tokens are pre-sorted by expert ID, and the kernel uses a statically-sized 2-D grid (`max_M_tiles × N_tiles`) that is safe for CUDA graph capture. Each M-tile determines its owning expert via an **O(log E) binary search** over the cumulative tile-offset array (unrolled at compile time), then loads the corresponding weight/scale slice directly. Activation quantization is fused per-row. Avoids both the weight gather and the per-expert kernel-launch overhead of the `eager` loop.

All kernels target Hopper (SM90) FP8 WGMMA instructions and are also compatible with ROCm and XPU backends via Triton.
