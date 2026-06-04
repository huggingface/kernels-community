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

Block-wise W8A8 FP8 matrix multiplication. Takes a pre-quantized activation matrix `A` (float8, `[M, K]`) with per-token-group scales `As`, and a pre-quantized weight matrix `B` (float8, `[N, K]`) with per-block scales `Bs`, and returns the result in the requested output dtype. The tile shape adapts to M so that small decode batches use the smallest valid FP8 WGMMA tile (16), while larger prefill batches use 128. Used by `FP8Linear` and by the `eager` per-expert loop in `FP8Experts`.

### `w8a8_tensor_fp8_matmul`

Tensor-scale W8A8 FP8 matrix multiplication. Expects pre-quantized activations and weights, plus tensor-scale activation/weight scales (`As`, `Bs`), and computes `C = A @ B.T`. This path is used for per-tensor quantization (`block_size=None` / full-matrix block).

### `w8a8_fp8_matmul`

Unified matmul dispatcher for the 2D case. Routes to tensor mode when `block_size is None` or `block_size == [N, K]`; otherwise routes to block mode.

### `w8a8_fp8_matmul_batched`

Batched expert matmul for MoE decode, designed for the case where each token routes to a single expert (S tokens × 1 expert each, or small top-k). Activation quantization is **fused** into the inner loop — no separate `fp8_act_quant` call or weight-gather is needed. Each program handles one token: it reads `expert_ids[batch_id]` to stride directly into the correct expert slice of the `[E, N, K]` weight tensor, quantizes its single activation row on the fly (per-block scale, M=1), and writes one output row. Eliminates the `[S, N, K]` gather copy required by non-FP8 batched implementations.

### `w8a8_tensor_fp8_matmul_batched`

Tensor-scale batched expert matmul. Quantizes activations once with per-token tensor scales and multiplies by expert-selected FP8 weights using per-expert tensor scales (`[E]` or `[E,1,1]`).

### `w8a8_fp8_matmul_grouped`

Grouped GEMM expert matmul for MoE prefill, designed for the case where many tokens share each expert. Tokens are pre-sorted by expert ID, and the kernel uses a statically-sized 2-D grid (`max_M_tiles × N_tiles`) that is safe for CUDA graph capture. Each M-tile determines its owning expert via an **O(log E) binary search** over the cumulative tile-offset array (unrolled at compile time), then loads the corresponding weight/scale slice directly. Activation quantization is fused per-row. Avoids both the weight gather and the per-expert kernel-launch overhead of the `eager` loop.

### `w8a8_tensor_fp8_matmul_grouped`

Tensor-scale grouped expert matmul. Uses the same grouped scheduling as block mode, but with per-token tensor activation scales and per-expert tensor weight scales.

### Dispatch semantics summary

- `w8a8_fp8_matmul`: tensor mode if `block_size is None` or `block_size == [N, K]`, else block mode.
- `w8a8_fp8_matmul_batched`: tensor mode if `block_size is None` or `block_size == [N, K]`, else block mode.
- `w8a8_fp8_matmul_grouped`: tensor mode if `block_size is None` or `block_size == [N, K]`, else block mode.

All kernels target Hopper (SM90) FP8 WGMMA instructions and are also compatible with ROCm and XPU backends via Triton.

## Naming convention

Every matmul kernel name spells out four axes:

```
w<W>a<A>_<weight_scale_layout>_<activation_quant>_<weight_dtype>_matmul[_batched|_grouped]
```

| Axis | Values |
|---|---|
| `w<W>` / `a<A>` | weight / activation bit-widths (`w8a8`, `w4a8`) |
| `<weight_scale_layout>` | `block` (per-`block_n × block_k` weight scale) or `tensor` (one scalar) |
| `<activation_quant>` | `dynamic` (kernel computes per-K-block / per-row scale inline) or `static` (caller passes a per-tensor scalar) |
| `<weight_dtype>` | `fp8` (`float8_e4m3fn`) or `fp4` (packed E2M1) |

The dispatch suffix selects the input layout:

- *no suffix*: single matmul `(M, K) @ (N, K).T → (M, N)`
- `_batched`: per-row expert dispatch `(S, K) + expert_ids → (S, N)`
- `_grouped`: expert-sorted grouped GEMM `(S, K) + offsets + tokens_per_expert → (S, N)`

Not every combination is implemented — current set:

- `w8a8_block_static_fp8_matmul` (single-matmul only)
- `w8a8_block_dynamic_fp8_matmul[_batched|_grouped]`
- `w4a8_block_dynamic_fp4_matmul[_batched|_grouped]`
- `w8a8_tensor_dynamic_fp8_matmul[_batched|_grouped]`

For convenience, neutral dispatchers `matmul`, `matmul_batched`, `matmul_grouped` route to the right kernel based on `B.dtype`, `block_size`, and the optional `activation_scale` kwarg.
