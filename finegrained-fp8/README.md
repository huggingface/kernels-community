---
license: apache-2.0
tags:
  - kernels
---

# finegrained-fp8

Triton kernels for fine-grained block-wise FP8 quantization and expert dispatch, developed as part of the HuggingFace Transformers FP8 + MoE optimization effort.

All kernels target Hopper (SM90) FP8 WGMMA instructions and are also compatible with Blackwell (SM100), ROCm, and XPU backends via Triton.

## Kernels

### Dispatchers

#### `matmul`, `matmul_batched`, `matmul_grouped`

Neutral entry points that route to the right kernel based on `B.dtype`, `block_size`, and an optional `activation_scale` kwarg:

- `B.dtype == int8` → FP4 path (block dynamic UE8M0).
- `activation_scale is not None` → block-static FP8 (single-matmul only).
- `block_size is None` or `[N, K]` → tensor-mode FP8.
- otherwise → block-dynamic FP8.

This is the recommended public surface — call `matmul(A, B, Bs, block_size, output_dtype, activation_scale=...)` and let the dispatcher pick the kernel.

### Single matmul (`(M, K) @ (N, K).T → (M, N)`)

#### `w8a8_block_dynamic_fp8_matmul`

Block-wise W8A8 FP8 GEMM with **inline** activation quant. `A` is raw bf16/fp16/fp32; the kernel computes per-K-tile per-row scales and casts to `float8_e4m3fn` inside the K-loop. `Bs` accepts fp32 or `float8_e8m0fnu` (UE8M0) — the kernel decodes UE8M0 inline. `BLOCK_SIZE_M` adapts to `M` (16 for decode, up to 128 for prefill).

#### `w8a8_block_static_fp8_matmul`

Same as `w8a8_block_dynamic_fp8_matmul` but takes an explicit per-tensor activation scalar `As`. `A` is still raw bf16/fp16/fp32; the kernel divides by the static scalar before casting to FP8. The scalar factors out of the K-loop and applies once at the end. Useful for calibration-based static activation quant.

#### `w8a8_tensor_dynamic_fp8_matmul`

Tensor-scale W8A8 FP8 GEMM. Pre-quantizes `A` via `fp8_act_quant(A, K)` to get a per-row activation scale, then runs the matmul with that and a single per-tensor weight scale `Bs`.

#### `w4a8_block_dynamic_fp4_matmul`

W4A8 FP4 GEMM. `B` is packed FP4 (`int8`, two E2M1 codes per byte); `Bs` is UE8M0 with a fixed K-group of 32. `A` is raw bf16/fp16/fp32 — quantized to FP8 (E4M3) inline with its own UE8M0 K-group-32 scales. Uses `tl.dot_scaled` for the scaled MMA. Tile shape `(BLOCK_SIZE_N, BLOCK_SIZE_K)` is autotuned.

### Batched (`(S, K) + expert_ids → (S, N)`)

Per-row expert dispatch for MoE decode. Each program handles one routed token, reads its expert id, strides into the matching slice of `(E, N, K)` weights, and writes one output row. EP-sentinel safe (rows with `expert_ids[i] >= num_experts` are skipped). Variants:

- `w8a8_block_dynamic_fp8_matmul_batched`
- `w8a8_tensor_dynamic_fp8_matmul_batched`
- `w4a8_block_dynamic_fp4_matmul_batched`

### Grouped (`(S, K) + offsets + tokens_per_expert → (S, N)`)

Grouped GEMM for MoE prefill: tokens are pre-sorted by expert. Each M-tile finds its owning expert via an O(log E) binary search over `tile_offsets` (unrolled at compile time), then loads the right `(N, K)` weight slice. Grid size is data-independent and CUDA-graph safe. Sentinel rows (`offsets[-1] < S`) are skipped by the early-return guard. Variants:

- `w8a8_block_dynamic_fp8_matmul_grouped`
- `w8a8_tensor_dynamic_fp8_matmul_grouped`
- `w4a8_block_dynamic_fp4_matmul_grouped`

## Naming convention

Every matmul kernel name spells out four axes:

```
w<W>a<A>_<weight_scale_layout>_<activation_quant>_<weight_dtype>_matmul[_batched|_grouped]
```

| Axis                    | Values                                                                                                         |
| ----------------------- | -------------------------------------------------------------------------------------------------------------- |
| `w<W>` / `a<A>`         | weight / activation bit-widths (`w8a8`, `w4a8`)                                                                |
| `<weight_scale_layout>` | `block` (per-`block_n × block_k` weight scale) or `tensor` (one scalar)                                        |
| `<activation_quant>`    | `dynamic` (kernel computes per-K-block / per-row scale inline) or `static` (caller passes a per-tensor scalar) |
| `<weight_dtype>`        | `fp8` (`float8_e4m3fn`) or `fp4` (packed E2M1)                                                                 |

The dispatch suffix selects the input layout:

- _no suffix_: single matmul `(M, K) @ (N, K).T → (M, N)`
- `_batched`: per-row expert dispatch `(S, K) + expert_ids → (S, N)`
- `_grouped`: expert-sorted grouped GEMM `(S, K) + offsets + tokens_per_expert → (S, N)`

Not every combination is implemented — current set:

- `w8a8_block_static_fp8_matmul` (single-matmul only)
- `w8a8_block_dynamic_fp8_matmul[_batched|_grouped]`
- `w4a8_block_dynamic_fp4_matmul[_batched|_grouped]`
- `w8a8_tensor_dynamic_fp8_matmul[_batched|_grouped]`

For convenience, neutral dispatchers `matmul`, `matmul_batched`, `matmul_grouped` route to the right kernel based on `B.dtype`, `block_size`, and the optional `activation_scale` kwarg.
