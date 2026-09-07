#pragma once

#include <cstddef>
#include <cstdint>

/* The ggml-facing boundary, mirroring gguf_cuda/ggml_dispatch.cu.
 *
 * Everything that touches ggml or Metal lives behind these functions, so torch headers and ggml
 * headers never meet in one translation unit. Buffers arrive as (MTLBuffer, byte offset) pairs
 * because a torch tensor's storage is a whole MTLBuffer that the tensor may only be a view into.
 *
 * Each returns 0 on success, or non-zero when the quantization has no kernel upstream, which the
 * caller reports rather than faulting.
 */

extern "C" {

// out[M, N] = x[M, K] @ dequant(blocks)[N, K]^T, using ggml's mul_mv (M small) or mul_mm kernels.
// `x` and `out` are f32; `blocks` is the GGUF weight exactly as stored.
int gguf_metal_mul_mat(void *blocks, size_t blocks_off, void *x, size_t x_off, void *out,
                       size_t out_off, int ggml_type, int64_t K, int64_t N, int64_t M);

// One dispatch for a whole bank of experts, via ggml's mul_mv_id -- what a MoE layer needs and what
// a loop of `gguf_metal_mul_mat` calls cannot be: the routed experts of one token are independent, so
// upstream runs them as a single grid rather than a dispatch each.
//
// `blocks` is `(E, N, bytes_per_row)`, `x` is `(T, K)` f32, `ids` is `(T, U)` i32 naming the expert
// each of a token's `U` slots selected, and `out` is `(T, U, N)` f32.
int gguf_metal_mul_mat_id(void *blocks, size_t blocks_off, void *x, size_t x_off, void *ids,
                          size_t ids_off, void *out, size_t out_off, int ggml_type, int64_t K,
                          int64_t N, int64_t E, int64_t T, int64_t U);

// dequant(blocks)[rows, cols] -> out, via ggml's get_rows. `indices` is an i32 buffer holding the
// row numbers to unpack, which for a whole weight is simply 0..rows-1.
int gguf_metal_get_rows(void *blocks, size_t blocks_off, void *indices, size_t indices_off,
                        void *out, size_t out_off, int ggml_type, int64_t rows, int64_t cols,
                        int out_dtype);

// True when the type has a gemv/gemm kernel in the metallib.
int gguf_metal_supports(int ggml_type);

// The ggml type ids above, written into `out` (up to `max`); returns how many there are. Called
// once at import to publish `GEMV_TYPES`, so the caller never has to guess this backend's coverage.
int gguf_metal_gemv_types(int *out, int max);
}
