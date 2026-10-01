#pragma once

#include <cstddef>
#include <cstdint>

/* The ggml-facing boundary.
 *
 * Everything that touches ggml or Metal lives behind these functions, so torch headers and ggml
 * headers never meet in one translation unit. Buffers arrive as (MTLBuffer, byte offset) pairs
 * because a torch tensor's storage is a whole MTLBuffer that the tensor may only be a view into.
 *
 * Returns 0 on success, or non-zero when the metallib has no such kernel, which the caller reports
 * rather than faulting.
 */

extern "C" {

// out[rows, cols] = rms_norm(x) * weight, in one dispatch -- ggml's `kernel_rms_norm_mul_f32`.
// `x`, `weight` and `out` are f32 and contiguous, `weight` a single row broadcast over every row of
// `x`.
int ggml_norm_metal_rms_norm(void *x, size_t x_off, void *w, size_t w_off, void *out, size_t out_off,
                             int32_t rows, int32_t cols, float eps);
}
