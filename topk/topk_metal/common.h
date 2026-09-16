#pragma once

#include <cstddef>
#include <cstdint>

/* The Metal-facing boundary.
 *
 * Everything that touches Metal lives behind this function, so torch headers and Metal headers never
 * meet in one translation unit. Buffers arrive as (MTLBuffer, byte offset) pairs because a torch
 * tensor's storage is a whole MTLBuffer that the tensor may only be a view into.
 *
 * Returns 0 on success, or non-zero when the metallib has no such kernel, which the caller reports
 * rather than faulting.
 */

extern "C" {

// Top-k over each row of `logits` (`rows x n` f32) -> `indices` (i32) and `values` (f32), both
// `rows x k`. With `softmax` set, `values` is softmaxed over the k that were selected, which is what
// a router wants and saves the caller a second dispatch.
int topk_metal_top_k(void *logits, size_t logits_off, void *indices, size_t indices_off,
                     void *values, size_t values_off, int64_t rows, int64_t n, int64_t k,
                     int softmax);
}
