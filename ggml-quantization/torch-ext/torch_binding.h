#pragma once

#include <torch/torch.h>

#include <vector>

// The ggml type ids this build's `mul_mat_vec` implements. Backend-specific: the CUDA port reaches
// everything upstream's `mul_mat_vec_q_switch_type` covers, while Metal's list is the type table in
// gguf_metal/ggml_dispatch.mm. A caller must ask rather than assume, or it will route a type into a
// gemv that has no kernel for it.
std::vector<int64_t> gemv_types();

// The two entry points every backend implements. Both take a GGUF weight exactly as it is stored
// in the file — `(rows, bytes_per_row)` uint8 blocks — so nothing has to be unpacked to use them.

// Blocks -> values, for the rows `indices` names: `(rows, bytes_per_row)` uint8 -> `(len, cols)` of
// `dtype`. ggml's `get_rows`, which unpacks as it gathers, so reading a handful of rows out of a large
// table never materializes the rest.
at::Tensor get_rows(const at::Tensor &blocks, const at::Tensor &indices, int64_t ggml_type,
                    int64_t cols, at::ScalarType dtype);

// Blocks -> values: `(rows, bytes_per_row)` uint8 -> `(rows, cols)` of `dtype`. `get_rows` over every
// row in order.
at::Tensor dequantize(const at::Tensor &blocks, int64_t ggml_type, int64_t rows, int64_t cols,
                      at::ScalarType dtype);

// Fused dequantize-gemv: `blocks` is `(out_features, bytes_per_row)`, `x` is `(rows, in_features)`
// with `rows <= MAX_GEMV_ROWS`. Returns `(rows, out_features)` f32, whatever `x`'s dtype.
at::Tensor mul_mat_vec(const at::Tensor &blocks, const at::Tensor &x, int64_t ggml_type,
                       int64_t out_features);

// One dispatch for a whole bank of routed experts. `blocks` is `(n_experts, out_features,
// bytes_per_row)`, `x` is `(n_tokens, in_features)`, and `ids` is `(n_tokens, n_used)` naming the
// expert each of a token's slots selected. Returns `(n_tokens, n_used, out_features)` f32.
//
// A MoE layer is otherwise a Python loop of `mul_mat_vec` calls -- one per expert per layer, each a
// gemv whose arithmetic is dwarfed by the dispatch around it.
at::Tensor mul_mat_id(const at::Tensor &blocks, const at::Tensor &x, const at::Tensor &ids,
                      int64_t ggml_type, int64_t out_features);
