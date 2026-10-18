#pragma once

#include <torch/torch.h>

#include <optional>

// Torch-facing entry points of the vendored aiter gfx1201 SageAttention
// kernels, defined in aiter/torch_itfs/gfx1201_sage_torch.hip.

void gfx1201_sage_prepare(const at::Tensor &query,
                          const at::Tensor &key,
                          const at::Tensor &value,
                          const std::optional<at::Tensor> &query_weight,
                          const std::optional<at::Tensor> &key_weight,
                          const std::optional<at::Tensor> &cosine,
                          const std::optional<at::Tensor> &sine,
                          at::Tensor &maximum,
                          at::Tensor &query_out,
                          at::Tensor &key_out,
                          at::Tensor &value_out,
                          at::Tensor &query_scale,
                          at::Tensor &key_scale,
                          at::Tensor &value_scale,
                          int64_t rope_dim,
                          double eps,
                          double sm_scale);

void gfx1201_sage_attention_fwd(const at::Tensor &q_int8,
                                const at::Tensor &q_scale,
                                const at::Tensor &k_int8,
                                const at::Tensor &k_scale,
                                const at::Tensor &v_fp8,
                                const at::Tensor &v_scale,
                                at::Tensor &out,
                                int64_t seq_len);
