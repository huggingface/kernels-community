#pragma once

#include <optional>

#include <torch/csrc/stable/tensor.h>

void
causal_conv1d_fwd(const torch::stable::Tensor &x,
                  const torch::stable::Tensor &weight,
                  std::optional<torch::stable::Tensor> bias_,
                  std::optional<torch::stable::Tensor> seq_idx_,
                  std::optional<torch::stable::Tensor> initial_states_,
                  torch::stable::Tensor &out,
                  std::optional<torch::stable::Tensor> final_states_out_,
                  bool silu_activation);

void
causal_conv1d_bwd(const torch::stable::Tensor &x,
                  const torch::stable::Tensor &weight,
                  std::optional<torch::stable::Tensor> bias_,
                  torch::stable::Tensor dout,
                  std::optional<torch::stable::Tensor> seq_idx_,
                  std::optional<torch::stable::Tensor> initial_states_,
                  std::optional<torch::stable::Tensor> dfinal_states_,
                  torch::stable::Tensor &dx,
                  torch::stable::Tensor &dweight,
                  std::optional<torch::stable::Tensor> dbias_,
                  std::optional<torch::stable::Tensor> dinitial_states_,
                  bool silu_activation);

void
causal_conv1d_update(const torch::stable::Tensor &x,
                     const torch::stable::Tensor &conv_state,
                     const torch::stable::Tensor &weight,
                     std::optional<torch::stable::Tensor> bias_,
                     torch::stable::Tensor &out,
                     bool silu_activation,
                     std::optional<torch::stable::Tensor> cache_seqlens_,
                     std::optional<torch::stable::Tensor> conv_state_indices_);
