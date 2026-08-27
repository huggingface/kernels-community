#pragma once

#include <torch/torch.h>

void
causal_conv1d_fwd(const at::Tensor &x,
                  const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias_,
                  const c10::optional<at::Tensor> &seq_idx_,
                  const c10::optional<at::Tensor> &initial_states_,
                  at::Tensor &out,
                  c10::optional<at::Tensor> &final_states_out_,
                  bool silu_activation);

void
causal_conv1d_bwd(const at::Tensor &x,
                  const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias_,
                  at::Tensor &dout,
                  const c10::optional<at::Tensor> &seq_idx_,
                  const c10::optional<at::Tensor> &initial_states_,
                  const c10::optional<at::Tensor> &dfinal_states_,
                  at::Tensor &dx,
                  at::Tensor &dweight,
                  c10::optional<at::Tensor> &dbias_,
                  c10::optional<at::Tensor> &dinitial_states_,
                  bool silu_activation);

void
causal_conv1d_update(const at::Tensor &x,
                     const at::Tensor &conv_state,
                     const at::Tensor &weight,
                     const c10::optional<at::Tensor> &bias_,
                     at::Tensor &out,
                     bool silu_activation,
                     const c10::optional<at::Tensor> &cache_seqlens_,
                     const c10::optional<at::Tensor> &conv_state_indices_
                     );

