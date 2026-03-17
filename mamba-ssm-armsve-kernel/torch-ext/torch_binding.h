#pragma once

#include <torch/torch.h>

void mamba_selective_scan(torch::Tensor &A,
                          torch::Tensor &B,
                          torch::Tensor &C,
                          torch::Tensor &hidden_states,
                          torch::Tensor &discrete_time_step,
                          torch::Tensor &ssm_state,
                          torch::Tensor &scan_output,
                          int64_t B_size,
                          int64_t D_size,
                          int64_t L_size,
                          int64_t N_size);