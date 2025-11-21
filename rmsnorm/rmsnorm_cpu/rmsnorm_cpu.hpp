#pragma once

#include "cpu_features.hpp"
#include "rmsnorm_avx2.hpp"
#include "rmsnorm_avx512.hpp"
#include <stdexcept>
#include <ATen/ATen.h>

namespace rmsnorm_cpu
{

    // Main dispatcher that selects the best implementation based on runtime CPU features
    void rmsnorm(torch::Tensor &out, const torch::Tensor &input, const torch::Tensor &weight,
                 float epsilon);

    void rmsnorm_backward(
        torch::Tensor &grad_input,
        torch::Tensor &grad_weight,
        const torch::Tensor &grad_output,
        const torch::Tensor &hidden_states,
        const torch::Tensor &weight,
        float variance_epsilon);
} // namespace rmsnorm_cpu
