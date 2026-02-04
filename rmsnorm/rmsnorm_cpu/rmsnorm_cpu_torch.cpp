#include <torch/all.h>
#include "rmsnorm_cpu.hpp"

// Forward implementation for CPU
torch::Tensor rmsnorm_cpu_forward(
    const torch::Tensor &hidden_states,
    const torch::Tensor &weight,
    double variance_epsilon)
{
    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    auto output = torch::empty_like(hidden_states);

    rmsnorm_cpu::rmsnorm(
        output,
        hidden_states,
        weight,
        static_cast<float>(variance_epsilon));

    return output;
}

// Backward implementation for CPU
std::tuple<torch::Tensor, torch::Tensor> rmsnorm_cpu_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &hidden_states,
    const torch::Tensor &weight,
    double variance_epsilon)
{
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    auto grad_input = torch::empty_like(hidden_states);
    auto grad_weight = torch::zeros_like(weight);

    rmsnorm_cpu::rmsnorm_backward(
        grad_input,
        grad_weight,
        grad_output,
        hidden_states,
        weight,
        static_cast<float>(variance_epsilon));

    return std::make_tuple(grad_input, grad_weight);
}

