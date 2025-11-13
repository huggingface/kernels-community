#include <torch/all.h>
#include "rmsnorm_cpu.hpp"

// Forward implementation for CPU
torch::Tensor rmsnorm_cpu_forward(
    const torch::Tensor& hidden_states,
    const torch::Tensor& weight,
    double variance_epsilon
) {
    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    
    auto output = torch::empty_like(hidden_states);

    rmsnorm_cpu::rmsnorm(
        output,
        hidden_states,
        weight,
        static_cast<float>(variance_epsilon)
    );

    return output;
}

// Backward implementation for CPU
std::tuple<torch::Tensor, torch::Tensor> rmsnorm_cpu_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& hidden_states,
    const torch::Tensor& weight,
    double variance_epsilon
) {
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
        static_cast<float>(variance_epsilon)
    );

    return std::make_tuple(grad_input, grad_weight);
}


// Custom autograd function for CPU RMSNorm
class RMSNormCPUFunction : public torch::autograd::Function<RMSNormCPUFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        const torch::Tensor& hidden_states,
        const torch::Tensor& weight,
        double variance_epsilon
    ) {
        ctx->save_for_backward({hidden_states, weight});
        ctx->saved_data["variance_epsilon"] = variance_epsilon;
        return rmsnorm_cpu_forward(hidden_states, weight, variance_epsilon);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto hidden_states = saved[0];
        auto weight = saved[1];
        auto variance_epsilon = ctx->saved_data["variance_epsilon"].toDouble();
        auto grad_output = grad_outputs[0];

        auto grads = rmsnorm_cpu_backward(grad_output, hidden_states, weight, variance_epsilon);
        auto grad_input = std::get<0>(grads);
        auto grad_weight = std::get<1>(grads);

        return {grad_input, grad_weight, torch::Tensor()};
    }
};


torch::Tensor apply_rms_norm_cpu(
    const torch::Tensor& hidden_states,
    const torch::Tensor& weight,
    double variance_epsilon) {

    auto output = RMSNormCPUFunction::apply(hidden_states, weight, variance_epsilon);
    return output;
}

