#include <torch/all.h>
#include "registration.h"
#if defined(XPU_KERNEL)
#include <c10/core/DeviceGuard.h>
#endif

#if defined(CPU_KERNEL)
torch::Tensor rmsnorm_cpu_forward(const torch::Tensor &hidden_states, const torch::Tensor &weight, double variance_epsilon);
std::tuple<torch::Tensor, torch::Tensor> rmsnorm_cpu_backward(const torch::Tensor &grad_output, const torch::Tensor &hidden_states, const torch::Tensor &weight, double variance_epsilon);
#endif

#if defined(XPU_KERNEL)
namespace at { namespace AtenTypeXPU {
    std::tuple<at::Tensor, at::Tensor> rms_norm_fw(const at::Tensor& input, at::IntArrayRef normalized_shape, const at::Tensor& weight, double epsilon);
    std::tuple<at::Tensor, at::Tensor> rms_norm_bw(const at::Tensor& grad_output, const at::Tensor& input, at::IntArrayRef normalized_shape, const at::Tensor& rstd, const at::Tensor& weight, const at::Tensor& output, std::array<bool, 2> grad_input_mask);
}}
#endif


std::vector<torch::Tensor> apply_rms_norm(torch::Tensor const &hidden_states, torch::Tensor const &weight,
                  double variance_epsilon) {

    if (hidden_states.is_meta()) {
        return {torch::empty_like(hidden_states), torch::empty({hidden_states.size(0)}, hidden_states.options())};
    }

#if defined(CPU_KERNEL)
    if (hidden_states.device().type() == torch::kCPU) {
        auto out = rmsnorm_cpu_forward(hidden_states, weight, variance_epsilon);
        return {out, torch::empty({hidden_states.size(0)}, hidden_states.options())};
    }
#endif

#if defined(XPU_KERNEL)
    if (hidden_states.device().type() == torch::kXPU) {
        c10::DeviceGuard device_guard{hidden_states.device()};
        auto res = at::AtenTypeXPU::rms_norm_fw(hidden_states, {hidden_states.size(-1)}, weight, variance_epsilon);
        return {std::get<0>(res), std::get<1>(res)};
    }
#endif
    TORCH_CHECK(false, "Unsupported device type");
}

std::vector<torch::Tensor> apply_rms_norm_backward(
    const torch::Tensor &grad_output,
    const torch::Tensor &hidden_states,
    const torch::Tensor &weight,
    const torch::Tensor &output,
    const torch::Tensor &rstd,
    double variance_epsilon,
    bool input_requires_grad,
    bool weight_requires_grad) {

    if (grad_output.is_meta()) {
        return {input_requires_grad ? torch::empty_like(hidden_states) : torch::Tensor(),
                weight_requires_grad ? torch::empty_like(weight) : torch::Tensor()};
    }

#if defined(CPU_KERNEL)
    if (grad_output.is_cpu()) {
        auto res = rmsnorm_cpu_backward(grad_output, hidden_states, weight, variance_epsilon);
        return {input_requires_grad ? std::get<0>(res) : torch::Tensor(),
                weight_requires_grad ? std::get<1>(res) : torch::Tensor()};
    }
#endif

#if defined(XPU_KERNEL)
    if (grad_output.is_xpu()) {
        c10::DeviceGuard device_guard{grad_output.device()};
        auto res = at::AtenTypeXPU::rms_norm_bw(
            grad_output, hidden_states, {hidden_states.size(-1)}, rstd, weight, output, {input_requires_grad, weight_requires_grad});
        return {std::get<0>(res), std::get<1>(res)};
    }
#endif
    TORCH_CHECK(false, "Unsupported device type");
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("apply_rms_norm(Tensor hidden_states, Tensor weight, float variance_epsilon) -> Tensor[]");
  ops.def("apply_rms_norm_backward(Tensor grad_output, Tensor hidden_states, Tensor weight, Tensor output, Tensor rstd, float variance_epsilon, bool input_requires_grad, bool weight_requires_grad) -> Tensor[]");

  ops.impl("apply_rms_norm", c10::DispatchKey::CompositeExplicitAutograd, &apply_rms_norm);
  ops.impl("apply_rms_norm_backward", c10::DispatchKey::CompositeExplicitAutograd, &apply_rms_norm_backward);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
