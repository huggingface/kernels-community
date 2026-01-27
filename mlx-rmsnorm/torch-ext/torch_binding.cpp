#include "torch_binding.h"

#include <c10/util/Optional.h>
#include <torch/library.h>

#include "registration.h"

namespace {

void launch_forward_kernel_binding(const at::Tensor& input,
                                   const at::Tensor& weight,
                                   at::Tensor& output,
                                   double epsilon) {
  launch_forward_kernel(input, weight, output, static_cast<float>(epsilon));
}

void launch_backward_kernel_binding(const at::Tensor& input,
                                    const at::Tensor& weight,
                                    const at::Tensor& grad_output,
                                    at::Tensor& grad_input,
                                    c10::optional<at::Tensor> grad_weight,
                                    double epsilon) {
  at::Tensor* grad_weight_ptr = nullptr;
  if (grad_weight.has_value()) {
    grad_weight_ptr = &grad_weight.value();
  }

  launch_backward_kernel(input, weight, grad_output, grad_input, grad_weight_ptr,
                         static_cast<float>(epsilon));
}

}  // namespace

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("launch_forward_kernel(Tensor input, Tensor weight, Tensor! output, float epsilon) -> ()");
  ops.impl("launch_forward_kernel", at::kMPS, &launch_forward_kernel_binding);

  ops.def("launch_backward_kernel(Tensor input, Tensor weight, Tensor grad_output, Tensor(a!) grad_input, Tensor(a!)? grad_weight, float epsilon) -> ()");
  ops.impl("launch_backward_kernel", at::kMPS, &launch_backward_kernel_binding);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)