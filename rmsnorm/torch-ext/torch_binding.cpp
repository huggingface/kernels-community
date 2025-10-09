#include <torch/all.h>
#include "registration.h"
#if defined(XPU_KERNEL)
#include <c10/core/DeviceGuard.h>
#endif

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().type() == torch::kXPU, #x " must be on XPU")

torch::Tensor _apply_rms_norm(torch::Tensor const &hidden_states, torch::Tensor const &weight,
                  double variance_epsilon);

torch::Tensor apply_rms_norm(torch::Tensor const &hidden_states, torch::Tensor const &weight,
                  double variance_epsilon) {
    CHECK_DEVICE(hidden_states); CHECK_DEVICE(weight);
#if defined(XPU_KERNEL)
    c10::DeviceGuard device_guard{hidden_states.device()};
#endif
    return _apply_rms_norm(hidden_states, weight, variance_epsilon);
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("apply_rms_norm(Tensor hidden_states, Tensor weight, float variance_epsilon) -> Tensor");
#if defined(XPU_KERNEL)
  ops.impl("apply_rms_norm", torch::kXPU, &apply_rms_norm);
  ops.impl("apply_rms_norm", c10::DispatchKey::Autograd, &apply_rms_norm);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
