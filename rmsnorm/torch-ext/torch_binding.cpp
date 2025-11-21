#include <torch/all.h>
#include "registration.h"
#if defined(XPU_KERNEL)
#include <c10/core/DeviceGuard.h>
#endif

// Forward declarations for XPU
#if defined(XPU_KERNEL)
#define CHECK_DEVICE(x) TORCH_CHECK(x.device().type() == torch::kXPU, #x " must be on XPU")
torch::Tensor _apply_rms_norm(torch::Tensor const &hidden_states, torch::Tensor const &weight,
                  double variance_epsilon);
#endif

#if defined(CPU_KERNEL)
torch::Tensor apply_rms_norm_cpu(
    torch::Tensor const &hidden_states,
    torch::Tensor const &weight,
    double variance_epsilon);
#endif
// Unified dispatcher for both CPU and XPU
torch::Tensor apply_rms_norm(torch::Tensor const &hidden_states, torch::Tensor const &weight,
                  double variance_epsilon) {
#if defined(CPU_KERNEL)
    if (hidden_states.device().type() == torch::kCPU) {
        // CPU path with autograd support
        TORCH_CHECK(weight.device().type() == torch::kCPU, "weight must be on CPU");
        return apply_rms_norm_cpu(hidden_states, weight, variance_epsilon);
    }
#elif defined(XPU_KERNEL)
    if (hidden_states.device().type() == torch::kXPU) {
        // XPU path
        CHECK_DEVICE(hidden_states); CHECK_DEVICE(weight);
        c10::DeviceGuard device_guard{hidden_states.device()};
        return _apply_rms_norm(hidden_states, weight, variance_epsilon);
    }
#endif
    else {
        TORCH_CHECK(false, "Unsupported device type: ", hidden_states.device().type());
    }
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("apply_rms_norm(Tensor hidden_states, Tensor weight, float variance_epsilon) -> Tensor");
#if defined(CPU_KERNEL)
  // Register CPU implementation
  ops.impl("apply_rms_norm", torch::kCPU, &apply_rms_norm);
  ops.impl("apply_rms_norm", c10::DispatchKey::Autograd, &apply_rms_norm);
#elif defined(XPU_KERNEL)
  // Register XPU implementation
  ops.impl("apply_rms_norm", torch::kXPU, &apply_rms_norm);
  ops.impl("apply_rms_norm", c10::DispatchKey::Autograd, &apply_rms_norm);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
