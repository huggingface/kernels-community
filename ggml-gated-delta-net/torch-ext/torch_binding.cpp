#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "gated_delta_net(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor state) "
      "-> Tensor");
  ops.def("delta_gates(Tensor b, Tensor a, Tensor a_log, Tensor dt_bias) -> Tensor[]");
  ops.def("rms_norm_gate(Tensor x, Tensor weight, Tensor gate, float eps) -> Tensor");
  ops.def("causal_conv_update(Tensor state, Tensor x, Tensor weight, Tensor? bias, bool silu) -> Tensor");
  ops.def("l2_norm(Tensor x, float eps) -> Tensor");
  // Take no tensor, so they have no device to dispatch on and are registered as catch-alls. Each
  // backend's shared object is its own library namespace, so there is one implementation per build.
  ops.def("supports_gated_delta_net(int head_dim) -> bool");
  ops.impl("supports_gated_delta_net", &supports_gated_delta_net);

  // The schema is the same for every backend; only the implementation differs.
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("gated_delta_net", torch::kCUDA, &gated_delta_net);
  ops.impl("l2_norm", torch::kCUDA, &l2_norm);
#elif defined(METAL_KERNEL)
  ops.impl("gated_delta_net", torch::kMPS, &gated_delta_net);
  ops.impl("delta_gates", torch::kMPS, &delta_gates);
  ops.impl("rms_norm_gate", torch::kMPS, &rms_norm_gate);
  ops.impl("causal_conv_update", torch::kMPS, &causal_conv_update);
  ops.impl("l2_norm", torch::kMPS, &l2_norm);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
