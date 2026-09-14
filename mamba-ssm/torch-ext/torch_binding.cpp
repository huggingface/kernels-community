#include <torch/library.h>

#include "registration.h"
#include "pytorch_shim.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
#if !defined(XPU_KERNEL)
  ops.def("selective_scan_fwd(Tensor u, Tensor delta, Tensor A, Tensor B,"
                          "Tensor C, Tensor? D_, Tensor? z_, Tensor? delta_bias_,"
                          "bool delta_softplus) -> Tensor[]");
  ops.impl("selective_scan_fwd", torch::kCUDA, &selective_scan_fwd);

  ops.def("selective_scan_bwd(Tensor u, Tensor delta, Tensor A, Tensor B,"
                          "Tensor C, Tensor? D_, Tensor? z_, Tensor? delta_bias_,"
                          "Tensor dout, Tensor? x_, Tensor? out_, Tensor!? dz_,"
                          "bool delta_softplus, bool recompute_out_z) -> Tensor[]");
  ops.impl("selective_scan_bwd", torch::kCUDA, &selective_scan_bwd);
#endif

  ops.def(
      "causal_conv1d_fwd("
      "    Tensor x, Tensor weight, Tensor? bias, Tensor? seq_idx,"
      "    Tensor? initial_states, Tensor! out, Tensor!? final_states_out,"
      "    bool silu_activation) -> ()");
#if defined(XPU_KERNEL)
  ops.impl("causal_conv1d_fwd", torch::kXPU, make_pytorch_shim(&causal_conv1d_fwd));
#else
  ops.impl("causal_conv1d_fwd", torch::kCUDA, make_pytorch_shim(&causal_conv1d_fwd));
#endif

  ops.def(
      "causal_conv1d_bwd("
      "    Tensor x, Tensor weight, Tensor? bias, Tensor! dout,"
      "    Tensor? seq_idx, Tensor? initial_states, Tensor? dfinal_states,"
      "    Tensor! dx, Tensor! dweight, Tensor!? dbias,"
      "    Tensor!? dinitial_states, bool silu_activation) -> ()");
#if defined(XPU_KERNEL)
  ops.impl("causal_conv1d_bwd", torch::kXPU, make_pytorch_shim(&causal_conv1d_bwd));
#else
  ops.impl("causal_conv1d_bwd", torch::kCUDA, make_pytorch_shim(&causal_conv1d_bwd));
#endif

  ops.def(
      "causal_conv1d_update("
      "    Tensor x, Tensor conv_state, Tensor weight, Tensor? bias,"
      "    Tensor! out, bool silu_activation, Tensor? cache_seqlens,"
      "    Tensor? conv_state_indices) -> ()");
#if defined(XPU_KERNEL)
  ops.impl("causal_conv1d_update", torch::kXPU, make_pytorch_shim(&causal_conv1d_update));
#else
  ops.impl("causal_conv1d_update", torch::kCUDA, make_pytorch_shim(&causal_conv1d_update));
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
