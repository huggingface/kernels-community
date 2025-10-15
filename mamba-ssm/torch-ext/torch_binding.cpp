#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("selective_scan_fwd(Tensor u, Tensor delta, Tensor A, Tensor B,"
                          "Tensor C, Tensor? D_, Tensor? z_, Tensor? delta_bias_,"
                          "bool delta_softplus) -> Tensor[]");
  ops.impl("selective_scan_fwd", torch::kCUDA, &selective_scan_fwd);

  ops.def("selective_scan_bwd(Tensor u, Tensor delta, Tensor A, Tensor B,"
                          "Tensor C, Tensor? D_, Tensor? z_, Tensor? delta_bias_,"
                          "Tensor dout, Tensor? x_, Tensor? out_, Tensor!? dz_,"
                          "bool delta_softplus, bool recompute_out_z) -> Tensor[]");
  ops.impl("selective_scan_bwd", torch::kCUDA, &selective_scan_bwd);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
