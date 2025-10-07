#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("sgmv_shrink(Tensor! y, Tensor x, Tensor w_ptr, Tensor s_start, "
                      "Tensor s_end, Tensor! tmp, int layer_idx) -> ()");
  ops.impl("sgmv_shrink", torch::kCUDA, &dispatch_sgmv_shrink);

  ops.def("sgmv_cutlass(Tensor! y, Tensor x, Tensor w_ptr, Tensor s_start, "
                       "Tensor s_end, Tensor! tmp, int layer_idx) -> ()");
  ops.impl("sgmv_cutlass", torch::kCUDA, &dispatch_sgmv_cutlass);

  ops.def("sgmv_cutlass_tmp_size(int num_problems) -> int");
  ops.impl("sgmv_cutlass_tmp_size", &sgmv_tmp_size);

  ops.def("dispatch_bgmv(Tensor! y, Tensor x, Tensor w_ptr, Tensor indices, "
                        "int layer_indices, float scale) -> ()");
  ops.impl("dispatch_bgmv", torch::kCUDA, &dispatch_bgmv);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
