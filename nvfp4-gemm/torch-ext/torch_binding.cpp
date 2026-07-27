#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("scaled_fp4_quant("
          "Tensor input, "
          "Tensor input_global_scale, "
          "bool is_sf_swizzled_layout) -> (Tensor, Tensor)");
  ops.impl("scaled_fp4_quant", torch::kCUDA, &scaled_fp4_quant);

  ops.def("cutlass_scaled_fp4_mm("
          "Tensor a, "
          "Tensor b, "
          "Tensor a_sf, "
          "Tensor b_sf, "
          "Tensor alpha) -> Tensor");
  ops.impl("cutlass_scaled_fp4_mm", torch::kCUDA, &cutlass_scaled_fp4_mm);

  ops.def("nvfp4_gemv("
          "Tensor a, "
          "Tensor b, "
          "Tensor b_sf, "
          "Tensor alpha) -> Tensor");
  ops.impl("nvfp4_gemv", torch::kCUDA, &nvfp4_gemv);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
