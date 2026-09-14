#include <torch/csrc/stable/library.h>

#include "registration.h"
#include "torch_binding.h"

STABLE_TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("scaled_fp4_quant("
          "Tensor input, "
          "Tensor input_global_scale, "
          "bool is_sf_swizzled_layout) -> (Tensor, Tensor)");

  ops.def("cutlass_scaled_fp4_mm("
          "Tensor a, "
          "Tensor b, "
          "Tensor a_sf, "
          "Tensor b_sf, "
          "Tensor alpha) -> Tensor");

  ops.def("nvfp4_gemv("
          "Tensor a, "
          "Tensor b, "
          "Tensor b_sf, "
          "Tensor alpha) -> Tensor");

  ops.def("nvfp4_gemv_swiglu("
          "Tensor a, "
          "Tensor b, "
          "Tensor b_sf, "
          "Tensor alpha) -> Tensor");

  ops.def("nvfp4_gemv_gated("
          "Tensor a, "
          "Tensor g, "
          "Tensor b, "
          "Tensor b_sf, "
          "Tensor alpha) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
  ops.impl("scaled_fp4_quant", TORCH_BOX(&scaled_fp4_quant));
  ops.impl("cutlass_scaled_fp4_mm", TORCH_BOX(&cutlass_scaled_fp4_mm));
  ops.impl("nvfp4_gemv", TORCH_BOX(&nvfp4_gemv));
  ops.impl("nvfp4_gemv_swiglu", TORCH_BOX(&nvfp4_gemv_swiglu));
  ops.impl("nvfp4_gemv_gated", TORCH_BOX(&nvfp4_gemv_gated));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
