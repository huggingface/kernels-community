#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("ms_deform_attn_forward(Tensor value, Tensor spatial_shapes,"
          "                       Tensor level_start_index, Tensor sampling_loc,"
          "                       Tensor attn_weight, int im2col_step) -> Tensor");
  ops.impl("ms_deform_attn_forward", torch::kCUDA, &ms_deform_attn_cuda_forward);

  ops.def("ms_deform_attn_backward(Tensor value, Tensor spatial_shapes,"
          "                       Tensor level_start_index, Tensor sampling_loc,"
          "                       Tensor attn_weight, Tensor grad_output,"
          "                       int im2col_step) -> Tensor[]");
  ops.impl("ms_deform_attn_backward", torch::kCUDA, &ms_deform_attn_cuda_backward);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
