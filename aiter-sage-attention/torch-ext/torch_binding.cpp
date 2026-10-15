#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("gfx1201_sage_prepare("
          "Tensor query, "
          "Tensor key, "
          "Tensor value, "
          "Tensor? query_weight, "
          "Tensor? key_weight, "
          "Tensor? cosine, "
          "Tensor? sine, "
          "Tensor! maximum, "
          "Tensor! query_out, "
          "Tensor! key_out, "
          "Tensor! value_out, "
          "Tensor! query_scale, "
          "Tensor! key_scale, "
          "Tensor! value_scale, "
          "int rope_dim, "
          "float eps, "
          "float sm_scale) -> ()");
  ops.impl("gfx1201_sage_prepare", torch::kCUDA, &gfx1201_sage_prepare);

  ops.def("gfx1201_sage_attention_fwd("
          "Tensor q_int8, "
          "Tensor q_scale, "
          "Tensor k_int8, "
          "Tensor k_scale, "
          "Tensor v_fp8, "
          "Tensor v_scale, "
          "Tensor! out, "
          "int seq_len) -> ()");
  ops.impl("gfx1201_sage_attention_fwd", torch::kCUDA,
           &gfx1201_sage_attention_fwd);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
