#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("flash_attention_varlen(Tensor! out, Tensor query, Tensor key, Tensor value, Tensor cu_seqlens_q, Tensor cu_seqlens_k, int max_seqlen_q, int max_seqlen_k, bool do_causal, float scale, float softcapping) -> ()");
  ops.impl("flash_attention_varlen", torch::kMPS, flash_attention_varlen);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
