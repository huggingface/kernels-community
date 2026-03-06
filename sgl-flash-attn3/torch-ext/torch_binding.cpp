#include <torch/library.h>

#include "pytorch_shim.h"
#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "fwd(Tensor   q,"
      "    Tensor   k,"
      "    Tensor   v,"
      "    Tensor?  k_new,"
      "    Tensor?  v_new,"
      "    Tensor?  q_v,"
      "    Tensor?  out,"
      "    Tensor?  cu_seqlens_q,"
      "    Tensor?  cu_seqlens_k,"
      "    Tensor?  cu_seqlens_k_new,"
      "    Tensor?  seqused_q,"
      "    Tensor?  seqused_k,"
      "    int?     max_seqlen_q,"
      "    int?     max_seqlen_k,"
      "    Tensor?  page_table,"
      "    Tensor?  kv_batch_idx,"
      "    Tensor?  leftpad_k,"
      "    Tensor?  rotary_cos,"
      "    Tensor?  rotary_sin,"
      "    Tensor?  seqlens_rotary,"
      "    Tensor?  q_descale,"
      "    Tensor?  k_descale,"
      "    Tensor?  v_descale,"
      "    float?   softmax_scale,"
      "    bool     is_causal,"
      "    int      window_size_left,"
      "    int      window_size_right,"
      "    int      attention_chunk,"
      "    float    softcap,"
      "    bool     is_rotary_interleaved,"
      "    Tensor?  scheduler_metadata,"
      "    int      num_splits,"
      "    bool?    pack_gqa,"
      "    int      sm_margin,"
      "    Tensor?  sinks"
      ") -> (Tensor, Tensor, Tensor, Tensor)");

  ops.impl("fwd", torch::kCUDA, make_pytorch_shim(&mha_fwd));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
