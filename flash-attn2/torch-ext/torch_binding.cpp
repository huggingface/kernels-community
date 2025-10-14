#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

// TODO: Add all of the functions listed
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.doc() = "FlashAttention";
//     m.def("fwd", &FLASH_NAMESPACE::mha_fwd, "Forward pass");
//     m.def("varlen_fwd", &FLASH_NAMESPACE::mha_varlen_fwd, "Forward pass (variable length)");
//     m.def("bwd", &FLASH_NAMESPACE::mha_bwd, "Backward pass");
//     m.def("varlen_bwd", &FLASH_NAMESPACE::mha_varlen_bwd, "Backward pass (variable length)");
//     m.def("fwd_kvcache", &FLASH_NAMESPACE::mha_fwd_kvcache, "Forward pass, with KV-cache");
// } 

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("fwd("
    "Tensor! q, "
    "Tensor k, "
    "Tensor v, "
    "Tensor(out_!)? out_, "
    "Tensor? alibi_slopes_, "
    "float p_dropout, "
    "float softmax_scale, "
    "bool is_causal,"
    "int window_size_left, "
    "int window_size_right, "
    "float softcap, "
    "bool return_softmax, "
    "Generator? gen_) -> Tensor[]");
  ops.impl("fwd", torch::kCUDA, &mha_fwd);

  ops.def("varlen_fwd("
    "Tensor! q, "
    "Tensor k, "
    "Tensor v, "
    "Tensor? out_, "
    "Tensor cu_seqlens_q, "
    "Tensor cu_seqlens_k, "
    "Tensor? seqused_k_, "
    "Tensor? leftpad_k_, "
    "Tensor? block_table_, "
    "Tensor? alibi_slopes_, "
    "int max_seqlen_q, "
    "int max_seqlen_k, "
    "float p_dropout, "
    "float softmax_scale, "
    "bool zero_tensors, "
    "bool is_causal, "
    "int window_size_left, "
    "int window_size_right, "
    "float softcap, "
    "bool return_softmax, "
    "Generator? gen_) -> Tensor[]");
  ops.impl("varlen_fwd", torch::kCUDA, &mha_varlen_fwd);

  ops.def("bwd("
    "Tensor! dout, "
    "Tensor! q, "
    "Tensor! k, "
    "Tensor! v, "
    "Tensor! out, "
    "Tensor! "
    "softmax_lse, "
    "Tensor? dq_, "
    "Tensor? dk_, "
    "Tensor? dv_, "
    "Tensor? alibi_slopes_, "
    "float p_dropout, "
    "float softmax_scale, "
    "bool is_causal, "
    "int window_size_left, "
    "int window_size_right, "
    "float softcap, "
    "bool deterministic, "
    "Generator? gen_, "
    "Tensor? rng_state) -> Tensor[]");
  ops.impl("bwd", torch::kCUDA, &mha_bwd);

  ops.def("varlen_bwd("
    "Tensor! dout, "
    "Tensor! q, "
    "Tensor! k, "
    "Tensor! v, "
    "Tensor! out, "
    "Tensor! softmax_lse, "
    "Tensor? dq_, "
    "Tensor? dk_, "
    "Tensor? dv_, "
    "Tensor cu_seqlens_q, "
    "Tensor cu_seqlens_k, "
    "Tensor? alibi_slopes_, "
    "int max_seqlen_q, "
    "int max_seqlen_k, "
    "float p_dropout, float softmax_scale, "
    "bool zero_tensors, "
    "bool is_causal, "
    "int window_size_left, "
    "int window_size_right, "
    "float softcap, "
    "bool deterministic, "
    "Generator? gen_, "
    "Tensor? rng_state) -> Tensor[]");
  ops.impl("varlen_bwd", torch::kCUDA, &mha_varlen_bwd);

  ops.def("fwd_kvcache("
    "Tensor! q, "
    "Tensor! kcache, "
    "Tensor! vcache, "
    "Tensor? k_, "
    "Tensor? v_, "
    "Tensor? seqlens_k_, "
    "Tensor? rotary_cos_, "
    "Tensor? rotary_sin_, "
    "Tensor? cache_batch_idx_, "
    "Tensor? leftpad_k_, "
    "Tensor? block_table_, "
    "Tensor? alibi_slopes_, "
    "Tensor? out_, "
    "float softmax_scale, "
    "bool is_causal, "
    "int window_size_left, "
    "int window_size_right, "
    "float softcap, "
    "bool is_rotary_interleaved, "
    "int num_splits) -> Tensor[]");
  ops.impl("fwd_kvcache", torch::kCUDA, &mha_fwd_kvcache);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
