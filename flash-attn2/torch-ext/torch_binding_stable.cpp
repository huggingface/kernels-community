// CUDA stable-ABI bindings. The XPU/CPU (ATen) bindings live in
// torch_binding.cpp; this file is active only for the CUDA backend.
#if defined(CUDA_KERNEL)

#include <cstdint>

#include <torch/csrc/stable/library.h>

#include "registration.h"

// Boxed entry points, defined in flash_attn/flash_api.cpp.
void boxed_mha_fwd(StableIValue *stack, uint64_t num_args, uint64_t num_outputs);
void boxed_mha_varlen_fwd(StableIValue *stack, uint64_t num_args, uint64_t num_outputs);
void boxed_mha_bwd(StableIValue *stack, uint64_t num_args, uint64_t num_outputs);
void boxed_mha_varlen_bwd(StableIValue *stack, uint64_t num_args, uint64_t num_outputs);
void boxed_mha_fwd_kvcache(StableIValue *stack, uint64_t num_args, uint64_t num_outputs);

// Schemas return Tensor tuples rather than Tensor[], which the stable ABI cannot box.
STABLE_TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
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
    "Generator? gen_) -> (Tensor, Tensor, Tensor, Tensor)");

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
    "Generator? gen_) -> (Tensor, Tensor, Tensor, Tensor)");

  ops.def("bwd("
    "Tensor! dout, "
    "Tensor! q, "
    "Tensor! k, "
    "Tensor! v, "
    "Tensor! out, "
    "Tensor! softmax_lse, "
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
    "Tensor? rng_state) -> (Tensor, Tensor, Tensor, Tensor)");

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
    "Tensor? rng_state) -> (Tensor, Tensor, Tensor, Tensor)");

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
    "int num_splits) -> (Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
  ops.impl("fwd", &boxed_mha_fwd);
  ops.impl("varlen_fwd", &boxed_mha_varlen_fwd);
  ops.impl("bwd", &boxed_mha_bwd);
  ops.impl("varlen_bwd", &boxed_mha_varlen_bwd);
  ops.impl("fwd_kvcache", &boxed_mha_fwd_kvcache);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

#endif  // defined(CUDA_KERNEL)
