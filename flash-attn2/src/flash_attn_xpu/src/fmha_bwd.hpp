#pragma once

#include "fmha_bwd_types.hpp"

namespace sycl {
  inline namespace _V1 {
    class queue;
  }
}

namespace at {
  class Tensor;
}

// Forward declare bwd_policy structs (defined in fmha_bwd_types.hpp)
struct bwd_policy_head32;
struct bwd_policy_head64;
struct bwd_policy_head96;
struct bwd_policy_head128;
struct bwd_policy_head160;
struct bwd_policy_head192;
struct bwd_policy_head256;
struct bwd_policy_head512;

// Dtype-specific bwd dispatch functions (instantiated in per-head TUs)
template <typename bwd_policy, int IsCausal, int IsLocal, int IsVarLen>
void bwd_policy_dispatch_fp16(
    sycl::queue& queue,
    const fmha_bwd_args_t& args);

template <typename bwd_policy, int IsCausal, int IsLocal, int IsVarLen>
void bwd_policy_dispatch_bf16(
    sycl::queue& queue,
    const fmha_bwd_args_t& args);

// Combined bwd dispatch (delegates to fp16/bf16 based on cuType)
// Defined inline in header so callers (fmha_bwd.cpp) can see the template body.
template <typename bwd_policy, int IsCausal, int IsLocal, int IsVarLen>
inline void bwd_policy_dispatch(
    sycl::queue& queue,
    BwdCutlassType cuType,
    const fmha_bwd_args_t& args) {
  if (cuType == BwdCutlassType::half) {
    bwd_policy_dispatch_fp16<bwd_policy, IsCausal, IsLocal, IsVarLen>(queue, args);
  } else {
    bwd_policy_dispatch_bf16<bwd_policy, IsCausal, IsLocal, IsVarLen>(queue, args);
  }
}

// Extern template declarations for all head dimensions (dtype-split)
#define EXTERN_BWD_DISPATCH(HDIM) \
  extern template void bwd_policy_dispatch_fp16<bwd_policy_head##HDIM, 0, 0, 0>(sycl::queue&, const fmha_bwd_args_t&); \
  extern template void bwd_policy_dispatch_fp16<bwd_policy_head##HDIM, 0, 1, 0>(sycl::queue&, const fmha_bwd_args_t&); \
  extern template void bwd_policy_dispatch_fp16<bwd_policy_head##HDIM, 1, 0, 0>(sycl::queue&, const fmha_bwd_args_t&); \
  extern template void bwd_policy_dispatch_fp16<bwd_policy_head##HDIM, 1, 1, 0>(sycl::queue&, const fmha_bwd_args_t&); \
  extern template void bwd_policy_dispatch_bf16<bwd_policy_head##HDIM, 0, 0, 0>(sycl::queue&, const fmha_bwd_args_t&); \
  extern template void bwd_policy_dispatch_bf16<bwd_policy_head##HDIM, 0, 1, 0>(sycl::queue&, const fmha_bwd_args_t&); \
  extern template void bwd_policy_dispatch_bf16<bwd_policy_head##HDIM, 1, 0, 0>(sycl::queue&, const fmha_bwd_args_t&); \
  extern template void bwd_policy_dispatch_bf16<bwd_policy_head##HDIM, 1, 1, 0>(sycl::queue&, const fmha_bwd_args_t&); \
  extern template void bwd_policy_dispatch_fp16<bwd_policy_head##HDIM, 0, 0, 1>(sycl::queue&, const fmha_bwd_args_t&); \
  extern template void bwd_policy_dispatch_fp16<bwd_policy_head##HDIM, 0, 1, 1>(sycl::queue&, const fmha_bwd_args_t&); \
  extern template void bwd_policy_dispatch_fp16<bwd_policy_head##HDIM, 1, 0, 1>(sycl::queue&, const fmha_bwd_args_t&); \
  extern template void bwd_policy_dispatch_fp16<bwd_policy_head##HDIM, 1, 1, 1>(sycl::queue&, const fmha_bwd_args_t&); \
  extern template void bwd_policy_dispatch_bf16<bwd_policy_head##HDIM, 0, 0, 1>(sycl::queue&, const fmha_bwd_args_t&); \
  extern template void bwd_policy_dispatch_bf16<bwd_policy_head##HDIM, 0, 1, 1>(sycl::queue&, const fmha_bwd_args_t&); \
  extern template void bwd_policy_dispatch_bf16<bwd_policy_head##HDIM, 1, 0, 1>(sycl::queue&, const fmha_bwd_args_t&); \
  extern template void bwd_policy_dispatch_bf16<bwd_policy_head##HDIM, 1, 1, 1>(sycl::queue&, const fmha_bwd_args_t&);

EXTERN_BWD_DISPATCH(32)
EXTERN_BWD_DISPATCH(64)
EXTERN_BWD_DISPATCH(96)
EXTERN_BWD_DISPATCH(128)
EXTERN_BWD_DISPATCH(160)
EXTERN_BWD_DISPATCH(192)
EXTERN_BWD_DISPATCH(256)
EXTERN_BWD_DISPATCH(512)
#undef EXTERN_BWD_DISPATCH

/**
 * Main backward pass implementation for fixed-length sequences
 * 
 * @param queue SYCL queue to submit work to
 * @param dout Gradient of output tensor (batch, seqlen_q, num_heads, head_dim)
 * @param q Query tensor
 * @param k Key tensor
 * @param v Value tensor
 * @param out Forward pass output
 * @param softmax_lse Log-sum-exp from forward pass
 * @param dq Output: gradient of query
 * @param dk Output: gradient of key
 * @param dv Output: gradient of value
 * @param softmax_d Intermediate buffer for softmax derivative
 * @param sm_scale Softmax scale (typically 1/sqrt(head_dim))
 * @param is_causal Whether to use causal masking
 */
void cutlass_fmha_bwd_fix_impl(
    sycl::queue& queue,
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
    at::Tensor& dq,
    at::Tensor& dk,
    at::Tensor& dv,
    at::Tensor& softmax_d,
    float sm_scale,
    int window_size_left,
    int window_size_right,
    bool is_causal,
    bool is_local,
    float p_dropout = 0.0f,
    uint64_t philox_seed = 0,
    uint64_t philox_offset = 0,
    bool deterministic = false);

void cutlass_fmha_bwd_varlen_impl(
    sycl::queue& queue,
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    at::Tensor& dq,
    at::Tensor& dk,
    at::Tensor& dv,
    at::Tensor& softmax_d,
    float sm_scale,
    int max_seqlen_q,
    int max_seqlen_k,
    int window_size_left,
    int window_size_right,
    bool is_causal,
    bool is_local,
    float p_dropout = 0.0f,
    uint64_t philox_seed = 0,
    uint64_t philox_offset = 0,
    bool deterministic = false);
