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

// Template declarations for different head dimensions
// IsCausal: 0 = non-causal, 1 = causal
// IsLocal: 0 = non-local, 1 = local (sliding window)
template <typename bwd_policy, int IsCausal = -1, int IsLocal = -1>
void bwd_policy_dispatch(
    sycl::queue& queue,
    BwdCutlassType cuType,
    const fmha_bwd_args_t& args);

// Extern template declarations for head32 (causal x local combinations)
extern template void bwd_policy_dispatch<bwd_policy_head32, 0, 0>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head32, 0, 1>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head32, 1, 0>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head32, 1, 1>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);

// Extern template declarations for head64 (causal x local combinations)
extern template void bwd_policy_dispatch<bwd_policy_head64, 0, 0>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head64, 0, 1>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head64, 1, 0>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head64, 1, 1>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);

// Extern template declarations for head96
extern template void bwd_policy_dispatch<bwd_policy_head96, 0, 0>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head96, 0, 1>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head96, 1, 0>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head96, 1, 1>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);

// Extern template declarations for head128
extern template void bwd_policy_dispatch<bwd_policy_head128, 0, 0>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head128, 0, 1>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head128, 1, 0>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head128, 1, 1>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);

// Extern template declarations for head160
extern template void bwd_policy_dispatch<bwd_policy_head160, 0, 0>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head160, 0, 1>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head160, 1, 0>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head160, 1, 1>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);

// Extern template declarations for head192
extern template void bwd_policy_dispatch<bwd_policy_head192, 0, 0>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head192, 0, 1>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head192, 1, 0>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head192, 1, 1>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);

// Extern template declarations for head256
extern template void bwd_policy_dispatch<bwd_policy_head256, 0, 0>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head256, 0, 1>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head256, 1, 0>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);
extern template void bwd_policy_dispatch<bwd_policy_head256, 1, 1>(
    sycl::queue&, BwdCutlassType, const fmha_bwd_args_t&);

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
    uint64_t philox_offset = 0);
