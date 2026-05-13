#pragma once

#include "fmha_fwd_types.hpp"

namespace sycl {
  inline namespace _V1 {
    class queue;
  }
}

namespace at {
  class Tensor;
}

namespace std {
  template<typename T> class optional;
}

struct prefill_policy_head32;
struct prefill_policy_head64;
struct prefill_policy_head96;
struct prefill_policy_head128;
struct prefill_policy_head160;
struct prefill_policy_head192;
struct prefill_policy_head256;
struct prefill_policy_head512;

struct decode_policy_head32;
struct decode_policy_head64;
struct decode_policy_head96;
struct decode_policy_head128;
struct decode_policy_head160;
struct decode_policy_head192;
struct decode_policy_head256;
struct decode_policy_head512;

struct decode_paged_policy_head32;
struct decode_paged_policy_head64;
struct decode_paged_policy_head96;
struct decode_paged_policy_head128;
struct decode_paged_policy_head160;
struct decode_paged_policy_head192;
struct decode_paged_policy_head256;
struct decode_paged_policy_head512;

// Dtype-specific dispatch functions (instantiated in per-head TUs)
template <typename chunk_policy, int PipelineStages, int IsVarLen, int IsPaged>
void policy_dispatch_fp16(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);

template <typename chunk_policy, int PipelineStages, int IsVarLen, int IsPaged>
void policy_dispatch_bf16(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);

// Combined dispatch (delegates to fp16/bf16 based on cuType)
// Defined inline in header so callers (fmha_fwd.cpp) can see the template body.
template <typename chunk_policy, int PipelineStages, int IsVarLen, int IsPaged>
inline void policy_dispatch(
    sycl::queue& queue,
    CutlassType cuType,
    const fmha_fwd_args_t& args) {
  if (cuType == CutlassType::half) {
    policy_dispatch_fp16<chunk_policy, PipelineStages, IsVarLen, IsPaged>(queue, args);
  } else {
    policy_dispatch_bf16<chunk_policy, PipelineStages, IsVarLen, IsPaged>(queue, args);
  }
}

// Varlen prefill mode extern declarations (IsVarLen=1, IsPaged=0/1)
#define EXTERN_DISPATCH_VARLEN_PREFILL(HDIM) \
  extern template void policy_dispatch_fp16<prefill_policy_head##HDIM, PipelineStages_Prefill, 1, 0>(sycl::queue&, const fmha_fwd_args_t&); \
  extern template void policy_dispatch_fp16<prefill_policy_head##HDIM, PipelineStages_Prefill, 1, 1>(sycl::queue&, const fmha_fwd_args_t&); \
  extern template void policy_dispatch_bf16<prefill_policy_head##HDIM, PipelineStages_Prefill, 1, 0>(sycl::queue&, const fmha_fwd_args_t&); \
  extern template void policy_dispatch_bf16<prefill_policy_head##HDIM, PipelineStages_Prefill, 1, 1>(sycl::queue&, const fmha_fwd_args_t&);

EXTERN_DISPATCH_VARLEN_PREFILL(32)
EXTERN_DISPATCH_VARLEN_PREFILL(64)
EXTERN_DISPATCH_VARLEN_PREFILL(96)
EXTERN_DISPATCH_VARLEN_PREFILL(128)
EXTERN_DISPATCH_VARLEN_PREFILL(160)
EXTERN_DISPATCH_VARLEN_PREFILL(192)
EXTERN_DISPATCH_VARLEN_PREFILL(256)
EXTERN_DISPATCH_VARLEN_PREFILL(512)
#undef EXTERN_DISPATCH_VARLEN_PREFILL

// Varlen decode mode extern declarations (IsVarLen=1, IsPaged=0/1)
#define EXTERN_DISPATCH_VARLEN_DECODE(HDIM) \
  extern template void policy_dispatch_fp16<decode_policy_head##HDIM, PipelineStages_Decode, 1, 0>(sycl::queue&, const fmha_fwd_args_t&); \
  extern template void policy_dispatch_fp16<decode_paged_policy_head##HDIM, PipelineStages_Decode, 1, 1>(sycl::queue&, const fmha_fwd_args_t&); \
  extern template void policy_dispatch_bf16<decode_policy_head##HDIM, PipelineStages_Decode, 1, 0>(sycl::queue&, const fmha_fwd_args_t&); \
  extern template void policy_dispatch_bf16<decode_paged_policy_head##HDIM, PipelineStages_Decode, 1, 1>(sycl::queue&, const fmha_fwd_args_t&);

EXTERN_DISPATCH_VARLEN_DECODE(32)
EXTERN_DISPATCH_VARLEN_DECODE(64)
EXTERN_DISPATCH_VARLEN_DECODE(96)
EXTERN_DISPATCH_VARLEN_DECODE(128)
EXTERN_DISPATCH_VARLEN_DECODE(160)
EXTERN_DISPATCH_VARLEN_DECODE(192)
EXTERN_DISPATCH_VARLEN_DECODE(256)
EXTERN_DISPATCH_VARLEN_DECODE(512)
#undef EXTERN_DISPATCH_VARLEN_DECODE

// Fixed mode extern declarations (IsVarLen=0, IsPaged=0)
#define EXTERN_DISPATCH_FIX(HDIM) \
  extern template void policy_dispatch_fp16<decode_policy_head##HDIM, PipelineStages_Decode, 0, 0>(sycl::queue&, const fmha_fwd_args_t&); \
  extern template void policy_dispatch_fp16<prefill_policy_head##HDIM, PipelineStages_Prefill, 0, 0>(sycl::queue&, const fmha_fwd_args_t&); \
  extern template void policy_dispatch_bf16<decode_policy_head##HDIM, PipelineStages_Decode, 0, 0>(sycl::queue&, const fmha_fwd_args_t&); \
  extern template void policy_dispatch_bf16<prefill_policy_head##HDIM, PipelineStages_Prefill, 0, 0>(sycl::queue&, const fmha_fwd_args_t&);

EXTERN_DISPATCH_FIX(32)
EXTERN_DISPATCH_FIX(64)
EXTERN_DISPATCH_FIX(96)
EXTERN_DISPATCH_FIX(128)
EXTERN_DISPATCH_FIX(160)
EXTERN_DISPATCH_FIX(192)
EXTERN_DISPATCH_FIX(256)
EXTERN_DISPATCH_FIX(512)
#undef EXTERN_DISPATCH_FIX

void cutlass_fmha_fwd_varlen_impl(
    sycl::queue& queue,
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    at::Tensor& out,
    at::Tensor& softmax_lse,
    const std::optional<at::Tensor>& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    double sm_scale,
    int window_size_left,
    int window_size_right,
    bool is_varlen,
    bool is_paged,
    bool is_causal,
    bool is_local,
    float p_dropout = 0.0f,
    uint64_t philox_seed = 0,
    uint64_t philox_offset = 0,
    void* rng_state = nullptr,
    void* s_dmask = nullptr,
    int seqlen_q_rounded = 0,
    int seqlen_k_rounded = 0);

void cutlass_fmha_fwd_fix_impl(
    sycl::queue& queue,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& out,
    at::Tensor& softmax_lse,
    float sm_scale,
    int window_size_left,
    int window_size_right,
    bool is_causal,
    bool is_local,
    float p_dropout = 0.0f,
    uint64_t philox_seed = 0,
    uint64_t philox_offset = 0,
    void* rng_state = nullptr,
    void* s_dmask = nullptr,
    int seqlen_q_rounded = 0,
    int seqlen_k_rounded = 0);
