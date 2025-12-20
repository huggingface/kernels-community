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

struct decode_policy_head32;
struct decode_policy_head64;
struct decode_policy_head96;
struct decode_policy_head128;
struct decode_policy_head160;
struct decode_policy_head192;
struct decode_policy_head256;

template <typename chunk_policy, int PipelineStages, int IsVarLen = -1, int IsPaged = -1>
void policy_dispatch(
    sycl::queue& queue, 
    CutlassType cuType, 
    const fmha_fwd_args_t& args);

// Varlen mode extern declarations (IsVarLen=1, IsPaged=0/1)
extern template void policy_dispatch<prefill_policy_head32, PipelineStages_Prefill, 1, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head32, PipelineStages_Prefill, 1, 1>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head64, PipelineStages_Prefill, 1, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head64, PipelineStages_Prefill, 1, 1>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head96, PipelineStages_Prefill, 1, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head96, PipelineStages_Prefill, 1, 1>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head128, PipelineStages_Prefill, 1, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head128, PipelineStages_Prefill, 1, 1>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head160, PipelineStages_Prefill, 1, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head160, PipelineStages_Prefill, 1, 1>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head192, PipelineStages_Prefill, 1, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head192, PipelineStages_Prefill, 1, 1>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head256, PipelineStages_Prefill, 1, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head256, PipelineStages_Prefill, 1, 1>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);

// Fixed mode extern declarations (IsVarLen=0, IsPaged=0)

extern template void policy_dispatch<decode_policy_head32, PipelineStages_Decode, 0, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<decode_policy_head64, PipelineStages_Decode, 0, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<decode_policy_head96, PipelineStages_Decode, 0, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<decode_policy_head128, PipelineStages_Decode, 0, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<decode_policy_head160, PipelineStages_Decode, 0, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<decode_policy_head192, PipelineStages_Decode, 0, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<decode_policy_head256, PipelineStages_Decode, 0, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);

extern template void policy_dispatch<prefill_policy_head32, PipelineStages_Prefill, 0, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head64, PipelineStages_Prefill, 0, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head96, PipelineStages_Prefill, 0, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head128, PipelineStages_Prefill, 0, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head160, PipelineStages_Prefill, 0, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head192, PipelineStages_Prefill, 0, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
extern template void policy_dispatch<prefill_policy_head256, PipelineStages_Prefill, 0, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);

void cutlass_fmha_fwd_varlen_impl(
    sycl::queue& queue,
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    at::Tensor& out,
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
    bool is_local);

void cutlass_fmha_fwd_fix_impl(
    sycl::queue& queue,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& out,
    float sm_scale,
    int window_size_left,
    int window_size_right,
    bool is_causal,
    bool is_local);
