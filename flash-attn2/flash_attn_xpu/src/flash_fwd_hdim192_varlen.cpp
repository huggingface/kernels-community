#include "fmha_fwd_impl.hpp"

// Varlen mode: IsVarLen=1, handles both paged and non-paged cases

// Varlen + non-paged
template void policy_dispatch<
    prefill_policy_head192,
    PipelineStages_Prefill,
    1, 0>(
    sycl::queue& queue,
    CutlassType cuType,
    const fmha_fwd_args_t& args);

// Varlen + paged
template void policy_dispatch<
    prefill_policy_head192,
    PipelineStages_Prefill,
    1, 1>(
    sycl::queue& queue,
    CutlassType cuType,
    const fmha_fwd_args_t& args);

// Varlen decode + non-paged
template void policy_dispatch<
    decode_policy_head192,
    PipelineStages_Decode,
    1, 0>(
    sycl::queue& queue,
    CutlassType cuType,
    const fmha_fwd_args_t& args);

// Varlen decode + paged
template void policy_dispatch<
    decode_paged_policy_head192,
    PipelineStages_Decode,
    1, 1>(
    sycl::queue& queue,
    CutlassType cuType,
    const fmha_fwd_args_t& args);
