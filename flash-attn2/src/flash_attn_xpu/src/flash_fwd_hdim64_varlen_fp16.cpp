#include "fmha_fwd_impl.hpp"

// Varlen mode: IsVarLen=1, dtype=fp16

// Varlen prefill + non-paged
template void policy_dispatch_fp16<
    prefill_policy_head64,
    PipelineStages_Prefill,
    1, 0>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);

// Varlen prefill + paged
template void policy_dispatch_fp16<
    prefill_policy_head64,
    PipelineStages_Prefill,
    1, 1>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);

// Varlen decode + non-paged
template void policy_dispatch_fp16<
    decode_policy_head64,
    PipelineStages_Decode,
    1, 0>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);

// Varlen decode + paged
template void policy_dispatch_fp16<
    decode_paged_policy_head64,
    PipelineStages_Decode,
    1, 1>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);
