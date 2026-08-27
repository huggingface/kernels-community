#include "fmha_fwd_impl.hpp"

// Varlen mode: IsVarLen=1, dtype=bf16

// Varlen prefill + non-paged
template void policy_dispatch_bf16<
    prefill_policy_head32,
    PipelineStages_Prefill,
    1, 0>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);

// Varlen prefill + paged
template void policy_dispatch_bf16<
    prefill_policy_head32,
    PipelineStages_Prefill,
    1, 1>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);

// Varlen decode + non-paged
template void policy_dispatch_bf16<
    decode_policy_head32,
    PipelineStages_Decode,
    1, 0>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);

// Varlen decode + paged
template void policy_dispatch_bf16<
    decode_paged_policy_head32,
    PipelineStages_Decode,
    1, 1>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);
