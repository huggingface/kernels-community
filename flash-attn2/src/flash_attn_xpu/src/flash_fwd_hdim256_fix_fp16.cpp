#include "fmha_fwd_impl.hpp"

// Fixed mode: IsVarLen=0, IsPaged=0, dtype=fp16

// Decode fixed mode
template void policy_dispatch_fp16<
    decode_policy_head256,
    PipelineStages_Decode,
    0, 0>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);

// Prefill fixed mode
template void policy_dispatch_fp16<
    prefill_policy_head256,
    PipelineStages_Prefill,
    0, 0>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);
