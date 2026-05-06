#include "fmha_fwd_impl.hpp"

// Non-varlen + paged: IsVarLen=0, IsPaged=1, dtype=bf16
// Used by mha_fwd_kvcache when block_table is provided.

// Prefill paged
template void policy_dispatch_bf16<
    prefill_policy_head64,
    PipelineStages_Prefill,
    0, 1>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);

// Decode paged (smaller K-tile to fit page boundaries)
template void policy_dispatch_bf16<
    decode_paged_policy_head64,
    PipelineStages_Decode,
    0, 1>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);
