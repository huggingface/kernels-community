#include "fmha_fwd_impl.hpp"

// Varlen mode: IsVarLen=1, handles both paged and non-paged cases
// No need for dynamic dispatch since we know it's varlen

// Varlen + non-paged
template void policy_dispatch<
    prefill_policy_head32, 
    PipelineStages_Prefill, 
    1, 0>(
    sycl::queue& queue, 
    CutlassType cuType, 
    const fmha_fwd_args_t& args);

// Varlen + paged
template void policy_dispatch<
    prefill_policy_head32, 
    PipelineStages_Prefill, 
    1, 1>(
    sycl::queue& queue, 
    CutlassType cuType, 
    const fmha_fwd_args_t& args);
