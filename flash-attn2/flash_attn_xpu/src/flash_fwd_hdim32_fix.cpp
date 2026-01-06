#include "fmha_fwd_impl.hpp"

// Fixed mode: non-varlen (IsVarLen=0), non-paged (IsPaged=0)
// Includes both decode and prefill policies

// Decode fixed mode
template void policy_dispatch<
    decode_policy_head32, 
    PipelineStages_Decode, 
    0, 0>(
    sycl::queue& queue, 
    CutlassType cuType, 
    const fmha_fwd_args_t& args);

// Prefill fixed mode
template void policy_dispatch<
    prefill_policy_head32, 
    PipelineStages_Prefill, 
    0, 0>(
    sycl::queue& queue, 
    CutlassType cuType, 
    const fmha_fwd_args_t& args);
