#include "fmha_fwd_impl.hpp"

template void policy_dispatch_dynamic<
    prefill_policy_head128, 
    PipelineStages_Prefill>(
    sycl::queue& queue, 
    CutlassType cuType, 
    const fmha_fwd_args_t& args);

template void policy_dispatch<prefill_policy_head128, PipelineStages_Prefill, 1, 1>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
template void policy_dispatch<prefill_policy_head128, PipelineStages_Prefill, 1, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
template void policy_dispatch<prefill_policy_head128, PipelineStages_Prefill, 0, 1>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
template void policy_dispatch<prefill_policy_head128, PipelineStages_Prefill, 0, 0>(
    sycl::queue&, CutlassType, const fmha_fwd_args_t&);
