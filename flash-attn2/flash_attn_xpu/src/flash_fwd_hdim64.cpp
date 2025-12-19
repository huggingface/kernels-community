#include "fmha_fwd_impl.hpp"

template void policy_dispatch_dynamic<
    prefill_policy_head64, 
    PipelineStages_Prefill>(
    sycl::queue& queue, 
    CutlassType cuType, 
    const fmha_fwd_args_t& args);
