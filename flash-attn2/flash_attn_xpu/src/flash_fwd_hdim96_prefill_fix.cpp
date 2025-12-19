#include "fmha_fwd_impl.hpp"

template void policy_dispatch<
    prefill_policy_head96, 
    PipelineStages_Prefill, 
    0, 0>(
    sycl::queue& queue, 
    CutlassType cuType, 
    const fmha_fwd_args_t& args);
