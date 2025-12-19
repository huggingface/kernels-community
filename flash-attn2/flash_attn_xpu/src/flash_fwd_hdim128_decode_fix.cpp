#include "fmha_fwd_impl.hpp"

template void policy_dispatch<
    decode_policy_head128, 
    PipelineStages_Decode, 
    0, 0>(
    sycl::queue& queue, 
    CutlassType cuType, 
    const fmha_fwd_args_t& args);
