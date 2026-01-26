#include "fmha_bwd_impl.hpp"

// Fixed mode backward for head_dim=64
// IsCausal=0 (non-causal)
template void bwd_policy_dispatch<
    bwd_policy_head64, 
    0>(
    sycl::queue& queue, 
    BwdCutlassType cuType, 
    const fmha_bwd_args_t& args);

// IsCausal=1 (causal)
template void bwd_policy_dispatch<
    bwd_policy_head64, 
    1>(
    sycl::queue& queue, 
    BwdCutlassType cuType, 
    const fmha_bwd_args_t& args);
