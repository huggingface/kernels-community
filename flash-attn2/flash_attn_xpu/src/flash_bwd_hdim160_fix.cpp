#include "fmha_bwd_impl.hpp"

// Fixed mode backward for head_dim=160
// IsCausal=0 (non-causal), IsLocal=0 (non-local)
template void bwd_policy_dispatch<
    bwd_policy_head160, 
    0, 0>(
    sycl::queue& queue, 
    BwdCutlassType cuType, 
    const fmha_bwd_args_t& args);

// IsCausal=0 (non-causal), IsLocal=1 (local)
template void bwd_policy_dispatch<
    bwd_policy_head160, 
    0, 1>(
    sycl::queue& queue, 
    BwdCutlassType cuType, 
    const fmha_bwd_args_t& args);

// IsCausal=1 (causal), IsLocal=0 (non-local)
template void bwd_policy_dispatch<
    bwd_policy_head160, 
    1, 0>(
    sycl::queue& queue, 
    BwdCutlassType cuType, 
    const fmha_bwd_args_t& args);

// IsCausal=1 (causal), IsLocal=1 (local)
template void bwd_policy_dispatch<
    bwd_policy_head160, 
    1, 1>(
    sycl::queue& queue, 
    BwdCutlassType cuType, 
    const fmha_bwd_args_t& args);
