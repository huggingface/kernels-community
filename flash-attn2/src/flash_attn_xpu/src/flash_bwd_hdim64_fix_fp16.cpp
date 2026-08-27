#include "fmha_bwd_impl.hpp"

// Fixed mode backward for head_dim=64, dtype=fp16

template void bwd_policy_dispatch_fp16<bwd_policy_head64, 0, 0, 0>(
    sycl::queue& queue, const fmha_bwd_args_t& args);

template void bwd_policy_dispatch_fp16<bwd_policy_head64, 0, 1, 0>(
    sycl::queue& queue, const fmha_bwd_args_t& args);

template void bwd_policy_dispatch_fp16<bwd_policy_head64, 1, 0, 0>(
    sycl::queue& queue, const fmha_bwd_args_t& args);

template void bwd_policy_dispatch_fp16<bwd_policy_head64, 1, 1, 0>(
    sycl::queue& queue, const fmha_bwd_args_t& args);
