#include "fmha_bwd_impl.hpp"

// Fixed mode backward for head_dim=256, dtype=fp16

template void bwd_policy_dispatch_fp16<bwd_policy_head256, 0, 0>(
    sycl::queue& queue, const fmha_bwd_args_t& args);

template void bwd_policy_dispatch_fp16<bwd_policy_head256, 0, 1>(
    sycl::queue& queue, const fmha_bwd_args_t& args);

template void bwd_policy_dispatch_fp16<bwd_policy_head256, 1, 0>(
    sycl::queue& queue, const fmha_bwd_args_t& args);

template void bwd_policy_dispatch_fp16<bwd_policy_head256, 1, 1>(
    sycl::queue& queue, const fmha_bwd_args_t& args);
