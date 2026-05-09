#include "fmha_bwd_impl.hpp"

// Fixed mode backward for head_dim=32, dtype=fp16

template void bwd_policy_dispatch_fp16<bwd_policy_head32, 0, 0>(
    sycl::queue& queue, const fmha_bwd_args_t& args);

template void bwd_policy_dispatch_fp16<bwd_policy_head32, 0, 1>(
    sycl::queue& queue, const fmha_bwd_args_t& args);

template void bwd_policy_dispatch_fp16<bwd_policy_head32, 1, 0>(
    sycl::queue& queue, const fmha_bwd_args_t& args);

template void bwd_policy_dispatch_fp16<bwd_policy_head32, 1, 1>(
    sycl::queue& queue, const fmha_bwd_args_t& args);
