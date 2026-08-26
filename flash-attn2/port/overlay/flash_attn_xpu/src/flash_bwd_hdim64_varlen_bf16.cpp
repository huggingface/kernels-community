#include "fmha_bwd_impl.hpp"

// Varlen mode backward for head_dim=64, dtype=bf16

template void bwd_policy_dispatch_bf16<bwd_policy_head64, 0, 0, 1>(
    sycl::queue& queue, const fmha_bwd_args_t& args);

template void bwd_policy_dispatch_bf16<bwd_policy_head64, 0, 1, 1>(
    sycl::queue& queue, const fmha_bwd_args_t& args);

template void bwd_policy_dispatch_bf16<bwd_policy_head64, 1, 0, 1>(
    sycl::queue& queue, const fmha_bwd_args_t& args);

template void bwd_policy_dispatch_bf16<bwd_policy_head64, 1, 1, 1>(
    sycl::queue& queue, const fmha_bwd_args_t& args);
