#include "fmha_bwd_impl.hpp"

// Varlen mode backward for head_dim=192, dtype=fp16

template void bwd_policy_dispatch_fp16<bwd_policy_head192, 0, 0, 1>(
    sycl::queue& queue, const fmha_bwd_args_t& args);

template void bwd_policy_dispatch_fp16<bwd_policy_head192, 0, 1, 1>(
    sycl::queue& queue, const fmha_bwd_args_t& args);

template void bwd_policy_dispatch_fp16<bwd_policy_head192, 1, 0, 1>(
    sycl::queue& queue, const fmha_bwd_args_t& args);

template void bwd_policy_dispatch_fp16<bwd_policy_head192, 1, 1, 1>(
    sycl::queue& queue, const fmha_bwd_args_t& args);
