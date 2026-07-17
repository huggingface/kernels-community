#include "fmha_fwd_impl.hpp"

// MXFP (block-scaled FP8/FP4) fixed mode: IsVarLen=0, IsPaged=0
// Instantiates prefill + decode for all three element types (e5m2, e4m3, e2m1).

template void policy_dispatch_mxfp_e5m2<prefill_policy_head64, PipelineStages_Prefill, 0, 0>(sycl::queue&, const fmha_fwd_args_t&);
template void policy_dispatch_mxfp_e5m2<decode_policy_head64,  PipelineStages_Decode,  0, 0>(sycl::queue&, const fmha_fwd_args_t&);
template void policy_dispatch_mxfp_e4m3<prefill_policy_head64, PipelineStages_Prefill, 0, 0>(sycl::queue&, const fmha_fwd_args_t&);
template void policy_dispatch_mxfp_e4m3<decode_policy_head64,  PipelineStages_Decode,  0, 0>(sycl::queue&, const fmha_fwd_args_t&);
template void policy_dispatch_mxfp_e2m1<prefill_policy_head64, PipelineStages_Prefill, 0, 0>(sycl::queue&, const fmha_fwd_args_t&);
template void policy_dispatch_mxfp_e2m1<decode_policy_head64,  PipelineStages_Decode,  0, 0>(sycl::queue&, const fmha_fwd_args_t&);
