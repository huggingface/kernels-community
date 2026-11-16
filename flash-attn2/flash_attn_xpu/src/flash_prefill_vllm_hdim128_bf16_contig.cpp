#include "prefill_vllm/chunk_prefill.hpp"

template bool kc_vllm_prefill::xe2::chunk_prefill_impl<128, true, false>(
    sycl::queue&, const kc_vllm_prefill::xe2::chunk_prefill_args_t&);
