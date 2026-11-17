#include "prefill/chunk_prefill.hpp"

template bool flash_xpu_prefill::xe2::chunk_prefill_impl<128, true, false>(
    sycl::queue&, const flash_xpu_prefill::xe2::chunk_prefill_args_t&);
