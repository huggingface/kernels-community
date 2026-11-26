#include "../../../prefill/chunk_prefill.hpp"

template bool flash_xpu_prefill::chunk_prefill_impl<35, 64, false, true>(
    sycl::queue&, const flash_xpu_prefill::chunk_prefill_args_t&);
