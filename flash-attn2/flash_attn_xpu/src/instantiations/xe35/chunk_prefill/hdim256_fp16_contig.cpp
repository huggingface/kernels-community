#include "../../../prefill/chunk_prefill.hpp"

template bool flash_xpu_prefill::chunk_prefill_impl<35, 256, false, false>(
    sycl::queue&, const flash_xpu_prefill::chunk_prefill_args_t&);
