// Xe35 (CRI) image: Xe3 chunk prefill.
#include "prefill/xe3/chunk_prefill.hpp"

static_assert(FLASH_XPU_SPLITKV_TARGET_ARCH == 35,
              "build with -D__SYCL_TARGET_INTEL_GPU_CRI__=1");

template bool flash_xpu_prefill::xe3::chunk_prefill_impl<128, true, false>(
    sycl::queue&, const flash_xpu_prefill::xe3::chunk_prefill_args_t&);
