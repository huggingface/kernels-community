// Xe35 (CRI) image: vllm-xpu-kernels Xe3 chunk prefill.
#include "prefill_vllm/xe3/chunk_prefill.hpp"

static_assert(FLASH_XPU_SPLITKV_TARGET_ARCH == 35,
              "build with -D__SYCL_TARGET_INTEL_GPU_CRI__=1");

template bool kc_vllm_prefill::xe3::chunk_prefill_impl<64, true, true>(
    sycl::queue&, const kc_vllm_prefill::xe3::chunk_prefill_args_t&);
