// Xe35 (CRI) image: paged decode.
#include "decode/paged_decode.hpp"

static_assert(FLASH_XPU_SPLITKV_TARGET_ARCH == 35,
              "build with -D__SYCL_TARGET_INTEL_GPU_CRI__=1");

// Paged decode, head_dim == 512, dtype=fp16
FLASH_XPU_PAGED_DECODE_INSTANTIATE(512, fp16, cute::half_t)
