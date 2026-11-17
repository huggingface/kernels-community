// Xe35 (CRI) image: paged decode.
#include "decode/paged_decode.hpp"

static_assert(FLASH_XPU_SPLITKV_TARGET_ARCH == 35,
              "build with -D__SYCL_TARGET_INTEL_GPU_CRI__=1");

// Paged decode, head_dim == 96, dtype=bf16
FLASH_XPU_PAGED_DECODE_INSTANTIATE(96, bf16, cute::bfloat16_t)
