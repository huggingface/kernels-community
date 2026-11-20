// Xe35 (CRI) image: built with -D__SYCL_TARGET_INTEL_GPU_CRI__=1 so sycl-tla
// emits the Xe35 block loads.
#include "decode/splitkv_launch.hpp"

static_assert(FLASH_XPU_SPLITKV_TARGET_ARCH == 35,
              "build with -D__SYCL_TARGET_INTEL_GPU_CRI__=1");

// Split-KV decode, head_dim_v <= 192, dtype=bf16
FLASH_XPU_SPLITKV_INSTANTIATE(192, bf16, cute::bfloat16_t)
