// Xe35 (CRI) image: vllm-xpu-kernels paged decode.
#include "decode/vllm/paged_decode.hpp"

static_assert(FLASH_XPU_SPLITKV_TARGET_ARCH == 35,
              "build with -D__SYCL_TARGET_INTEL_GPU_CRI__=1");

// vllm paged decode, head_dim == 512, dtype=fp16
FLASH_XPU_VLLM_INSTANTIATE(512, fp16, cute::half_t)
