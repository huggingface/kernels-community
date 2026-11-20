#include "decode/splitkv_launch.hpp"

// Split-KV decode, head_dim_v <= 192, dtype=fp16
FLASH_XPU_SPLITKV_INSTANTIATE(192, fp16, cute::half_t)
