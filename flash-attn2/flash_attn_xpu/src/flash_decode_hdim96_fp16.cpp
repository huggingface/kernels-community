#include "decode/splitkv_launch.hpp"

// Split-KV decode, head_dim_v <= 96, dtype=fp16
FLASH_XPU_SPLITKV_INSTANTIATE(96, fp16, cute::half_t)
