#include "decode/splitkv_launch.hpp"

// Split-KV decode, head_dim_v <= 128, dtype=fp16
FLASH_XPU_SPLITKV_INSTANTIATE(128, fp16, cute::half_t)
