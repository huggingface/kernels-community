#include "decode/splitkv_launch.hpp"

// Split-KV decode, head_dim_v <= 192, dtype=bf16
FLASH_XPU_SPLITKV_INSTANTIATE(192, bf16, cute::bfloat16_t)
