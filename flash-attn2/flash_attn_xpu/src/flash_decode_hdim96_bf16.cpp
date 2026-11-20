#include "decode/splitkv_launch.hpp"

// Split-KV decode, head_dim_v <= 96, dtype=bf16
FLASH_XPU_SPLITKV_INSTANTIATE(96, bf16, cute::bfloat16_t)
