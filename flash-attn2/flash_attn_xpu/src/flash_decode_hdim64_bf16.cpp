#include "decode/splitkv_launch.hpp"

// Split-KV decode, head_dim_v <= 64, dtype=bf16
FLASH_XPU_SPLITKV_INSTANTIATE(64, bf16, cute::bfloat16_t)
