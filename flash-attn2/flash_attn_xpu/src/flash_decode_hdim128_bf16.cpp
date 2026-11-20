#include "decode/splitkv_launch.hpp"

// Split-KV decode, head_dim_v <= 128, dtype=bf16
FLASH_XPU_SPLITKV_INSTANTIATE(128, bf16, cute::bfloat16_t)
