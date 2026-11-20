#include "decode/splitkv_launch.hpp"

// Split-KV decode, head_dim_v <= 256, dtype=bf16
FLASH_XPU_SPLITKV_INSTANTIATE(256, bf16, cute::bfloat16_t)
