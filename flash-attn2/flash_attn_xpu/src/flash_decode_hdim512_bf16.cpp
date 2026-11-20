#include "decode/splitkv_launch.hpp"

// Split-KV decode, head_dim_v <= 512, dtype=bf16
FLASH_XPU_SPLITKV_INSTANTIATE(512, bf16, cute::bfloat16_t)
