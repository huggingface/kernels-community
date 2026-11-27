#include "../../../decode/splitkv_launch.hpp"

FLASH_XPU_SPLITKV_INSTANTIATE(20, 512, bf16, cute::bfloat16_t)
