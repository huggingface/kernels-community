#include "../../../decode/splitkv_launch.hpp"

FLASH_XPU_SPLITKV_INSTANTIATE(20, 512, fp16, cute::half_t)
