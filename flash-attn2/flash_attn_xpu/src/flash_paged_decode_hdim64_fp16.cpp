// Xe2 (BMG) image: paged decode, page size 16 or 32.
#include "decode/xe2/paged_decode.hpp"

// Paged decode, head_dim == 64, dtype=fp16
FLASH_XPU_PAGED_DECODE_XE2_INSTANTIATE(64, fp16, cute::half_t)
