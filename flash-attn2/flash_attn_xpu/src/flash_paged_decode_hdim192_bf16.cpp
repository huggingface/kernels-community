// Xe2 (BMG) image: paged decode, page size 16 or 32.
#include "decode/xe2/paged_decode.hpp"

// Paged decode, head_dim == 192, dtype=bf16
FLASH_XPU_PAGED_DECODE_XE2_INSTANTIATE(192, bf16, cute::bfloat16_t)
