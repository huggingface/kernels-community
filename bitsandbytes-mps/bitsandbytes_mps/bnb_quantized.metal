// bitsandbytes MPS Metal kernels - template instantiations
// Instantiates kernel variants for all (type, blocksize, quant_type) combos.

// clang-format off
#include "utils.h"
#include "gemm/gemm.h"
#include "quantized_utils.h"
#include "bnb_quantized.h"

// ============================================================================
// Instantiation macros
// ============================================================================

#define instantiate_bnb_kernel(name, type, blocksize, quant_type) \
  template [[host_name(                                           \
      #name "_" #type "_bs_" #blocksize "_qt_" #quant_type        \
  )]] [[kernel]] decltype(name<type, blocksize, quant_type>)      \
      name<type, blocksize, quant_type>;

// ---- Instantiate all kernel types for a given (type, blocksize, quant_type) ----

#define instantiate_bnb_all_kernels(type, blocksize, quant_type)     \
  instantiate_bnb_kernel(bnb_quantize_blockwise, type, blocksize, quant_type)   \
  instantiate_bnb_kernel(bnb_dequantize_blockwise, type, blocksize, quant_type) \
  instantiate_bnb_kernel(bnb_qmv, type, blocksize, quant_type)                 \
  instantiate_bnb_kernel(bnb_qmm_t, type, blocksize, quant_type)

// ---- Instantiate for all quant types (FP4=1, NF4=2) ----

#define instantiate_bnb_quant_types(type, blocksize)  \
  instantiate_bnb_all_kernels(type, blocksize, 1)     \
  instantiate_bnb_all_kernels(type, blocksize, 2)

// ---- Instantiate for all blocksizes ----

#define instantiate_bnb_blocksizes(type)     \
  instantiate_bnb_quant_types(type, 64)      \
  instantiate_bnb_quant_types(type, 128)

// ---- Instantiate for all scalar types ----

instantiate_bnb_blocksizes(half)
instantiate_bnb_blocksizes(bfloat16_t)
instantiate_bnb_blocksizes(float)

// clang-format on
