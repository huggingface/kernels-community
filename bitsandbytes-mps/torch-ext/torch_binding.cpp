#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // 4-bit quantization
  ops.def(
      "bnb_quantize_4bit(Tensor input, int blocksize, int quant_type) "
      "-> (Tensor, Tensor)");

  // 4-bit dequantization
  ops.def(
      "bnb_dequantize_4bit(Tensor packed, Tensor absmax, int blocksize, "
      "int quant_type, int numel, ScalarType output_dtype) -> Tensor");

  // Fused GEMV with 4-bit weights
  ops.def(
      "bnb_gemv_4bit(Tensor x, Tensor w, Tensor absmax, int blocksize, "
      "int quant_type, int output_features) -> Tensor");

  // Fused GEMM with 4-bit transposed weights
  ops.def(
      "bnb_gemm_4bit(Tensor x, Tensor w, Tensor absmax, int blocksize, "
      "int quant_type, int output_features) -> Tensor");
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, MPS, ops) {
  ops.impl("bnb_quantize_4bit", bnb_quantize_4bit);
  ops.impl("bnb_dequantize_4bit", bnb_dequantize_4bit);
  ops.impl("bnb_gemv_4bit", bnb_gemv_4bit);
  ops.impl("bnb_gemm_4bit", bnb_gemm_4bit);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
