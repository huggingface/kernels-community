#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // The default behavior in PyTorch 2.6 is "requires_contiguous", so we need
  // to override this for many GEMMs with the following tag. Otherwise,
  // torch.compile will force all input tensors to be contiguous(), which
  // will break many custom ops that require column-major weight matrices.
  // TODO: remove this for PyTorch 2.8, when the default is planned to switch
  // to match exact eager-mode strides.
  at::Tag stride_tag = at::Tag::needs_fixed_stride_order;

  #ifndef USE_ROCM

  // CUTLASS w8a8 GEMM, supporting symmetric per-tensor or per-row/column                                                                                                                                            
  // quantization, as well as bias                                                                                                                                                                                   
  ops.def(                                                                                                                                                                                                           
      "cutlass_scaled_mm(Tensor! out, Tensor a,"                                                                                                                                                                     
      "                  Tensor b, Tensor a_scales,"                                                                                                                                                                 
      "                  Tensor b_scales, Tensor? bias) -> ()", {stride_tag});                                                                                                                                                     
  ops.impl("cutlass_scaled_mm", torch::kCUDA, &cutlass_scaled_mm);                                                                                                                                                   
                                                                                                                                                                                                                     
  // CUTLASS w8a8 GEMM, supporting asymmetric per-tensor or per-row/column                                                                                                                                           
  // quantization.                                                                                                                                                                                                   
  ops.def(                                                                                                                                                                                                           
      "cutlass_scaled_mm_azp(Tensor! out, Tensor a,"                                                                                                                                                                 
      "                  Tensor b, Tensor a_scales,"                                                                                                                                                                 
      "                  Tensor b_scales, Tensor azp_adj,"                                                                                                                                                           
      "                  Tensor? azp, Tensor? bias) -> ()", {stride_tag});                                                                                                                                                         
  ops.impl("cutlass_scaled_mm_azp", torch::kCUDA, &cutlass_scaled_mm_azp);                                                                                                                                           
                                                                                                                                                                                                                     
  // Check if cutlass scaled_mm is supported for CUDA devices of the given                                                                                                                                           
  // capability                                                                                                                                                                                                      
  ops.def("cutlass_scaled_mm_supports_fp8(int cuda_device_capability) -> bool");                                                                                                                                     
  ops.impl("cutlass_scaled_mm_supports_fp8", &cutlass_scaled_mm_supports_fp8);                                            

  // Check if cutlass scaled_mm supports block quantization (used by DeepSeekV3)
  ops.def(
      "cutlass_scaled_mm_supports_block_fp8(int cuda_device_capability) -> "
      "bool");
  ops.impl("cutlass_scaled_mm_supports_block_fp8", &cutlass_scaled_mm_supports_block_fp8);

  #endif

  // Compute FP8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale) -> "
      "()");
  ops.impl("static_scaled_fp8_quant", torch::kCUDA, &static_scaled_fp8_quant);

  // Compute dynamic-per-tensor FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_scaled_fp8_quant(Tensor! result, Tensor input, Tensor! scale) "
      "-> "
      "()");
  ops.impl("dynamic_scaled_fp8_quant", torch::kCUDA, &dynamic_scaled_fp8_quant);

  // Compute dynamic-per-token FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! scale, Tensor? scale_ub) -> "
      "()");
  ops.impl("dynamic_per_token_scaled_fp8_quant", torch::kCUDA,
           &dynamic_per_token_scaled_fp8_quant);

  // Compute int8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_int8_quant(Tensor! result, Tensor input, Tensor scale,"
      "Tensor? azp) -> ()");
  ops.impl("static_scaled_int8_quant", torch::kCUDA, &static_scaled_int8_quant);

  // Compute int8 quantized tensor and scaling factor
  ops.def(
      "dynamic_scaled_int8_quant(Tensor! result, Tensor input, Tensor! scale, "
      "Tensor!? azp) -> ()");
  ops.impl("dynamic_scaled_int8_quant", torch::kCUDA,
           &dynamic_scaled_int8_quant);

  #ifndef USE_ROCM

  // awq_marlin repack from AWQ.
  ops.def(
      "awq_marlin_repack(Tensor b_q_weight, SymInt size_k, "
      "SymInt size_n, int num_bits) -> Tensor");

  // gptq_marlin Optimized Quantized GEMM for GPTQ.
  ops.def(
      "gptq_marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, "
      "Tensor b_scales, Tensor? global_scale, Tensor? b_zeros_or_none, Tensor? "
      "g_idx_or_none, Tensor? perm_or_none, Tensor workspace, int b_q_type, "
      "SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full, "
      "bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) -> Tensor",
      {stride_tag});

  // gptq_marlin repack from GPTQ.
  ops.def(
      "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, "
      "SymInt size_k, SymInt size_n, int num_bits) -> Tensor");

  // Marlin (Dense) Optimized Quantized GEMM for GPTQ.
  ops.def(
      "marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, "
      "Tensor! workspace, SymInt size_m, SymInt size_n, SymInt size_k) -> "
      "Tensor", {stride_tag});

  // Marlin_24 (Sparse) Optimized Quantized GEMM for GPTQ.
  ops.def(
      "gptq_marlin_24_gemm(Tensor a, Tensor b_q_weight, Tensor b_meta, "
      "Tensor b_scales, Tensor workspace, "
      "int b_q_type, "
      "SymInt size_m, SymInt size_n, SymInt size_k) -> Tensor", {stride_tag});

  // marlin_qqq_gemm for QQQ.
  ops.def(
      "marlin_qqq_gemm(Tensor a, Tensor b_q_weight, "
      "Tensor s_tok, Tensor s_ch, Tensor s_group, "
      "Tensor! workspace, SymInt size_m, SymInt size_n, "
      "SymInt size_k) -> Tensor", {stride_tag});
  #endif
}

#ifndef USE_ROCM

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
  ops.impl("awq_marlin_repack", &awq_marlin_repack);
  ops.impl("gptq_marlin_24_gemm", &gptq_marlin_24_gemm);
  ops.impl("gptq_marlin_gemm", &gptq_marlin_gemm);
  ops.impl("gptq_marlin_repack", &gptq_marlin_repack);
  ops.impl("marlin_gemm", &marlin_gemm);
  ops.impl("marlin_qqq_gemm", &marlin_qqq_gemm);
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Meta, ops) {
  ops.impl("awq_marlin_repack", &awq_marlin_repack_meta);
  ops.impl("gptq_marlin_repack", &gptq_marlin_repack_meta);
}

#endif

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
