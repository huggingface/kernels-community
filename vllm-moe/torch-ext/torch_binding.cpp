#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // Activation used in fused MoE layers.
  ops.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  // Apply topk softmax to the gating outputs.
  ops.def("topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
          "token_expert_indices, Tensor gating_output) -> ()");
  ops.impl("topk_softmax", torch::kCUDA, &topk_softmax);

  // Calculate the result of moe by summing up the partial results
  // from all selected experts.
  ops.def("moe_sum(Tensor! input, Tensor output) -> ()");
  ops.impl("moe_sum", torch::kCUDA, &moe_sum);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  ops.def("moe_align_block_size(Tensor topk_ids, int num_experts,"
          "                     int block_size, Tensor! sorted_token_ids,"
          "                     Tensor! experts_ids,"
          "                     Tensor! num_tokens_post_pad) -> ()");
  ops.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

  // temporarily adapted from
  // https://github.com/sgl-project/sglang/commit/ded9fcd09a43d5e7d5bb31a2bc3e9fc21bf65d2a
  ops.def("sgl_moe_align_block_size(Tensor topk_ids, int num_experts,"
          "                         int block_size, Tensor! sorted_token_ids,"
          "                         Tensor! experts_ids,"
          "                         Tensor! num_tokens_post_pad) -> ()");
  ops.impl("sgl_moe_align_block_size", torch::kCUDA, &sgl_moe_align_block_size);

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
  ops.def("dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input, "
          "Tensor! scale, Tensor? scale_ub) -> "
          "()");
  ops.impl("dynamic_per_token_scaled_fp8_quant", torch::kCUDA,
           &dynamic_per_token_scaled_fp8_quant);

#ifndef USE_ROCM
  ops.def(
      "moe_wna16_gemm(Tensor input, Tensor! output, Tensor b_qweight, "
      "Tensor b_scales, Tensor? b_qzeros, "
      "Tensor? topk_weights, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_post_pad, "
      "int top_k, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, "
      "int bit) -> Tensor");

  ops.impl("moe_wna16_gemm", torch::kCUDA, &moe_wna16_gemm);

  ops.def("marlin_gemm_moe(Tensor! a, Tensor! b_q_weights, Tensor! sorted_ids, "
          "Tensor! topk_weights, Tensor! topk_ids, Tensor! b_scales, Tensor! "
          "b_zeros, Tensor! g_idx, Tensor! perm, Tensor! workspace, "
          "int b_q_type, SymInt size_m, "
          "SymInt size_n, SymInt size_k, bool is_k_full, int num_experts, int "
          "topk, "
          "int moe_block_size, bool replicate_input, bool apply_weights)"
          " -> Tensor");
#endif
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
  ops.impl("marlin_gemm_moe", &marlin_gemm_moe);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
