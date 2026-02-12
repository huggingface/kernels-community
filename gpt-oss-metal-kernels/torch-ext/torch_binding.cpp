#include <ATen/Tensor.h>
#include "torch_binding.h"
#include "registration.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("f32_bf16w_matmul(Tensor input, Tensor weight_bf16, Tensor bias_bf16, "
          "Tensor! output, int num_tokens, int num_cols, int num_rows, int threadgroup_size) -> ()");
  ops.impl("f32_bf16w_matmul", torch::kMPS, &f32_bf16w_matmul_torch);

  ops.def("bf16_f32_embeddings(Tensor token_ids, Tensor weight_bf16, Tensor! output, "
          "int threadgroup_size) -> ()");
  ops.impl("bf16_f32_embeddings", torch::kMPS, &bf16_f32_embeddings_torch);

  ops.def("f32_bf16w_rmsnorm(Tensor input, Tensor weight_bf16, Tensor! output, float epsilon) -> ()");
  ops.impl("f32_bf16w_rmsnorm", torch::kMPS, &f32_bf16w_rmsnorm_torch);

  ops.def("f32_bf16w_dense_matmul_qkv(Tensor input, Tensor weight_bf16, Tensor bias_bf16, Tensor! output) -> ()");
  ops.impl("f32_bf16w_dense_matmul_qkv", torch::kMPS, &f32_bf16w_dense_matmul_qkv_torch);

  ops.def("f32_bf16w_dense_matmul_attn_output(Tensor input, Tensor weight_bf16, Tensor bias_bf16, Tensor! output) -> ()");
  ops.impl("f32_bf16w_dense_matmul_attn_output", torch::kMPS, &f32_bf16w_dense_matmul_attn_output_torch);

  ops.def("f32_bf16w_dense_matmul_mlp_gate(Tensor input, Tensor weight_bf16, Tensor bias_bf16, Tensor! output) -> ()");
  ops.impl("f32_bf16w_dense_matmul_mlp_gate", torch::kMPS, &f32_bf16w_dense_matmul_mlp_gate_torch);

  ops.def("f32_rope(Tensor! activations, float rope_base, float interpolation_scale, float yarn_offset, "
          "float yarn_scale, float yarn_multiplier, int num_tokens, int num_q_heads, int num_kv_heads, "
          "int attn_head_dim, int token_offset, int threadgroup_size) -> ()");
  ops.impl("f32_rope", torch::kMPS, &f32_rope_torch);

  ops.def("f32_bf16w_matmul_qkv(Tensor input, Tensor weight_bf16, Tensor bias_bf16, Tensor! output, Tensor kv_cache, "
          "int kv_cache_offset_bytes, int num_tokens, int num_cols, int num_q_heads, int num_kv_heads, "
          "int attn_head_dim, int token_offset, int max_tokens, float rope_base, float interpolation_scale, "
          "float yarn_offset, float yarn_scale, float yarn_multiplier, int threadgroup_size) -> ()");
  ops.impl("f32_bf16w_matmul_qkv", torch::kMPS, &f32_bf16w_matmul_qkv_torch);

  ops.def("f32_sdpa(Tensor q, int q_offset_bytes, Tensor kv, int kv_offset_bytes, Tensor s_bf16, int s_offset_bytes, "
          "Tensor! output, int output_offset_bytes, int window, int kv_stride, int num_q_tokens, int num_kv_tokens, "
          "int num_q_heads, int num_kv_heads, int head_dim) -> ()");
  ops.impl("f32_sdpa", torch::kMPS, &f32_sdpa_torch);

  ops.def("f32_topk(Tensor scores, Tensor expert_ids, Tensor expert_scores, int num_tokens, int num_experts, "
          "int num_active_experts) -> ()");
  ops.impl("f32_topk", torch::kMPS, &f32_topk_torch);

  ops.def("expert_routing_metadata(Tensor expert_ids, Tensor expert_scores, Tensor expert_offsets, "
          "Tensor intra_expert_offsets, int num_tokens, int num_experts) -> ()");
  ops.impl("expert_routing_metadata", torch::kMPS, &expert_routing_metadata_torch);

  ops.def("f32_scatter(Tensor input, Tensor expert_ids, Tensor expert_scores, Tensor expert_offsets, "
          "Tensor intra_expert_offsets, Tensor! output, int num_channels, int num_tokens, "
          "int num_active_experts) -> ()");
  ops.impl("f32_scatter", torch::kMPS, &f32_scatter_torch);

  ops.def("f32_bf16w_matmul_add(Tensor input, Tensor weight_bf16, Tensor bias_bf16, Tensor! output, "
          "int num_tokens, int num_cols, int num_rows, int threadgroup_size) -> ()");
  ops.impl("f32_bf16w_matmul_add", torch::kMPS, &f32_bf16w_matmul_add_torch);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)