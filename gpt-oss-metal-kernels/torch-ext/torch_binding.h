#pragma once

#include <torch/torch.h>

void f32_bf16w_matmul_torch(const at::Tensor& input,
                            const at::Tensor& weight_bf16,
                            const at::Tensor& bias_bf16,
                            at::Tensor& output,
                            int64_t num_tokens,
                            int64_t num_cols,
                            int64_t num_rows,
                            int64_t threadgroup_size);

void bf16_f32_embeddings_torch(const at::Tensor& token_ids,
                               const at::Tensor& weight_bf16,
                               at::Tensor& output,
                               int64_t threadgroup_size);

void f32_bf16w_rmsnorm_torch(const at::Tensor& input,
                             const at::Tensor& weight_bf16,
                             at::Tensor& output,
                             double epsilon);

void f32_bf16w_dense_matmul_qkv_torch(const at::Tensor& input,
                                      const at::Tensor& weight_bf16,
                                      const at::Tensor& bias_bf16,
                                      at::Tensor& output);

void f32_bf16w_dense_matmul_attn_output_torch(const at::Tensor& input,
                                              const at::Tensor& weight_bf16,
                                              const at::Tensor& bias_bf16,
                                              at::Tensor& output);

void f32_bf16w_dense_matmul_mlp_gate_torch(const at::Tensor& input,
                                           const at::Tensor& weight_bf16,
                                           const at::Tensor& bias_bf16,
                                           at::Tensor& output);

void f32_rope_torch(at::Tensor& activations,
                    double rope_base,
                    double interpolation_scale,
                    double yarn_offset,
                    double yarn_scale,
                    double yarn_multiplier,
                    int64_t num_tokens,
                    int64_t num_q_heads,
                    int64_t num_kv_heads,
                    int64_t attn_head_dim,
                    int64_t token_offset,
                    int64_t threadgroup_size);

void f32_bf16w_matmul_qkv_torch(const at::Tensor& input,
                                const at::Tensor& weight_bf16,
                                const at::Tensor& bias_bf16,
                                at::Tensor& output,
                                at::Tensor& kv_cache,
                                int64_t kv_cache_offset_bytes,
                                int64_t num_tokens,
                                int64_t num_cols,
                                int64_t num_q_heads,
                                int64_t num_kv_heads,
                                int64_t attn_head_dim,
                                int64_t token_offset,
                                int64_t max_tokens,
                                double rope_base,
                                double interpolation_scale,
                                double yarn_offset,
                                double yarn_scale,
                                double yarn_multiplier,
                                int64_t threadgroup_size);

void f32_sdpa_torch(const at::Tensor& q,
                    int64_t q_offset_bytes,
                    const at::Tensor& kv,
                    int64_t kv_offset_bytes,
                    const at::Tensor& s_bf16,
                    int64_t s_offset_bytes,
                    at::Tensor& output,
                    int64_t output_offset_bytes,
                    int64_t window,
                    int64_t kv_stride,
                    int64_t num_q_tokens,
                    int64_t num_kv_tokens,
                    int64_t num_q_heads,
                    int64_t num_kv_heads,
                    int64_t head_dim);

void f32_topk_torch(const at::Tensor& scores,
                    at::Tensor& expert_ids,
                    at::Tensor& expert_scores,
                    int64_t num_tokens,
                    int64_t num_experts,
                    int64_t num_active_experts);

void expert_routing_metadata_torch(const at::Tensor& expert_ids,
                                   const at::Tensor& expert_scores,
                                   at::Tensor& expert_offsets,
                                   at::Tensor& intra_expert_offsets,
                                   int64_t num_tokens,
                                   int64_t num_experts);

void f32_scatter_torch(const at::Tensor& input,
                       const at::Tensor& expert_ids,
                       const at::Tensor& expert_scores,
                       const at::Tensor& expert_offsets,
                       const at::Tensor& intra_expert_offsets,
                       at::Tensor& output,
                       int64_t num_channels,
                       int64_t num_tokens,
                       int64_t num_active_experts);

void f32_bf16w_matmul_add_torch(const at::Tensor& input,
                                const at::Tensor& weight_bf16,
                                const at::Tensor& bias_bf16,
                                at::Tensor& output,
                                int64_t num_tokens,
                                int64_t num_cols,
                                int64_t num_rows,
                                int64_t threadgroup_size);