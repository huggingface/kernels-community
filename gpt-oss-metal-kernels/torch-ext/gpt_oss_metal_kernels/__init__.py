from ._ops import ops
import torch

def f32_bf16w_matmul(input: torch.Tensor,
                     weight_bf16: torch.Tensor,
                     bias_bf16: torch.Tensor,
                     output: torch.Tensor,
                     num_tokens: int,
                     num_cols: int,
                     num_rows: int,
                     threadgroup_size: int) -> torch.Tensor:
    ops.f32_bf16w_matmul(input, weight_bf16, bias_bf16, output,
                         num_tokens, num_cols, num_rows, threadgroup_size)
    return output

def bf16_f32_embeddings(token_ids: torch.Tensor,
                        weight_bf16: torch.Tensor,
                        output: torch.Tensor,
                        threadgroup_size: int) -> torch.Tensor:
    ops.bf16_f32_embeddings(token_ids, weight_bf16, output, threadgroup_size)
    return output

def f32_bf16w_rmsnorm(input: torch.Tensor,
                      weight_bf16: torch.Tensor,
                      output: torch.Tensor,
                      epsilon: float) -> torch.Tensor:
    ops.f32_bf16w_rmsnorm(input, weight_bf16, output, epsilon)
    return output

def f32_bf16w_dense_matmul_qkv(input: torch.Tensor,
                               weight_bf16: torch.Tensor,
                               bias_bf16: torch.Tensor,
                               output: torch.Tensor) -> torch.Tensor:
    ops.f32_bf16w_dense_matmul_qkv(input, weight_bf16, bias_bf16, output)
    return output

def f32_bf16w_dense_matmul_attn_output(input: torch.Tensor,
                                       weight_bf16: torch.Tensor,
                                       bias_bf16: torch.Tensor,
                                       output: torch.Tensor) -> torch.Tensor:
    ops.f32_bf16w_dense_matmul_attn_output(input, weight_bf16, bias_bf16, output)
    return output

def f32_bf16w_dense_matmul_mlp_gate(input: torch.Tensor,
                                    weight_bf16: torch.Tensor,
                                    bias_bf16: torch.Tensor,
                                    output: torch.Tensor) -> torch.Tensor:
    ops.f32_bf16w_dense_matmul_mlp_gate(input, weight_bf16, bias_bf16, output)
    return output

def f32_rope(activations: torch.Tensor,
             rope_base: float,
             interpolation_scale: float,
             yarn_offset: float,
             yarn_scale: float,
             yarn_multiplier: float,
             num_tokens: int,
             num_q_heads: int,
             num_kv_heads: int,
             attn_head_dim: int,
             token_offset: int,
             threadgroup_size: int) -> torch.Tensor:
    ops.f32_rope(activations, rope_base, interpolation_scale, yarn_offset,
                 yarn_scale, yarn_multiplier, num_tokens, num_q_heads,
                 num_kv_heads, attn_head_dim, token_offset, threadgroup_size)
    return activations

def f32_bf16w_matmul_qkv(input: torch.Tensor,
                         weight_bf16: torch.Tensor,
                         bias_bf16: torch.Tensor,
                         output: torch.Tensor,
                         kv_cache: torch.Tensor,
                         kv_cache_offset_bytes: int,
                         num_tokens: int,
                         num_cols: int,
                         num_q_heads: int,
                         num_kv_heads: int,
                         attn_head_dim: int,
                         token_offset: int,
                         max_tokens: int,
                         rope_base: float,
                         interpolation_scale: float,
                         yarn_offset: float,
                         yarn_scale: float,
                         yarn_multiplier: float,
                         threadgroup_size: int) -> torch.Tensor:
    ops.f32_bf16w_matmul_qkv(input, weight_bf16, bias_bf16, output, kv_cache,
                             kv_cache_offset_bytes, num_tokens, num_cols,
                             num_q_heads, num_kv_heads, attn_head_dim,
                             token_offset, max_tokens, rope_base,
                             interpolation_scale, yarn_offset, yarn_scale,
                             yarn_multiplier, threadgroup_size)
    return output

def f32_sdpa(q: torch.Tensor,
             q_offset_bytes: int,
             kv: torch.Tensor,
             kv_offset_bytes: int,
             s_bf16: torch.Tensor,
             s_offset_bytes: int,
             output: torch.Tensor,
             output_offset_bytes: int,
             window: int,
             kv_stride: int,
             num_q_tokens: int,
             num_kv_tokens: int,
             num_q_heads: int,
             num_kv_heads: int,
             head_dim: int) -> torch.Tensor:
    ops.f32_sdpa(q, q_offset_bytes, kv, kv_offset_bytes, s_bf16, s_offset_bytes,
                 output, output_offset_bytes, window, kv_stride,
                 num_q_tokens, num_kv_tokens, num_q_heads, num_kv_heads, head_dim)
    return output

def f32_topk(scores: torch.Tensor,
             expert_ids: torch.Tensor,
             expert_scores: torch.Tensor,
             num_tokens: int,
             num_experts: int,
             num_active_experts: int) -> None:
    ops.f32_topk(scores, expert_ids, expert_scores,
                 num_tokens, num_experts, num_active_experts)

def expert_routing_metadata(expert_ids: torch.Tensor,
                            expert_scores: torch.Tensor,
                            expert_offsets: torch.Tensor,
                            intra_expert_offsets: torch.Tensor,
                            num_tokens: int,
                            num_experts: int) -> None:
    ops.expert_routing_metadata(expert_ids, expert_scores,
                                expert_offsets, intra_expert_offsets,
                                num_tokens, num_experts)

def f32_scatter(input: torch.Tensor,
                expert_ids: torch.Tensor,
                expert_scores: torch.Tensor,
                expert_offsets: torch.Tensor,
                intra_expert_offsets: torch.Tensor,
                output: torch.Tensor,
                num_channels: int,
                num_tokens: int,
                num_active_experts: int) -> torch.Tensor:
    ops.f32_scatter(input, expert_ids, expert_scores,
                    expert_offsets, intra_expert_offsets,
                    output, num_channels, num_tokens, num_active_experts)
    return output

def f32_bf16w_matmul_add(input: torch.Tensor,
                         weight_bf16: torch.Tensor,
                         bias_bf16: torch.Tensor,
                         output: torch.Tensor,
                         num_tokens: int,
                         num_cols: int,
                         num_rows: int,
                         threadgroup_size: int) -> torch.Tensor:
    ops.f32_bf16w_matmul_add(input, weight_bf16, bias_bf16, output,
                             num_tokens, num_cols, num_rows, threadgroup_size)
    return output

__all__ = [
    "f32_bf16w_matmul",
    "bf16_f32_embeddings",
    "f32_bf16w_rmsnorm",
    "f32_bf16w_dense_matmul_qkv",
    "f32_bf16w_dense_matmul_attn_output",
    "f32_bf16w_dense_matmul_mlp_gate",
    "f32_rope",
    "f32_bf16w_matmul_qkv",
    "f32_sdpa",
    "f32_topk",
    "expert_routing_metadata",
    "f32_scatter",
    "f32_bf16w_matmul_add",
]