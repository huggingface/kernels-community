#pragma once

#include <cstdint>

struct fmha_fwd_args_t {
  void* query;
  void* key;
  void* value;
  void* out;
  void* softmax_lse;  // Output: log-sum-exp for backward pass (batch, num_heads, seqlen_q)
  void* block_table;
  void* cu_seqlens_q;
  void* cu_seqlens_k;
  int max_queries;
  int max_keys;
  int total_seqlen_q;
  int total_seqlen_k;
  float sm_scale;
  int batch_size;
  int num_heads_q;
  int num_heads_k;
  int head_size;
  int max_blocks_per_seq;
  int block_size;
  int window_size_left = -1;
  int window_size_right = -1;
  bool is_varlen = false;
  bool is_paged = false;
  bool is_causal = false;
  bool is_local = false;
  
  // Dropout parameters
  float p_dropout = 0.0f;     // Probability of dropping (NOT keeping)
  uint64_t philox_seed = 0;   // Philox RNG seed
  uint64_t philox_offset = 0; // Philox RNG offset
  void* rng_state = nullptr;  // Output: RNG state for backward pass (2 x int64)

  // S_dmask parameters (for return_softmax with dropout)
  void* s_dmask = nullptr;    // Output: attention matrix with dropout sign-bit encoding
  int seqlen_q_rounded = 0;   // Q sequence length rounded up (stride for s_dmask)
  int seqlen_k_rounded = 0;   // K sequence length rounded up (stride for s_dmask)

  // KV Cache fields (optional, set to nullptr for non-kvcache paths)
  int* cache_seqlens = nullptr;    // (batch_size,) per-batch effective KV length
  int* cache_batch_idx = nullptr;  // (batch_size,) indices into KV cache batch dim
  int* cache_leftpad = nullptr;    // (batch_size,) left padding per batch in KV cache

  // Fused KV cache append: new K/V to scatter into cache inside the kernel
  void* knew = nullptr;
  void* vnew = nullptr;
  int seqlen_knew = 0;
  int64_t knew_batch_stride = 0;
  int64_t knew_head_stride = 0;
  int64_t knew_row_stride = 0;
  int64_t vnew_batch_stride = 0;
  int64_t vnew_head_stride = 0;
  int64_t vnew_row_stride = 0;

  // Fused rotary embedding for kvcache append and Q load
  void* rotary_cos = nullptr;
  void* rotary_sin = nullptr;
  int rotary_dim = 0;
  bool is_rotary_interleaved = true;
};

enum class CutlassType {
  half,
  bfloat16
};

constexpr int PipelineStages_Decode = 1;
constexpr int PipelineStages_Prefill = 2;
