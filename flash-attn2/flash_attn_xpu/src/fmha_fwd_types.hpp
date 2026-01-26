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
};

enum class CutlassType {
  half,
  bfloat16
};

constexpr int PipelineStages_Decode = 1;
constexpr int PipelineStages_Prefill = 2;
