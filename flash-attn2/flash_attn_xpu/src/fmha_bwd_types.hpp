#pragma once

#include <cstdint>

struct fmha_bwd_args_t {
  // Input tensors
  void* dout;        // grad output
  void* out;         // forward output
  void* query;
  void* key;
  void* value;
  void* softmax_lse; // logsumexp from forward

  // Output gradient tensors
  void* dq;
  void* dk;
  void* dv;

  // Intermediate buffers
  void* softmax_d;   // sum of dout * out
  void* dq_accum;    // accumulator for dq (float)

  // Dimensions
  int batch_size;
  int num_heads_q;
  int num_heads_k;
  int seqlen_q;
  int seqlen_k;
  int head_size;

  // Padded dimensions
  int seqlen_q_rounded;
  int seqlen_k_rounded;

  // Scale factor
  float sm_scale;

  // Flags
  bool is_causal = false;
  bool is_local = false;
  bool is_bf16 = false;
  bool deterministic = false;

  // Window size for local attention
  int window_size_left = -1;
  int window_size_right = -1;
  
  // Dropout parameters
  float p_dropout = 0.0f;     // Probability of dropping (NOT keeping)
  uint64_t philox_seed = 0;   // Philox RNG seed (from forward pass)
  uint64_t philox_offset = 0; // Philox RNG offset (from forward pass)
};

enum class BwdCutlassType {
  half,
  bfloat16
};

constexpr int PipelineStages_Bwd = 1;

// Block configuration for backward pass
struct bwd_policy_head32 {
  static constexpr int kBlockM = 64;
  static constexpr int kBlockN = 32;
  static constexpr int kHeadDim = 32;
  static constexpr int kNSGs = 8;
  static constexpr int AtomLayoutMSdP = 4;
  static constexpr int AtomLayoutNdKV = 2;
  static constexpr int AtomLayoutMdQ = 2;
};

struct bwd_policy_head64 {
  static constexpr int kBlockM = 64;
  static constexpr int kBlockN = 32;
  static constexpr int kHeadDim = 64;
  static constexpr int kNSGs = 8;
  static constexpr int AtomLayoutMSdP = 4;
  static constexpr int AtomLayoutNdKV = 2;
  static constexpr int AtomLayoutMdQ = 2;
};

struct bwd_policy_head96 {
  static constexpr int kBlockM = 64;
  static constexpr int kBlockN = 64;
  static constexpr int kHeadDim = 96;
  static constexpr int kNSGs = 8;
  static constexpr int AtomLayoutMSdP = 2;
  static constexpr int AtomLayoutNdKV = 4;
  static constexpr int AtomLayoutMdQ = 4;
};

struct bwd_policy_head128 {
  static constexpr int kBlockM = 64;
  static constexpr int kBlockN = 64;
  static constexpr int kHeadDim = 128;
  static constexpr int kNSGs = 8;
  static constexpr int AtomLayoutMSdP = 2;
  static constexpr int AtomLayoutNdKV = 2;
  static constexpr int AtomLayoutMdQ = 4;
};

struct bwd_policy_head160 {
  static constexpr int kBlockM = 64;
  static constexpr int kBlockN = 32;
  static constexpr int kHeadDim = 160;
  static constexpr int kNSGs = 8;
  static constexpr int AtomLayoutMSdP = 4;
  static constexpr int AtomLayoutNdKV = 2;
  static constexpr int AtomLayoutMdQ = 2;
};

struct bwd_policy_head192 {
  static constexpr int kBlockM = 64;
  static constexpr int kBlockN = 32;
  static constexpr int kHeadDim = 192;
  static constexpr int kNSGs = 8;
  static constexpr int AtomLayoutMSdP = 4;
  static constexpr int AtomLayoutNdKV = 2;
  static constexpr int AtomLayoutMdQ = 2;
};

struct bwd_policy_head256 {
  static constexpr int kBlockM = 64;
  static constexpr int kBlockN = 32;
  static constexpr int kHeadDim = 256;
  static constexpr int kNSGs = 8;
  static constexpr int AtomLayoutMSdP = 4;
  static constexpr int AtomLayoutNdKV = 2;
  static constexpr int AtomLayoutMdQ = 2;
};
