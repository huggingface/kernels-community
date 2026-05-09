#include "fmha_fwd.hpp"
#include "fmha_utils.hpp"
#include "torch/all.h"
#include <sycl/sycl.hpp>

namespace {

/// Round `x` up to the nearest multiple of `m` (plain version, no 224→256 special case).
constexpr int round_up(int x, int m) {
  return (x + m - 1) / m * m;
}

int paged_k_tile_size_n(int head_size, int max_seqlen_q) {
  if (max_seqlen_q == 1) {
    return 64;
  }
  if (head_size <= 96) {
    return 64;
  }
  if (head_size <= 128) {
    return 32;
  }
  if (head_size <= 256) {
    return 64;
  }
  if (head_size == 512) {
    return 32;
  }
  return 0;
}

bool is_supported_paged_block_size(int block_size) {
  return block_size >= 64 && block_size % 64 == 0;
}

/// Dispatch the varlen forward kernel by head_size, delegating paged/non-paged variants.
/// The Policy template parameter is selected by the caller based on head_size.
template <typename Policy, int PipelineStages>
void dispatch_varlen_paged(sycl::queue& queue, CutlassType cuType,
                           const fmha_fwd_args_t& args) {
  if (args.is_paged) {
    policy_dispatch<Policy, PipelineStages, /*IsVarLen=*/1, /*IsPaged=*/1>(queue, cuType, args);
  } else {
    policy_dispatch<Policy, PipelineStages, /*IsVarLen=*/1, /*IsPaged=*/0>(queue, cuType, args);
  }
}

template <typename Policy, typename PagedPolicy, int PipelineStages>
void dispatch_varlen_decode_paged(sycl::queue& queue, CutlassType cuType,
                                  const fmha_fwd_args_t& args) {
  if (args.is_paged) {
    policy_dispatch<PagedPolicy, PipelineStages, /*IsVarLen=*/1, /*IsPaged=*/1>(queue, cuType, args);
  } else {
    policy_dispatch<Policy, PipelineStages, /*IsVarLen=*/1, /*IsPaged=*/0>(queue, cuType, args);
  }
}

template <typename Policy, int PipelineStages, int IsPaged>
void dispatch_kvcache_policy(sycl::queue& queue, CutlassType cuType,
                             const fmha_fwd_args_t& args,
                             bool has_rotary) {
  if (has_rotary) {
    policy_dispatch<Policy, PipelineStages, /*IsVarLen=*/0, IsPaged,
                    /*HasRotary=*/true>(queue, cuType, args);
  } else {
    policy_dispatch<Policy, PipelineStages, /*IsVarLen=*/0, IsPaged,
                    /*HasRotary=*/false>(queue, cuType, args);
  }
}

/// Dispatch forward kernel by head_size for the varlen prefill path.
/// Supported head dimensions are bucketed to the corresponding prefill policies.
void dispatch_fwd_varlen_by_head(sycl::queue& queue, CutlassType cuType,
                                 const fmha_fwd_args_t& args, int head_size) {
  if      (head_size <=  32) dispatch_varlen_paged<prefill_policy_head32,  PipelineStages_Prefill>(queue, cuType, args);
  else if (head_size <=  64) dispatch_varlen_paged<prefill_policy_head64,  PipelineStages_Prefill>(queue, cuType, args);
  else if (head_size <=  96) dispatch_varlen_paged<prefill_policy_head96,  PipelineStages_Prefill>(queue, cuType, args);
  else if (head_size <= 128) dispatch_varlen_paged<prefill_policy_head128, PipelineStages_Prefill>(queue, cuType, args);
  else if (head_size <= 160) dispatch_varlen_paged<prefill_policy_head160, PipelineStages_Prefill>(queue, cuType, args);
  else if (head_size <= 192) dispatch_varlen_paged<prefill_policy_head192, PipelineStages_Prefill>(queue, cuType, args);
  else if (head_size <= 256) dispatch_varlen_paged<prefill_policy_head256, PipelineStages_Prefill>(queue, cuType, args);
  else if (head_size == 512) dispatch_varlen_paged<prefill_policy_head512, PipelineStages_Prefill>(queue, cuType, args);
  else throw std::runtime_error("Unsupported head_size: " + std::to_string(head_size) + ". Only <= 256 or exactly 512 is supported");
}

/// Dispatch forward kernel by head_size for the varlen decode path (max_seqlen_q == 1).
void dispatch_fwd_varlen_decode_by_head(sycl::queue& queue, CutlassType cuType,
                                        const fmha_fwd_args_t& args, int head_size) {
  if      (head_size <=  32) dispatch_varlen_decode_paged<decode_policy_head32,  decode_paged_policy_head32,  PipelineStages_Decode>(queue, cuType, args);
  else if (head_size <=  64) dispatch_varlen_decode_paged<decode_policy_head64,  decode_paged_policy_head64,  PipelineStages_Decode>(queue, cuType, args);
  else if (head_size <=  96) dispatch_varlen_decode_paged<decode_policy_head96,  decode_paged_policy_head96,  PipelineStages_Decode>(queue, cuType, args);
  else if (head_size <= 128) dispatch_varlen_decode_paged<decode_policy_head128, decode_paged_policy_head128, PipelineStages_Decode>(queue, cuType, args);
  else if (head_size <= 160) dispatch_varlen_decode_paged<decode_policy_head160, decode_paged_policy_head160, PipelineStages_Decode>(queue, cuType, args);
  else if (head_size <= 192) dispatch_varlen_decode_paged<decode_policy_head192, decode_paged_policy_head192, PipelineStages_Decode>(queue, cuType, args);
  else if (head_size <= 256) dispatch_varlen_decode_paged<decode_policy_head256, decode_paged_policy_head256, PipelineStages_Decode>(queue, cuType, args);
  else if (head_size == 512) dispatch_varlen_decode_paged<decode_policy_head512, decode_paged_policy_head512, PipelineStages_Decode>(queue, cuType, args);
  else throw std::runtime_error("Unsupported head_size: " + std::to_string(head_size) + ". Only <= 256 or exactly 512 is supported");
}

/// Dispatch forward kernel by head_size for the decode path (seqlen_q == 1).
void dispatch_fwd_decode_by_head(sycl::queue& queue, CutlassType cuType,
                                 const fmha_fwd_args_t& args, int head_size) {
  if      (head_size <=  32) policy_dispatch<decode_policy_head32,  PipelineStages_Decode, 0, 0>(queue, cuType, args);
  else if (head_size <=  64) policy_dispatch<decode_policy_head64,  PipelineStages_Decode, 0, 0>(queue, cuType, args);
  else if (head_size <=  96) policy_dispatch<decode_policy_head96,  PipelineStages_Decode, 0, 0>(queue, cuType, args);
  else if (head_size <= 128) policy_dispatch<decode_policy_head128, PipelineStages_Decode, 0, 0>(queue, cuType, args);
  else if (head_size <= 160) policy_dispatch<decode_policy_head160, PipelineStages_Decode, 0, 0>(queue, cuType, args);
  else if (head_size <= 192) policy_dispatch<decode_policy_head192, PipelineStages_Decode, 0, 0>(queue, cuType, args);
  else if (head_size <= 256) policy_dispatch<decode_policy_head256, PipelineStages_Decode, 0, 0>(queue, cuType, args);
  else if (head_size == 512) policy_dispatch<decode_policy_head512, PipelineStages_Decode, 0, 0>(queue, cuType, args);
  else throw std::runtime_error("Unsupported head_size: " + std::to_string(head_size) + ". Only <= 256 or exactly 512 is supported");
}

/// Dispatch forward kernel by head_size for the prefill path (seqlen_q > 1).
void dispatch_fwd_prefill_by_head(sycl::queue& queue, CutlassType cuType,
                                  const fmha_fwd_args_t& args, int head_size) {
  if      (head_size <=  32) policy_dispatch<prefill_policy_head32,  PipelineStages_Prefill, 0, 0>(queue, cuType, args);
  else if (head_size <=  64) policy_dispatch<prefill_policy_head64,  PipelineStages_Prefill, 0, 0>(queue, cuType, args);
  else if (head_size <=  96) policy_dispatch<prefill_policy_head96,  PipelineStages_Prefill, 0, 0>(queue, cuType, args);
  else if (head_size <= 128) policy_dispatch<prefill_policy_head128, PipelineStages_Prefill, 0, 0>(queue, cuType, args);
  else if (head_size <= 160) policy_dispatch<prefill_policy_head160, PipelineStages_Prefill, 0, 0>(queue, cuType, args);
  else if (head_size <= 192) policy_dispatch<prefill_policy_head192, PipelineStages_Prefill, 0, 0>(queue, cuType, args);
  else if (head_size <= 256) policy_dispatch<prefill_policy_head256, PipelineStages_Prefill, 0, 0>(queue, cuType, args);
  else if (head_size == 512) policy_dispatch<prefill_policy_head512, PipelineStages_Prefill, 0, 0>(queue, cuType, args);
  else throw std::runtime_error("Unsupported head_size: " + std::to_string(head_size) + ". Only <= 256 or exactly 512 is supported");
}

/// Dispatch forward kernel by head_size for the kvcache prefill paged path
/// (non-varlen, seqlen_q > 1, IsPaged=1).
void dispatch_fwd_kvcache_prefill_paged_by_head(sycl::queue& queue, CutlassType cuType,
                                                const fmha_fwd_args_t& args, int head_size,
                                                bool has_rotary) {
  if      (head_size <=  32) dispatch_kvcache_policy<prefill_policy_head32,  PipelineStages_Prefill, 1>(queue, cuType, args, has_rotary);
  else if (head_size <=  64) dispatch_kvcache_policy<prefill_policy_head64,  PipelineStages_Prefill, 1>(queue, cuType, args, has_rotary);
  else if (head_size <=  96) dispatch_kvcache_policy<prefill_policy_head96,  PipelineStages_Prefill, 1>(queue, cuType, args, has_rotary);
  else if (head_size <= 128) dispatch_kvcache_policy<prefill_policy_head128, PipelineStages_Prefill, 1>(queue, cuType, args, has_rotary);
  else if (head_size <= 160) dispatch_kvcache_policy<prefill_policy_head160, PipelineStages_Prefill, 1>(queue, cuType, args, has_rotary);
  else if (head_size <= 192) dispatch_kvcache_policy<prefill_policy_head192, PipelineStages_Prefill, 1>(queue, cuType, args, has_rotary);
  else if (head_size <= 256) dispatch_kvcache_policy<prefill_policy_head256, PipelineStages_Prefill, 1>(queue, cuType, args, has_rotary);
  else if (head_size == 512) dispatch_kvcache_policy<prefill_policy_head512, PipelineStages_Prefill, 1>(queue, cuType, args, has_rotary);
  else throw std::runtime_error("Unsupported head_size: " + std::to_string(head_size) + ". Only <= 256 or exactly 512 is supported");
}

/// Dispatch forward kernel by head_size for the kvcache decode paged path
/// (non-varlen, seqlen_q == 1, IsPaged=1). Uses decode_paged_policy with a
/// smaller K-tile so block_size=64 multiples don't overshoot a page.
void dispatch_fwd_kvcache_decode_paged_by_head(sycl::queue& queue, CutlassType cuType,
                                               const fmha_fwd_args_t& args, int head_size,
                                               bool has_rotary) {
  if      (head_size <=  32) dispatch_kvcache_policy<decode_paged_policy_head32,  PipelineStages_Decode, 1>(queue, cuType, args, has_rotary);
  else if (head_size <=  64) dispatch_kvcache_policy<decode_paged_policy_head64,  PipelineStages_Decode, 1>(queue, cuType, args, has_rotary);
  else if (head_size <=  96) dispatch_kvcache_policy<decode_paged_policy_head96,  PipelineStages_Decode, 1>(queue, cuType, args, has_rotary);
  else if (head_size <= 128) dispatch_kvcache_policy<decode_paged_policy_head128, PipelineStages_Decode, 1>(queue, cuType, args, has_rotary);
  else if (head_size <= 160) dispatch_kvcache_policy<decode_paged_policy_head160, PipelineStages_Decode, 1>(queue, cuType, args, has_rotary);
  else if (head_size <= 192) dispatch_kvcache_policy<decode_paged_policy_head192, PipelineStages_Decode, 1>(queue, cuType, args, has_rotary);
  else if (head_size <= 256) dispatch_kvcache_policy<decode_paged_policy_head256, PipelineStages_Decode, 1>(queue, cuType, args, has_rotary);
  else if (head_size == 512) dispatch_kvcache_policy<decode_paged_policy_head512, PipelineStages_Decode, 1>(queue, cuType, args, has_rotary);
  else throw std::runtime_error("Unsupported head_size: " + std::to_string(head_size) + ". Only <= 256 or exactly 512 is supported");
}

void dispatch_fwd_kvcache_decode_by_head(sycl::queue& queue, CutlassType cuType,
                                         const fmha_fwd_args_t& args, int head_size,
                                         bool has_rotary) {
  if      (head_size <=  32) dispatch_kvcache_policy<decode_policy_head32,  PipelineStages_Decode, 0>(queue, cuType, args, has_rotary);
  else if (head_size <=  64) dispatch_kvcache_policy<decode_policy_head64,  PipelineStages_Decode, 0>(queue, cuType, args, has_rotary);
  else if (head_size <=  96) dispatch_kvcache_policy<decode_policy_head96,  PipelineStages_Decode, 0>(queue, cuType, args, has_rotary);
  else if (head_size <= 128) dispatch_kvcache_policy<decode_policy_head128, PipelineStages_Decode, 0>(queue, cuType, args, has_rotary);
  else if (head_size <= 160) dispatch_kvcache_policy<decode_policy_head160, PipelineStages_Decode, 0>(queue, cuType, args, has_rotary);
  else if (head_size <= 192) dispatch_kvcache_policy<decode_policy_head192, PipelineStages_Decode, 0>(queue, cuType, args, has_rotary);
  else if (head_size <= 256) dispatch_kvcache_policy<decode_policy_head256, PipelineStages_Decode, 0>(queue, cuType, args, has_rotary);
  else if (head_size == 512) dispatch_kvcache_policy<decode_policy_head512, PipelineStages_Decode, 0>(queue, cuType, args, has_rotary);
  else throw std::runtime_error("Unsupported head_size: " + std::to_string(head_size) + ". Only <= 256 or exactly 512 is supported");
}

void dispatch_fwd_kvcache_prefill_by_head(sycl::queue& queue, CutlassType cuType,
                                          const fmha_fwd_args_t& args, int head_size,
                                          bool has_rotary) {
  if      (head_size <=  32) dispatch_kvcache_policy<prefill_policy_head32,  PipelineStages_Prefill, 0>(queue, cuType, args, has_rotary);
  else if (head_size <=  64) dispatch_kvcache_policy<prefill_policy_head64,  PipelineStages_Prefill, 0>(queue, cuType, args, has_rotary);
  else if (head_size <=  96) dispatch_kvcache_policy<prefill_policy_head96,  PipelineStages_Prefill, 0>(queue, cuType, args, has_rotary);
  else if (head_size <= 128) dispatch_kvcache_policy<prefill_policy_head128, PipelineStages_Prefill, 0>(queue, cuType, args, has_rotary);
  else if (head_size <= 160) dispatch_kvcache_policy<prefill_policy_head160, PipelineStages_Prefill, 0>(queue, cuType, args, has_rotary);
  else if (head_size <= 192) dispatch_kvcache_policy<prefill_policy_head192, PipelineStages_Prefill, 0>(queue, cuType, args, has_rotary);
  else if (head_size <= 256) dispatch_kvcache_policy<prefill_policy_head256, PipelineStages_Prefill, 0>(queue, cuType, args, has_rotary);
  else if (head_size == 512) dispatch_kvcache_policy<prefill_policy_head512, PipelineStages_Prefill, 0>(queue, cuType, args, has_rotary);
  else throw std::runtime_error("Unsupported head_size: " + std::to_string(head_size) + ". Only <= 256 or exactly 512 is supported");
}

/// Clamp window sizes for local attention and fold causal into local when both are set.
void normalize_window_params(int& window_size_left, int& window_size_right,
                             bool& is_causal, bool is_local, int max_seqlen_k) {
  if (!is_local) return;
  if (window_size_left  == -1) window_size_left  = max_seqlen_k;
  if (window_size_right == -1) window_size_right = max_seqlen_k;
  if (is_causal) {
    window_size_right = 0;
    is_causal = false;
  }
}

}  // anonymous namespace

void cutlass_fmha_fwd_varlen_impl(
    sycl::queue& queue,
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    at::Tensor& out,
    at::Tensor& softmax_lse,
    const std::optional<at::Tensor>& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    double sm_scale,
    int window_size_left,
    int window_size_right,
    bool is_varlen,
    bool is_paged,
    bool is_causal,
    bool is_local) {
  int batch_size, num_heads_q, num_heads_kv, head_size;
  int total_seqlen_q, total_seqlen_k;
  int num_blocks, block_size, max_blocks_per_seq;

  if (is_varlen) {
    batch_size     = cu_seqlens_q.numel() - 1;
    num_heads_q    = query.size(1);
    num_heads_kv   = key_cache.size(1);
    head_size      = query.size(2);
    total_seqlen_q = query.size(0);
    total_seqlen_k = key_cache.size(0);
  } else {
    batch_size     = query.size(0);
    num_heads_q    = query.size(1);
    num_heads_kv   = key_cache.size(1);
    head_size      = query.size(3);
    max_seqlen_q   = query.size(2);
    max_seqlen_k   = key_cache.size(2);
  }

  if (is_paged) {
    num_blocks         = key_cache.size(0);
    block_size         = key_cache.size(1);
    num_heads_kv       = key_cache.size(2);
    max_blocks_per_seq = block_table->size(1);
    total_seqlen_k     = num_blocks * block_size;
    TORCH_CHECK(
      is_supported_paged_block_size(block_size),
      "Unsupported paged KV block_size=", block_size,
      ". This FA2 XPU path supports block_size values that are positive multiples of 64 ",
      "(64, 128, 256, ...). block_size=16/32 need dedicated paged policies and are not enabled.");
    const int k_tile_n = paged_k_tile_size_n(head_size, max_seqlen_q);
    TORCH_CHECK(k_tile_n > 0, "Unsupported head_size for paged FA2 forward: ", head_size);
    TORCH_CHECK(
      block_size % k_tile_n == 0,
      "Paged KV block_size must be a multiple of the kernel K tile. Got block_size=",
      block_size, ", K tile=", k_tile_n,
      ". Current paged mapping does not support tiles crossing page boundaries.");
  } else {
    num_blocks         = 0;
    block_size         = 0;
    num_heads_kv       = key_cache.size(1);
    max_blocks_per_seq = 0;
    total_seqlen_k     = key_cache.size(0);
  }

  normalize_window_params(window_size_left, window_size_right,
                          is_causal, is_local, max_seqlen_k);

  fmha_fwd_args_t args = {
      query.data_ptr(),
      key_cache.data_ptr(),
      value_cache.data_ptr(),
      out.data_ptr(),
      softmax_lse.data_ptr(),
      is_paged && block_table.has_value() ? block_table->data_ptr() : nullptr,
      cu_seqlens_q.data_ptr(),
      cu_seqlens_k.data_ptr(),
      max_seqlen_q,
      max_seqlen_k,
      total_seqlen_q,
      total_seqlen_k,
      static_cast<float>(sm_scale),
      batch_size,
      num_heads_q,
      num_heads_kv,
      head_size,
      max_blocks_per_seq,
      block_size,
      window_size_left,
      window_size_right,
      is_varlen,
      is_paged,
      is_causal,
      is_local};

  const CutlassType cuType = aten_to_Cutlass_dtype(query);
  if (args.max_queries == 1) {
    dispatch_fwd_varlen_decode_by_head(queue, cuType, args, args.head_size);
  } else {
    dispatch_fwd_varlen_by_head(queue, cuType, args, args.head_size);
  }
}

void cutlass_fmha_fwd_fix_impl(
    sycl::queue& queue,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& out,
    at::Tensor& softmax_lse,
    float sm_scale,
    int window_size_left,
    int window_size_right,
    bool is_causal,
    bool is_local,
    float p_dropout,
    uint64_t philox_seed,
    uint64_t philox_offset,
    void* rng_state,
    void* s_dmask,
    int seqlen_q_rounded,
    int seqlen_k_rounded) {
  const int batch_size   = query.size(0);
  const int max_seqlen_q = query.size(1);
  const int num_heads_q  = query.size(2);
  const int head_size    = query.size(3);

  const int max_seqlen_k = key.size(1);
  const int num_heads_kv = key.size(2);

  const int total_seqlen_q = batch_size * max_seqlen_q;
  const int total_seqlen_k = batch_size * max_seqlen_k;

  normalize_window_params(window_size_left, window_size_right,
                          is_causal, is_local, max_seqlen_k);

  fmha_fwd_args_t args = {
      query.data_ptr(),
      key.data_ptr(),
      value.data_ptr(),
      out.data_ptr(),
      softmax_lse.data_ptr(),
      nullptr,
      nullptr,
      nullptr,
      max_seqlen_q,
      max_seqlen_k,
      total_seqlen_q,
      total_seqlen_k,
      sm_scale,
      batch_size,
      num_heads_q,
      num_heads_kv,
      head_size,
      0,
      0,
      window_size_left,
      window_size_right,
      false,
      false,
      is_causal,
      is_local,
      p_dropout,
      philox_seed,
      philox_offset,
      rng_state,
      s_dmask,
      seqlen_q_rounded,
      seqlen_k_rounded};

  const CutlassType cuType = aten_to_Cutlass_dtype(query);
  const int h = args.head_size;

  // Decode path (single query token) vs. prefill path
  if (max_seqlen_q == 1) {
    dispatch_fwd_decode_by_head(queue, cuType, args, h);
  } else {
    dispatch_fwd_prefill_by_head(queue, cuType, args, h);
  }
}

void cutlass_fmha_fwd_kvcache_impl(
    sycl::queue& queue,
    const at::Tensor& query,
    const at::Tensor& kcache,
    const at::Tensor& vcache,
    at::Tensor& out,
    at::Tensor& softmax_lse,
    const at::Tensor& seqlens_k,
    const std::optional<at::Tensor>& cache_batch_idx,
    const std::optional<at::Tensor>& cache_leftpad,
    const std::optional<at::Tensor>& knew,
    const std::optional<at::Tensor>& vnew,
    const std::optional<at::Tensor>& block_table,
    const std::optional<at::Tensor>& rotary_cos,
    const std::optional<at::Tensor>& rotary_sin,
    int rotary_dim,
    bool is_rotary_interleaved,
    int max_seqlen_k_paged,
    float sm_scale,
    int window_size_left,
    int window_size_right,
    bool is_causal,
    bool is_local) {
  const bool is_paged = block_table.has_value() && block_table->defined();

  const int batch_size   = query.size(0);
  const int max_seqlen_q = query.size(1);
  const int num_heads_q  = query.size(2);
  const int head_size    = query.size(3);

  int max_seqlen_k;
  int num_heads_kv;
  int num_blocks = 0;
  int block_size = 0;
  int max_blocks_per_seq = 0;
  if (is_paged) {
    // Paged layout: kcache is (num_blocks, page_block_size, num_heads_kv, head_size)
    num_blocks         = kcache.size(0);
    block_size         = kcache.size(1);
    num_heads_kv       = kcache.size(2);
    max_blocks_per_seq = block_table->size(1);
    max_seqlen_k       = max_seqlen_k_paged;
    TORCH_CHECK(num_blocks * block_size >= max_seqlen_k,
                "Paged KV pool too small for max_seqlen_k: ",
                num_blocks * block_size, " < ", max_seqlen_k);
    TORCH_CHECK(!cache_batch_idx.has_value(),
                "Paged KVcache does not support cache_batch_idx");
  } else {
    max_seqlen_k = kcache.size(1);
    num_heads_kv = kcache.size(2);
  }

  const int total_seqlen_q = batch_size * max_seqlen_q;
  const int total_seqlen_k = is_paged
      ? num_blocks * block_size
      : batch_size * max_seqlen_k;

  normalize_window_params(window_size_left, window_size_right,
                          is_causal, is_local, max_seqlen_k);

  fmha_fwd_args_t args = {
      query.data_ptr(),
      kcache.data_ptr(),
      vcache.data_ptr(),
      out.data_ptr(),
      softmax_lse.data_ptr(),
      is_paged ? block_table->data_ptr() : nullptr,
      nullptr,      // cu_seqlens_q
      nullptr,      // cu_seqlens_k
      max_seqlen_q,
      max_seqlen_k,
      total_seqlen_q,
      total_seqlen_k,
      sm_scale,
      batch_size,
      num_heads_q,
      num_heads_kv,
      head_size,
      max_blocks_per_seq,
      block_size,
      window_size_left,
      window_size_right,
      false,        // is_varlen
      is_paged,
      is_causal,
      is_local,
      0.0f, 0, 0, nullptr, nullptr, 0, 0,  // dropout & s_dmask defaults
      static_cast<int*>(seqlens_k.data_ptr()),
      cache_batch_idx.has_value()
          ? static_cast<int*>(cache_batch_idx->data_ptr())
          : nullptr,
      cache_leftpad.has_value()
          ? static_cast<int*>(cache_leftpad->data_ptr())
          : nullptr,
      knew.has_value() ? knew->data_ptr() : nullptr,
      vnew.has_value() ? vnew->data_ptr() : nullptr,
      knew.has_value() ? static_cast<int>(knew->size(1)) : 0,
      knew.has_value() ? knew->stride(0) : 0,
      knew.has_value() ? knew->stride(2) : 0,
      knew.has_value() ? knew->stride(1) : 0,
      vnew.has_value() ? vnew->stride(0) : 0,
      vnew.has_value() ? vnew->stride(2) : 0,
      vnew.has_value() ? vnew->stride(1) : 0,
      rotary_cos.has_value() ? rotary_cos->data_ptr() : nullptr,
      rotary_sin.has_value() ? rotary_sin->data_ptr() : nullptr,
      rotary_dim,
      is_rotary_interleaved};

  const CutlassType cuType = aten_to_Cutlass_dtype(query);
  const int h = args.head_size;
  const bool has_rotary = args.rotary_dim > 0 && args.rotary_cos != nullptr &&
                          args.rotary_sin != nullptr;

  if (is_paged) {
    // Paged dispatch requires the K-tile to evenly divide block_size,
    // because each tile is loaded from a single page.
    const int k_tile_n = paged_k_tile_size_n(h, max_seqlen_q);
    TORCH_CHECK(k_tile_n > 0,
                "Unsupported head_size for paged FA2 kvcache: ", h);
    TORCH_CHECK(is_supported_paged_block_size(block_size),
                "Unsupported paged KV block_size=", block_size,
                ". Supported values are positive multiples of 64.");
    TORCH_CHECK(block_size % k_tile_n == 0,
                "Paged KV block_size must be a multiple of the kernel K tile. "
                "Got block_size=", block_size, ", K tile=", k_tile_n);
    if (max_seqlen_q == 1) {
      dispatch_fwd_kvcache_decode_paged_by_head(queue, cuType, args, h, has_rotary);
    } else {
      dispatch_fwd_kvcache_prefill_paged_by_head(queue, cuType, args, h, has_rotary);
    }
  } else {
    if (max_seqlen_q == 1) {
      dispatch_fwd_kvcache_decode_by_head(queue, cuType, args, h, has_rotary);
    } else {
      dispatch_fwd_kvcache_prefill_by_head(queue, cuType, args, h, has_rotary);
    }
  }
}
