#include "fmha_fwd.hpp"
#include "fmha_utils.hpp"
#include "torch/all.h"
#include <sycl/sycl.hpp>

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
    batch_size = cu_seqlens_q.numel() - 1;
    num_heads_q = query.size(1);
    num_heads_kv = key_cache.size(1);
    head_size = query.size(2);
    total_seqlen_q = query.size(0);
    total_seqlen_k = key_cache.size(0);
  } else {
    batch_size = query.size(0);
    num_heads_q = query.size(1);
    num_heads_kv = key_cache.size(1);
    head_size = query.size(3);
    max_seqlen_q = query.size(2);
    max_seqlen_k = key_cache.size(2);
  }
  if (is_paged) {
    num_blocks = key_cache.size(0);
    block_size = key_cache.size(1);
    num_heads_kv = key_cache.size(2);
    max_blocks_per_seq = block_table->size(1);
    total_seqlen_k = num_blocks * block_size;
  } else {
    num_blocks = 0;
    block_size = 0;
    num_heads_kv = key_cache.size(1);
    max_blocks_per_seq = 0;
    total_seqlen_k = key_cache.size(0);
  }

  if (is_local) {
    window_size_left = window_size_left == -1 ? max_seqlen_k : window_size_left;
    window_size_right = window_size_right == -1 ? max_seqlen_k : window_size_right;
    if (is_causal) {
      window_size_right = 0;
      is_causal = false;
    }
  }

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

  CutlassType cuType = aten_to_Cutlass_dtype(query);

  const int h = args.head_size;
  
  // Note: is_varlen is always true in this function (cutlass_fmha_fwd_varlen_impl)
  // Only need to dispatch based on head size and paged mode
  // if (h <= 32) {
  //   if (args.is_paged) {
  //     policy_dispatch<prefill_policy_head32, PipelineStages_Prefill, 1, 1>(queue, cuType, args);
  //   } else {
  //     policy_dispatch<prefill_policy_head32, PipelineStages_Prefill, 1, 0>(queue, cuType, args);
  //   }
  // }
  // else if (h <= 64) {
  //   if (args.is_paged) {
  //     policy_dispatch<prefill_policy_head64, PipelineStages_Prefill, 1, 1>(queue, cuType, args);
  //   } else {
  //     policy_dispatch<prefill_policy_head64, PipelineStages_Prefill, 1, 0>(queue, cuType, args);
  //   }
  // }
  // else if (h <= 96) {
  //   if (args.is_paged) {
  //     policy_dispatch<prefill_policy_head96, PipelineStages_Prefill, 1, 1>(queue, cuType, args);
  //   } else {
  //     policy_dispatch<prefill_policy_head96, PipelineStages_Prefill, 1, 0>(queue, cuType, args);
  //   }
  // }
  // else if (h <= 128) {
  //   if (args.is_paged) {
  //     policy_dispatch<prefill_policy_head128, PipelineStages_Prefill, 1, 1>(queue, cuType, args);
  //   } else {
  //     policy_dispatch<prefill_policy_head128, PipelineStages_Prefill, 1, 0>(queue, cuType, args);
  //   }
  // }
  // else if (h <= 160) {
  //   if (args.is_paged) {
  //     policy_dispatch<prefill_policy_head160, PipelineStages_Prefill, 1, 1>(queue, cuType, args);
  //   } else {
  //     policy_dispatch<prefill_policy_head160, PipelineStages_Prefill, 1, 0>(queue, cuType, args);
  //   }
  // }
  // else if (h <= 192) {
  //   if (args.is_paged) {
  //     policy_dispatch<prefill_policy_head192, PipelineStages_Prefill, 1, 1>(queue, cuType, args);
  //   } else {
  //     policy_dispatch<prefill_policy_head192, PipelineStages_Prefill, 1, 0>(queue, cuType, args);
  //   }
  // }
  // else if (h <= 256) {
  //   if (args.is_paged) {
  //     policy_dispatch<prefill_policy_head256, PipelineStages_Prefill, 1, 1>(queue, cuType, args);
  //   } else {
  //     policy_dispatch<prefill_policy_head256, PipelineStages_Prefill, 1, 0>(queue, cuType, args);
  //   }
  // }
  // else {
  //   throw std::runtime_error("Unsupported head_size: " + std::to_string(h) + ". Max supported head_size is 256");
  // }
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
    void* rng_state) {
  int batch_size = query.size(0);
  int max_seqlen_q = query.size(1);
  int num_heads_q = query.size(2);
  int head_size = query.size(3);

  int max_seqlen_k = key.size(1);
  int num_heads_kv = key.size(2);

  int total_seqlen_q = batch_size * max_seqlen_q;
  int total_seqlen_k = batch_size * max_seqlen_k;

  if (is_local) {
    window_size_left = window_size_left == -1 ? max_seqlen_k : window_size_left;
    window_size_right = window_size_right == -1 ? max_seqlen_k : window_size_right;
    if (is_causal) {
      window_size_right = 0;
      is_causal = false;
    }
  }

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
      rng_state};

  CutlassType cuType = aten_to_Cutlass_dtype(query);
  const int h = args.head_size;

  if (max_seqlen_q == 1) {
    if (h <= 32) {
      policy_dispatch<decode_policy_head32, PipelineStages_Decode, 0, 0>(queue, cuType, args);
    }
    // else if (h <= 64) {
    //   policy_dispatch<decode_policy_head64, PipelineStages_Decode, 0, 0>(queue, cuType, args);
    // }
    // else if (h <= 96) {
    //   policy_dispatch<decode_policy_head96, PipelineStages_Decode, 0, 0>(queue, cuType, args);
    // }
    // else if (h <= 128) {
    //   policy_dispatch<decode_policy_head128, PipelineStages_Decode, 0, 0>(queue, cuType, args);
    // }
    // else if (h <= 160) {
    //   policy_dispatch<decode_policy_head160, PipelineStages_Decode, 0, 0>(queue, cuType, args);
    // }
    // else if (h <= 192) {
    //   policy_dispatch<decode_policy_head192, PipelineStages_Decode, 0, 0>(queue, cuType, args);
    // }
    // else if (h <= 256) {
    //   policy_dispatch<decode_policy_head256, PipelineStages_Decode, 0, 0>(queue, cuType, args);
    // }
    else {
      throw std::runtime_error("Unsupported head_size: " + std::to_string(h) + ". Max supported head_size is 256");
    }
  } else {
    if (h <= 32) {
      policy_dispatch<prefill_policy_head32, PipelineStages_Prefill, 0, 0>(queue, cuType, args);
    }
    // else if (h <= 64) {
    //   policy_dispatch<prefill_policy_head64, PipelineStages_Prefill, 0, 0>(queue, cuType, args);
    // }
    // else if (h <= 96) {
    //   policy_dispatch<prefill_policy_head96, PipelineStages_Prefill, 0, 0>(queue, cuType, args);
    // }
    // else if (h <= 128) {
    //   policy_dispatch<prefill_policy_head128, PipelineStages_Prefill, 0, 0>(queue, cuType, args);
    // }
    // else if (h <= 160) {
    //   policy_dispatch<prefill_policy_head160, PipelineStages_Prefill, 0, 0>(queue, cuType, args);
    // }
    // else if (h <= 192) {
    //   policy_dispatch<prefill_policy_head192, PipelineStages_Prefill, 0, 0>(queue, cuType, args);
    // }
    // else if (h <= 256) {
    //   policy_dispatch<prefill_policy_head256, PipelineStages_Prefill, 0, 0>(queue, cuType, args);
    // }
    else {
      throw std::runtime_error("Unsupported head_size: " + std::to_string(h) + ". Max supported head_size is 256");
    }
  }
}
