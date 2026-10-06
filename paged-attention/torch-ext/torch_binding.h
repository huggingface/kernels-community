#pragma once

#ifdef TORCH_TARGET_VERSION
#include <optional>
#include <string>
#include <vector>

#include <torch/csrc/stable/tensor.h>

using Tensor = torch::stable::Tensor;
#else
#include <torch/torch.h>

using Tensor = torch::Tensor;
#endif

void paged_attention_v1(
    Tensor& out, Tensor& query,
    Tensor& key_cache, Tensor& value_cache,
    int64_t num_kv_heads, double scale, Tensor& block_tables,
    Tensor& seq_lens, int64_t block_size, int64_t max_seq_len,
    const std::optional<Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, Tensor& k_scale,
    Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void paged_attention_v2(
    Tensor& out, Tensor& exp_sums,
    Tensor& max_logits, Tensor& tmp_out,
    Tensor& query, Tensor& key_cache,
    Tensor& value_cache, int64_t num_kv_heads, double scale,
    Tensor& block_tables, Tensor& seq_lens,
    int64_t block_size, int64_t max_seq_len,
    const std::optional<Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, Tensor& k_scale,
    Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void swap_blocks(Tensor& src, Tensor& dst,
                 const Tensor& block_mapping);

// Note: the key_caches and value_caches vectors are constant but
// not the Tensors they contain. The vectors need to be const refs
// in order to satisfy pytorch's C++ operator registration code.
void copy_blocks(std::vector<Tensor> const& key_caches,
                 std::vector<Tensor> const& value_caches,
                 const Tensor& block_mapping);

void reshape_and_cache(Tensor& key,
                       Tensor& value,
                       Tensor& key_cache,
                       Tensor& value_cache,
                       Tensor& slot_mapping,
                       const std::string& kv_cache_dtype,
                       Tensor& k_scale,
                       Tensor& v_scale);

void reshape_and_cache_flash(Tensor& key,
                             Tensor& value,
                             Tensor& key_cache,
                             Tensor& value_cache,
                             Tensor& slot_mapping,
                             const std::string& kv_cache_dtype,
                             Tensor& k_scale,
                             Tensor& v_scale);

int64_t get_device_attribute(int64_t attribute, int64_t device_id);

int64_t get_max_shared_memory_per_block_device_attribute(int64_t device_id);

void convert_fp8(Tensor& dst_cache,
                 Tensor& src_cache, const double scale,
                 const std::string& kv_cache_dtype);
