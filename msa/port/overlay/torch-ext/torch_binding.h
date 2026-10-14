#pragma once

#include <torch/torch.h>

#include <tuple>
#include <vector>

// q2k -> k2q CSR build (sorted within row). Fills row_ptr and q_idx in place.
void run_build_k2q_csr(
    torch::Tensor q2k,
    torch::Tensor cu_q,
    torch::Tensor cu_k,
    torch::Tensor row_ptr,
    torch::Tensor q_idx,
    int64_t topk,
    int64_t blk_kv,
    int64_t total_rows,
    int64_t max_kv_blocks);

// q2k -> k2q CSR build with fused attention schedule metadata. Fills
// row_ptr, q_idx, scheduler_metadata, work_count, qsplit_idx and
// split_counts in place.
void run_build_k2q_csr_with_schedule(
    torch::Tensor q2k,
    torch::Tensor cu_q,
    torch::Tensor cu_k,
    torch::Tensor row_ptr,
    torch::Tensor q_idx,
    torch::Tensor scheduler_metadata,
    torch::Tensor work_count,
    torch::Tensor qsplit_idx,
    torch::Tensor split_counts,
    int64_t topk,
    int64_t blk_kv,
    int64_t total_rows,
    int64_t max_kv_blocks,
    int64_t target_q_per_cta,
    int64_t work_capacity,
    int64_t max_seqlen_q);

// (request_indices, qo_tile_indices, kv_tile_indices, block_valid_mask,
//  split_counts, kv_pages, merge_indptr, o_indptr, scalar summary)
using BuildDecodeScheduleResult = std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    std::vector<int64_t>>;

// Build the paged decode split-KV schedule on GPU.
BuildDecodeScheduleResult build_decode_schedule(
    torch::Tensor seqused_k,
    int64_t page_size,
    int64_t seqlen_q,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t max_seqlen_k,
    bool enable_cuda_graph,
    int64_t max_grid_size_override,
    int64_t fixed_split_size,
    bool disable_split_kv);
