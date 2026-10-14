#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "run_build_k2q_csr(Tensor q2k, Tensor cu_q, Tensor cu_k, "
      "Tensor! row_ptr, Tensor! q_idx, int topk, int blk_kv, "
      "int total_rows, int max_kv_blocks) -> ()");
  ops.impl("run_build_k2q_csr", torch::kCUDA, &run_build_k2q_csr);

  ops.def(
      "run_build_k2q_csr_with_schedule(Tensor q2k, Tensor cu_q, Tensor cu_k, "
      "Tensor! row_ptr, Tensor! q_idx, Tensor! scheduler_metadata, "
      "Tensor! work_count, Tensor! qsplit_idx, Tensor! split_counts, "
      "int topk, int blk_kv, int total_rows, int max_kv_blocks, "
      "int target_q_per_cta, int work_capacity, int max_seqlen_q) -> ()");
  ops.impl("run_build_k2q_csr_with_schedule", torch::kCUDA,
           &run_build_k2q_csr_with_schedule);

  ops.def(
      "build_decode_schedule(Tensor seqused_k, int page_size, int seqlen_q, "
      "int num_qo_heads, int num_kv_heads, int head_dim, int max_seqlen_k, "
      "bool enable_cuda_graph, int max_grid_size, int fixed_split_size, "
      "bool disable_split_kv) -> (Tensor, Tensor, Tensor, Tensor, Tensor, "
      "Tensor, Tensor, Tensor, int[])");
  ops.impl("build_decode_schedule", torch::kCUDA, &build_decode_schedule);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
