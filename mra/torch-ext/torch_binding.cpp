#include <torch/torch.h>
#include <ATen/ATen.h>
#include <vector>
#include <torch/library.h>

#include "registration.h"
#include "cuda_launch.h"

std::vector<at::Tensor> index_max(
  at::Tensor index_vals,
  at::Tensor indices,
  int64_t A_num_block,
  int64_t B_num_block
) {
  return index_max_kernel(
    index_vals,
    indices,
    static_cast<int>(A_num_block),
    static_cast<int>(B_num_block)
  );
}

at::Tensor mm_to_sparse(
  at::Tensor dense_A,
  at::Tensor dense_B,
  at::Tensor indices
) {
  return mm_to_sparse_kernel(
    dense_A,
    dense_B,
    indices
  );
}

at::Tensor sparse_dense_mm(
  at::Tensor sparse_A,
  at::Tensor indices,
  at::Tensor dense_B,
  int64_t A_num_block
) {
  return sparse_dense_mm_kernel(
    sparse_A,
    indices,
    dense_B,
    static_cast<int>(A_num_block)
  );
}

at::Tensor reduce_sum(
  at::Tensor sparse_A,
  at::Tensor indices,
  int64_t A_num_block,
  int64_t B_num_block
) {
  return reduce_sum_kernel(
    sparse_A,
    indices,
    static_cast<int>(A_num_block),
    static_cast<int>(B_num_block)
  );
}

at::Tensor scatter(
  at::Tensor dense_A,
  at::Tensor indices,
  int64_t B_num_block
) {
  return scatter_kernel(
    dense_A,
    indices,
    static_cast<int>(B_num_block)
  );
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("index_max(Tensor index_vals, Tensor indices, int A_num_block, int B_num_block) -> Tensor[]");
  ops.impl("index_max", torch::kCUDA, &index_max);

  ops.def("mm_to_sparse(Tensor dense_A, Tensor dense_B, Tensor indices) -> Tensor");
  ops.impl("mm_to_sparse", torch::kCUDA, &mm_to_sparse);

  ops.def("sparse_dense_mm(Tensor sparse_A, Tensor indices, Tensor dense_B, int A_num_block) -> Tensor");
  ops.impl("sparse_dense_mm", torch::kCUDA, &sparse_dense_mm);

  ops.def("reduce_sum(Tensor sparse_A, Tensor indices, int A_num_block, int B_num_block) -> Tensor");
  ops.impl("reduce_sum", torch::kCUDA, &reduce_sum);

  ops.def("scatter(Tensor dense_A, Tensor indices, int B_num_block) -> Tensor");
  ops.impl("scatter", torch::kCUDA, &scatter);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME);