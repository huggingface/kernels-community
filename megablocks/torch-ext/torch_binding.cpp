#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

#include "cumsum.h"
#include "histogram.h"
#include "indices.h"
#include "replicate.h"
#include "sort.h"

#include "grouped_gemm/grouped_gemm.h"

// void exclusive_cumsum(torch::Tensor x, int dim, torch::Tensor out) {
torch::Tensor exclusive_cumsum_wrapper(torch::Tensor x, int64_t dim, torch::Tensor out) {
  megablocks::exclusive_cumsum(x, dim, out);
  return out;
}

// void inclusive_cumsum(torch::Tensor x, int dim, torch::Tensor out) {
torch::Tensor inclusive_cumsum_wrapper(torch::Tensor x, int64_t dim, torch::Tensor out) {
  megablocks::inclusive_cumsum(x, dim, out);
  return out;
}

// torch::Tensor histogram(torch::Tensor x, int num_bins);
torch::Tensor histogram_wrapper(torch::Tensor x, int64_t num_bins) {
  return megablocks::histogram(x, num_bins);
}

// void indices(torch::Tensor padded_bins,
//   int block_size,
//   int output_block_rows,
//   int output_block_columns,
//   torch::Tensor out);
torch::Tensor indices_wrapper(torch::Tensor padded_bins,
                               int64_t block_size,
                               int64_t output_block_rows,
                               int64_t output_block_columns,
                               torch::Tensor out) {
  megablocks::indices(padded_bins, block_size, output_block_rows, output_block_columns, out);
  return out;
}



// Forward pass: replicate values from x according to bin sizes
// void replicate_forward(torch::Tensor x,
//   torch::Tensor bins,
//   torch::Tensor out);
torch::Tensor replicate_forward_wrapper(torch::Tensor x, torch::Tensor bins, torch::Tensor out) {
  megablocks::replicate_forward(x, bins, out);
  return out;
}

// // Backward pass: reduce gradients back to bins using segmented reduction
// void replicate_backward(torch::Tensor grad,
//    torch::Tensor bins,
//    torch::Tensor out);
torch::Tensor replicate_backward_wrapper(torch::Tensor grad, torch::Tensor bins, torch::Tensor out) {
  megablocks::replicate_backward(grad, bins, out);
  return out;
}

// // Public interface function for radix sorting with indices
// void sort(torch::Tensor x,
//   int end_bit,
//   torch::Tensor x_out,
//   torch::Tensor iota_out);
torch::Tensor sort_wrapper(torch::Tensor x, int64_t end_bit, torch::Tensor x_out, torch::Tensor iota_out) {
  megablocks::sort(x, end_bit, x_out, iota_out);
  return x_out;
}

// GroupedGemm operation
torch::Tensor gmm(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor batch_sizes, bool trans_a, bool trans_b) {
  grouped_gemm::GroupedGemm(a, b, c, batch_sizes, trans_a, trans_b);
  return c;
}

// Reference implementation:
//
// m.def("exclusive_cumsum", &exclusive_cumsum, "batched exclusive cumsum.");
// m.def("histogram", &histogram, "even width histogram.");
// m.def("inclusive_cumsum", &inclusive_cumsum, "batched inclusive cumsum");
// m.def("indices", &indices, "indices construction for sparse matrix.");
// m.def("replicate_forward", &replicate_forward, "(fwd) replicate a vector dynamically.");
// m.def("replicate_backward", &replicate_backward, "(bwd) replicate a vector dynamically.");
// m.def("sort", &sort, "key/value sort.");

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("exclusive_cumsum(Tensor x, int dim, Tensor(a!) out) -> Tensor(a!)");
  ops.impl("exclusive_cumsum", torch::kCUDA, &exclusive_cumsum_wrapper);

  ops.def("inclusive_cumsum(Tensor x, int dim, Tensor(a!) out) -> Tensor(a!)");
  ops.impl("inclusive_cumsum", torch::kCUDA, &inclusive_cumsum_wrapper);

  ops.def("histogram(Tensor x, int num_bins) -> Tensor");
  ops.impl("histogram", torch::kCUDA, &histogram_wrapper);

  ops.def("indices(Tensor padded_bins, int block_size, int output_block_rows, int output_block_columns, Tensor(a!) out) -> Tensor(a!)");
  ops.impl("indices", torch::kCUDA, &indices_wrapper);

  ops.def("replicate_forward(Tensor x, Tensor bins, Tensor(a!) out) -> Tensor(a!)");
  ops.impl("replicate_forward", torch::kCUDA, &replicate_forward_wrapper);

  ops.def("replicate_backward(Tensor grad, Tensor bins, Tensor(a!) out) -> Tensor(a!)");
  ops.impl("replicate_backward", torch::kCUDA, &replicate_backward_wrapper);
  
  ops.def("sort(Tensor x, int end_bit, Tensor x_out, Tensor iota_out) -> Tensor(x_out)");
  ops.impl("sort", torch::kCUDA, &sort_wrapper);

  // Register the gmm GroupedGemm operation
  ops.def("gmm(Tensor (a!) a, Tensor (b!) b, Tensor(c!) c, Tensor batch_sizes, bool trans_a, bool trans_b) -> Tensor(c!)");
  ops.impl("gmm", torch::kCUDA, &gmm);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)