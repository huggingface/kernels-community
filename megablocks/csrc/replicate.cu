#undef CUB_WRAPPED_NAMESPACE
#define CUB_WRAPPED_NAMESPACE megablocks

#include "replicate.h"
#include <cstdint>
#include <cub/cub.cuh>
#include <torch/headeronly/util/Half.h>

#define CUDA_CALL(code)					    \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    STD_TORCH_CHECK(status == cudaSuccess, err);	    \
  } while (0)

using torch::headeronly::ScalarType;

// __ldg (read-only cache load) is a CUDA intrinsic that hipify does not
// translate; on AMD fall back to a plain dereference.
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
  #define _LDG(arg) (*(arg))
#else
  #define _LDG(arg) __ldg(arg)
#endif

namespace megablocks {
namespace replicate {

template <typename T, int kThreadsPerBlock>
__global__ void __launch_bounds__(kThreadsPerBlock)
  ReplicateForwardKernel(T * __restrict__ x,
			 int * __restrict__ bins,
			 T * __restrict__ out,
			 int columns) {
  // Offset to this threadblocks batch.
  //
  // x is [batch_size, num_bins]
  // out is [batch_size, columns]
  // bins is [num_bins]
  int batch_idx = blockIdx.y;
  int num_bins = gridDim.x;
  x += batch_idx * num_bins;
  out += batch_idx * columns;

  // Load the start/end for this bin.
  int bin_idx = blockIdx.x;
  int start = 0;
  if (bin_idx > 0) start = _LDG(bins + bin_idx - 1);
  int end = _LDG(bins + bin_idx);

  // Load the value to replicate.
  T value = _LDG((T*)x + bin_idx);

  // Offset to this threadblocks bin and this threads
  // offset within the bin.
  int bin_offset = blockIdx.z * kThreadsPerBlock + threadIdx.x;
  out += start + bin_offset;

  // Replicate the value to the output.
  //
  // TODO(tgale): Vectorize these stores.
  int num_elements = end - start;
  const int kElementsPerLoop = gridDim.z * kThreadsPerBlock;
  T *out_ptr = (T*)out;
  for (; bin_offset < num_elements; num_elements -= kElementsPerLoop) {
    *out_ptr = value;
    out_ptr += kElementsPerLoop;
  }
}

template <typename T>
cudaError_t ReplicateForward(T *x,
			     int batch_size,
			     int num_bins,
			     int *bins,
			     T *out,
			     int columns,
			     cudaStream_t stream) {
  const int kThreadsPerBlock = 64;
  dim3 block_dim(kThreadsPerBlock, 1, 1);
  int group_size = std::ceil((float)columns / (num_bins * kThreadsPerBlock));
  dim3 grid_dim(num_bins, batch_size, group_size);
  ReplicateForwardKernel<T, kThreadsPerBlock><<<
    grid_dim, block_dim, 0, stream>>>(x, bins, out, columns);
  return cudaGetLastError();
}

void cub_segmented_reduce(torch::stable::Tensor grad,
			  torch::stable::Tensor bins,
			  torch::stable::Tensor out,
			  cudaStream_t stream) {
  // Append a zero to the bin boundaries for CUB.
  torch::stable::Tensor offsets = torch::stable::new_empty(bins, {bins.numel() + 1});
  CUDA_CALL(cudaMemsetAsync(offsets.mutable_data_ptr<int>(),
			    0,
			    offsets.numel() * sizeof(int),
			    stream));
  CUDA_CALL(cudaMemcpyAsync(offsets.mutable_data_ptr<int>() + 1,
			    bins.const_data_ptr<int>(),
			    bins.numel() * sizeof(int),
			    cudaMemcpyDeviceToDevice,
			    stream));

  // Get temporary buffer size.
  size_t scratchpad_bytes = 0;
  CUDA_CALL(cub::DeviceSegmentedReduce::Sum(nullptr,
					    scratchpad_bytes,
					    grad.const_data_ptr<c10::Half>(),
					    out.mutable_data_ptr<c10::Half>(),
					    bins.numel(),
					    offsets.const_data_ptr<int>(),
					    offsets.const_data_ptr<int>() + 1,
					    stream));

  // Allocate scratchpad.
  torch::stable::Tensor scratchpad = torch::stable::new_empty(
      grad, {static_cast<int64_t>(scratchpad_bytes)}, ScalarType::Char);

  // Run the kernel for each batch item.
  for (int i = 0; i < grad.size(0); ++i) {
    int num_bins = out.size(1);
    int num_values = grad.size(1);
    CUDA_CALL(cub::DeviceSegmentedReduce::Sum(scratchpad.mutable_data_ptr<int8_t>(),
					      scratchpad_bytes,
					      grad.const_data_ptr<c10::Half>() + i * num_values,
					      out.mutable_data_ptr<c10::Half>() + i * num_bins,
					      bins.numel(),
					      offsets.const_data_ptr<int>(),
					      offsets.const_data_ptr<int>() + 1,
					      stream));
  }
}

} // namespace replicate

void replicate_forward(torch::stable::Tensor x,
		       torch::stable::Tensor bins,
		       torch::stable::Tensor out) {
  // Validate the inputs.
  STD_TORCH_CHECK(x.is_cuda());
  STD_TORCH_CHECK(x.dim() == 2);
  STD_TORCH_CHECK(x.scalar_type() == ScalarType::Half ||
	          x.scalar_type() == ScalarType::Short ||
	          x.scalar_type() == ScalarType::Int);
  STD_TORCH_CHECK(bins.is_cuda());
  STD_TORCH_CHECK(bins.dim() == 1);
  STD_TORCH_CHECK(bins.scalar_type() == ScalarType::Int);
  STD_TORCH_CHECK(out.is_cuda());
  STD_TORCH_CHECK(out.dim() == 2);
  STD_TORCH_CHECK(out.scalar_type() == x.scalar_type());

  // Batch dimensions should match for input/output.
  STD_TORCH_CHECK(x.size(0) == out.size(0));

  // One input for each bin (in each batch).
  STD_TORCH_CHECK(x.size(1) == bins.size(0));

  // Exit early if there is no work to do.
  if (out.numel() == 0) return;

  switch (x.scalar_type()) {
  case ScalarType::Half:
    CUDA_CALL(replicate::ReplicateForward(x.mutable_data_ptr<c10::Half>(),
					  x.size(0),
					  x.size(1),
					  bins.mutable_data_ptr<int>(),
					  out.mutable_data_ptr<c10::Half>(),
					  out.size(1),
					  static_cast<cudaStream_t>(current_stream_ptr(x))));
    return;
  case ScalarType::Int:
    CUDA_CALL(replicate::ReplicateForward(x.mutable_data_ptr<int>(),
					  x.size(0),
					  x.size(1),
					  bins.mutable_data_ptr<int>(),
					  out.mutable_data_ptr<int>(),
					  out.size(1),
					  static_cast<cudaStream_t>(current_stream_ptr(x))));
    return;
  }
  STD_TORCH_CHECK(x.scalar_type() == ScalarType::Short);
  CUDA_CALL(replicate::ReplicateForward(x.mutable_data_ptr<short>(),
					x.size(0),
					x.size(1),
					bins.mutable_data_ptr<int>(),
					out.mutable_data_ptr<short>(),
					out.size(1),
					static_cast<cudaStream_t>(current_stream_ptr(x))));
}

void replicate_backward(torch::stable::Tensor grad,
			torch::stable::Tensor bins,
			torch::stable::Tensor out) {
  // Validate the inputs.
  STD_TORCH_CHECK(grad.is_cuda());
  STD_TORCH_CHECK(grad.dim() == 2);
  STD_TORCH_CHECK(grad.scalar_type() == ScalarType::Half);
  STD_TORCH_CHECK(bins.is_cuda());
  STD_TORCH_CHECK(bins.dim() == 1);
  STD_TORCH_CHECK(bins.scalar_type() == ScalarType::Int);
  STD_TORCH_CHECK(out.is_cuda());
  STD_TORCH_CHECK(out.dim() == 2);
  STD_TORCH_CHECK(out.scalar_type() == ScalarType::Half);

  // Batch dimensions should match for input/output.
  STD_TORCH_CHECK(grad.size(0) == out.size(0));

  // One output for each bin (in each batch).
  STD_TORCH_CHECK(out.size(1) == bins.size(0));

  replicate::cub_segmented_reduce(grad, bins, out, static_cast<cudaStream_t>(current_stream_ptr(grad)));
}

} // namespace megablocks

#undef CUDA_CALL
#undef CUB_WRAPPED_NAMESPACE