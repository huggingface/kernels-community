#undef CUB_WRAPPED_NAMESPACE
#define CUB_WRAPPED_NAMESPACE megablocks

#include "sort.h"
#include <algorithm>
#include <cstdint>
#include <cub/cub.cuh>

#define CUDA_CALL(code)					    \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    STD_TORCH_CHECK(status == cudaSuccess, err);	    \
  } while (0)

using torch::headeronly::ScalarType;

namespace megablocks {

// Replaces torch::arange, which is not part of the stable ABI.
template <typename T>
__global__ void IotaKernel(T * __restrict__ out, int64_t n) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
       i += (int64_t)blockDim.x * gridDim.x) {
    out[i] = static_cast<T>(i);
  }
}

template <typename T>
void Iota(T *out, int64_t n, cudaStream_t stream) {
  const int kThreadsPerBlock = 256;
  int num_blocks = (int)std::min<int64_t>((n + kThreadsPerBlock - 1) / kThreadsPerBlock, 65535);
  IotaKernel<T><<<num_blocks, kThreadsPerBlock, 0, stream>>>(out, n);
  CUDA_CALL(cudaGetLastError());
}

template <typename T>
void cub_radix_sort(torch::stable::Tensor x,
		    int end_bit,
		    torch::stable::Tensor x_out,
		    torch::stable::Tensor iota_out) {
  // Get iota for values in sort.
  torch::stable::Tensor iota = torch::stable::new_empty(x, {x.numel()});
  Iota(iota.mutable_data_ptr<T>(), x.numel(), static_cast<cudaStream_t>(current_stream_ptr(x)));

  // Get temporary buffer size.
  size_t scratchpad_bytes = 0;
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(nullptr,
  					    scratchpad_bytes,
  					    x.const_data_ptr<T>(),
  					    x_out.mutable_data_ptr<T>(),
  					    iota.const_data_ptr<T>(),
  					    iota_out.mutable_data_ptr<T>(),
  					    x.numel(),
  					    /*begin_bit*/0,
  					    /*end_bit=*/end_bit,
  					    static_cast<cudaStream_t>(current_stream_ptr(x))));

  // Allocate scratchpad.
  torch::stable::Tensor scratchpad = torch::stable::new_empty(
      x, {static_cast<int64_t>(scratchpad_bytes)}, ScalarType::Char);

  // Run the kernel.
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(scratchpad.mutable_data_ptr(),
  					    scratchpad_bytes,
  					    x.const_data_ptr<T>(),
  					    x_out.mutable_data_ptr<T>(),
  					    iota.const_data_ptr<T>(),
  					    iota_out.mutable_data_ptr<T>(),
  					    x.numel(),
  					    /*begin_bit=*/0,
  					    /*end_bit=*/end_bit,
  					    static_cast<cudaStream_t>(current_stream_ptr(x))));
}

void sort(torch::stable::Tensor x,
	  int end_bit,
	  torch::stable::Tensor x_out,
	  torch::stable::Tensor iota_out) {
  STD_TORCH_CHECK(x.is_cuda());
  STD_TORCH_CHECK(x.dim() == 1);
  STD_TORCH_CHECK(x.scalar_type() == ScalarType::Short ||
  	          x.scalar_type() == ScalarType::Int ||
  	          x.scalar_type() == ScalarType::Long);
  STD_TORCH_CHECK(x_out.is_cuda());
  STD_TORCH_CHECK(x_out.dim() == 1);
  STD_TORCH_CHECK(x_out.scalar_type() == x.scalar_type());
  STD_TORCH_CHECK(iota_out.is_cuda());
  STD_TORCH_CHECK(iota_out.dim() == 1);
  STD_TORCH_CHECK(iota_out.scalar_type() == x.scalar_type());

  // Exit early if there is not work to do.
  if (x_out.numel() == 0) return;

  switch (x.scalar_type()) {
  case ScalarType::Short:
    return cub_radix_sort<short>(x, end_bit, x_out, iota_out);
  case ScalarType::Int:
    return cub_radix_sort<int>(x, end_bit, x_out, iota_out);
  }
  STD_TORCH_CHECK(x.scalar_type() == ScalarType::Long);
  return cub_radix_sort<long>(x, end_bit, x_out, iota_out);
}

} // namespace megablocks

#undef CUDA_CALL
#undef CUB_WRAPPED_NAMESPACE