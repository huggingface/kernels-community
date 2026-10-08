#undef CUB_WRAPPED_NAMESPACE
#define CUB_WRAPPED_NAMESPACE megablocks

#include "histogram.h"
#include <cstdint>
#include <cub/cub.cuh>

#define CUDA_CALL(code)					    \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    STD_TORCH_CHECK(status == cudaSuccess, err);	    \
  } while (0)

using torch::headeronly::ScalarType;

// hipify's CUDA->HIP symbol table maps cub::DeviceRadixSort / cub::DeviceScan
// but is missing cub::DeviceHistogram, so the namespace is selected explicitly.
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
  #define MEGABLOCKS_CUB hipcub
#else
  #define MEGABLOCKS_CUB cub
#endif

namespace megablocks {

template <typename T>
torch::stable::Tensor cub_histogram(torch::stable::Tensor x, int num_bins) {
  // Allocate the count buffer.
  torch::stable::Tensor out = torch::stable::new_empty(x, {x.size(0), num_bins}, ScalarType::Int);

  // Exit early if there is not work to do.
  if (out.numel() == 0) return out;

  // Get scratchpad size.
  size_t scratchpad_bytes = 0;
  CUDA_CALL(MEGABLOCKS_CUB::DeviceHistogram::HistogramEven(nullptr,
						scratchpad_bytes,
						x.const_data_ptr<T>(),
						out.mutable_data_ptr<int>(),
						/*num_levels=*/num_bins + 1,
						/*lower_level=*/0,
						/*upper_level=*/num_bins,
						/*num_samples=*/int(x.size(1)),
						static_cast<cudaStream_t>(current_stream_ptr(x))));

  // Allocate scratchpad.
  torch::stable::Tensor scratchpad = torch::stable::new_empty(
      x, {static_cast<int64_t>(scratchpad_bytes)}, ScalarType::Char);

  // Run the kernel.
  for (int i = 0; i < x.size(0); ++i) {
    CUDA_CALL(MEGABLOCKS_CUB::DeviceHistogram::HistogramEven(scratchpad.mutable_data_ptr(),
						  scratchpad_bytes,
						  x.const_data_ptr<T>() + x.size(1) * i,
						  out.mutable_data_ptr<int>() + out.size(1) * i,
						  /*num_levels=*/num_bins + 1,
						  /*lower_level=*/0,
						  /*upper_level=*/num_bins,
						  /*num_samples=*/int(x.size(1)),
						  static_cast<cudaStream_t>(current_stream_ptr(x))));
  }
  return out;
}

torch::stable::Tensor histogram(torch::stable::Tensor x, int num_bins) {
  STD_TORCH_CHECK(x.is_cuda());
  STD_TORCH_CHECK(x.dim() == 1 || x.dim() == 2);
  STD_TORCH_CHECK(x.scalar_type() == ScalarType::Short ||
	          x.scalar_type() == ScalarType::Int ||
	          x.scalar_type() == ScalarType::Long);
  bool no_batch = x.dim() == 1;
  if (no_batch) x = torch::stable::view(x, {1, x.numel()});

  if (x.scalar_type() == ScalarType::Short) {
    auto out = cub_histogram<short>(x, num_bins);
    return no_batch ? torch::stable::flatten(out) : out;
  } else if (x.scalar_type() == ScalarType::Int) {
    auto out = cub_histogram<int>(x, num_bins);
    return no_batch ? torch::stable::flatten(out) : out;
  } else {
    STD_TORCH_CHECK(x.scalar_type() == ScalarType::Long);
    auto out = cub_histogram<long>(x, num_bins);
    return no_batch ? torch::stable::flatten(out) : out;
  }
}

} // namespace megablocks

#undef CUDA_CALL
#undef CUB_WRAPPED_NAMESPACE