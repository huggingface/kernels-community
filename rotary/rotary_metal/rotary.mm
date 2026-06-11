#include <torch/torch.h>
#include <ATen/ExpandUtils.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Include the auto-generated header with embedded metallib
#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#else
#error "EMBEDDED_METALLIB_HEADER not defined"
#endif

namespace {

constexpr uint32_t MAX_DIMS = 8;
constexpr uint32_t NUM_TENSORS = 6;
// [0] ndim, [1] conj, [2..9] sizes, then strides for x1, x2, cos, sin,
// out1, out2 (MAX_DIMS entries each). Must match rotary.metal.
constexpr uint32_t PARAMS_LEN = 2 + MAX_DIMS + NUM_TENSORS * MAX_DIMS;

inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

const char *kernelNameForDtype(torch::ScalarType dtype) {
  switch (dtype) {
  case torch::kFloat:
    return "rotary_float";
  case torch::kHalf:
    return "rotary_half";
  case torch::kBFloat16:
    return "rotary_bfloat";
  default:
    TORCH_CHECK(false, "Unsupported data type: ", dtype);
  }
}

} // namespace

void _apply_rotary(torch::Tensor const &x1, torch::Tensor const &x2,
                   torch::Tensor const &cos, torch::Tensor const &sin,
                   torch::Tensor &out1, torch::Tensor &out2,
                   bool const conj) {
  // Broadcast cos/sin against x1/x2 the way TensorIterator does on CUDA.
  auto shape = at::infer_size(x1.sizes(), cos.sizes());
  TORCH_CHECK(torch::IntArrayRef(shape) == out1.sizes(),
              "out1 must have the broadcast shape of x1 and cos");
  const auto ndim = shape.size();
  TORCH_CHECK(ndim <= MAX_DIMS, "rotary-metal supports at most ", MAX_DIMS,
              " dimensions, got ", ndim);

  std::array<torch::Tensor, NUM_TENSORS> tensors = {
      x1.expand(shape), x2.expand(shape), cos.expand(shape),
      sin.expand(shape), out1,            out2};

  std::array<uint32_t, PARAMS_LEN> params{};
  params[0] = static_cast<uint32_t>(ndim);
  params[1] = conj ? 1 : 0;
  for (size_t d = 0; d < ndim; ++d) {
    params[2 + d] = static_cast<uint32_t>(shape[d]);
  }
  for (size_t t = 0; t < NUM_TENSORS; ++t) {
    const auto strides = tensors[t].strides();
    for (size_t d = 0; d < ndim; ++d) {
      params[2 + MAX_DIMS + t * MAX_DIMS + d] =
          static_cast<uint32_t>(strides[d]);
    }
  }

  int64_t numel = 1;
  for (auto s : shape) {
    numel *= s;
  }
  if (numel == 0) {
    return;
  }

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    NSError *error = nil;
    id<MTLLibrary> library =
        EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
    TORCH_CHECK(library, "Failed to create Metal library from embedded data: ",
                error.localizedDescription.UTF8String);

    const char *kernelName = kernelNameForDtype(x1.scalar_type());
    id<MTLFunction> function = [library
        newFunctionWithName:[NSString stringWithUTF8String:kernelName]];
    TORCH_CHECK(function, "Failed to create function state object for ",
                kernelName);

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:function error:&error];
    TORCH_CHECK(pso, error.localizedDescription.UTF8String);

    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

    dispatch_sync(serialQueue, ^() {
      id<MTLComputeCommandEncoder> encoder =
          [commandBuffer computeCommandEncoder];
      TORCH_CHECK(encoder, "Failed to create compute command encoder");

      [encoder setComputePipelineState:pso];
      for (uint32_t t = 0; t < NUM_TENSORS; ++t) {
        [encoder setBuffer:getMTLBufferStorage(tensors[t])
                    offset:tensors[t].storage_offset() *
                           tensors[t].element_size()
                   atIndex:t];
      }
      [encoder setBytes:params.data()
                 length:sizeof(params)
                atIndex:NUM_TENSORS];

      MTLSize gridSize = MTLSizeMake(numel, 1, 1);
      NSUInteger threadGroupSize = pso.maxTotalThreadsPerThreadgroup;
      if (threadGroupSize > static_cast<NSUInteger>(numel)) {
        threadGroupSize = numel;
      }
      MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
      [encoder endEncoding];

      torch::mps::commit();
    });
  }
}
