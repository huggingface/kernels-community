#include <torch/torch.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Include the auto-generated header with embedded metallib
#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#else
#error "EMBEDDED_METALLIB_HEADER not defined"
#endif

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

static void dispatchGatedKernel(const std::string &kernelName,
                                torch::Tensor &out,
                                torch::Tensor const &input) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    const uint32_t d = static_cast<uint32_t>(input.size(-1) / 2);
    const uint32_t numTokens =
        static_cast<uint32_t>(input.numel() / input.size(-1));

    if (numTokens == 0 || d == 0) return;

    NSError *error = nil;
    id<MTLLibrary> library =
        EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
    TORCH_CHECK(library, "Failed to create Metal library: ",
                error.localizedDescription.UTF8String);

    std::string kernel_name =
        kernelName + (input.scalar_type() == torch::kFloat ? "_f32" : "_f16");
    id<MTLFunction> function = [library
        newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
    TORCH_CHECK(function, "Failed to create function: ", kernel_name.c_str());

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:function error:&error];
    TORCH_CHECK(pso, "Failed to create pipeline state: ",
                error.localizedDescription.UTF8String);

    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to get command buffer");

    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

    dispatch_sync(serialQueue, ^() {
      id<MTLComputeCommandEncoder> encoder =
          [commandBuffer computeCommandEncoder];
      TORCH_CHECK(encoder, "Failed to create compute encoder");

      [encoder setComputePipelineState:pso];
      [encoder setBuffer:getMTLBufferStorage(out)
                  offset:out.storage_offset() * out.element_size()
                 atIndex:0];
      [encoder setBuffer:getMTLBufferStorage(input)
                  offset:input.storage_offset() * input.element_size()
                 atIndex:1];
      [encoder setBytes:&d length:sizeof(uint32_t) atIndex:2];

      uint32_t numChunks = (d + 7) / 8;
      MTLSize gridSize = MTLSizeMake(numChunks, numTokens, 1);
      NSUInteger threadGroupWidth =
          std::min<NSUInteger>(numChunks, pso.maxTotalThreadsPerThreadgroup);
      MTLSize threadGroupSize = MTLSizeMake(threadGroupWidth, 1, 1);

      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
      [encoder endEncoding];

      torch::mps::commit();
    });
  }
}

static void dispatchGatedKernelWithThreshold(const std::string &kernelName,
                                             torch::Tensor &out,
                                             torch::Tensor const &input,
                                             float threshold) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    const uint32_t d = static_cast<uint32_t>(input.size(-1) / 2);
    const uint32_t numTokens =
        static_cast<uint32_t>(input.numel() / input.size(-1));

    if (numTokens == 0 || d == 0) return;

    NSError *error = nil;
    id<MTLLibrary> library =
        EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
    TORCH_CHECK(library, "Failed to create Metal library: ",
                error.localizedDescription.UTF8String);

    std::string kernel_name =
        kernelName + (input.scalar_type() == torch::kFloat ? "_f32" : "_f16");
    id<MTLFunction> function = [library
        newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
    TORCH_CHECK(function, "Failed to create function: ", kernel_name.c_str());

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:function error:&error];
    TORCH_CHECK(pso, "Failed to create pipeline state: ",
                error.localizedDescription.UTF8String);

    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to get command buffer");

    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

    dispatch_sync(serialQueue, ^() {
      id<MTLComputeCommandEncoder> encoder =
          [commandBuffer computeCommandEncoder];
      TORCH_CHECK(encoder, "Failed to create compute encoder");

      [encoder setComputePipelineState:pso];
      [encoder setBuffer:getMTLBufferStorage(out)
                  offset:out.storage_offset() * out.element_size()
                 atIndex:0];
      [encoder setBuffer:getMTLBufferStorage(input)
                  offset:input.storage_offset() * input.element_size()
                 atIndex:1];
      [encoder setBytes:&d length:sizeof(uint32_t) atIndex:2];
      [encoder setBytes:&threshold length:sizeof(float) atIndex:3];

      uint32_t numChunks = (d + 7) / 8;
      MTLSize gridSize = MTLSizeMake(numChunks, numTokens, 1);
      NSUInteger threadGroupWidth =
          std::min<NSUInteger>(numChunks, pso.maxTotalThreadsPerThreadgroup);
      MTLSize threadGroupSize = MTLSizeMake(threadGroupWidth, 1, 1);

      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
      [encoder endEncoding];

      torch::mps::commit();
    });
  }
}

static void dispatchElementwiseKernel(const std::string &kernelName,
                                      torch::Tensor &out,
                                      torch::Tensor const &input) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    const uint32_t d = static_cast<uint32_t>(input.size(-1));
    const uint32_t numTokens =
        static_cast<uint32_t>(input.numel() / input.size(-1));

    if (numTokens == 0 || d == 0) return;

    NSError *error = nil;
    id<MTLLibrary> library =
        EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
    TORCH_CHECK(library, "Failed to create Metal library: ",
                error.localizedDescription.UTF8String);

    std::string kernel_name =
        kernelName + (input.scalar_type() == torch::kFloat ? "_f32" : "_f16");
    id<MTLFunction> function = [library
        newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
    TORCH_CHECK(function, "Failed to create function: ", kernel_name.c_str());

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:function error:&error];
    TORCH_CHECK(pso, "Failed to create pipeline state: ",
                error.localizedDescription.UTF8String);

    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to get command buffer");

    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

    dispatch_sync(serialQueue, ^() {
      id<MTLComputeCommandEncoder> encoder =
          [commandBuffer computeCommandEncoder];
      TORCH_CHECK(encoder, "Failed to create compute encoder");

      [encoder setComputePipelineState:pso];
      [encoder setBuffer:getMTLBufferStorage(out)
                  offset:out.storage_offset() * out.element_size()
                 atIndex:0];
      [encoder setBuffer:getMTLBufferStorage(input)
                  offset:input.storage_offset() * input.element_size()
                 atIndex:1];
      [encoder setBytes:&d length:sizeof(uint32_t) atIndex:2];

      uint32_t numChunks = (d + 7) / 8;
      MTLSize gridSize = MTLSizeMake(numChunks, numTokens, 1);
      NSUInteger threadGroupWidth =
          std::min<NSUInteger>(numChunks, pso.maxTotalThreadsPerThreadgroup);
      MTLSize threadGroupSize = MTLSizeMake(threadGroupWidth, 1, 1);

      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
      [encoder endEncoding];

      torch::mps::commit();
    });
  }
}

static void checkInputs(torch::Tensor &out, torch::Tensor const &input) {
  TORCH_CHECK(input.device().is_mps(), "input must be a MPS tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.device().is_mps(), "output must be a MPS tensor");
  TORCH_CHECK(out.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kFloat ||
                  input.scalar_type() == torch::kHalf,
              "Unsupported data type: ", input.scalar_type());
  TORCH_CHECK(input.scalar_type() == out.scalar_type(),
              "Input and output must have the same dtype");
}

void silu_and_mul(torch::Tensor &out, torch::Tensor &input) {
  checkInputs(out, input);
  dispatchGatedKernel("silu_and_mul", out, input);
}

void mul_and_silu(torch::Tensor &out, torch::Tensor &input) {
  checkInputs(out, input);
  dispatchGatedKernel("mul_and_silu", out, input);
}

void gelu_and_mul(torch::Tensor &out, torch::Tensor &input) {
  checkInputs(out, input);
  dispatchGatedKernel("gelu_and_mul", out, input);
}

void gelu_tanh_and_mul(torch::Tensor &out, torch::Tensor &input) {
  checkInputs(out, input);
  dispatchGatedKernel("gelu_tanh_and_mul", out, input);
}

void fatrelu_and_mul(torch::Tensor &out, torch::Tensor &input,
                     double threshold) {
  checkInputs(out, input);
  dispatchGatedKernelWithThreshold("fatrelu_and_mul", out, input,
                                   static_cast<float>(threshold));
}

void silu(torch::Tensor &out, torch::Tensor &input) {
  checkInputs(out, input);
  dispatchElementwiseKernel("silu", out, input);
}

void gelu(torch::Tensor &out, torch::Tensor &input) {
  checkInputs(out, input);
  dispatchElementwiseKernel("gelu", out, input);
}

void gelu_tanh(torch::Tensor &out, torch::Tensor &input) {
  checkInputs(out, input);
  dispatchElementwiseKernel("gelu_tanh", out, input);
}

void gelu_new(torch::Tensor &out, torch::Tensor &input) {
  checkInputs(out, input);
  dispatchElementwiseKernel("gelu_new", out, input);
}

void gelu_fast(torch::Tensor &out, torch::Tensor &input) {
  checkInputs(out, input);
  dispatchElementwiseKernel("gelu_fast", out, input);
}

void gelu_quick(torch::Tensor &out, torch::Tensor &input) {
  checkInputs(out, input);
  dispatchElementwiseKernel("gelu_quick", out, input);
}
