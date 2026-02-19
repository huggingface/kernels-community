// bitsandbytes MPS Metal kernels - ObjC++ dispatch
// Interfaces between PyTorch MPS tensors and Metal compute kernels.
// Uses the same dispatch pattern as kernels-community/activation, with
// get_command_buffer() moved inside dispatch_sync to avoid race conditions
// during model loading.

#include <torch/torch.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <unordered_map>

#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#endif

// ============================================================================
// Metal helpers
// ============================================================================

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& t) {
  return __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
}

namespace {

static id<MTLLibrary> library = nil;

id<MTLLibrary> get_library() {
  if (library != nil)
    return library;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  NSError* error = nil;

#ifdef EMBEDDED_METALLIB_HEADER
  library = EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
  if (library == nil) {
    std::cerr << "Failed to create Metal library from embedded header"
              << std::endl;
    if (error)
      std::cerr << "Error: " << [[error localizedDescription] UTF8String]
                << std::endl;
  }
#else
  library = [device newDefaultLibrary];
  if (library == nil) {
    std::cerr << "Failed to load Metal library" << std::endl;
    if (error)
      std::cerr << "Error: " << [[error localizedDescription] UTF8String]
                << std::endl;
  }
#endif
  return library;
}

id<MTLComputePipelineState> get_pipeline(const std::string& name) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  auto it = cache.find(name);
  if (it != cache.end())
    return it->second;

  id<MTLLibrary> lib = get_library();
  if (!lib)
    return nil;

  id<MTLFunction> func =
      [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
  if (!func) {
    std::cerr << "Kernel not found: " << name << std::endl;
    return nil;
  }

  NSError* error = nil;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLComputePipelineState> state =
      [device newComputePipelineStateWithFunction:func error:&error];
  if (!state) {
    std::cerr << "Failed to create pipeline for " << name << std::endl;
    return nil;
  }
  cache[name] = state;
  return state;
}

std::string type_str(torch::ScalarType type) {
  switch (type) {
    case torch::kFloat32:
      return "float";
    case torch::kFloat16:
      return "half";
    case torch::kBFloat16:
      return "bfloat16_t";
    default:
      throw std::runtime_error("Unsupported dtype for BnB MPS kernels");
  }
}

void set_tensor(
    id<MTLComputeCommandEncoder> enc,
    const torch::Tensor& t,
    int index) {
  [enc setBuffer:getMTLBufferStorage(t)
          offset:t.storage_offset() * t.element_size()
         atIndex:index];
}

} // namespace

// ============================================================================
// Public API: quantize_4bit
// ============================================================================

std::tuple<at::Tensor, at::Tensor> bnb_quantize_4bit(
    at::Tensor input,
    int64_t blocksize,
    int64_t quant_type) {
  TORCH_CHECK(input.is_mps(), "Input must be on MPS device");
  TORCH_CHECK(
      blocksize == 64 || blocksize == 128 || blocksize == 256 || blocksize == 512,
      "Only blocksize 64, 128, 256, and 512 are supported");
  TORCH_CHECK(
      quant_type == 1 || quant_type == 2,
      "quant_type must be 1 (FP4) or 2 (NF4)");

  int n = static_cast<int>(input.numel());
  int num_blocks =
      (n + static_cast<int>(blocksize) - 1) / static_cast<int>(blocksize);
  int packed_size = (n + 1) / 2;

  auto absmax =
      torch::empty({num_blocks}, input.options().dtype(torch::kFloat32));
  auto packed =
      torch::empty({packed_size}, input.options().dtype(torch::kUInt8));

  std::stringstream ss;
  ss << "bnb_quantize_blockwise_" << type_str(input.scalar_type()) << "_bs_"
     << blocksize << "_qt_" << quant_type;

  auto pipeline = get_pipeline(ss.str());
  TORCH_CHECK(pipeline, "Kernel not found: ", ss.str());

  @autoreleasepool {
    dispatch_sync(torch::mps::get_dispatch_queue(), ^{
      @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer =
            torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to get MPS command buffer");

        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        TORCH_CHECK(encoder, "Failed to create compute encoder");

        [encoder setComputePipelineState:pipeline];

        int idx = 0;
        set_tensor(encoder, input, idx++);
        set_tensor(encoder, absmax, idx++);
        set_tensor(encoder, packed, idx++);
        [encoder setBytes:&n length:sizeof(int) atIndex:idx++];

        NSUInteger threads_per_tg = pipeline.threadExecutionWidth;
        MTLSize grid = MTLSizeMake(num_blocks, 1, 1);
        MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
        [encoder endEncoding];

        torch::mps::commit();
      }
    });
  }

  return std::make_tuple(packed, absmax);
}

// ============================================================================
// Public API: dequantize_blockwise
// ============================================================================

at::Tensor bnb_dequantize_4bit(
    at::Tensor packed,
    at::Tensor absmax,
    int64_t blocksize,
    int64_t quant_type,
    int64_t numel,
    torch::ScalarType output_dtype) {
  TORCH_CHECK(packed.is_mps(), "packed must be on MPS device");
  TORCH_CHECK(absmax.is_mps(), "absmax must be on MPS device");
  TORCH_CHECK(
      blocksize == 64 || blocksize == 128 || blocksize == 256 || blocksize == 512,
      "Only blocksize 64, 128, 256, and 512 are supported");

  int n = static_cast<int>(numel);
  int num_blocks =
      (n + static_cast<int>(blocksize) - 1) / static_cast<int>(blocksize);

  auto output = torch::empty({n}, packed.options().dtype(output_dtype));

  std::stringstream ss;
  ss << "bnb_dequantize_blockwise_" << type_str(output_dtype) << "_bs_"
     << blocksize << "_qt_" << quant_type;

  auto pipeline = get_pipeline(ss.str());
  TORCH_CHECK(pipeline, "Kernel not found: ", ss.str());

  @autoreleasepool {
    dispatch_sync(torch::mps::get_dispatch_queue(), ^{
      @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer =
            torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to get MPS command buffer");

        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        TORCH_CHECK(encoder, "Failed to create compute encoder");

        [encoder setComputePipelineState:pipeline];

        int idx = 0;
        set_tensor(encoder, packed, idx++);
        set_tensor(encoder, absmax, idx++);
        set_tensor(encoder, output, idx++);
        [encoder setBytes:&n length:sizeof(int) atIndex:idx++];

        NSUInteger max_tg = pipeline.maxTotalThreadsPerThreadgroup;
        NSUInteger desired = (blocksize + 1) / 2;
        NSUInteger tg_size =
            std::min(max_tg, std::max(static_cast<NSUInteger>(1), desired));
        if (tg_size < pipeline.threadExecutionWidth) {
          tg_size = std::min(pipeline.threadExecutionWidth, max_tg);
        }

        MTLSize grid = MTLSizeMake(tg_size * num_blocks, 1, 1);
        MTLSize tg = MTLSizeMake(tg_size, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
        [encoder endEncoding];

        torch::mps::commit();
      }
    });
  }

  return output;
}

// ============================================================================
// Public API: GEMV (matrix-vector multiply)
// y = dequant(W) @ x
// ============================================================================

at::Tensor bnb_gemv_4bit(
    at::Tensor x,
    at::Tensor w,
    at::Tensor absmax,
    int64_t blocksize,
    int64_t quant_type,
    int64_t output_features) {
  TORCH_CHECK(
      x.is_mps() && w.is_mps() && absmax.is_mps(),
      "All tensors must be on MPS device");

  int K = static_cast<int>(x.size(-1));
  int N = static_cast<int>(output_features);

  auto out_sizes = x.sizes().vec();
  out_sizes.back() = N;
  auto y = torch::zeros(out_sizes, x.options());

  std::stringstream ss;
  ss << "bnb_qmv_" << type_str(x.scalar_type()) << "_bs_" << blocksize
     << "_qt_" << quant_type;

  auto pipeline = get_pipeline(ss.str());
  TORCH_CHECK(pipeline, "Kernel not found: ", ss.str());

  @autoreleasepool {
    dispatch_sync(torch::mps::get_dispatch_queue(), ^{
      @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer =
            torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to get MPS command buffer");

        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        TORCH_CHECK(encoder, "Failed to create compute encoder");

        [encoder setComputePipelineState:pipeline];

        int idx = 0;
        set_tensor(encoder, w, idx++);
        set_tensor(encoder, absmax, idx++);
        set_tensor(encoder, x, idx++);
        set_tensor(encoder, y, idx++);
        [encoder setBytes:&K length:sizeof(int) atIndex:idx++];
        [encoder setBytes:&N length:sizeof(int) atIndex:idx++];

        int rows_per_tg = 8;
        int grid_y = (N + rows_per_tg - 1) / rows_per_tg;

        [encoder dispatchThreadgroups:MTLSizeMake(1, grid_y, 1)
                threadsPerThreadgroup:MTLSizeMake(32 * 2, 1, 1)];
        [encoder endEncoding];

        torch::mps::commit();
      }
    });
  }

  return y;
}

// ============================================================================
// Public API: GEMM (matrix-matrix multiply with transposed weight)
// Y = X @ dequant(W).T
// ============================================================================

at::Tensor bnb_gemm_4bit(
    at::Tensor x,
    at::Tensor w,
    at::Tensor absmax,
    int64_t blocksize,
    int64_t quant_type,
    int64_t output_features) {
  TORCH_CHECK(
      x.is_mps() && w.is_mps() && absmax.is_mps(),
      "All tensors must be on MPS device");
  TORCH_CHECK(x.dim() >= 2, "Input must be at least 2D for GEMM");

  int K = static_cast<int>(x.size(-1));
  int M = static_cast<int>(x.size(-2));
  int N = static_cast<int>(output_features);

  auto out_sizes = x.sizes().vec();
  out_sizes.back() = N;
  auto y = torch::zeros(out_sizes, x.options());

  std::stringstream ss;
  ss << "bnb_qmm_t_" << type_str(x.scalar_type()) << "_bs_" << blocksize
     << "_qt_" << quant_type;

  auto pipeline = get_pipeline(ss.str());
  TORCH_CHECK(pipeline, "Kernel not found: ", ss.str());

  @autoreleasepool {
    dispatch_sync(torch::mps::get_dispatch_queue(), ^{
      @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer =
            torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to get MPS command buffer");

        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        TORCH_CHECK(encoder, "Failed to create compute encoder");

        [encoder setComputePipelineState:pipeline];

        int idx = 0;
        set_tensor(encoder, w, idx++);
        set_tensor(encoder, absmax, idx++);
        set_tensor(encoder, x, idx++);
        set_tensor(encoder, y, idx++);
        [encoder setBytes:&K length:sizeof(int) atIndex:idx++];
        [encoder setBytes:&N length:sizeof(int) atIndex:idx++];
        [encoder setBytes:&M length:sizeof(int) atIndex:idx++];

        int grid_x = (N + 31) / 32;
        int grid_y = (M + 31) / 32;

        [encoder dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];

        torch::mps::commit();
      }
    });
  }

  return y;
}
