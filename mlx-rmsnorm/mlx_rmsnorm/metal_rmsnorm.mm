#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <torch/version.h>
#include <ATen/Functions.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#endif

#if defined(TORCH_VERSION_MAJOR) && defined(TORCH_VERSION_MINOR) && \
    (TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 10))
#define HAS_GLOBAL_DISPATCH_SYNC_WITH_RETHROW 1
#endif

static inline void dispatch_sync_with_rethrow_compat(dispatch_queue_t q, void (^block)()) {
#ifdef HAS_GLOBAL_DISPATCH_SYNC_WITH_RETHROW
  dispatch_sync_with_rethrow(q, block);
#else
  at::native::mps::dispatch_sync_with_rethrow(q, block);
#endif
}

#import "metal_rmsnorm.h"

namespace {

constexpr uint32_t kSimdSize = 32;
constexpr uint32_t kNumReads = 4;
constexpr uint32_t kLoopedLimit = 4096;
constexpr uint32_t kFunctionConstantWeightGradEnabled = 20;


// uses_weight_grad_constant mirrors that optional: true when we pass the flag (backward path), false when we don’t (forward path).
// weight_grad_enabled becomes true only when the backward launcher receives a valid grad_weight

struct PipelineKey {
  std::string name;
  bool weight_grad_enabled;
  bool uses_weight_grad_constant;

  bool operator==(const PipelineKey& other) const {
    return name == other.name &&
        weight_grad_enabled == other.weight_grad_enabled &&
        uses_weight_grad_constant == other.uses_weight_grad_constant;
  }
};

struct PipelineKeyHasher {
  size_t operator()(const PipelineKey& key) const noexcept {
    size_t h = std::hash<std::string>()(key.name);
    h ^= std::hash<bool>()(key.uses_weight_grad_constant) + 0x9e3779b97f4a7c15ULL +
        (h << 6) + (h >> 2);
    if (key.uses_weight_grad_constant) {
      h ^= std::hash<bool>()(key.weight_grad_enabled) + 0x9e3779b97f4a7c15ULL +
          (h << 6) + (h >> 2);
    }
    return h;
  }
};

id<MTLLibrary> loadMetalLibrary(id<MTLDevice> device) {
  static id<MTLLibrary> library = nil;

#ifdef EMBEDDED_METALLIB_HEADER
    NSError* error = nil;
    library = EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
    if (library == nil && error != nil) {
        std::cerr << "failed to create Metal library from embedded header: " << [[error localizedDescription] UTF8String] << "\n";
    }
#endif
  return library;
}

id<MTLComputePipelineState> getPipeline(id<MTLLibrary> library,
                                        const PipelineKey& key) {
  static std::unordered_map<PipelineKey, id<MTLComputePipelineState>, PipelineKeyHasher>
      pipeline_cache;
  static std::mutex cache_mutex;

  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = pipeline_cache.find(key);
    if (it != pipeline_cache.end()) {
      return it->second;
    }
  }

  NSString* func_name = [NSString stringWithUTF8String:key.name.c_str()];
  NSError* function_error = nil;
  id<MTLFunction> function = nil;
  if (key.uses_weight_grad_constant) {  // only for backward path
    MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
    bool weight_grad_enabled = key.weight_grad_enabled;
    [constants setConstantValue:&weight_grad_enabled
                          type:MTLDataTypeBool
                        atIndex:kFunctionConstantWeightGradEnabled];
    function = [library newFunctionWithName:func_name
                             constantValues:constants
                                      error:&function_error];
    [constants release];
  } else {
    function = [library newFunctionWithName:func_name];
  }

  TORCH_CHECK(function,
              "Missing Metal function ", key.name.c_str(),
              " in MLX RMSNorm metallib",
              function_error ? ": " : "",
              function_error ? [[function_error localizedDescription] UTF8String] : "");

  NSError* error = nil;
  id<MTLComputePipelineState> pipeline = [library.device newComputePipelineStateWithFunction:function
                                                                                          error:&error];

  [function release];

  TORCH_CHECK(pipeline,
              "Failed to create compute pipeline for ", key.name.c_str(),
              ": ", error ? [[error localizedDescription] UTF8String] : "unknown error");

  [pipeline retain];
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    pipeline_cache.emplace(key, pipeline);
  }
  return pipeline;
}

std::string dtype_suffix(at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::kFloat:
      return "float32";
    case at::kHalf:
      return "float16";
    case at::kBFloat16:
      return "bfloat16";
    default:
      TORCH_CHECK(false, "Unsupported dtype for MLX RMSNorm kernel: ",
                  c10::toString(scalar_type));
  }
}

size_t feature_size(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.dim() >= 1, "Tensor must have at least 1 dimension");
  return static_cast<size_t>(tensor.size(tensor.dim() - 1));
}

size_t batch_size(const at::Tensor& tensor) {
  if (tensor.dim() == 1) {
    return 1;
  }
  size_t result = 1;
  for (int64_t i = 0; i < tensor.dim() - 1; ++i) {
    result *= static_cast<size_t>(tensor.size(i));
  }
  return result;
}

void dispatch_kernel(const std::string& name,
                     std::optional<bool> weight_grad_enabled,
                     const std::function<void(id<MTLComputeCommandEncoder>,
                                              id<MTLComputePipelineState>)>& encode_and_dispatch) {
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream, "MPS stream is unavailable");

  dispatch_sync_with_rethrow_compat(stream->queue(), ^{
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      id<MTLDevice> device = stream->device();
      id<MTLLibrary> library = loadMetalLibrary(device);

      PipelineKey key{.name = name,
                      .weight_grad_enabled = weight_grad_enabled.value_or(false),
                      .uses_weight_grad_constant = weight_grad_enabled.has_value()};
      id<MTLComputePipelineState> pipeline = getPipeline(library, key);
      [encoder setComputePipelineState:pipeline];
      encode_and_dispatch(encoder, pipeline);
    }
  });

  stream->synchronize(at::mps::SyncType::COMMIT_AND_CONTINUE);
}

MTLSize make_size(NSUInteger value) {
  return MTLSizeMake(value, 1, 1);
}

void ensure_device(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.device().is_mps(),
              "Expected tensor on MPS device but got ", tensor.device());
}

}  // namespace

void launch_forward_kernel(const at::Tensor& input,
                           const at::Tensor& weight,
                           at::Tensor& output,
                           float epsilon) {
  ensure_device(input);
  ensure_device(weight);
  ensure_device(output);

  TORCH_CHECK(input.sizes().equals(output.sizes()),
              "Input and output must have the same shape");
  TORCH_CHECK(input.size(-1) == weight.size(0),
              "Weight length must match normalized dimension");

  const auto suffix = dtype_suffix(input.scalar_type());
  const bool looped = feature_size(input) > kLoopedLimit;
  const std::string kernel_name = (looped ? "rms_looped" : "rms") + suffix;

  at::Tensor input_contig = input.contiguous();
  at::Tensor weight_contig = weight.contiguous();
  at::Tensor output_contig = output.contiguous();

  const uint32_t axis_size = static_cast<uint32_t>(feature_size(input_contig));
  const uint32_t weight_stride = static_cast<uint32_t>(weight_contig.stride(0));
  const size_t batches = batch_size(input_contig);

  dispatch_kernel(kernel_name, std::nullopt,
                  [&](id<MTLComputeCommandEncoder> encoder,
                      id<MTLComputePipelineState> pipeline) {
                    at::native::mps::mtl_setArgs(encoder, input_contig, weight_contig, output_contig,
                                epsilon, axis_size, weight_stride);

                    const size_t max_threads = static_cast<size_t>([pipeline maxTotalThreadsPerThreadgroup]);
                    size_t threadgroup_size = max_threads;
                    if (!looped) {
                      const size_t groups_needed = (axis_size + kNumReads - 1) / kNumReads;
                      const size_t simd_groups = (groups_needed + kSimdSize - 1) / kSimdSize;
                      threadgroup_size = kSimdSize * simd_groups;
                      TORCH_CHECK(threadgroup_size <= max_threads,
                                  "Computed threadgroup size exceeds pipeline limit");
                    }
                    const size_t total_threads = batches * threadgroup_size;
                    [encoder dispatchThreads:make_size(static_cast<NSUInteger>(total_threads))
                                  threadsPerThreadgroup:make_size(static_cast<NSUInteger>(threadgroup_size))];
                  });

  if (!output_contig.is_alias_of(output)) {
    output.copy_(output_contig);
  }
}

void launch_backward_kernel(const at::Tensor& input,
                            const at::Tensor& weight,
                            const at::Tensor& grad_output,
                            at::Tensor& grad_input,
                            at::Tensor* grad_weight,
                            float epsilon) {
  ensure_device(input);
  ensure_device(weight);
  ensure_device(grad_output);
  ensure_device(grad_input);
  TORCH_CHECK(!grad_weight || grad_weight->defined(),
              "grad_weight tensor must be defined when provided");
  if (grad_weight) {
    ensure_device(*grad_weight);
  }

  TORCH_CHECK(input.sizes().equals(grad_output.sizes()),
              "grad_output shape must match input shape");
  TORCH_CHECK(input.sizes().equals(grad_input.sizes()),
              "grad_input shape must match input shape");
  TORCH_CHECK(input.size(-1) == weight.size(0),
              "Weight length must match normalized dimension");
  if (grad_weight) {
    TORCH_CHECK(grad_weight->size(0) == weight.size(0),
                "grad_weight length must match weight length");
  }

  const auto suffix = dtype_suffix(input.scalar_type());
  const bool looped = feature_size(input) > kLoopedLimit;
  const std::string kernel_name = (looped ? "vjp_rms_looped" : "vjp_rms") + suffix;

  at::Tensor input_contig = input.contiguous();
  at::Tensor weight_contig = weight.contiguous();
  at::Tensor grad_output_contig = grad_output.contiguous();
  at::Tensor grad_input_contig = grad_input.contiguous();
  at::Tensor grad_weight_contig;
  if (grad_weight) {
    grad_weight_contig = grad_weight->contiguous();
  } else {
    grad_weight_contig = at::empty({weight.size(0)}, weight.options());
  }

  const uint32_t axis_size = static_cast<uint32_t>(feature_size(input_contig));
  const uint32_t weight_stride = static_cast<uint32_t>(weight_contig.stride(0));
  const size_t batches = batch_size(input_contig);
  const bool weight_grad_enabled = grad_weight != nullptr;

  dispatch_kernel(kernel_name, weight_grad_enabled,
                  [&](id<MTLComputeCommandEncoder> encoder,
                      id<MTLComputePipelineState> pipeline) {
                    at::native::mps::mtl_setArgs(encoder, input_contig, weight_contig, grad_output_contig,
                                grad_input_contig, grad_weight_contig,
                                epsilon, axis_size, weight_stride);

                    const size_t max_threads = static_cast<size_t>([pipeline maxTotalThreadsPerThreadgroup]);
                    size_t threadgroup_size = max_threads;
                    if (!looped) {
                      const size_t groups_needed = (axis_size + kNumReads - 1) / kNumReads;
                      const size_t simd_groups = (groups_needed + kSimdSize - 1) / kSimdSize;
                      threadgroup_size = kSimdSize * simd_groups;
                      TORCH_CHECK(threadgroup_size <= max_threads,
                                  "Computed threadgroup size exceeds pipeline limit");
                    }
                    const size_t total_threads = batches * threadgroup_size;
                    [encoder dispatchThreads:make_size(static_cast<NSUInteger>(total_threads))
                                  threadsPerThreadgroup:make_size(static_cast<NSUInteger>(threadgroup_size))];
                  });

  if (!grad_input_contig.is_alias_of(grad_input)) {
    grad_input.copy_(grad_input_contig);
  }
  if (grad_weight) {
    if (!grad_weight_contig.is_alias_of(*grad_weight)) {
      grad_weight->copy_(grad_weight_contig);
    }
  }
}


