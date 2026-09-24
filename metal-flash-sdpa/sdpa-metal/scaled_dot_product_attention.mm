#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <torch/torch.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>

// Include the auto-generated header with embedded metallib
#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#else
#error "EMBEDDED_METALLIB_HEADER not defined"
#endif

namespace {

// Must match VarlenAttnParams in scaled_dot_product_attention.metal.
struct VarlenAttnParams {
  int64_t q_strides[2]; // (token, head)
  int64_t k_strides[2];
  int64_t v_strides[2];
  int64_t o_strides[2];
  int32_t H;
  int32_t H_kv;
  float scale;
  float softcap;
};
static_assert(sizeof(VarlenAttnParams) == 80, "VarlenAttnParams layout");

// Function constant indices, see scaled_dot_product_attention.metal.
constexpr NSUInteger kFcDoCausal = 301;
constexpr NSUInteger kFcHasSinks = 302;
constexpr NSUInteger kFcHasSoftcap = 303;

struct TileConfig {
  int bq;
  int bk;
  int wm;
  int wn;
};

// Must match the instantiations in scaled_dot_product_attention.metal.
TileConfig getTileConfig(torch::ScalarType dtype, int64_t head_dim) {
  if (head_dim == 192 || head_dim == 256) {
    // The larger tiles do not fit in threadgroup memory for float32.
    return dtype == torch::kFloat ? TileConfig{16, 8, 2, 1}
                                  : TileConfig{32, 16, 4, 1};
  }
  return TileConfig{32, head_dim >= 128 ? 16 : 32, 4, 1};
}

std::string getKernelDtypeString(torch::ScalarType dtype) {
  switch (dtype) {
  case torch::kFloat:
    return "float32";
  case torch::kHalf:
    return "float16";
  case torch::kBFloat16:
    return "bfloat16";
  default:
    TORCH_CHECK(false, "Unsupported dtype for flash attention: ", dtype);
  }
}

id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

id<MTLComputePipelineState> getPipeline(const std::string &kernel_name,
                                        bool do_causal, bool has_sinks,
                                        bool has_softcap) {
  static std::mutex mutex;
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  static id<MTLLibrary> lib = nil;

  std::string key = kernel_name + (do_causal ? "_causal" : "") +
                    (has_sinks ? "_sinks" : "") +
                    (has_softcap ? "_softcap" : "");

  std::lock_guard<std::mutex> lock(mutex);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }

  id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
  NSError *error = nil;
  if (!lib) {
    lib = EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
    TORCH_CHECK(lib, "Failed to create Metal library from embedded data: ",
                error ? error.localizedDescription.UTF8String : "unknown");
  }

  MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
  [constants setConstantValue:&do_causal
                         type:MTLDataTypeBool
                      atIndex:kFcDoCausal];
  [constants setConstantValue:&has_sinks
                         type:MTLDataTypeBool
                      atIndex:kFcHasSinks];
  [constants setConstantValue:&has_softcap
                         type:MTLDataTypeBool
                      atIndex:kFcHasSoftcap];

  id<MTLFunction> function =
      [lib newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]
                constantValues:constants
                         error:&error];
  TORCH_CHECK(function, "Failed to get Metal function: ", kernel_name,
              " Error: ",
              error ? error.localizedDescription.UTF8String : "unknown");

  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  TORCH_CHECK(pipeline, "Failed to create compute pipeline: ",
              error ? error.localizedDescription.UTF8String : "unknown");

  cache.emplace(key, pipeline);
  return pipeline;
}

void checkAttentionTensor(const torch::Tensor &t, const char *name,
                          torch::ScalarType dtype) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on the MPS device");
  TORCH_CHECK(t.scalar_type() == dtype, name, " must have dtype ", dtype,
              ", got ", t.scalar_type());
  TORCH_CHECK(t.dim() == 3, name,
              " must have shape [tokens, heads, head_dim], got ", t.sizes());
  TORCH_CHECK(t.stride(2) == 1, name, " must be contiguous in head_dim");
  TORCH_CHECK(t.stride(0) <= std::numeric_limits<int32_t>::max(), name,
              " token stride is too large");
}

void checkCuSeqlens(const torch::Tensor &t, const char *name) {
  TORCH_CHECK(t.device().is_mps(), name, " must be on the MPS device");
  TORCH_CHECK(t.scalar_type() == torch::kInt, name,
              " must have dtype torch.int32, got ", t.scalar_type());
  TORCH_CHECK(t.dim() == 1 && t.size(0) >= 1, name,
              " must be a 1D tensor of size batch_size + 1");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

} // namespace

void flash_attention_varlen(
    torch::Tensor &out,          // [total_q_tokens, num_heads, head_size]
    torch::Tensor &query,        // [total_q_tokens, num_heads, head_size]
    torch::Tensor &key,          // [total_k_tokens, num_heads_kv, head_size]
    torch::Tensor &value,        // [total_k_tokens, num_heads_kv, head_size]
    torch::Tensor &cu_seqlens_q, // [batch_size + 1]
    torch::Tensor &cu_seqlens_k, // [batch_size + 1]
    int64_t max_seqlen_q,        // Maximum query sequence length
    int64_t max_seqlen_k,        // Maximum key sequence length
    bool do_causal,              // Whether to use causal mask
    double scale,                // Attention scale
    double softcapping,          // Softcap value, <= 0 disables softcapping
    const std::optional<torch::Tensor> &s_aux) { // [num_heads] sinks
  (void)max_seqlen_k;

  const auto dtype = query.scalar_type();
  checkAttentionTensor(query, "query", dtype);
  checkAttentionTensor(key, "key", dtype);
  checkAttentionTensor(value, "value", dtype);
  checkAttentionTensor(out, "out", dtype);
  checkCuSeqlens(cu_seqlens_q, "cu_seqlens_q");
  checkCuSeqlens(cu_seqlens_k, "cu_seqlens_k");

  const int64_t num_heads = query.size(1);
  const int64_t head_dim = query.size(2);
  const int64_t num_heads_kv = key.size(1);
  const int64_t batch_size = cu_seqlens_q.size(0) - 1;

  // Check if we support this head dimension
  const std::vector<int64_t> supported_head_dims = {32, 64,  72,  80,
                                                    96, 128, 192, 256};
  TORCH_CHECK(std::find(supported_head_dims.begin(), supported_head_dims.end(),
                        head_dim) != supported_head_dims.end(),
              "Head dimension ", head_dim, " is not supported");
  TORCH_CHECK(key.size(2) == head_dim && value.size(2) == head_dim,
              "query, key and value must have the same head_dim");
  TORCH_CHECK(key.sizes() == value.sizes(),
              "key and value must have the same shape");
  TORCH_CHECK(out.sizes() == query.sizes(),
              "out must have the same shape as query");
  TORCH_CHECK(num_heads_kv > 0 && num_heads % num_heads_kv == 0,
              "num_heads (", num_heads,
              ") must be divisible by num_heads_kv (", num_heads_kv, ")");
  TORCH_CHECK(cu_seqlens_k.size(0) == cu_seqlens_q.size(0),
              "cu_seqlens_q and cu_seqlens_k must have the same size");
  TORCH_CHECK(max_seqlen_q >= 0, "max_seqlen_q must be non-negative");

  // The kernel reads sinks as float32. The conversion is cheap: one value
  // per head.
  torch::Tensor sinks;
  if (s_aux.has_value()) {
    TORCH_CHECK(s_aux->device().is_mps(), "s_aux must be on the MPS device");
    TORCH_CHECK(at::isFloatingType(s_aux->scalar_type()),
                "s_aux must have a floating point dtype");
    TORCH_CHECK(s_aux->dim() == 1 && s_aux->size(0) == num_heads,
                "s_aux must have shape [num_heads]");
    sinks = s_aux->to(torch::kFloat).contiguous();
  }
  const bool has_sinks = sinks.defined();

  const TileConfig tiles = getTileConfig(dtype, head_dim);
  const int64_t num_q_blocks = (max_seqlen_q + tiles.bq - 1) / tiles.bq;
  if (batch_size == 0 || num_q_blocks == 0 || num_heads == 0) {
    return;
  }

  const bool has_softcap = softcapping > 0.0;

  VarlenAttnParams params = {};
  params.q_strides[0] = query.stride(0);
  params.q_strides[1] = query.stride(1);
  params.k_strides[0] = key.stride(0);
  params.k_strides[1] = key.stride(1);
  params.v_strides[0] = value.stride(0);
  params.v_strides[1] = value.stride(1);
  params.o_strides[0] = out.stride(0);
  params.o_strides[1] = out.stride(1);
  params.H = static_cast<int32_t>(num_heads);
  params.H_kv = static_cast<int32_t>(num_heads_kv);
  params.scale = static_cast<float>(scale);
  params.softcap = has_softcap ? static_cast<float>(softcapping) : 1.0f;

  const std::string kernel_name =
      "attention_varlen_" + getKernelDtypeString(dtype) + "_bq" +
      std::to_string(tiles.bq) + "_bk" + std::to_string(tiles.bk) + "_bd" +
      std::to_string(head_dim) + "_wm" + std::to_string(tiles.wm) + "_wn" +
      std::to_string(tiles.wn);
  id<MTLComputePipelineState> pipeline =
      getPipeline(kernel_name, do_causal, has_sinks, has_softcap);

  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream, "Failed to get current MPS stream");

  // Nothing in this block may throw: an exception escaping dispatch_sync
  // terminates the process.
  dispatch_sync(stream->queue(), ^{
    // Reuse PyTorch's encoder; the stream owns its lifetime.
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

    [encoder setComputePipelineState:pipeline];

    const torch::Tensor *tensors[] = {&query,  &key,          &value,
                                      &out,    &cu_seqlens_q, &cu_seqlens_k};
    const NSUInteger indices[] = {0, 1, 2, 3, 5, 6};
    for (int i = 0; i < 6; ++i) {
      const torch::Tensor &t = *tensors[i];
      [encoder setBuffer:getMTLBufferStorage(t)
                  offset:t.storage_offset() * t.element_size()
                 atIndex:indices[i]];
    }
    [encoder setBytes:&params length:sizeof(VarlenAttnParams) atIndex:4];
    if (has_sinks) {
      [encoder setBuffer:getMTLBufferStorage(sinks)
                  offset:sinks.storage_offset() * sinks.element_size()
                 atIndex:7];
    }

    MTLSize gridSize = MTLSizeMake(num_q_blocks, num_heads, batch_size);
    MTLSize threadgroupSize = MTLSizeMake(32, tiles.wm, tiles.wn);

    [encoder dispatchThreadgroups:gridSize
            threadsPerThreadgroup:threadgroupSize];
    stream->synchronize(at::mps::SyncType::COMMIT);
  });
}
