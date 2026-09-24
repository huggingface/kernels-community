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

// Must match VarlenAttnParams in varlen_params.h.
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

// Function constants shared by scaled_dot_product_attention.metal and
// sdpa_vector.metal.
struct FunctionConstants {
  bool do_causal;
  bool has_sinks;
  bool has_softcap;
  int blocks; // Number of key blocks of the 2-pass vector kernel.
};
constexpr NSUInteger kFcDoCausal = 301;
constexpr NSUInteger kFcHasSinks = 302;
constexpr NSUInteger kFcHasSoftcap = 303;
constexpr NSUInteger kFcBlocks = 304;

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
                                        FunctionConstants fc) {
  static std::mutex mutex;
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  static id<MTLLibrary> lib = nil;

  std::string key = kernel_name + (fc.do_causal ? "_causal" : "") +
                    (fc.has_sinks ? "_sinks" : "") +
                    (fc.has_softcap ? "_softcap" : "") + "_blocks" +
                    std::to_string(fc.blocks);

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

  // Constants that a function does not use are ignored.
  MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
  [constants setConstantValue:&fc.do_causal
                         type:MTLDataTypeBool
                      atIndex:kFcDoCausal];
  [constants setConstantValue:&fc.has_sinks
                         type:MTLDataTypeBool
                      atIndex:kFcHasSinks];
  [constants setConstantValue:&fc.has_softcap
                         type:MTLDataTypeBool
                      atIndex:kFcHasSoftcap];
  [constants setConstantValue:&fc.blocks type:MTLDataTypeInt atIndex:kFcBlocks];

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

// The last character of the GPU architecture name, e.g. 's' for
// applegpu_g15s. MLX uses it to pick the number of blocks of the 2-pass
// vector kernel: 'p' phone, 'g' base/Pro, 's' Max, 'd' Ultra.
char getArchitectureSuffix() {
  static const char suffix = [] {
    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    NSString *name = device.architecture.name;
    return name.length > 0 ? static_cast<char>([name characterAtIndex:name.length - 1])
                           : 'g';
  }();
  return suffix;
}

// Decode kernels, following MLX's dispatch in
// mlx/backend/metal/scaled_dot_product_attention.cpp.
enum class VectorKernel { None, OnePass, TwoPass, TwoPassGqa };

VectorKernel chooseVectorKernel(int64_t head_dim, int64_t gqa_factor,
                                int64_t max_seqlen_q, int64_t max_seqlen_k) {
  const bool supported_head_dim = head_dim == 32 || head_dim == 64 ||
                                  head_dim == 96 || head_dim == 128 ||
                                  head_dim == 192 || head_dim == 256;
  if (!supported_head_dim || max_seqlen_q > 8 ||
      gqa_factor * max_seqlen_q > 32) {
    return VectorKernel::None;
  }

  const char arch = getArchitectureSuffix();
  const bool two_pass =
      ((arch == 'd' || arch == 's') && max_seqlen_k >= 1024) ||
      (gqa_factor > 1 && max_seqlen_k >= 4096);
  if (!two_pass) {
    return VectorKernel::OnePass;
  }

  const bool gqa_dims =
      (gqa_factor == 8 && (head_dim == 64 || head_dim == 128)) ||
      ((gqa_factor == 12 || gqa_factor == 16) && head_dim == 128);
  if (gqa_dims && max_seqlen_q == 1 && max_seqlen_k >= 8192) {
    return VectorKernel::TwoPassGqa;
  }
  return VectorKernel::TwoPass;
}

int twoPassBlocks(int64_t n_simds, int64_t max_seqlen_k) {
  const char arch = getArchitectureSuffix();
  const int64_t N = max_seqlen_k;
  int blocks;
  if (arch == 's') {
    blocks = 64;
    if (N > 1024 && n_simds > 4) {
      if (N <= 8192) {
        blocks = 128;
      } else if (N <= 32768) {
        blocks = 256;
      } else if (N <= 65536) {
        blocks = 512;
      } else {
        blocks = 1024;
      }
    }
  } else if (arch == 'd') {
    blocks = 128;
    if (n_simds <= 2 && N > 8192) {
      blocks = 256;
    } else if (n_simds >= 6) {
      if (N >= 16384 && N < 65536) {
        blocks = 512;
      } else if (N >= 65536) {
        blocks = 1024;
      }
    }
  } else {
    blocks = n_simds >= 4 ? 64 : 32;
  }
  // All counts are multiples of 32, which the second pass requires.
  return blocks;
}

void setTensor(id<MTLComputeCommandEncoder> encoder, const torch::Tensor &t,
               NSUInteger index) {
  [encoder setBuffer:getMTLBufferStorage(t)
              offset:t.storage_offset() * t.element_size()
             atIndex:index];
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
  const auto dtype = query.scalar_type();
  checkAttentionTensor(query, "query", dtype);
  checkAttentionTensor(key, "key", dtype);
  checkAttentionTensor(value, "value", dtype);
  checkAttentionTensor(out, "out", dtype);
  checkCuSeqlens(cu_seqlens_q, "cu_seqlens_q");
  checkCuSeqlens(cu_seqlens_k, "cu_seqlens_k");

  const int64_t total_q_tokens = query.size(0);
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
  TORCH_CHECK(max_seqlen_q >= 0 && max_seqlen_k >= 0,
              "max_seqlen_q and max_seqlen_k must be non-negative");

  // The kernels read sinks as float32. The conversion is cheap: one value
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

  if (batch_size == 0 || max_seqlen_q == 0 || total_q_tokens == 0 ||
      num_heads == 0) {
    return;
  }

  const bool has_softcap = softcapping > 0.0;
  FunctionConstants fc = {do_causal, sinks.defined(), has_softcap, 0};

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

  const std::string dtype_name = getKernelDtypeString(dtype);
  const int64_t gqa_factor = num_heads / num_heads_kv;

  // Pick the kernels. Decode (a few query tokens per sequence) uses the
  // vector kernels, everything else the tiled steel kernel.
  VectorKernel vector_kernel =
      chooseVectorKernel(head_dim, gqa_factor, max_seqlen_q, max_seqlen_k);
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLComputePipelineState> reduce_pipeline = nil;
  MTLSize grid_size;
  MTLSize threadgroup_size;
  torch::Tensor partials, sums, maxs;

  if (vector_kernel == VectorKernel::OnePass) {
    pipeline = getPipeline("sdpa_vector_varlen_" + dtype_name + "_" +
                               std::to_string(head_dim),
                           fc);
    grid_size = MTLSizeMake(num_heads, max_seqlen_q, batch_size);
    threadgroup_size = MTLSizeMake(1024, 1, 1);
  } else if (vector_kernel != VectorKernel::None) {
    const bool gqa_variant = vector_kernel == VectorKernel::TwoPassGqa;
    fc.blocks = twoPassBlocks(gqa_factor * max_seqlen_q, max_seqlen_k);
    const std::string suffix =
        "varlen_" + dtype_name + "_" + std::to_string(head_dim);
    pipeline = getPipeline(
        gqa_variant ? "sdpa_vector_2pass_1_gqa_" + std::to_string(gqa_factor) +
                          "_" + suffix
                    : "sdpa_vector_2pass_1_" + suffix,
        fc);
    reduce_pipeline = getPipeline("sdpa_vector_2pass_2_" + suffix, fc);
    grid_size = MTLSizeMake(num_heads_kv, batch_size, fc.blocks);
    threadgroup_size =
        MTLSizeMake(32, gqa_factor, gqa_variant ? 1 : max_seqlen_q);

    partials = torch::empty({total_q_tokens, num_heads, fc.blocks, head_dim},
                            query.options());
    sums = torch::empty({total_q_tokens, num_heads, fc.blocks},
                        query.options().dtype(torch::kFloat));
    maxs = torch::empty_like(sums);
  }

  // Register pressure can lower the maximum threadgroup size of the vector
  // kernels, e.g. on older GPUs. Fall back to the steel kernel then.
  if (pipeline != nil &&
      (pipeline.maxTotalThreadsPerThreadgroup <
           threadgroup_size.width * threadgroup_size.height *
               threadgroup_size.depth ||
       (reduce_pipeline != nil &&
        reduce_pipeline.maxTotalThreadsPerThreadgroup < 1024))) {
    pipeline = nil;
    reduce_pipeline = nil;
  }

  if (pipeline == nil) {
    const TileConfig tiles = getTileConfig(dtype, head_dim);
    pipeline = getPipeline(
        "attention_varlen_" + dtype_name + "_bq" + std::to_string(tiles.bq) +
            "_bk" + std::to_string(tiles.bk) + "_bd" +
            std::to_string(head_dim) + "_wm" + std::to_string(tiles.wm) +
            "_wn" + std::to_string(tiles.wn),
        fc);
    grid_size = MTLSizeMake((max_seqlen_q + tiles.bq - 1) / tiles.bq, num_heads,
                            batch_size);
    threadgroup_size = MTLSizeMake(32, tiles.wm, tiles.wn);
  }

  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream, "Failed to get current MPS stream");

  // Nothing in this block may throw: an exception escaping dispatch_sync
  // terminates the process.
  dispatch_sync(stream->queue(), ^{
    // Reuse PyTorch's encoder; the stream owns its lifetime.
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

    [encoder setComputePipelineState:pipeline];
    setTensor(encoder, query, 0);
    setTensor(encoder, key, 1);
    setTensor(encoder, value, 2);
    setTensor(encoder, reduce_pipeline != nil ? partials : out, 3);
    [encoder setBytes:&params length:sizeof(VarlenAttnParams) atIndex:4];
    setTensor(encoder, cu_seqlens_q, 5);
    setTensor(encoder, cu_seqlens_k, 6);
    if (sinks.defined()) {
      setTensor(encoder, sinks, 7);
    }
    if (reduce_pipeline != nil) {
      setTensor(encoder, sums, 8);
      setTensor(encoder, maxs, 9);
    }
    [encoder dispatchThreadgroups:grid_size
            threadsPerThreadgroup:threadgroup_size];

    if (reduce_pipeline != nil) {
      const int32_t num_blocks = fc.blocks;
      [encoder setComputePipelineState:reduce_pipeline];
      setTensor(encoder, partials, 0);
      setTensor(encoder, sums, 1);
      setTensor(encoder, maxs, 2);
      setTensor(encoder, out, 3);
      [encoder setBytes:&params length:sizeof(VarlenAttnParams) atIndex:4];
      [encoder setBytes:&num_blocks length:sizeof(int32_t) atIndex:5];
      [encoder dispatchThreadgroups:MTLSizeMake(num_heads, total_q_tokens, 1)
              threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    }
    stream->synchronize(at::mps::SyncType::COMMIT);
  });
}
