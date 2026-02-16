#include <torch/torch.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#endif

namespace {

// ---------------------------------------------------------------------------
// Metal library / pipeline helpers
// ---------------------------------------------------------------------------

static id<MTLLibrary> library = nil;

id<MTLLibrary> get_library() {
  if (library != nil) return library;
  auto device = at::mps::MPSDevice::getInstance()->device();
  NSError* error = nil;

#ifdef EMBEDDED_METALLIB_HEADER
  library = EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
  if (library == nil) {
    std::cerr << "Failed to create Metal library from embedded header" << std::endl;
    if (error) std::cerr << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
  }
#else
  library = [device newDefaultLibrary];
  if (library == nil) {
    std::cerr << "Failed to load Metal library" << std::endl;
    if (error) std::cerr << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
  }
#endif
  return library;
}

id<MTLComputePipelineState> get_pipeline(const std::string& name) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  auto it = cache.find(name);
  if (it != cache.end()) return it->second;

  id<MTLLibrary> lib = get_library();
  if (!lib) return nil;

  id<MTLFunction> func = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
  if (!func) {
    std::cerr << "Kernel not found: " << name << std::endl;
    return nil;
  }

  NSError* error = nil;
  id<MTLComputePipelineState> state =
      [at::mps::MPSDevice::getInstance()->device() newComputePipelineStateWithFunction:func error:&error];
  if (!state) {
    std::cerr << "Failed to create pipeline for " << name << std::endl;
    return nil;
  }
  cache[name] = state;
  return state;
}

std::string type_str(torch::ScalarType type) {
  switch (type) {
    case torch::kFloat32:  return "float";
    case torch::kFloat16:  return "float16_t";
    case torch::kBFloat16: return "bfloat16_t";
    default: throw std::runtime_error("Unsupported dtype");
  }
}

// ---------------------------------------------------------------------------
// Encoder helpers
// ---------------------------------------------------------------------------

void set_tensor(id<MTLComputeCommandEncoder> enc,
                const torch::Tensor& t, int index) {
  [enc setBuffer:at::native::mps::getMTLBufferStorage(t)
          offset:t.storage_offset() * t.element_size()
         atIndex:index];
}

// Batch dims: ndim, shape, strides of the first (tensor.dim() - matrix_dims) dims.
void set_batch_dims(id<MTLComputeCommandEncoder> enc,
                    const torch::Tensor& t,
                    int matrix_dims,
                    int& idx) {
  int total = t.dim();
  int batch_ndim = std::max(total - matrix_dims, 0);
  if (batch_ndim == 0) batch_ndim = 1;

  std::vector<int>     shape(batch_ndim, 1);
  std::vector<int64_t> strides(batch_ndim, 0);
  if (total > matrix_dims) {
    for (int i = 0; i < total - matrix_dims; ++i) {
      shape[i]   = static_cast<int>(t.size(i));
      strides[i] = t.stride(i);
    }
  }
  [enc setBytes:&batch_ndim length:sizeof(int)                     atIndex:idx++];
  [enc setBytes:shape.data()   length:batch_ndim * sizeof(int)     atIndex:idx++];
  [enc setBytes:strides.data() length:batch_ndim * sizeof(int64_t) atIndex:idx++];
}

void set_strides(id<MTLComputeCommandEncoder> enc,
                 const torch::Tensor& t,
                 int batch_ndim,
                 int& idx) {
  std::vector<int64_t> strides(batch_ndim, 0);
  for (int i = 0; i < std::min(batch_ndim, (int)t.dim()); ++i) {
    strides[i] = t.stride(i);
  }
  [enc setBytes:strides.data() length:batch_ndim * sizeof(int64_t) atIndex:idx++];
}

} // namespace

// ===========================================================================
// Affine qmv dispatch
// Kernel: affine_qmv<T, group_size, bits, batched>
// Buffers: w(0), scales(1), biases(2), x(3), y(4),
//          in_vec_size(5), out_vec_size(6),
//          x_batch_ndims(7), x_shape(8), x_strides(9),
//          w_batch_ndims(10), w_shape(11), w_strides(12),
//          s_strides(13), b_strides(14)
// ===========================================================================

static void affine_qmv_dispatch(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& scales,
    const torch::Tensor& biases,
    torch::Tensor& y,
    int group_size,
    int bits) {
  bool batched = x.dim() > 1;
  int K = static_cast<int>(x.size(-1));
  int N = static_cast<int>(y.size(-1));

  std::stringstream ss;
  ss << "affine_qmv_" << type_str(x.scalar_type())
     << "_gs_" << group_size << "_b_" << bits
     << "_batch_" << (batched ? "1" : "0");

  auto pipeline = get_pipeline(ss.str());
  TORCH_CHECK(pipeline, "Kernel not found: ", ss.str());

  @autoreleasepool {
    auto stream = at::mps::getCurrentMPSStream();
    auto encoder = stream->commandEncoder();
    [encoder setComputePipelineState:pipeline];

    int idx = 0;
    set_tensor(encoder, w, idx++);       // 0
    set_tensor(encoder, scales, idx++);  // 1
    set_tensor(encoder, biases, idx++);  // 2
    set_tensor(encoder, x, idx++);       // 3
    set_tensor(encoder, y, idx++);       // 4

    [encoder setBytes:&K length:sizeof(int) atIndex:idx++]; // 5: in_vec_size
    [encoder setBytes:&N length:sizeof(int) atIndex:idx++]; // 6: out_vec_size

    // x batch dims (7,8,9)
    set_batch_dims(encoder, x, 1, idx);
    // w batch dims (10,11,12)
    set_batch_dims(encoder, w, 2, idx);
    // scales strides (13)
    int w_batch_ndim = std::max((int)w.dim() - 2, 1);
    set_strides(encoder, scales, w_batch_ndim, idx);
    // biases strides (14)
    set_strides(encoder, biases, w_batch_ndim, idx);

    // Grid: each TG handles 8 output rows (num_simdgroups=2 * results_per=4)
    int rows_per_tg = 8;
    int grid_y = (N + rows_per_tg - 1) / rows_per_tg;
    int grid_z = batched ? static_cast<int>(x.numel() / K) : 1;

    [encoder dispatchThreadgroups:MTLSizeMake(1, grid_y, grid_z)
            threadsPerThreadgroup:MTLSizeMake(32 * 2, 1, 1)];
  }
}

// ===========================================================================
// Affine qmm_t dispatch (transposed weight)
// Kernel: affine_qmm_t<T, group_size, bits, aligned_N, batched>
// Buffers: w(0), scales(1), biases(2), x(3), y(4),
//          K(5), N(6), M(7),
//          x_batch_ndims(8), x_shape(9), x_strides(10),
//          w_batch_ndims(11), w_shape(12), w_strides(13),
//          s_strides(14), b_strides(15)
// ===========================================================================

static void affine_qmm_t_dispatch(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& scales,
    const torch::Tensor& biases,
    torch::Tensor& y,
    int group_size,
    int bits) {
  bool batched = x.dim() > 2;
  int K = static_cast<int>(x.size(-1));
  int N = static_cast<int>(y.size(-1));
  int M = static_cast<int>(x.size(-2));
  bool aligned_N = (N % 32 == 0);

  std::stringstream ss;
  ss << "affine_qmm_t_" << type_str(x.scalar_type())
     << "_gs_" << group_size << "_b_" << bits
     << "_alN_" << (aligned_N ? "true" : "false")
     << "_batch_" << (batched ? "1" : "0");

  auto pipeline = get_pipeline(ss.str());
  TORCH_CHECK(pipeline, "Kernel not found: ", ss.str());

  @autoreleasepool {
    auto stream = at::mps::getCurrentMPSStream();
    auto encoder = stream->commandEncoder();
    [encoder setComputePipelineState:pipeline];

    int idx = 0;
    set_tensor(encoder, w, idx++);       // 0
    set_tensor(encoder, scales, idx++);  // 1
    set_tensor(encoder, biases, idx++);  // 2
    set_tensor(encoder, x, idx++);       // 3
    set_tensor(encoder, y, idx++);       // 4

    [encoder setBytes:&K length:sizeof(int) atIndex:idx++]; // 5
    [encoder setBytes:&N length:sizeof(int) atIndex:idx++]; // 6
    [encoder setBytes:&M length:sizeof(int) atIndex:idx++]; // 7

    // x batch dims (8,9,10)
    set_batch_dims(encoder, x, 2, idx);
    // w batch dims (11,12,13)
    set_batch_dims(encoder, w, 2, idx);
    // scales strides (14)
    int w_batch_ndim = std::max((int)w.dim() - 2, 1);
    set_strides(encoder, scales, w_batch_ndim, idx);
    // biases strides (15)
    set_strides(encoder, biases, w_batch_ndim, idx);

    // BM=BK=BN=32 (kernel defaults)
    int grid_x = (N + 31) / 32;
    int grid_y = (M + 31) / 32;
    int grid_z = batched ? static_cast<int>(x.numel() / (M * K)) : 1;

    [encoder dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, grid_z)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)]; // WM*WN*SIMD = 2*2*32
  }
}

// ===========================================================================
// Affine qmm_n dispatch (non-transposed weight)
// Same buffer layout as qmm_t but without aligned_N template parameter.
// ===========================================================================

static void affine_qmm_n_dispatch(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& scales,
    const torch::Tensor& biases,
    torch::Tensor& y,
    int group_size,
    int bits) {
  bool batched = x.dim() > 2;
  int K = static_cast<int>(x.size(-1));
  int N = static_cast<int>(y.size(-1));
  int M = static_cast<int>(x.size(-2));

  std::stringstream ss;
  ss << "affine_qmm_n_" << type_str(x.scalar_type())
     << "_gs_" << group_size << "_b_" << bits
     << "_batch_" << (batched ? "1" : "0");

  auto pipeline = get_pipeline(ss.str());
  TORCH_CHECK(pipeline, "Kernel not found: ", ss.str());

  @autoreleasepool {
    auto stream = at::mps::getCurrentMPSStream();
    auto encoder = stream->commandEncoder();
    [encoder setComputePipelineState:pipeline];

    int idx = 0;
    set_tensor(encoder, w, idx++);       // 0
    set_tensor(encoder, scales, idx++);  // 1
    set_tensor(encoder, biases, idx++);  // 2
    set_tensor(encoder, x, idx++);       // 3
    set_tensor(encoder, y, idx++);       // 4

    [encoder setBytes:&K length:sizeof(int) atIndex:idx++]; // 5
    [encoder setBytes:&N length:sizeof(int) atIndex:idx++]; // 6
    [encoder setBytes:&M length:sizeof(int) atIndex:idx++]; // 7

    // x batch dims (8,9,10)
    set_batch_dims(encoder, x, 2, idx);
    // w batch dims (11,12,13)
    set_batch_dims(encoder, w, 2, idx);
    // scales strides (14)
    int w_batch_ndim = std::max((int)w.dim() - 2, 1);
    set_strides(encoder, scales, w_batch_ndim, idx);
    // biases strides (15)
    set_strides(encoder, biases, w_batch_ndim, idx);

    int grid_x = (N + 31) / 32;
    int grid_y = (M + 31) / 32;
    int grid_z = batched ? static_cast<int>(x.numel() / (M * K)) : 1;

    [encoder dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, grid_z)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  }
}

// ===========================================================================
// Public high-level ops
// ===========================================================================

at::Tensor affine_qmv(
    at::Tensor x, at::Tensor w, at::Tensor scales, at::Tensor biases,
    int64_t group_size, int64_t bits, int64_t output_features) {
  auto out_sizes = x.sizes().vec();
  out_sizes.back() = output_features;
  auto y = torch::zeros(out_sizes, x.options());
  affine_qmv_dispatch(x, w, scales, biases, y,
                       static_cast<int>(group_size), static_cast<int>(bits));
  return y;
}

at::Tensor affine_qmm_t(
    at::Tensor x, at::Tensor w, at::Tensor scales, at::Tensor biases,
    int64_t group_size, int64_t bits) {
  // For transposed weight: w = [N, K_packed], N = w.size(0)
  int64_t N = w.size(0);
  auto out_sizes = x.sizes().vec();
  out_sizes.back() = N;
  auto y = torch::zeros(out_sizes, x.options());
  affine_qmm_t_dispatch(x, w, scales, biases, y,
                         static_cast<int>(group_size), static_cast<int>(bits));
  return y;
}

at::Tensor affine_qmm_n(
    at::Tensor x, at::Tensor w, at::Tensor scales, at::Tensor biases,
    int64_t group_size, int64_t bits, int64_t output_features) {
  // For non-transposed weight: w = [K_packed, N_packed]
  // output_features = N (the logical unpacked output dimension)
  auto out_sizes = x.sizes().vec();
  out_sizes.back() = output_features;
  auto y = torch::zeros(out_sizes, x.options());
  affine_qmm_n_dispatch(x, w, scales, biases, y,
                         static_cast<int>(group_size), static_cast<int>(bits));
  return y;
}
