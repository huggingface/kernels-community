#include <torch/torch.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <dlfcn.h>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Metal library / pipeline helpers
// ---------------------------------------------------------------------------

static id<MTLLibrary> library = nil;

id<MTLLibrary> get_library() {
  if (library != nil) return library;
  auto device = at::mps::getMPSDevice();
  NSError* error = nil;

  Dl_info info;
  if (dladdr((void*)get_library, &info)) {
    NSString* path = [NSString stringWithUTF8String:info.dli_fname];
    NSString* dir  = [path stringByDeletingLastPathComponent];
    NSString* libPath = [dir stringByAppendingPathComponent:@"quantization_mlx.metallib"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
      library = [device newLibraryWithFile:libPath error:&error];
    }
  }
  if (library == nil) library = [device newDefaultLibrary];
  if (library == nil) {
    std::cerr << "Failed to load Metal library" << std::endl;
    if (error) std::cerr << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
  }
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
      [at::mps::getMPSDevice() newComputePipelineStateWithFunction:func error:&error];
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
  [enc setBuffer:at::mps::getMTLBuffer(t)
          offset:t.storage_offset() * t.element_size()
         atIndex:index];
}

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
// affine_qmm_t_nax dispatch (transposed weight, NAX GEMM)
// Kernel: affine_qmm_t_nax<T, group_size, bits, aligned_N, batched, BM, BK, BN, WM, WN>
// Name: affine_qmm_t_nax_{type}_gs_{gs}_b_{bits}_bm64_bn64_bk64_wm2_wn2_alN_{aligned}_batch_{batched}
// Buffers: same as affine_qmm_t (0-15)
// ===========================================================================

static void affine_qmm_t_nax_dispatch(
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
  bool aligned_N = (N % 64 == 0);  // BN=64 for NAX

  std::stringstream ss;
  ss << "affine_qmm_t_nax_" << type_str(x.scalar_type())
     << "_gs_" << group_size << "_b_" << bits
     << "_bm64_bn64_bk64_wm2_wn2"
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

    // BM=BN=BK=64 for NAX
    int grid_x = (N + 63) / 64;
    int grid_y = (M + 63) / 64;
    int grid_z = batched ? static_cast<int>(x.numel() / (M * K)) : 1;

    // WM*WN*SIMD = 2*2*32 = 128 threads
    [encoder dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, grid_z)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  }
}

// ===========================================================================
// affine_qmm_n_nax dispatch (non-transposed weight, NAX GEMM)
// ===========================================================================

static void affine_qmm_n_nax_dispatch(
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
  ss << "affine_qmm_n_nax_" << type_str(x.scalar_type())
     << "_gs_" << group_size << "_b_" << bits
     << "_bm64_bn64_bk64_wm2_wn2"
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

    int grid_x = (N + 63) / 64;
    int grid_y = (M + 63) / 64;
    int grid_z = batched ? static_cast<int>(x.numel() / (M * K)) : 1;

    [encoder dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, grid_z)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  }
}

// ===========================================================================
// affine_gather_qmm_rhs_nax dispatch
// Kernel: affine_gather_qmm_rhs_nax<T, group_size, bits, BM, BN, BK, WM, WN, transpose>
// Name:   affine_gather_qmm_rhs_nax_{nt|nn}_{type}_gs_{gs}_b_{bits}_bm_64_bn_64_bk_64_wm_2_wn_2
// Buffers: x(0), w(1), scales(2), biases(3), indices(4), y(5), M(6), N(7), K(8)
// ===========================================================================

static void affine_gather_qmm_rhs_nax_dispatch(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& scales,
    const torch::Tensor& biases,
    const torch::Tensor& indices,
    torch::Tensor& y,
    int group_size,
    int bits,
    bool transpose) {
  int M_val = static_cast<int>(x.size(0));
  int N_val = static_cast<int>(y.size(1));
  int K_val = static_cast<int>(x.size(1));

  std::stringstream ss;
  ss << "affine_gather_qmm_rhs_nax_" << (transpose ? "nt" : "nn")
     << "_" << type_str(x.scalar_type())
     << "_gs_" << group_size << "_b_" << bits
     << "_bm_64_bn_64_bk_64_wm_2_wn_2";

  auto pipeline = get_pipeline(ss.str());
  TORCH_CHECK(pipeline, "Kernel not found: ", ss.str());

  @autoreleasepool {
    auto stream = at::mps::getCurrentMPSStream();
    auto encoder = stream->commandEncoder();
    [encoder setComputePipelineState:pipeline];

    int idx = 0;
    set_tensor(encoder, x, idx++);        // 0
    set_tensor(encoder, w, idx++);        // 1
    set_tensor(encoder, scales, idx++);   // 2
    set_tensor(encoder, biases, idx++);   // 3
    set_tensor(encoder, indices, idx++);  // 4
    set_tensor(encoder, y, idx++);        // 5

    [encoder setBytes:&M_val length:sizeof(int) atIndex:idx++]; // 6
    [encoder setBytes:&N_val length:sizeof(int) atIndex:idx++]; // 7
    [encoder setBytes:&K_val length:sizeof(int) atIndex:idx++]; // 8

    int grid_x = (N_val + 63) / 64;
    int grid_y = (M_val + 63) / 64;

    [encoder dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  }
}

// ===========================================================================
// Public high-level ops
// ===========================================================================

at::Tensor affine_qmm_t_nax(
    at::Tensor x, at::Tensor w, at::Tensor scales, at::Tensor biases,
    int64_t group_size, int64_t bits) {
  // Transposed weight: w = [N, K_packed], N = w.size(0)
  int64_t N = w.size(0);
  auto out_sizes = x.sizes().vec();
  out_sizes.back() = N;
  auto y = torch::zeros(out_sizes, x.options());
  affine_qmm_t_nax_dispatch(x, w, scales, biases, y,
                             static_cast<int>(group_size), static_cast<int>(bits));
  return y;
}

at::Tensor affine_qmm_n_nax(
    at::Tensor x, at::Tensor w, at::Tensor scales, at::Tensor biases,
    int64_t group_size, int64_t bits, int64_t output_features) {
  auto out_sizes = x.sizes().vec();
  out_sizes.back() = output_features;
  auto y = torch::zeros(out_sizes, x.options());
  affine_qmm_n_nax_dispatch(x, w, scales, biases, y,
                             static_cast<int>(group_size), static_cast<int>(bits));
  return y;
}

at::Tensor affine_gather_qmm_rhs_nax(
    at::Tensor x, at::Tensor w, at::Tensor scales, at::Tensor biases,
    at::Tensor indices,
    int64_t group_size, int64_t bits, int64_t output_features,
    bool transpose) {
  // x: [M, K], y: [M, N]
  int64_t M = x.size(0);
  auto y = torch::zeros({M, output_features}, x.options());
  affine_gather_qmm_rhs_nax_dispatch(x, w, scales, biases, indices, y,
                                     static_cast<int>(group_size),
                                     static_cast<int>(bits), transpose);
  return y;
}
