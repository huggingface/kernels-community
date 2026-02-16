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

// Set batch metadata: ndim, shape, strides for the first `batch_dims` dims.
// If tensor has no batch dims, passes ndim=1, shape={1}, strides={0}.
void set_batch_dims(id<MTLComputeCommandEncoder> enc,
                    const torch::Tensor& t,
                    int matrix_dims,  // 2 for matmul, 1 for matvec
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
  [enc setBytes:&batch_ndim length:sizeof(int)                        atIndex:idx++];
  [enc setBytes:shape.data()   length:batch_ndim * sizeof(int)        atIndex:idx++];
  [enc setBytes:strides.data() length:batch_ndim * sizeof(int64_t)    atIndex:idx++];
}

// Set only strides for the first `batch_ndim` dimensions.
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
// FP-quantized dispatch: mxfp4_qmm_n
// ===========================================================================

static void mxfp4_qmm_n_dispatch(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& scales,
    torch::Tensor& y) {
  bool batched = x.dim() > 2;
  int K  = static_cast<int>(x.size(-1));
  int N  = static_cast<int>(y.size(-1));
  int M  = static_cast<int>(x.size(-2));

  std::stringstream ss;
  ss << "mxfp4_qmm_n_" << type_str(x.scalar_type())
     << "_gs_32_b_4_batch_" << (batched ? "1" : "0");

  auto pipeline = get_pipeline(ss.str());
  TORCH_CHECK(pipeline, "Kernel not found: ", ss.str());

  @autoreleasepool {
    auto stream = at::mps::getCurrentMPSStream();
    auto encoder = stream->commandEncoder();
    [encoder setComputePipelineState:pipeline];

    int idx = 0;
    set_tensor(encoder, w, idx++);       // 0: w
    set_tensor(encoder, scales, idx++);  // 1: scales
    set_tensor(encoder, x, idx++);       // 2: x
    set_tensor(encoder, y, idx++);       // 3: y

    [encoder setBytes:&K length:sizeof(int) atIndex:idx++]; // 4
    [encoder setBytes:&N length:sizeof(int) atIndex:idx++]; // 5
    [encoder setBytes:&M length:sizeof(int) atIndex:idx++]; // 6

    // Batch metadata for x (7,8,9) and w (10,11,12)
    set_batch_dims(encoder, x, 2, idx);
    set_batch_dims(encoder, w, 2, idx);

    // scales strides (13)
    set_strides(encoder, scales, std::max((int)w.dim() - 2, 1), idx);

    // Grid
    int grid_x = (N + 31) / 32;
    int grid_y = (M + 31) / 32;
    int grid_z = batched ? static_cast<int>(x.numel() / (M * K)) : 1;

    [encoder dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, grid_z)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  }
}

// ===========================================================================
// FP-quantized dispatch: mxfp4_qmv
// ===========================================================================

static void mxfp4_qmv_dispatch(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& scales,
    torch::Tensor& y) {
  // For qmv, w is [N, K_packed], x is [..., K], y is [..., N]
  bool batched = x.dim() > 1;
  int K = static_cast<int>(x.size(-1));
  int N = static_cast<int>(y.size(-1));

  std::stringstream ss;
  ss << "mxfp4_qmv_" << type_str(x.scalar_type())
     << "_gs_32_b_4_batch_" << (batched ? "1" : "0");

  auto pipeline = get_pipeline(ss.str());
  TORCH_CHECK(pipeline, "Kernel not found: ", ss.str());

  @autoreleasepool {
    auto stream = at::mps::getCurrentMPSStream();
    auto encoder = stream->commandEncoder();
    [encoder setComputePipelineState:pipeline];

    int idx = 0;
    set_tensor(encoder, w, idx++);       // 0
    set_tensor(encoder, scales, idx++);  // 1
    set_tensor(encoder, x, idx++);       // 2
    set_tensor(encoder, y, idx++);       // 3

    [encoder setBytes:&K length:sizeof(int) atIndex:idx++]; // 4: in_vec_size
    [encoder setBytes:&N length:sizeof(int) atIndex:idx++]; // 5: out_vec_size

    // Batch metadata (6,7,8 = x; 9,10,11 = w; 12 = s_strides)
    set_batch_dims(encoder, x, 1, idx);
    set_batch_dims(encoder, w, 2, idx);
    int w_batch_ndim = std::max((int)w.dim() - 2, 1);
    set_strides(encoder, scales, w_batch_ndim, idx);

    // Grid: each threadgroup handles results_per_simdgroup=4 output rows,
    // with num_simdgroups=2.
    int rows_per_tg = 8; // num_simdgroups(2) * results_per_simdgroup(4)
    int grid_y = (N + rows_per_tg - 1) / rows_per_tg;
    int grid_z = batched ? static_cast<int>(x.numel() / K) : 1;

    [encoder dispatchThreadgroups:MTLSizeMake(1, grid_y, grid_z)
            threadsPerThreadgroup:MTLSizeMake(32 * 2, 1, 1)]; // 2 simdgroups
  }
}

// ===========================================================================
// Public high-level ops
// ===========================================================================

at::Tensor mxfp4_qmm_n(
    at::Tensor x, at::Tensor w, at::Tensor scales, int64_t output_features) {
  // x: [..., M, K], y: [..., M, N]
  auto out_sizes = x.sizes().vec();
  out_sizes.back() = output_features;
  auto y = torch::zeros(out_sizes, x.options());
  mxfp4_qmm_n_dispatch(x, w, scales, y);
  return y;
}

at::Tensor mxfp4_qmv(
    at::Tensor x, at::Tensor w, at::Tensor scales, int64_t output_features) {
  // x: [..., K], y: [..., N]
  auto out_sizes = x.sizes().vec();
  out_sizes.back() = output_features;
  auto y = torch::zeros(out_sizes, x.options());
  mxfp4_qmv_dispatch(x, w, scales, y);
  return y;
}
