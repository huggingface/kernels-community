/* Metal side: launch MLX's quantized kernels on torch's own stream.
 *
 * The kernels are upstream's, compiled as they ship (vendor/UPSTREAM pins the revision). What lives
 * here is the host side, which cannot be vendored: MLX's launchers are written against its own
 * array and device types. So this file transcribes them, keeping upstream's choices of kernel,
 * grid and buffer layout:
 *
 *   - QuantizedMatmul::eval_gpu    -> quantized_matmul  (the qmv, qvm and qmm families, split-K, NAX)
 *   - GatherQMM::eval_gpu          -> gather_qmm        (gather_qmv/qvm/qmm, the sorted rhs path)
 *   - fast::Quantize::eval_gpu     -> quantize / dequantize
 *   - strided_reduce_general_dispatch (reduce.cpp), for the split-K sums
 *
 * plus the op-level checks and defaults from mlx/ops.cpp. tests/test_vendor_drift.py checks the
 * transcribed pieces against vendor/mlx.
 *
 * Every launcher runs twice: a planning pass with no encoder, which allocates the temporaries the
 * launch needs (outside torch's stream queue, where allocation is safe) and records kernel names,
 * then the encoding pass, which binds and dispatches inside the queue.
 */

#import <Metal/Metal.h>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "torch_binding.h"

#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#endif

#define CHECK(cond, ...) TORCH_CHECK(cond, "mlx-quantization-metal-kernels: ", __VA_ARGS__)

// Just enough of MLX's `metal::Device` for the helpers below to be upstream's code verbatim.
namespace metal {

struct Device {
  std::string arch;
  int arch_gen;
  const std::string &get_architecture() const { return arch; }
  int get_architecture_gen() const { return arch_gen; }
};

}  // namespace metal

namespace {

// Upstream helpers from outside the vendored files: mlx/backend/common/quantized.h and mlx/utils.h.
inline constexpr short get_pack_factor(int bits, int wsize = 8) {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

inline bool is_power_of_2(int n) {
  return ((n & (n - 1)) == 0) && n != 0;
}

// ---------------------------------------------------------------------------------------------------
// Verbatim from vendor/mlx/backend/metal/quantized.cpp (test_vendor_drift.py compares them).
// ---------------------------------------------------------------------------------------------------

inline int get_qmv_batch_limit(int D, int O, metal::Device& d) {
  auto arch_size = d.get_architecture().back();
  auto arch_gen = d.get_architecture_gen();
  if (arch_gen >= 17 && arch_size != 'd') {
    if (D <= 2048 && O <= 2048) {
      return 33;
    } else if (D <= 4096 && O <= 4096) {
      return 25;
    } else {
      return 13;
    }
  } else if (arch_gen >= 15 && arch_size != 'd') {
    if (D <= 2048 && O <= 2048) {
      return 13;
    } else if (D <= 4096 && O <= 4096) {
      return 15;
    } else {
      return 13;
    }
  } else if (arch_gen >= 13) {
    switch (arch_size) {
      case 'd':
        if (D <= 2048 && O <= 2048) {
          return 32;
        } else if (D <= 4096 && O <= 4096) {
          return 18;
        } else {
          return 12;
        }
      default:
        if (D <= 2048 && O <= 2048) {
          return 14;
        } else if (D <= 4096 && O <= 4096) {
          return 10;
        } else {
          return 6;
        }
    }
  } else {
    switch (arch_size) {
      case 'd':
        if (D <= 2048 && O <= 2048) {
          return 32;
        } else if (D <= 4096 && O <= 4096) {
          return 18;
        } else {
          return 12;
        }
      default:
        if (D <= 2048 && O <= 2048) {
          return 18;
        } else if (D <= 4096 && O <= 4096) {
          return 12;
        } else {
          return 10;
        }
    }
  }
}

inline int qmv_fast_k_alignment(int bits) {
  return get_pack_factor(bits, 32) * (bits == 2 ? 1 : 2) * 32;
}

// affine qmv_wide only beats qmv on gen-15+; fp benefits on every gen.
inline bool use_qmv_wide(const std::string& mode, metal::Device& d) {
  return mode != "affine" || d.get_architecture_gen() >= 15;
}


// ---------------------------------------------------------------------------------------------------
// Device, library and pipelines
// ---------------------------------------------------------------------------------------------------

// Upstream's Device constructor: the arch string (e.g. "applegpu_g14s") is MLX_METAL_GPU_ARCH if set,
// else the GPU's own, and the generation is the two digits before the size letter. Read per call
// rather than once, so a test can steer every branch of the dispatch from one process.
metal::Device device_info() {
  static const std::string native = [] {
    id<MTLDevice> dev = at::mps::MPSDevice::getInstance()->device();
    return std::string(dev.architecture.name.UTF8String);
  }();
  const char *env = std::getenv("MLX_METAL_GPU_ARCH");
  std::string arch = (env && *env) ? std::string(env) : native;
  int gen = 0;
  if (arch.size() >= 3) {
    gen = (arch[arch.size() - 3] - '0') * 10 + (arch[arch.size() - 2] - '0');
  }
  return {arch, gen};
}

// Upstream's metal::is_nax_available (device.cpp).
bool is_nax_available() {
  bool can_use_nax = false;
  if (@available(macOS 26.2, *)) {
    can_use_nax = true;
  }
  metal::Device d = device_info();
  auto arch = d.get_architecture().back();
  auto gen = d.get_architecture_gen();
  can_use_nax &= gen >= (arch == 'p' ? 18 : 17);
  return can_use_nax;
}

// Upstream's env::enable_tf32: MLX_ENABLE_TF32, on unless set to 0.
bool enable_tf32() {
  const char *v = std::getenv("MLX_ENABLE_TF32");
  return !(v && *v) || std::atoi(v) != 0;
}

id<MTLLibrary> library() {
  static id<MTLLibrary> lib = [] {
    id<MTLDevice> dev = at::mps::MPSDevice::getInstance()->device();
    NSError *error = nil;
#ifdef EMBEDDED_METALLIB_HEADER
    id<MTLLibrary> l = EMBEDDED_METALLIB_NAMESPACE::createLibrary(dev, &error);
#else
    id<MTLLibrary> l = [dev newDefaultLibrary];
#endif
    CHECK(l != nil, "failed to load the metallib: ",
          error ? error.localizedDescription.UTF8String : "unknown error");
    return l;
  }();
  return lib;
}

// The pipeline for `name`. `aligns` are the (align_N, align_K) function constants (201, 202) the
// gather_qmm_rhs kernels are specialised on; upstream keys those pipelines by both.
id<MTLComputePipelineState> pipeline(const std::string &name,
                                     std::optional<std::pair<bool, bool>> aligns = std::nullopt) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  static std::mutex mu;
  std::string key = name;
  if (aligns) {
    key += std::string("_align_N_") + (aligns->first ? 't' : 'n') + "_align_K_" + (aligns->second ? 't' : 'n');
  }
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;

  NSError *error = nil;
  id<MTLFunction> fn = nil;
  if (aligns) {
    MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
    bool align_N = aligns->first, align_K = aligns->second;
    [fc setConstantValue:&align_N type:MTLDataTypeBool atIndex:201];
    [fc setConstantValue:&align_K type:MTLDataTypeBool atIndex:202];
    fn = [library() newFunctionWithName:@(name.c_str()) constantValues:fc error:&error];
  } else {
    fn = [library() newFunctionWithName:@(name.c_str())];
  }
  CHECK(fn != nil, "no kernel named ", name, " in the metallib");
  id<MTLComputePipelineState> state =
      [at::mps::MPSDevice::getInstance()->device() newComputePipelineStateWithFunction:fn error:&error];
  CHECK(state != nil, "failed to build pipeline ", name, ": ",
        error ? error.localizedDescription.UTF8String : "unknown error");
  cache[key] = state;
  return state;
}

// Upstream's get_type_string, for the dtypes the kernels are instantiated for.
std::string type_string(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return "float";
    case at::kHalf: return "float16_t";
    case at::kBFloat16: return "bfloat16_t";
    default: CHECK(false, "unsupported dtype ", t);
  }
}

// Upstream's type_to_name, which the reduce kernels are named with.
std::string type_to_name(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return "float32";
    case at::kHalf: return "float16";
    case at::kBFloat16: return "bfloat16";
    default: CHECK(false, "unsupported dtype ", t);
  }
}

// ---------------------------------------------------------------------------------------------------
// The encoder a launcher writes to, run twice (see the top of the file)
// ---------------------------------------------------------------------------------------------------

struct Enc {
  id<MTLComputeCommandEncoder> enc;  // nil in the planning pass
  std::vector<at::Tensor> *temps;
  std::vector<std::string> *names;
  size_t next = 0;
  id<MTLComputePipelineState> pso = nil;

  bool planning() const { return enc == nil; }

  // A tensor the launch needs that is not an input or the output: a temporary, or a copy upstream
  // makes (ensure_row_contiguous). Built in the planning pass, handed back in the encoding pass.
  at::Tensor make(const std::function<at::Tensor()> &build) {
    if (planning()) {
      temps->push_back(build());
      return temps->back();
    }
    return (*temps)[next++];
  }

  void kernel(const std::string &name, std::optional<std::pair<bool, bool>> aligns = std::nullopt) {
    if (planning()) {
      names->push_back(name);
      return;
    }
    pso = pipeline(name, aligns);
    [enc setComputePipelineState:pso];
  }
  size_t max_threads() const { return planning() ? 1024 : pso.maxTotalThreadsPerThreadgroup; }

  void array(const at::Tensor &t, int i) {
    if (planning()) return;
    // an MPS tensor's storage is a whole MTLBuffer that the tensor may be a view into
    [enc setBuffer:(__bridge id<MTLBuffer>)t.storage().data()
            offset:t.storage_offset() * t.element_size()
           atIndex:i];
  }
  template <typename T>
  void bytes(const T &v, int i) {
    if (!planning()) [enc setBytes:&v length:sizeof(T) atIndex:i];
  }
  template <typename T>
  void vec(const std::vector<T> &v, int i) {
    if (!planning()) [enc setBytes:v.data() length:v.size() * sizeof(T) atIndex:i];
  }
  void groups(MTLSize grid, MTLSize group) {
    if (!planning()) [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
  }
  void threads(MTLSize grid, MTLSize group) {
    if (!planning()) [enc dispatchThreads:grid threadsPerThreadgroup:group];
  }
};

// Plans `launch`, then (if `encode`) encodes it on torch's current stream. Returns the kernel names.
std::vector<std::string> run(const std::function<void(Enc &)> &launch, bool encode) {
  std::vector<at::Tensor> temps;
  std::vector<std::string> names;
  Enc plan{nil, &temps, &names};
  launch(plan);
  if (encode) {
    auto *temps_p = &temps;
    auto *names_p = &names;
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^{
      @autoreleasepool {
        Enc e{stream->commandEncoder(), temps_p, names_p};
        launch(e);
      }
    });
  }
  return names;
}

// ---------------------------------------------------------------------------------------------------
// Upstream's array helpers, on torch tensors
// ---------------------------------------------------------------------------------------------------

std::vector<int> shape_of(const at::Tensor &t) {
  return std::vector<int>(t.sizes().begin(), t.sizes().end());
}
std::vector<int64_t> strides_of(const at::Tensor &t) { return t.strides().vec(); }

at::Tensor ensure_row_contiguous(const at::Tensor &t) { return t.contiguous(); }

at::Tensor ensure_row_contiguous_matrix(const at::Tensor &t) {
  if (t.is_contiguous()) return t;
  if (t.dim() < 2) {
    if (t.stride(0) == 1) return t;
  } else if (t.stride(-2) == t.size(-1) && t.stride(-1) == 1) {
    return t;
  }
  return t.contiguous();
}

int add_strides_and_shapes(Enc &e, bool skip, const at::Tensor &x, const at::Tensor &w,
                           const at::Tensor &scales, const std::optional<at::Tensor> &biases, int offset) {
  if (skip) return offset;
  int x_batch_ndims = x.dim() - 2;
  int w_batch_ndims = w.dim() - 2;
  e.bytes(x_batch_ndims, offset++);
  e.vec(shape_of(x), offset++);
  e.vec(strides_of(x), offset++);
  e.bytes(w_batch_ndims, offset++);
  e.vec(shape_of(w), offset++);
  e.vec(strides_of(w), offset++);
  e.vec(strides_of(scales), offset++);
  if (biases) e.vec(strides_of(*biases), offset++);
  return offset;
}

// collapse_contiguous_dims over the two index arrays' shared shape.
int add_gather_strides_and_shapes(Enc &e, const at::Tensor &lhs, const at::Tensor &rhs, int offset) {
  std::vector<int> shape;
  std::vector<int64_t> s0, s1;
  for (int64_t i = 0; i < lhs.dim(); ++i) {
    if (lhs.size(i) == 1) continue;
    if (!shape.empty() && s0.back() == lhs.stride(i) * lhs.size(i) && s1.back() == rhs.stride(i) * rhs.size(i)) {
      shape.back() *= lhs.size(i);
      s0.back() = lhs.stride(i);
      s1.back() = rhs.stride(i);
    } else {
      shape.push_back(lhs.size(i));
      s0.push_back(lhs.stride(i));
      s1.push_back(rhs.stride(i));
    }
  }
  if (shape.empty()) {
    shape = {1};
    s0 = {0};
    s1 = {0};
  }
  int ndims = shape.size();
  e.bytes(ndims, offset++);
  e.vec(shape, offset++);
  e.vec(s0, offset++);
  e.vec(s1, offset++);
  return offset;
}

std::string kname(const std::string &mode, const std::string &func, const std::string &type,
                  int group_size, int bits) {
  return mode + "_" + func + "_" + type + "_gs_" + std::to_string(group_size) + "_b_" + std::to_string(bits);
}

// ---------------------------------------------------------------------------------------------------
// strided_reduce_general_dispatch (reduce.cpp), for the sum upstream runs over a row-contiguous
// [outer..., S, inner] intermediate along S. ColReduceArgs then comes out as reduction_size = S,
// reduction_stride = inner, the outer dims collapsed to one of stride S * inner (none when there is
// a single outer block), non_col_reductions 1; each kernel appends (S, inner) as its one reduce
// dim. output_grid_for_col_reduce is (outer, 1, 1): every output stride below `inner` is dropped.
// Sum keeps the input dtype (remap_reduce_types), so fp16 partials are summed in fp16.
// ---------------------------------------------------------------------------------------------------

void encode_col_reduce_args(Enc &e, size_t reduction_size, int64_t reduction_stride, int outer,
                            int64_t outer_stride, int reduce_dim, int64_t reduce_dim_stride) {
  const int ndim = outer > 1 ? 1 : 0;
  const std::vector<int> shape = {outer > 1 ? outer : 0};  // empty vectors are pushed as {0}
  const std::vector<int64_t> strides = {outer > 1 ? outer_stride : 0};
  const int reduce_ndim = 1;
  const size_t non_col_reductions = 1;
  e.bytes(reduction_size, 2);
  e.bytes(reduction_stride, 3);
  e.vec(shape, 4);
  e.vec(strides, 5);
  e.bytes(ndim, 6);
  e.vec(std::vector<int>{reduce_dim}, 7);
  e.vec(std::vector<int64_t>{reduce_dim_stride}, 8);
  e.bytes(reduce_ndim, 9);
  e.bytes(non_col_reductions, 10);
}

void col_sum(Enc &e, const at::Tensor &in, const at::Tensor &out, int S, int64_t inner, int outer) {
  const at::ScalarType t = in.scalar_type();
  const bool large = in.numel() > INT32_MAX;
  const std::string sfx = "_reduce_sum" + type_to_name(t);
  const int BN = 32, BM = 1024 / BN, threadgroup_size = 8 * 32;
  const std::string tile = "_" + std::to_string(BM) + "_" + std::to_string(BN);
  const size_t total = S;  // reduction_size * non_col_reductions

  // Small column
  if (total < 32) {
    e.kernel(std::string("col_reduce_small") + (large ? "_large" : "") + "_1" + sfx);
    e.array(in, 0);
    e.array(out, 1);
    encode_col_reduce_args(e, S, inner, outer, S * inner, S, inner);
    const int n_reads = 4;
    const size_t reduction_stride_blocks = (inner + n_reads - 1) / n_reads;
    const size_t threadgroup_x = std::min<size_t>(reduction_stride_blocks, 32);
    const size_t threadgroup_y = std::min<size_t>(8, std::min<size_t>(e.max_threads() / threadgroup_x, total));
    e.groups(MTLSizeMake((reduction_stride_blocks + threadgroup_x - 1) / threadgroup_x, outer, 1),
             MTLSizeMake(threadgroup_x, threadgroup_y, 1));
    return;
  }

  // Long column but small row
  CHECK(!(inner < 32 && total >= 1024), "a split-K sum over ", S, " partitions of ", inner,
        " values needs upstream's col_reduce_longcolumn, which is not transcribed");

  if (total > 256 && out.numel() / 32 < 1024) {
    // strided_reduce_2pass
    const int outer_blocks = 32;
    std::vector<int64_t> scratch_shape = {32};
    for (auto s : out.sizes()) scratch_shape.push_back(s);
    at::Tensor scratch = e.make([&] { return at::empty(scratch_shape, out.options()); });
    e.kernel(std::string("col_reduce_2pass") + (large ? "_large" : "") + "_1" + tile + sfx);
    e.array(in, 0);
    e.array(scratch, 1);
    encode_col_reduce_args(e, S, inner, outer, S * inner, S, inner);
    const size_t out_size = out.numel() / inner;
    e.bytes(out_size, 11);
    e.threads(MTLSizeMake(threadgroup_size * ((inner + BN - 1) / BN), outer * outer_blocks, 1),
              MTLSizeMake(threadgroup_size, 1, 1));
    // the second pass: ColReduceArgs(intermediate), plus the outer_blocks reduce dim
    const bool large2 = scratch.numel() > INT32_MAX;
    e.kernel(std::string("col_reduce_looped") + (large2 ? "_large" : "") + "_1_32_32" + sfx);
    e.array(scratch, 0);
    e.array(out, 1);
    encode_col_reduce_args(e, outer_blocks, out.numel(), 1, 0, outer_blocks, out.numel());
    e.threads(MTLSizeMake(threadgroup_size * ((out.numel() + BN - 1) / BN), 1, 1),
              MTLSizeMake(threadgroup_size, 1, 1));
    return;
  }

  // strided_reduce_looped
  e.kernel(std::string("col_reduce_looped") + (large ? "_large" : "") + "_1" + tile + sfx);
  e.array(in, 0);
  e.array(out, 1);
  encode_col_reduce_args(e, S, inner, outer, S * inner, S, inner);
  e.threads(MTLSizeMake(threadgroup_size * ((inner + BN - 1) / BN), outer, 1),
            MTLSizeMake(threadgroup_size, 1, 1));
}

// ---------------------------------------------------------------------------------------------------
// QuantizedMatmul (quantized.cpp)
// ---------------------------------------------------------------------------------------------------

struct MM {
  at::Tensor x, w, scales;
  std::optional<at::Tensor> biases;
  at::Tensor out;
  std::string mode, type;
  int group_size, bits, M, N, K;
  bool transpose;
};

int batch_of(const MM &a) { return a.out.numel() / a.M / a.N; }

void bind_wsb(Enc &e, const MM &a, int &c) {
  e.array(a.w, c++);
  e.array(a.scales, c++);
  if (a.biases) e.array(*a.biases, c++);
}

void qmv_quad(Enc &e, const MM &a) {
  int B = batch_of(a);
  constexpr int quads_per_simd = 8;
  constexpr int results_per_quadgroup = 8;
  int bn = quads_per_simd * results_per_quadgroup;
  int simdgroup_size = 32;
  e.kernel(kname(a.mode, "qmv_quad", a.type, a.group_size, a.bits) + "_d_" + std::to_string(a.K) +
           (B > 1 ? "_batch_1" : "_batch_0"));
  int c = 0;
  bind_wsb(e, a, c);
  e.array(a.x, c++);
  e.array(a.out, c++);
  e.bytes(a.K, c++);
  e.bytes(a.N, c++);
  add_strides_and_shapes(e, B <= 1, a.x, a.w, a.scales, a.biases, c++);
  e.groups(MTLSizeMake(a.M, (a.N + bn - 1) / bn, B), MTLSizeMake(simdgroup_size, 1, 1));
}

void qmv(Enc &e, const MM &a, metal::Device &d) {
  int B = batch_of(a);
  int bn = 8;
  int bk = 32;
  bool fast = a.N % bn == 0 && a.K % qmv_fast_k_alignment(a.bits) == 0;
  bool use_narrow_qmv = fast && a.N >= 4096 && d.get_architecture_gen() == 17 &&
                        d.get_architecture().back() == 's' && a.mode == "nvfp4";
  int results_per_simdgroup = use_narrow_qmv ? 2 : 4;
  bn = 2 * results_per_simdgroup;
  e.kernel(kname(a.mode, fast ? "qmv_fast" : "qmv", a.type, a.group_size, a.bits) +
           (use_narrow_qmv ? "_r_2" : "") + (B > 1 ? "_batch_1" : "_batch_0"));
  e.array(a.w, 0);
  e.array(a.scales, 1);
  if (a.biases) e.array(*a.biases, 2);
  int c = 3;
  e.array(a.x, c++);
  e.array(a.out, c++);
  e.bytes(a.K, c++);
  e.bytes(a.N, c++);
  add_strides_and_shapes(e, B <= 1, a.x, a.w, a.scales, a.biases, c);
  e.groups(MTLSizeMake(a.M, (a.N + bn - 1) / bn, B), MTLSizeMake(bk, 2, 1));
}

void qmv_wide(Enc &e, const MM &a) {
  int n_tiles = (a.M + 4) / 5;  // ceil(M / 5); tile size caps at 5
  int vecs_per_tg = (a.M + n_tiles - 1) / n_tiles;
  int k_lanes = a.mode == "affine" ? 8 : 16;
  constexpr int num_simdgroups = 2;
  int B = batch_of(a);
  bool batched = B > 1;
  int rows_per_tg = (32 / k_lanes) * num_simdgroups;
  e.kernel(kname(a.mode, "qmv_wide", a.type, a.group_size, a.bits) + "_nv_" + std::to_string(vecs_per_tg) +
           "_kl_" + std::to_string(k_lanes) + (batched ? "_batch_1" : "_batch_0"));
  int c = 0;
  bind_wsb(e, a, c);
  e.array(a.x, c++);
  e.array(a.out, c++);
  e.bytes(a.K, c++);
  e.bytes(a.N, c++);
  e.bytes(a.M, c++);
  add_strides_and_shapes(e, !batched, a.x, a.w, a.scales, a.biases, c);
  e.groups(MTLSizeMake((a.M + vecs_per_tg - 1) / vecs_per_tg, (a.N + rows_per_tg - 1) / rows_per_tg, B),
           MTLSizeMake(32, num_simdgroups, 1));
}

void dispatch_qmv(Enc &e, const MM &a, metal::Device &d) {
  // It is a qmv with a small inner dimension so route to qmv_quad kernel
  if ((a.K == 128 || a.K == 64) && is_power_of_2(a.bits)) {
    qmv_quad(e, a);
    return;
  }
  // Small batch so route to qmv_wide, which reuses each weight group across the M vectors.
  if (a.M >= 2 && use_qmv_wide(a.mode, d)) {
    qmv_wide(e, a);
    return;
  }
  qmv(e, a, d);
}

void qvm(Enc &e, const MM &a) {
  int B = batch_of(a);
  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(a.group_size, 32) * num_simdgroups;
  e.kernel(kname(a.mode, "qvm", a.type, a.group_size, a.bits) + (B > 1 ? "_batch_1" : "_batch_0"));
  e.array(a.w, 0);
  e.array(a.scales, 1);
  if (a.biases) e.array(*a.biases, 2);
  int c = 3;
  e.array(a.x, c++);
  e.array(a.out, c++);
  e.bytes(a.K, c++);
  e.bytes(a.N, c++);
  add_strides_and_shapes(e, B <= 1, a.x, a.w, a.scales, a.biases, c++);
  e.groups(MTLSizeMake(a.M, (a.N + bn - 1) / bn, B), MTLSizeMake(bk, num_simdgroups, 1));
}

void qvm_split_k(Enc &e, const MM &a) {
  int split_k = a.K > 8192 ? 32 : 8;
  int split_D = (a.K + split_k - 1) / split_k;
  int B = batch_of(a);
  B *= split_k;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(a.group_size, 32) * num_simdgroups;

  auto x_shape = shape_of(a.x);
  auto x_strides = strides_of(a.x);
  if (x_shape.size() == 1) {
    x_shape.insert(x_shape.begin(), 1);
    x_strides.insert(x_strides.begin(), 0);
  }

  int x_ndim = x_shape.size();
  int x_batch_ndims = x_ndim - 2;
  int w_batch_ndims = a.w.dim() - 2;
  auto w_shape = shape_of(a.w);
  auto w_strides = strides_of(a.w);
  auto s_strides = strides_of(a.scales);

  // Add split_k dim with reshapes
  x_shape.insert(x_shape.end() - 2, split_k);
  x_shape.back() /= split_k;
  x_strides.insert(x_strides.end() - 2, split_D);
  x_strides[x_ndim - 1] = split_D;
  x_batch_ndims += 1;

  w_shape.insert(w_shape.end() - 2, split_k);
  w_shape[a.w.dim() - 1] /= split_k;
  w_strides.insert(w_strides.end() - 2, split_D * a.w.size(-1));
  w_batch_ndims += 1;
  s_strides.insert(s_strides.end() - 2, split_D * a.scales.size(-1));

  int final_block_size = a.K - (split_k - 1) * split_D;

  std::vector<int64_t> temp_shape = a.out.sizes().vec();
  if (temp_shape.size() == 1) temp_shape.insert(temp_shape.begin(), 1);
  temp_shape.insert(temp_shape.end() - 2, split_k);
  at::Tensor intermediate = e.make([&] { return at::empty(temp_shape, a.x.options()); });

  e.kernel(kname(a.mode, "qvm_split_k", a.type, a.group_size, a.bits) + "_spk_" + std::to_string(split_k));
  int c = 0;
  bind_wsb(e, a, c);
  e.array(a.x, c++);
  e.array(intermediate, c++);
  e.bytes(split_D, c++);
  e.bytes(a.N, c++);
  e.bytes(x_batch_ndims, c++);
  e.vec(x_shape, c++);
  e.vec(x_strides, c++);
  e.bytes(w_batch_ndims, c++);
  e.vec(w_shape, c++);
  e.vec(w_strides, c++);
  e.vec(s_strides, c++);
  if (a.biases) {
    auto b_strides = strides_of(*a.biases);
    b_strides.insert(b_strides.end() - 2, split_D * a.biases->size(-1));
    e.vec(b_strides, c++);
  }
  e.bytes(final_block_size, c++);
  e.groups(MTLSizeMake(a.M, (a.N + bn - 1) / bn, B), MTLSizeMake(bk, num_simdgroups, 1));

  // the sum over the split_k axis (intermediate.ndim() - 3)
  int axis = intermediate.dim() - 3;
  int64_t outer = 1;
  for (int i = 0; i < axis; ++i) outer *= intermediate.size(i);
  col_sum(e, intermediate, a.out, split_k, intermediate.stride(axis), outer);
}

void qmm_nax(Enc &e, const MM &a) {
  int B = batch_of(a);
  int wm = 2;
  int wn = 2;
  // Use smaller bm when one block covers all of M. Only qmm_t_nax has a 32-row instantiation.
  int bm = (a.transpose && a.M <= 32) ? 32 : 64;
  int bn = 64;
  int bk = 64;
  bool aligned = a.N % 64 == 0;
  bool batched = B > 1;
  e.kernel(kname(a.mode, a.transpose ? "qmm_t_nax" : "qmm_n_nax", a.type, a.group_size, a.bits) + "_bm" +
           std::to_string(bm) + "_bn" + std::to_string(bn) + "_bk" + std::to_string(bk) + "_wm" +
           std::to_string(wm) + "_wn" + std::to_string(wn) +
           (a.transpose ? (aligned ? "_alN_true" : "_alN_false") : "") + (batched ? "_batch_1" : "_batch_0"));
  int c = 0;
  e.array(a.w, c++);
  e.array(a.scales, c++);
  if (a.biases) {
    e.array(*a.biases, c++);
  } else if (a.transpose) {
    c++;
  }
  e.array(a.x, c++);
  e.array(a.out, c++);
  e.bytes(a.K, c++);
  e.bytes(a.N, c++);
  e.bytes(a.M, c++);
  add_strides_and_shapes(e, B <= 1, a.x, a.w, a.scales, a.biases, c);
  e.groups(MTLSizeMake((a.N + bn - 1) / bn, (a.M + bm - 1) / bm, B), MTLSizeMake(32, wn, wm));
}

void qmm(Enc &e, const MM &a) {
  bool has_nax_kernel = is_nax_available() && (a.transpose || a.mode == "affine");
  bool nax_aligned = (a.K % 64 == 0) && (a.transpose || a.N % 64 == 0);
  if (has_nax_kernel && nax_aligned && (enable_tf32() || a.x.scalar_type() != at::kFloat)) {
    qmm_nax(e, a);
    return;
  }

  int B = batch_of(a);
  int wm = 2;
  int wn = 2;
  int bm = 32;
  int bn = 32;
  bool aligned = a.N % 32 == 0;
  bool batched = B > 1;
  e.kernel(kname(a.mode, a.transpose ? "qmm_t" : "qmm_n", a.type, a.group_size, a.bits) +
           (a.transpose ? (aligned ? "_alN_true" : "_alN_false") : "") + (batched ? "_batch_1" : "_batch_0"));
  int c = 0;
  e.array(a.w, c++);
  e.array(a.scales, c++);
  if (a.biases) {
    e.array(*a.biases, c++);
  } else if (a.transpose) {
    c++;
  }
  e.array(a.x, c++);
  e.array(a.out, c++);
  e.bytes(a.K, c++);
  e.bytes(a.N, c++);
  e.bytes(a.M, c++);
  add_strides_and_shapes(e, B <= 1, a.x, a.w, a.scales, a.biases, c);
  e.groups(MTLSizeMake((a.N + bn - 1) / bn, (a.M + bm - 1) / bm, B), MTLSizeMake(32, wn, wm));
}

void qmm_splitk(Enc &e, const MM &a) {
  // Choose split_k to target ~512 threadgroups
  int bm = 32, bn = 32;
  int n_tiles = (a.N + bn - 1) / bn;
  int m_tiles = (a.M + bm - 1) / bm;
  int current_tgs = n_tiles * m_tiles;
  int split_k = std::max(1, 512 / current_tgs);

  // Each K partition must be a whole number of BK-wide (32) K-tiles as well as whole quantization
  // groups.
  int k_align = a.group_size > 32 ? a.group_size : 32;
  split_k = std::min(split_k, a.K / k_align);

  // Ensure K divides evenly by split_k * k_align
  while (split_k > 1 && (a.K % (split_k * k_align) != 0)) {
    split_k--;
  }
  if (split_k <= 1) {
    qmm(e, a);
    return;
  }

  int k_partition_size = a.K / split_k;
  int split_k_partition_stride = a.M * a.N;

  // Intermediate buffer: split_k at the front so that partition_stride = M * N
  std::vector<int64_t> temp_shape = a.out.sizes().vec();
  if (temp_shape.size() == 1) temp_shape.insert(temp_shape.begin(), 1);
  temp_shape.insert(temp_shape.begin(), split_k);
  at::Tensor intermediate = e.make([&] { return at::empty(temp_shape, a.x.options()); });

  bool aligned = a.N % 32 == 0;
  e.kernel(kname(a.mode, "qmm_t_splitk", a.type, a.group_size, a.bits) + (aligned ? "_alN_true" : "_alN_false"));
  int c = 0;
  bind_wsb(e, a, c);
  e.array(a.x, c++);
  e.array(intermediate, c++);
  e.bytes(a.K, c++);
  e.bytes(a.N, c++);
  e.bytes(a.M, c++);
  e.bytes(k_partition_size, c++);
  e.bytes(split_k_partition_stride, c++);
  e.groups(MTLSizeMake(n_tiles, m_tiles, split_k), MTLSizeMake(32, 2, 2));

  // Sum across split_k dimension (axis 0)
  col_sum(e, intermediate, a.out, split_k, intermediate.stride(0), 1);
}

// QuantizedMatmul::eval_gpu, from "Extract the matmul shapes" on.
void quantized_matmul_eval(Enc &e, const MM &a) {
  metal::Device d = device_info();
  int vector_limit = a.transpose ? get_qmv_batch_limit(a.K, a.N, d) : 4;
  // It is a matrix matrix product.
  if (a.M >= vector_limit) {
    // Use split-K qmm for small M with transposed weights (non-batched only)
    int B = batch_of(a);
    if (a.transpose && B == 1) {
      qmm_splitk(e, a);
      return;
    }
    qmm(e, a);
    return;
  }
  // Run of the mill qmv
  if (a.transpose) {
    dispatch_qmv(e, a, d);
    return;
  }
  // Run of the mill qvm
  if (a.K < 1024) {
    qvm(e, a);
    return;
  }
  // Qvm with large dimension so route to a split K kernel for more parallelism
  qvm_split_k(e, a);
}

// ---------------------------------------------------------------------------------------------------
// GatherQMM (quantized.cpp)
// ---------------------------------------------------------------------------------------------------

struct GMM : MM {
  std::optional<at::Tensor> global_scale;
  at::Tensor lhs_indices, rhs_indices;
  bool right_sorted;
};

void bind_wsb_gs(Enc &e, const GMM &a) {
  e.array(a.w, 0);
  e.array(a.scales, 1);
  if (a.biases) {
    e.array(*a.biases, 2);
  } else if (a.global_scale) {
    e.array(*a.global_scale, 2);
  }
}

std::string hgs(const GMM &a) { return a.global_scale ? "_hgs" : ""; }

void gather_qmm_nax(Enc &e, const GMM &a) {
  int B = batch_of(a);
  int wm = 2, wn = 2, bm = 64, bn = 64;
  // The gather qmm NAX kernels are instantiated with BK = 64 only.
  int bk = 64;
  bool aligned = a.N % 64 == 0;
  e.kernel(kname(a.mode, a.transpose ? "gather_qmm_t_nax" : "gather_qmm_n_nax", a.type, a.group_size, a.bits) +
           "_bm" + std::to_string(bm) + "_bn" + std::to_string(bn) + "_bk" + std::to_string(bk) + "_wm" +
           std::to_string(wm) + "_wn" + std::to_string(wn) +
           (a.transpose ? (aligned ? "_alN_true" : "_alN_false") : "") + hgs(a));
  bind_wsb_gs(e, a);
  int c = 3;
  e.array(a.x, c++);
  e.array(a.lhs_indices, c++);
  e.array(a.rhs_indices, c++);
  e.array(a.out, c++);
  e.bytes(a.K, c++);
  e.bytes(a.N, c++);
  e.bytes(a.M, c++);
  c = add_strides_and_shapes(e, false, a.x, a.w, a.scales, a.biases, c);
  add_gather_strides_and_shapes(e, a.lhs_indices, a.rhs_indices, c);
  e.groups(MTLSizeMake((a.N + bn - 1) / bn, (a.M + bm - 1) / bm, B), MTLSizeMake(32, wn, wm));
}

void gather_qmm(Enc &e, const GMM &a) {
  if (is_nax_available() && a.transpose && (a.K % 64 == 0) &&
      (enable_tf32() || a.x.scalar_type() != at::kFloat)) {
    gather_qmm_nax(e, a);
    return;
  }
  int B = batch_of(a);
  int wm = 2, wn = 2, bm = 32, bn = 32;
  bool aligned = a.N % 32 == 0;
  e.kernel(kname(a.mode, a.transpose ? "gather_qmm_t" : "gather_qmm_n", a.type, a.group_size, a.bits) +
           (a.transpose ? (aligned ? "_alN_true" : "_alN_false") : "") + hgs(a));
  bind_wsb_gs(e, a);
  int c = 3;
  e.array(a.x, c++);
  e.array(a.lhs_indices, c++);
  e.array(a.rhs_indices, c++);
  e.array(a.out, c++);
  e.bytes(a.K, c++);
  e.bytes(a.N, c++);
  e.bytes(a.M, c++);
  c = add_strides_and_shapes(e, false, a.x, a.w, a.scales, a.biases, c);
  add_gather_strides_and_shapes(e, a.lhs_indices, a.rhs_indices, c);
  e.groups(MTLSizeMake((a.N + bn - 1) / bn, (a.M + bm - 1) / bm, B), MTLSizeMake(32, wn, wm));
}

void gather_qmv(Enc &e, const GMM &a) {
  int B = batch_of(a);
  int bn = 8, bk = 32;
  bool fast = a.N % bn == 0 && a.K % qmv_fast_k_alignment(a.bits) == 0;
  e.kernel(kname(a.mode, fast ? "gather_qmv_fast" : "gather_qmv", a.type, a.group_size, a.bits) + hgs(a));
  bind_wsb_gs(e, a);
  int c = 3;
  e.array(a.x, c++);
  e.array(a.lhs_indices, c++);
  e.array(a.rhs_indices, c++);
  e.array(a.out, c++);
  e.bytes(a.K, c++);
  e.bytes(a.N, c++);
  c = add_strides_and_shapes(e, false, a.x, a.w, a.scales, a.biases, c);
  add_gather_strides_and_shapes(e, a.lhs_indices, a.rhs_indices, c);
  e.groups(MTLSizeMake(a.M, (a.N + bn - 1) / bn, B), MTLSizeMake(bk, 2, 1));
}

void gather_qvm(Enc &e, const GMM &a) {
  int B = batch_of(a);
  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(a.group_size, 32) * num_simdgroups;
  e.kernel(kname(a.mode, "gather_qvm", a.type, a.group_size, a.bits) + hgs(a));
  bind_wsb_gs(e, a);
  int c = 3;
  e.array(a.x, c++);
  e.array(a.lhs_indices, c++);
  e.array(a.rhs_indices, c++);
  e.array(a.out, c++);
  e.bytes(a.K, c++);
  e.bytes(a.N, c++);
  c = add_strides_and_shapes(e, false, a.x, a.w, a.scales, a.biases, c);
  add_gather_strides_and_shapes(e, a.lhs_indices, a.rhs_indices, c);
  e.groups(MTLSizeMake(a.M, (a.N + bn - 1) / bn, B), MTLSizeMake(bk, num_simdgroups, 1));
}

// gather_qmm_rhs and gather_qmm_rhs_nax, which share everything but their tiles.
void gather_qmm_rhs(Enc &e, const GMM &a, int M) {
  const bool nax = is_nax_available() && a.transpose && (enable_tf32() || a.x.scalar_type() != at::kFloat);

  // Start by normalizing the indices
  at::Tensor indices = e.make([&] { return ensure_row_contiguous(a.rhs_indices); });

  // Broadcast x with indices. If we are here that means lhs_indices were not provided so the
  // lhs_indices are implied to be the shape of x broadcasted with rhs_indices.
  at::Tensor x = e.make([&] {
    if (a.x.numel() / a.x.size(-2) / a.x.size(-1) == indices.numel()) {
      return ensure_row_contiguous(a.x);
    }
    auto x_shape = indices.sizes().vec();
    x_shape.push_back(a.x.size(-2));
    x_shape.push_back(a.x.size(-1));
    return a.x.expand(x_shape).contiguous();
  });
  at::Tensor w = e.make([&] { return ensure_row_contiguous(a.w); });
  at::Tensor scales = e.make([&] { return ensure_row_contiguous(a.scales); });
  std::optional<at::Tensor> biases, gs;
  if (a.biases) biases = e.make([&] { return ensure_row_contiguous(*a.biases); });
  if (a.global_scale) gs = e.make([&] { return ensure_row_contiguous(*a.global_scale); });

  int E = w.numel() / w.size(-1) / w.size(-2);
  int bm, bn, bk, wm, wn;
  if (nax) {
    // Use smaller bm for many experts and few tokens.
    bm = (M / E < 64) ? 32 : 64;
    bn = 64, bk = 64;
    wm = 2, wn = 2;
  } else {
    bm = 16, bn = 32, bk = 32;
    wm = 1, wn = 2;
  }
  const bool align_N = (a.N % bn) == 0;
  const bool align_K = (a.K % bk) == 0;

  // gather_mm_offsets (matmul.cpp)
  at::Tensor offsets = e.make([&] { return at::empty({E}, a.x.options().dtype(at::kInt)); });
  e.kernel("gather_mm_offsets");
  e.array(indices, 0);
  e.array(offsets, 1);
  e.bytes(M, 2);
  size_t group_size = std::min<size_t>(E, e.max_threads());
  e.threads(MTLSizeMake(E, 1, 1), MTLSizeMake(group_size, 1, 1));

  const char *func = nax ? (a.transpose ? "gather_qmm_rhs_nax_nt" : "gather_qmm_rhs_nax_nn")
                         : (a.transpose ? "gather_qmm_rhs_nt" : "gather_qmm_rhs_nn");
  e.kernel(kname(a.mode, func, type_string(x.scalar_type()), a.group_size, a.bits) + "_bm_" + std::to_string(bm) +
               "_bn_" + std::to_string(bn) + "_bk_" + std::to_string(bk) + "_wm_" + std::to_string(wm) + "_wn_" +
               std::to_string(wn) + hgs(a),
           std::make_pair(align_N, align_K));
  e.array(x, 0);
  e.array(w, 1);
  e.array(scales, 2);
  if (biases) {
    e.array(*biases, 3);
  } else if (gs) {
    e.array(*gs, 3);
  }
  int c = 4;
  e.array(offsets, c++);
  e.array(a.out, c++);
  e.bytes(M, c++);
  e.bytes(a.N, c++);
  e.bytes(a.K, c++);
  e.bytes(E, c++);
  e.groups(MTLSizeMake((a.N + bn - 1) / bn, std::min(M, (M + bm - 1) / bm + E - 1), 1), MTLSizeMake(32, wn, wm));
}

// GatherQMM::eval_gpu, after its ensure_row_contiguous_matrix calls.
void gather_qmm_eval(Enc &e, const GMM &a) {
  metal::Device d = device_info();
  int B = batch_of(a);
  int E = a.w.numel() / a.w.size(-1) / a.w.size(-2);
  int vector_limit = a.transpose ? get_qmv_batch_limit(a.K, a.N, d) : 4;

  // We are walking x in order and w is also in order so we can batch up the matmuls and reuse
  // reading x and w.
  if (a.M == 1 && B >= 16 && a.right_sorted == true && B / E >= 4) {
    gather_qmm_rhs(e, a, a.x.numel() / a.K);
    return;
  }
  // It is a matrix matrix product
  if (a.M >= vector_limit) {
    gather_qmm(e, a);
    return;
  }
  if (a.transpose) {
    gather_qmv(e, a);
    return;
  }
  gather_qvm(e, a);
}

// ---------------------------------------------------------------------------------------------------
// The op-level checks and defaults (mlx/ops.cpp)
// ---------------------------------------------------------------------------------------------------

struct Quant {
  std::string mode;
  bool affine;
  int group_size, bits;
};

// string_to_quantization_mode + quantization_params_from_mode, then what upstream instantiates.
Quant quant_params(const std::string &mode, std::optional<int64_t> group_size, std::optional<int64_t> bits) {
  int default_group_size, default_bits;
  if (mode == "affine") {
    default_group_size = 64, default_bits = 4;
  } else if (mode == "nvfp4") {
    default_group_size = 16, default_bits = 4;
  } else if (mode == "mxfp4") {
    default_group_size = 32, default_bits = 4;
  } else if (mode == "mxfp8") {
    default_group_size = 32, default_bits = 8;
  } else {
    CHECK(false, "unknown quantization mode '", mode, "'; expected affine, mxfp4, mxfp8 or nvfp4");
  }
  Quant q{mode, mode == "affine", int(group_size.value_or(default_group_size)), int(bits.value_or(default_bits))};
  if (q.affine) {
    CHECK(q.group_size == 32 || q.group_size == 64 || q.group_size == 128,
          "affine group_size must be 32, 64 or 128, got ", q.group_size);
    CHECK(q.bits == 2 || q.bits == 3 || q.bits == 4 || q.bits == 5 || q.bits == 6 || q.bits == 8,
          "affine bits must be one of 2, 3, 4, 5, 6, 8, got ", q.bits);
  } else {
    CHECK(q.group_size == default_group_size && q.bits == default_bits, mode, " requires group_size ",
          default_group_size, " and bits ", default_bits, ", got ", q.group_size, " and ", q.bits);
  }
  return q;
}

bool is_kernel_float(at::ScalarType t) { return t == at::kFloat || t == at::kHalf || t == at::kBFloat16; }

// validate_mode_with_type: the scales/biases contract, and the dtype they imply.
at::ScalarType validate_mode_with_type(const Quant &q, const at::Tensor &scales,
                                       const std::optional<at::Tensor> &biases,
                                       std::optional<at::ScalarType> out_type) {
  if (q.affine) {
    CHECK(biases.has_value(), "biases must be provided for affine quantization");
    auto dtype = at::result_type(scales, *biases);
    CHECK(is_kernel_float(dtype), "scales and biases must be float32, float16 or bfloat16, got ",
          scales.scalar_type(), " and ", biases->scalar_type());
    return out_type.value_or(dtype);
  }
  CHECK(scales.scalar_type() == at::kByte, "scales must be uint8 for mode '", q.mode, "', got ",
        scales.scalar_type());
  CHECK(!biases.has_value(), "biases must be None for mode '", q.mode, "'");
  return out_type.value_or(at::kBFloat16);
}

void validate_quantized_input(const at::Tensor &w, const at::Tensor &scales, int group_size, int bits,
                              const std::optional<at::Tensor> &biases) {
  CHECK(w.scalar_type() == at::kUInt32, "the weight matrix should be uint32 but got ", w.scalar_type());
  CHECK(w.dim() >= 2, "the weight matrix must have at least 2 dimensions, got ", w.sizes());
  CHECK(!biases || scales.sizes() == biases->sizes(), "scales and biases should have the same shape, got ",
        scales.sizes(), " and ", biases->sizes());
  CHECK(scales.dim() == w.dim() && std::equal(w.sizes().begin(), w.sizes().end() - 2, scales.sizes().begin()),
        "weight and scales should have the same batch shape, got ", w.sizes(), " and ", scales.sizes());
  CHECK(w.size(-1) * 32 / bits == scales.size(-1) * group_size,
        "the shapes of the weight and scales are incompatible: w ", w.sizes(), ", scales ", scales.sizes(),
        " with group_size=", group_size, " and bits=", bits);
}

// extract_quantized_matmul_dims: (w_inner_dims, w_outer_dims).
std::pair<int64_t, int64_t> quantized_matmul_dims(const at::Tensor &x, const at::Tensor &w,
                                                   const at::Tensor &scales,
                                                   const std::optional<at::Tensor> &biases, bool transpose,
                                                   int group_size, int bits) {
  validate_quantized_input(w, scales, group_size, bits, biases);
  int64_t w_inner_dims = transpose ? w.size(-1) * 32 / bits : w.size(-2);
  int64_t w_outer_dims = transpose ? w.size(-2) : w.size(-1) * 32 / bits;
  CHECK(x.dim() >= 1 && x.size(-1) == w_inner_dims, "last dimension of x ", x.sizes(),
        " does not match the expanded quantized matrix (", w_inner_dims, ", ", w_outer_dims,
        ") computed from w ", w.sizes(), " with group_size=", group_size, ", bits=", bits,
        " and transpose=", transpose);
  return {w_inner_dims, w_outer_dims};
}

// broadcast_arrays(inputs, {-2, -1}): broadcast every leading dimension, keep the last two.
std::vector<at::Tensor> broadcast_batch(const std::vector<at::Tensor> &inputs) {
  std::vector<int64_t> batch;
  for (const auto &t : inputs) {
    std::vector<int64_t> b(t.sizes().begin(), t.sizes().end() - 2);
    batch = at::infer_size(batch, b);
  }
  std::vector<at::Tensor> out;
  for (const auto &t : inputs) {
    auto shape = batch;
    shape.push_back(t.size(-2));
    shape.push_back(t.size(-1));
    out.push_back(t.expand(shape));
  }
  return out;
}

// Nothing to launch: an empty output, or an empty reduction (which is all zeros).
bool empty_product(const MM &a) { return a.out.numel() == 0 || a.K == 0; }

void check_mps(std::initializer_list<const at::Tensor *> ts) {
  for (const at::Tensor *t : ts) {
    if (t) CHECK(t->is_mps(), "all inputs must be on mps, got one on ", t->device());
  }
}

// quantized_matmul's op-level preparation, then QuantizedMatmul::eval_gpu's array normalization.
MM prepare_quantized_matmul(const at::Tensor &x_in, const at::Tensor &w_in,
                                           const at::Tensor &scales_in,
                                           const std::optional<at::Tensor> &biases_in, bool transpose,
                                           std::optional<int64_t> group_size_, std::optional<int64_t> bits_,
                                           const std::string &mode_name) {
  Quant q = quant_params(mode_name, group_size_, bits_);
  at::ScalarType dtype = validate_mode_with_type(q, scales_in, biases_in, std::nullopt);
  auto [w_inner_dims, w_outer_dims] =
      quantized_matmul_dims(x_in, w_in, scales_in, biases_in, transpose, q.group_size, q.bits);
  dtype = q.affine ? at::promote_types(x_in.scalar_type(), dtype) : x_in.scalar_type();
  CHECK(is_kernel_float(dtype), "x must be float32, float16 or bfloat16, got ", x_in.scalar_type());

  std::vector<at::Tensor> inputs;
  if (q.affine) {
    inputs = {x_in.to(dtype), w_in, scales_in.to(dtype), biases_in->to(dtype)};
  } else {
    inputs = {x_in, w_in, scales_in};
  }
  if (x_in.dim() > 2 && w_in.dim() > 2) inputs = broadcast_batch(inputs);
  auto out_shape = inputs[0].sizes().vec();
  out_shape.back() = w_outer_dims;

  MM a;
  // Make sure the last two dims of x and w, s, b are contiguous.
  a.x = ensure_row_contiguous_matrix(inputs[0]);
  a.w = ensure_row_contiguous_matrix(inputs[1]);
  a.scales = ensure_row_contiguous_matrix(inputs[2]);
  if (q.affine) a.biases = ensure_row_contiguous_matrix(inputs[3]);
  a.out = at::empty(out_shape, a.x.options());
  a.mode = q.mode;
  a.type = type_string(dtype);
  a.group_size = q.group_size;
  a.bits = q.bits;
  a.transpose = transpose;

  // Extract the matmul shapes
  bool non_batched = a.w.dim() == 2 && a.x.is_contiguous();
  a.K = a.x.size(-1);
  CHECK(non_batched || a.x.dim() >= 2, "x must have at least 2 dimensions when w is batched or x is strided");
  int64_t M = non_batched ? a.x.numel() / std::max<int64_t>(a.K, 1) : a.x.size(-2);
  int64_t N = a.out.size(-1);
  CHECK(M <= INT32_MAX && N <= INT32_MAX && a.K <= INT32_MAX, "dimensions must fit in int32");
  a.M = M;
  a.N = N;
  return a;
}

at::Tensor indices_or_default(const std::optional<at::Tensor> &indices, const at::Tensor &x) {
  if (indices) {
    CHECK(!at::isFloatingType(indices->scalar_type()) && indices->scalar_type() != at::kBool,
          "indices must be integers, got ", indices->scalar_type());
    // the kernels read uint32; int32 has the same bits for every valid index
    return indices->to(at::kInt);
  }
  std::vector<int64_t> shape(x.sizes().begin(), x.sizes().end() - 2);
  int64_t total = 1;
  for (auto s : shape) total *= s;
  return at::arange(total, x.options().dtype(at::kInt)).reshape(shape);
}

GMM prepare_gather_qmm(const at::Tensor &x_in, const at::Tensor &w_in, const at::Tensor &scales_in,
                                      const std::optional<at::Tensor> &biases_in,
                                      const std::optional<at::Tensor> &lhs_in,
                                      const std::optional<at::Tensor> &rhs_in, bool transpose,
                                      std::optional<int64_t> group_size_, std::optional<int64_t> bits_,
                                      const std::string &mode_name,
                                      const std::optional<at::Tensor> &global_scale, bool sorted_indices) {
  Quant q = quant_params(mode_name, group_size_, bits_);
  at::ScalarType out_type = validate_mode_with_type(q, scales_in, biases_in, std::nullopt);
  auto [w_inner_dims, w_outer_dims] =
      quantized_matmul_dims(x_in, w_in, scales_in, biases_in, transpose, q.group_size, q.bits);
  CHECK(x_in.dim() >= 2, "x must have at least 2 dimensions, got ", x_in.sizes());
  if (global_scale) {
    CHECK(q.mode == "nvfp4", "global scale is only supported for 'nvfp4' quantization mode");
    CHECK(global_scale->scalar_type() == at::kFloat, "global scale must have dtype float32, got ",
          global_scale->scalar_type());
    std::vector<int64_t> expected(w_in.sizes().begin(), w_in.sizes().end() - 2);
    CHECK(global_scale->sizes().vec() == expected, "global scale must have one entry per expert with shape ",
          expected, " but got ", global_scale->sizes());
  }
  out_type = q.affine ? at::promote_types(x_in.scalar_type(), out_type) : x_in.scalar_type();
  CHECK(is_kernel_float(out_type), "x must be float32, float16 or bfloat16, got ", x_in.scalar_type());

  // Extract indices and broadcast them
  auto lhs_rhs = at::broadcast_tensors({indices_or_default(lhs_in, x_in), indices_or_default(rhs_in, w_in)});

  GMM a;
  a.x = ensure_row_contiguous_matrix(x_in.to(out_type));
  a.w = ensure_row_contiguous_matrix(w_in);
  a.scales = ensure_row_contiguous_matrix(q.affine ? scales_in.to(out_type) : scales_in);
  if (q.affine) a.biases = ensure_row_contiguous_matrix(biases_in->to(out_type));
  if (global_scale) a.global_scale = ensure_row_contiguous(*global_scale);
  a.lhs_indices = lhs_rhs[0];
  a.rhs_indices = lhs_rhs[1];
  a.right_sorted = sorted_indices && !lhs_in;

  auto out_shape = a.lhs_indices.sizes().vec();
  out_shape.push_back(x_in.size(-2));
  out_shape.push_back(w_outer_dims);
  a.out = at::empty(out_shape, a.x.options());
  a.mode = q.mode;
  a.type = type_string(out_type);
  a.group_size = q.group_size;
  a.bits = q.bits;
  a.transpose = transpose;
  a.K = a.x.size(-1);
  a.M = a.x.size(-2);
  a.N = a.out.size(-1);
  return a;
}

// fast::Quantize::eval_gpu (quantize_impl and get_quantize_kernel_dims).
void quantize_launch(Enc &e, const Quant &q, bool dequantize, const at::Tensor &w, const at::Tensor &out,
                     const at::Tensor &scales, const std::optional<at::Tensor> &biases,
                     const std::optional<at::Tensor> &global_scale) {
  bool has_biases = q.affine;
  bool has_global_scale = !has_biases && global_scale.has_value();
  std::string type = type_string(dequantize ? out.scalar_type() : w.scalar_type());
  std::string name = kname(q.mode, dequantize ? "dequantize" : "quantize", type, q.group_size, q.bits);
  if (!has_biases) name += has_global_scale ? "_hgs_true" : "_hgs_false";
  e.kernel(name);
  if (dequantize) {
    if (has_biases) {
      e.array(*biases, 2);
    } else if (has_global_scale) {
      e.array(*global_scale, 2);
    }
    e.array(w, 0);
    e.array(scales, 1);
    e.array(out, 3);
  } else {
    if (has_biases) {
      e.array(*biases, 3);
    } else if (has_global_scale) {
      e.array(*global_scale, 3);
    }
    e.array(w, 0);
    e.array(out, 1);
    e.array(scales, 2);
  }

  // Treat uint32 as uint8 in kernel
  constexpr int simd_size = 32;
  int packs_per_int = (q.bits == 3 || q.bits == 5) ? 8 : q.bits == 6 ? 4 : 8 / q.bits;
  int per_thread = dequantize ? packs_per_int : std::max(q.group_size / simd_size, 1);
  size_t nthreads = dequantize ? out.numel() / packs_per_int : w.numel() / per_thread;
  CHECK(nthreads <= UINT_MAX, "tensor too large for a 1D grid");
  size_t thread_group_size = std::min<size_t>(e.max_threads(), nthreads);
  e.threads(MTLSizeMake(nthreads, 1, 1), MTLSizeMake(thread_group_size, 1, 1));
}

}  // namespace

// ---------------------------------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------------------------------

at::Tensor quantized_matmul(const at::Tensor &x, const at::Tensor &w, const at::Tensor &scales,
                            const std::optional<at::Tensor> &biases, bool transpose,
                            std::optional<int64_t> group_size, std::optional<int64_t> bits,
                            const std::string &mode) {
  check_mps({&x, &w, &scales, biases ? &*biases : nullptr});
  const MM a = prepare_quantized_matmul(x, w, scales, biases, transpose, group_size, bits, mode);
  if (empty_product(a)) return a.out.zero_();
  run([&](Enc &e) { quantized_matmul_eval(e, a); }, true);
  return a.out;
}

at::Tensor gather_qmm(const at::Tensor &x, const at::Tensor &w, const at::Tensor &scales,
                      const std::optional<at::Tensor> &biases, const std::optional<at::Tensor> &lhs_indices,
                      const std::optional<at::Tensor> &rhs_indices, bool transpose,
                      std::optional<int64_t> group_size, std::optional<int64_t> bits, const std::string &mode,
                      const std::optional<at::Tensor> &global_scale, bool sorted_indices) {
  if (!lhs_indices && !rhs_indices) {
    CHECK(!global_scale, "global scale is not supported without indices");
    return quantized_matmul(x, w, scales, biases, transpose, group_size, bits, mode);
  }
  check_mps({&x, &w, &scales, biases ? &*biases : nullptr, lhs_indices ? &*lhs_indices : nullptr,
             rhs_indices ? &*rhs_indices : nullptr, global_scale ? &*global_scale : nullptr});
  const GMM a = prepare_gather_qmm(x, w, scales, biases, lhs_indices, rhs_indices, transpose, group_size, bits,
                                   mode, global_scale, sorted_indices);
  if (empty_product(a)) return a.out.zero_();
  run([&](Enc &e) { gather_qmm_eval(e, a); }, true);
  return a.out;
}

std::vector<at::Tensor> quantize(const at::Tensor &w_in, std::optional<int64_t> group_size,
                                 std::optional<int64_t> bits, const std::string &mode,
                                 const std::optional<at::Tensor> &global_scale) {
  check_mps({&w_in, global_scale ? &*global_scale : nullptr});
  Quant q = quant_params(mode, group_size, bits);
  CHECK(is_kernel_float(w_in.scalar_type()), "only float32, float16 or bfloat16 can be quantized, got ",
        w_in.scalar_type());
  CHECK(w_in.dim() >= 2, "the matrix to be quantized must have at least 2 dimensions, got ", w_in.sizes());
  CHECK(w_in.size(-1) % q.group_size == 0, "the last dimension of the matrix (", w_in.size(-1),
        ") must be divisible by the group size ", q.group_size);
  if (global_scale) {
    CHECK(q.mode == "nvfp4", "global scale is only supported for 'nvfp4' quantization mode");
    CHECK(global_scale->numel() == 1, "global scale must be a scalar, got shape ", global_scale->sizes());
    CHECK(global_scale->scalar_type() == at::kFloat, "global scale must be float32, got ",
          global_scale->scalar_type());
  }
  at::Tensor w = ensure_row_contiguous(w_in);
  auto wq_shape = w.sizes().vec();
  wq_shape.back() = w.size(-1) * q.bits / 32;
  auto scales_shape = w.sizes().vec();
  scales_shape.back() = w.size(-1) / q.group_size;
  at::Tensor wq = at::empty(wq_shape, w.options().dtype(at::kUInt32));
  at::Tensor scales = at::empty(scales_shape, w.options().dtype(q.affine ? w.scalar_type() : at::kByte));
  std::optional<at::Tensor> biases;
  if (q.affine) biases = at::empty(scales_shape, w.options());
  std::optional<at::Tensor> gs;
  if (global_scale) gs = ensure_row_contiguous(*global_scale);
  if (w.numel() > 0) {
    run([&](Enc &e) { quantize_launch(e, q, false, w, wq, scales, biases, gs); }, true);
  }
  if (biases) return {wq, scales, *biases};
  return {wq, scales};
}

at::Tensor dequantize(const at::Tensor &w_in, const at::Tensor &scales_in,
                      const std::optional<at::Tensor> &biases_in, std::optional<int64_t> group_size,
                      std::optional<int64_t> bits, const std::string &mode,
                      const std::optional<at::Tensor> &global_scale, std::optional<at::ScalarType> dtype) {
  check_mps({&w_in, &scales_in, biases_in ? &*biases_in : nullptr, global_scale ? &*global_scale : nullptr});
  Quant q = quant_params(mode, group_size, bits);
  at::ScalarType out_type = validate_mode_with_type(q, scales_in, biases_in, dtype);
  CHECK(is_kernel_float(out_type), "dtype must be float32, float16 or bfloat16, got ", out_type);
  CHECK(w_in.scalar_type() == at::kUInt32, "the matrix should be given as uint32, got ", w_in.scalar_type());
  CHECK(w_in.dim() >= 2, "the matrix to be dequantized must have at least 2 dimensions, got ", w_in.sizes());
  if (global_scale) {
    CHECK(q.mode == "nvfp4", "global scale is only supported for 'nvfp4' quantization mode");
    CHECK(global_scale->numel() == 1 && global_scale->scalar_type() == at::kFloat,
          "global scale must be a float32 scalar");
  }
  auto w_shape = w_in.sizes().vec(), s_shape = scales_in.sizes().vec();
  w_shape.back() = s_shape.back() = -1;
  CHECK(w_shape == s_shape, "shape of scales ", scales_in.sizes(), " does not match the matrix ", w_in.sizes());
  int64_t out_size = w_in.size(-1) * 32 / q.bits;
  CHECK(out_size == scales_in.size(-1) * q.group_size, "shape of scales ", scales_in.sizes(),
        " does not match the matrix ", w_in.sizes(), " with group_size=", q.group_size, " and bits=", q.bits);

  at::Tensor w = ensure_row_contiguous(w_in);
  at::Tensor scales = ensure_row_contiguous(scales_in);
  std::optional<at::Tensor> biases;
  if (q.affine) {
    CHECK(biases_in->sizes() == scales_in.sizes(), "shape of biases ", biases_in->sizes(),
          " does not match scales ", scales_in.sizes());
    // the affine kernel reads scales, biases and its output as one type: the scales'
    biases = ensure_row_contiguous(biases_in->to(scales.scalar_type()));
  }
  std::optional<at::Tensor> gs;
  if (global_scale) gs = ensure_row_contiguous(*global_scale);
  auto out_shape = w.sizes().vec();
  out_shape.back() = out_size;
  // affine dequantizes in the scales' dtype and then casts, as upstream's astype does
  at::Tensor out = at::empty(out_shape, w.options().dtype(q.affine ? scales.scalar_type() : out_type));
  if (out.numel() > 0) {
    run([&](Enc &e) { quantize_launch(e, q, true, w, out, scales, biases, gs); }, true);
  }
  return out.scalar_type() == out_type ? out : out.to(out_type);
}

std::vector<std::string> trace_quantized_matmul(const at::Tensor &x, const at::Tensor &w, const at::Tensor &scales,
                                                const std::optional<at::Tensor> &biases, bool transpose,
                                                std::optional<int64_t> group_size, std::optional<int64_t> bits,
                                                const std::string &mode) {
  const MM a = prepare_quantized_matmul(x, w, scales, biases, transpose, group_size, bits, mode);
  if (empty_product(a)) return {};
  return run([&](Enc &e) { quantized_matmul_eval(e, a); }, false);
}

std::vector<std::string> trace_gather_qmm(const at::Tensor &x, const at::Tensor &w, const at::Tensor &scales,
                                          const std::optional<at::Tensor> &biases,
                                          const std::optional<at::Tensor> &lhs_indices,
                                          const std::optional<at::Tensor> &rhs_indices, bool transpose,
                                          std::optional<int64_t> group_size, std::optional<int64_t> bits,
                                          const std::string &mode, const std::optional<at::Tensor> &global_scale,
                                          bool sorted_indices) {
  if (!lhs_indices && !rhs_indices) {
    return trace_quantized_matmul(x, w, scales, biases, transpose, group_size, bits, mode);
  }
  const GMM a = prepare_gather_qmm(x, w, scales, biases, lhs_indices, rhs_indices, transpose, group_size, bits,
                                   mode, global_scale, sorted_indices);
  if (empty_product(a)) return {};
  return run([&](Enc &e) { gather_qmm_eval(e, a); }, false);
}
