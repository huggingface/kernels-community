/* Metal side: launch MLX's quantized kernels on torch's MPS stream.
 *
 * The kernels are upstream's, compiled as they ship (vendor/UPSTREAM pins the revision). This file
 * is the host side, which cannot be vendored, transcribed from upstream function by function:
 *
 *   vendor/mlx/backend/metal/quantized.cpp   QuantizedMatmul, GatherQMM, fast::Quantize::eval_gpu
 *                                            and the qmv / qvm / qmm / gather launchers they call
 *   vendor/mlx/backend/metal/reduce.cpp      strided_reduce_general_dispatch, for split-K sums
 *   vendor/mlx/backend/metal/device.cpp      is_nax_available
 *
 * Helpers marked verbatim are upstream's text; tests/test_vendor_drift.py checks them and the
 * transcribed lines against vendor/, so a pin bump that changes either fails loudly.
 *
 * Not transcribed: col_reduce_longcolumn, which a split-K sum only reaches with over 1024
 * partitions (it raises instead), and MLX's JIT mode -- the kernels here are all precompiled.
 */

#import <Metal/Metal.h>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"

#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#endif

#define MLXQ_CHECK(cond, ...) TORCH_CHECK(cond, "mlx-quantization-metal-kernels: ", __VA_ARGS__)

using mlxq::array;
using mlxq::Dtype;

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
// Device, library, pipelines
// ---------------------------------------------------------------------------------------------------

// Upstream's Device constructor: the arch (e.g. "applegpu_g14s") is MLX_METAL_GPU_ARCH if set, else
// the GPU's own, and the generation is the two digits before the size letter. Read per call rather
// than once, so the tests can steer every branch from one process.
metal::Device device() {
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

// is_nax_available (device.cpp)
bool is_nax_available() {
  bool can_use_nax = false;
  if (@available(macOS 26.2, *)) {
    can_use_nax = true;
  }
  auto d = device();
  auto arch = d.get_architecture().back();
  auto gen = d.get_architecture_gen();
  can_use_nax &= gen >= (arch == 'p' ? 18 : 17);
  return can_use_nax;
}

// env::enable_tf32: MLX_ENABLE_TF32, on unless set to 0.
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
    MLXQ_CHECK(l != nil, "failed to load the metallib: ", error ? error.localizedDescription.UTF8String : "unknown");
    return l;
  }();
  return lib;
}

// `aligns` are the (align_N, align_K) function constants 201/202 the gather_qmm_rhs kernels are
// specialised on; upstream keys those pipelines by both (get_gather_qmm_kernel's hash_name).
id<MTLComputePipelineState> pipeline(const std::string &name, std::optional<std::pair<bool, bool>> aligns) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  static std::mutex mu;
  std::string key = name;
  if (aligns) {
    key += std::string("_align_N_") + (aligns->first ? 't' : 'n') + "_align_K_" + (aligns->second ? 't' : 'n');
  }
  std::lock_guard<std::mutex> lock(mu);
  if (auto it = cache.find(key); it != cache.end()) return it->second;

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
  MLXQ_CHECK(fn != nil, "no kernel named ", name, " in the metallib");
  id<MTLComputePipelineState> state =
      [at::mps::MPSDevice::getInstance()->device() newComputePipelineStateWithFunction:fn error:&error];
  MLXQ_CHECK(state != nil, "failed to build ", name, ": ", error ? error.localizedDescription.UTF8String : "unknown");
  return cache[key] = state;
}

// get_type_string
std::string get_type_string(Dtype t) {
  switch (t) {
    case Dtype::float32: return "float";
    case Dtype::float16: return "float16_t";
    case Dtype::bfloat16: return "bfloat16_t";
    default: MLXQ_CHECK(false, "no kernels for this dtype");
  }
}

// type_to_name, which the reduce kernels are named with
std::string type_to_name(Dtype t) {
  switch (t) {
    case Dtype::float32: return "float32";
    case Dtype::float16: return "float16";
    case Dtype::bfloat16: return "bfloat16";
    default: MLXQ_CHECK(false, "no kernels for this dtype");
  }
}

template <typename... Args>
std::string concatenate(Args &&...args) {
  std::string s;
  auto add = [&](const auto &a) {
    if constexpr (std::is_arithmetic_v<std::decay_t<decltype(a)>>) {
      s += std::to_string(a);
    } else {
      s += a;
    }
  };
  (add(args), ...);
  return s;
}

// ---------------------------------------------------------------------------------------------------
// The command encoder. Every launch runs twice: a planning pass with no encoder, outside torch's
// stream queue, which makes the workspace allocations and records kernel names; then the encoding
// pass inside the queue, which gets the same allocations back in the same order.
// ---------------------------------------------------------------------------------------------------

struct CommandEncoder {
  id<MTLComputeCommandEncoder> enc;  // nil while planning
  mlxq::Workspace &ws;
  std::vector<array> &temps;
  std::vector<std::string> &names;
  size_t next = 0;
  id<MTLComputePipelineState> pso = nil;

  bool planning() const { return enc == nil; }

  array temporary(const std::function<array()> &make) {
    if (planning()) return temps.emplace_back(make());
    return temps[next++];
  }
  array empty(const std::vector<int> &shape, Dtype dtype) {
    return temporary([&] { return ws.empty(shape, dtype); });
  }
  array contiguous(const array &a, const std::optional<std::vector<int>> &shape = std::nullopt) {
    return temporary([&] { return ws.contiguous(a, shape); });
  }

  void set_compute_pipeline_state(const std::string &name,
                                  std::optional<std::pair<bool, bool>> aligns = std::nullopt) {
    if (planning()) {
      names.push_back(name);
      return;
    }
    pso = pipeline(name, aligns);
    [enc setComputePipelineState:pso];
  }
  size_t max_threads() const { return planning() ? 1024 : pso.maxTotalThreadsPerThreadgroup; }

  void set_input_array(const array &a, int i) {
    if (!planning()) [enc setBuffer:(__bridge id<MTLBuffer>)a.buffer offset:a.offset atIndex:i];
  }
  void set_output_array(const array &a, int i) { set_input_array(a, i); }
  template <typename T>
  void set_bytes(const T &v, int i) {
    if (!planning()) [enc setBytes:&v length:sizeof(T) atIndex:i];
  }
  template <typename T>
  void set_vector_bytes(const std::vector<T> &v, int i) {
    if (!planning()) [enc setBytes:v.data() length:v.size() * sizeof(T) atIndex:i];
  }
  void dispatch_threadgroups(MTLSize grid, MTLSize group) {
    if (!planning()) [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
  }
  void dispatch_threads(MTLSize grid, MTLSize group) {
    if (!planning()) [enc dispatchThreads:grid threadsPerThreadgroup:group];
  }
};

std::vector<std::string> run(mlxq::Workspace &ws, bool encode, const std::function<void(CommandEncoder &)> &launch) {
  std::vector<array> temps;
  std::vector<std::string> names;
  CommandEncoder plan{nil, ws, temps, names};
  launch(plan);
  if (encode) {
    auto *ws_p = &ws;
    auto *temps_p = &temps;
    auto *names_p = &names;
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^{
      @autoreleasepool {
        CommandEncoder e{stream->commandEncoder(), *ws_p, *temps_p, *names_p};
        launch(e);
      }
    });
  }
  return names;
}

// ---------------------------------------------------------------------------------------------------
// quantized.cpp's helpers
// ---------------------------------------------------------------------------------------------------

int add_strides_and_shapes(CommandEncoder &compute_encoder, bool skip, const array &x, const array &w,
                           const array &scales, const std::optional<array> &biases, int offset) {
  if (skip) {
    return offset;
  }
  int x_batch_ndims = x.ndim() - 2;
  int w_batch_ndims = w.ndim() - 2;
  compute_encoder.set_bytes(x_batch_ndims, offset++);
  compute_encoder.set_vector_bytes(x.shape(), offset++);
  compute_encoder.set_vector_bytes(x.strides(), offset++);
  compute_encoder.set_bytes(w_batch_ndims, offset++);
  compute_encoder.set_vector_bytes(w.shape(), offset++);
  compute_encoder.set_vector_bytes(w.strides(), offset++);
  compute_encoder.set_vector_bytes(scales.strides(), offset++);
  if (biases) {
    compute_encoder.set_vector_bytes(biases->strides(), offset++);
  }
  return offset;
}

// collapse_contiguous_dims over both index arrays (which share a shape), then encoded.
int add_gather_strides_and_shapes(CommandEncoder &compute_encoder, const array &lhs_indices,
                                  const array &rhs_indices, int offset) {
  std::vector<int> shape;
  std::vector<int64_t> s0, s1;
  for (int i = 0; i < lhs_indices.ndim(); ++i) {
    int n = lhs_indices.shape(i);
    if (n == 1) continue;
    if (!shape.empty() && s0.back() == lhs_indices.strides(i) * n && s1.back() == rhs_indices.strides(i) * n) {
      shape.back() *= n;
      s0.back() = lhs_indices.strides(i);
      s1.back() = rhs_indices.strides(i);
    } else {
      shape.push_back(n);
      s0.push_back(lhs_indices.strides(i));
      s1.push_back(rhs_indices.strides(i));
    }
  }
  if (shape.empty()) {
    shape = {1}, s0 = {0}, s1 = {0};
  }
  int ndims = shape.size();
  compute_encoder.set_bytes(ndims, offset++);
  compute_encoder.set_vector_bytes(shape, offset++);
  compute_encoder.set_vector_bytes(s0, offset++);
  compute_encoder.set_vector_bytes(s1, offset++);
  return offset;
}

// array::flags().row_contiguous
bool row_contiguous(const array &a) {
  int64_t stride = 1;
  for (int i = a.ndim() - 1; i >= 0; --i) {
    if (a.shape(i) != 1 && a.strides(i) != stride) return false;
    stride *= a.shape(i);
  }
  return true;
}

// The arguments every launcher takes; upstream passes them one by one.
struct Args {
  array x, w, scales;
  std::optional<array> biases, global_scale;
  array out;
  int group_size, bits, M, N, K;
  std::string mode, type_string;
  bool transpose;
  int B() const { return out.size() / M / N; }
};

void set_w_scales_biases(CommandEncoder &compute_encoder, const Args &a, int &c) {
  compute_encoder.set_input_array(a.w, c++);
  compute_encoder.set_input_array(a.scales, c++);
  if (a.biases) compute_encoder.set_input_array(*a.biases, c++);
}

// ---------------------------------------------------------------------------------------------------
// strided_reduce_general_dispatch (reduce.cpp), for the sum qmm_splitk and qvm_split_k run over a
// row-contiguous [outer..., S, inner] intermediate along S. Their ColReduceArgs come out as
// reduction_size = S, reduction_stride = inner, the outer dims collapsed to one of stride S * inner
// (none when outer is 1), non_col_reductions 1, and each kernel appends (S, inner) as its one reduce
// dim; output_grid_for_col_reduce is (outer, 1, 1). Sum keeps the input dtype (remap_reduce_types).
// ---------------------------------------------------------------------------------------------------

void encode_col_reduce_args(CommandEncoder &compute_encoder, size_t reduction_size, int64_t reduction_stride,
                            int outer, int reduce_dim, int64_t reduce_dim_stride) {
  // ColReduceArgs::encode; empty shape vectors are pushed as {0}
  std::vector<int> shape = {outer > 1 ? outer : 0};
  std::vector<int64_t> strides = {outer > 1 ? int64_t(reduction_size) * reduction_stride : 0};
  int ndim = outer > 1 ? 1 : 0;
  int reduce_ndim = 1;
  size_t non_col_reductions = 1;
  compute_encoder.set_bytes(reduction_size, 2);
  compute_encoder.set_bytes(reduction_stride, 3);
  compute_encoder.set_vector_bytes(shape, 4);
  compute_encoder.set_vector_bytes(strides, 5);
  compute_encoder.set_bytes(ndim, 6);
  compute_encoder.set_vector_bytes(std::vector<int>{reduce_dim}, 7);
  compute_encoder.set_vector_bytes(std::vector<int64_t>{reduce_dim_stride}, 8);
  compute_encoder.set_bytes(reduce_ndim, 9);
  compute_encoder.set_bytes(non_col_reductions, 10);
}

void strided_sum(CommandEncoder &compute_encoder, const array &in, const array &out, int S, int64_t inner,
                 int outer) {
  const std::string op = "_reduce_sum" + type_to_name(in.dtype());
  const bool large = in.size() > INT32_MAX;
  const size_t total = S;  // reduction_size * non_col_reductions
  int BN = 32;
  int BM = 1024 / BN;
  int threadgroup_size = 8 * 32;

  // Small column
  if (total < 32) {
    compute_encoder.set_compute_pipeline_state(concatenate("col_reduce_small", large ? "_large" : "", "_1", op));
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(out, 1);
    encode_col_reduce_args(compute_encoder, S, inner, outer, S, inner);
    const int n_reads = 4;
    size_t reduction_stride_blocks = (inner + n_reads - 1) / n_reads;
    size_t threadgroup_x = std::min<size_t>(reduction_stride_blocks, 32);
    size_t threadgroup_y = std::min<size_t>(8, std::min<size_t>(compute_encoder.max_threads() / threadgroup_x, total));
    compute_encoder.dispatch_threadgroups(
        MTLSizeMake((reduction_stride_blocks + threadgroup_x - 1) / threadgroup_x, outer, 1),
        MTLSizeMake(threadgroup_x, threadgroup_y, 1));
    return;
  }

  // Long column but small row
  MLXQ_CHECK(!(inner < 32 && total >= 1024), "a split-K sum of ", S, " partitions of ", inner,
        " values needs col_reduce_longcolumn, which is not transcribed");

  if (total > 256 && out.size() / 32 < 1024) {
    // strided_reduce_2pass
    int outer_blocks = 32;
    std::vector<int> intermediate_shape = {32};
    intermediate_shape.insert(intermediate_shape.end(), out.shape().begin(), out.shape().end());
    array intermediate = compute_encoder.empty(intermediate_shape, out.dtype());
    compute_encoder.set_compute_pipeline_state(
        concatenate("col_reduce_2pass", large ? "_large" : "", "_1_", BM, "_", BN, op));
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(intermediate, 1);
    encode_col_reduce_args(compute_encoder, S, inner, outer, S, inner);
    size_t out_size = out.size() / inner;
    compute_encoder.set_bytes(out_size, 11);
    compute_encoder.dispatch_threads(MTLSizeMake(threadgroup_size * ((inner + BN - 1) / BN), outer * outer_blocks, 1),
                                     MTLSizeMake(threadgroup_size, 1, 1));
    // the 2nd pass: ColReduceArgs(intermediate), plus the outer_blocks reduce dim
    bool large2 = intermediate.size() > INT32_MAX;
    compute_encoder.set_compute_pipeline_state(concatenate("col_reduce_looped", large2 ? "_large" : "", "_1_32_32", op));
    compute_encoder.set_input_array(intermediate, 0);
    compute_encoder.set_output_array(out, 1);
    encode_col_reduce_args(compute_encoder, outer_blocks, out.size(), 1, outer_blocks, out.size());
    compute_encoder.dispatch_threads(MTLSizeMake(threadgroup_size * ((out.size() + BN - 1) / BN), 1, 1),
                                     MTLSizeMake(threadgroup_size, 1, 1));
    return;
  }

  // strided_reduce_looped
  compute_encoder.set_compute_pipeline_state(concatenate("col_reduce_looped", large ? "_large" : "", "_1_", BM, "_", BN, op));
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  encode_col_reduce_args(compute_encoder, S, inner, outer, S, inner);
  compute_encoder.dispatch_threads(MTLSizeMake(threadgroup_size * ((inner + BN - 1) / BN), outer, 1),
                                   MTLSizeMake(threadgroup_size, 1, 1));
}

// ---------------------------------------------------------------------------------------------------
// QuantizedMatmul (quantized.cpp)
// ---------------------------------------------------------------------------------------------------

void qmv_quad(CommandEncoder &compute_encoder, const Args &a) {
  int B = a.B();
  constexpr int quads_per_simd = 8;
  constexpr int results_per_quadgroup = 8;
  int bn = quads_per_simd * results_per_quadgroup;
  int simdgroup_size = 32;
  compute_encoder.set_compute_pipeline_state(concatenate(a.mode, "_qmv_quad_", a.type_string, "_gs_", a.group_size,
                                                         "_b_", a.bits, "_d_", a.K, B > 1 ? "_batch_1" : "_batch_0"));
  int c = 0;
  set_w_scales_biases(compute_encoder, a, c);
  compute_encoder.set_input_array(a.x, c++);
  compute_encoder.set_output_array(a.out, c++);
  compute_encoder.set_bytes(a.K, c++);
  compute_encoder.set_bytes(a.N, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, a.x, a.w, a.scales, a.biases, c++);
  compute_encoder.dispatch_threadgroups(MTLSizeMake(a.M, (a.N + bn - 1) / bn, B), MTLSizeMake(simdgroup_size, 1, 1));
}

void qmv(CommandEncoder &compute_encoder, const Args &a, metal::Device &d) {
  int B = a.B();
  int bn = 8;
  int bk = 32;
  bool fast = a.N % bn == 0 && a.K % qmv_fast_k_alignment(a.bits) == 0;
  // A narrower output tile reduces register pressure for large floating-point quantized
  // matrix-vector products on M5 Max GPUs.
  bool use_narrow_qmv = fast && a.N >= 4096 && d.get_architecture_gen() == 17 &&
                        d.get_architecture().back() == 's' && a.mode == "nvfp4";
  int results_per_simdgroup = use_narrow_qmv ? 2 : 4;
  bn = 2 * results_per_simdgroup;
  compute_encoder.set_compute_pipeline_state(concatenate(a.mode, fast ? "_qmv_fast_" : "_qmv_", a.type_string, "_gs_",
                                                         a.group_size, "_b_", a.bits, use_narrow_qmv ? "_r_2" : "",
                                                         B > 1 ? "_batch_1" : "_batch_0"));
  compute_encoder.set_input_array(a.w, 0);
  compute_encoder.set_input_array(a.scales, 1);
  if (a.biases) compute_encoder.set_input_array(*a.biases, 2);
  int c = 3;
  compute_encoder.set_input_array(a.x, c++);
  compute_encoder.set_output_array(a.out, c++);
  compute_encoder.set_bytes(a.K, c++);
  compute_encoder.set_bytes(a.N, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, a.x, a.w, a.scales, a.biases, c);
  compute_encoder.dispatch_threadgroups(MTLSizeMake(a.M, (a.N + bn - 1) / bn, B), MTLSizeMake(bk, 2, 1));
}

void qmv_wide(CommandEncoder &compute_encoder, const Args &a) {
  // vecs_per_tg is the per-threadgroup input-vector tile: the fewest tiles, then the smallest tile
  // that fills them.
  int n_tiles = (a.M + 4) / 5;  // ceil(M / 5); tile size caps at 5
  int vecs_per_tg = (a.M + n_tiles - 1) / n_tiles;
  int k_lanes = a.mode == "affine" ? 8 : 16;
  constexpr int num_simdgroups = 2;
  int B = a.B();
  bool batched = B > 1;
  int rows_per_tg = (32 / k_lanes) * num_simdgroups;
  compute_encoder.set_compute_pipeline_state(concatenate(a.mode, "_qmv_wide_", a.type_string, "_gs_", a.group_size,
                                                         "_b_", a.bits, "_nv_", vecs_per_tg, "_kl_", k_lanes,
                                                         batched ? "_batch_1" : "_batch_0"));
  int c = 0;
  set_w_scales_biases(compute_encoder, a, c);
  compute_encoder.set_input_array(a.x, c++);
  compute_encoder.set_output_array(a.out, c++);
  compute_encoder.set_bytes(a.K, c++);
  compute_encoder.set_bytes(a.N, c++);
  compute_encoder.set_bytes(a.M, c++);
  add_strides_and_shapes(compute_encoder, !batched, a.x, a.w, a.scales, a.biases, c);
  compute_encoder.dispatch_threadgroups(
      MTLSizeMake((a.M + vecs_per_tg - 1) / vecs_per_tg, (a.N + rows_per_tg - 1) / rows_per_tg, B),
      MTLSizeMake(32, num_simdgroups, 1));
}

void dispatch_qmv(CommandEncoder &compute_encoder, const Args &a, metal::Device &d) {
  // It is a qmv with a small inner dimension so route to qmv_quad kernel
  if ((a.K == 128 || a.K == 64) && is_power_of_2(a.bits)) {
    qmv_quad(compute_encoder, a);
    return;
  }
  // Small batch so route to qmv_wide, which reuses each weight group across the M vectors.
  if (a.M >= 2 && use_qmv_wide(a.mode, d)) {
    qmv_wide(compute_encoder, a);
    return;
  }
  qmv(compute_encoder, a, d);
}

void qvm(CommandEncoder &compute_encoder, const Args &a) {
  int B = a.B();
  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(a.group_size, 32) * num_simdgroups;
  compute_encoder.set_compute_pipeline_state(concatenate(a.mode, "_qvm_", a.type_string, "_gs_", a.group_size, "_b_",
                                                         a.bits, B > 1 ? "_batch_1" : "_batch_0"));
  compute_encoder.set_input_array(a.w, 0);
  compute_encoder.set_input_array(a.scales, 1);
  if (a.biases) compute_encoder.set_input_array(*a.biases, 2);
  int c = 3;
  compute_encoder.set_input_array(a.x, c++);
  compute_encoder.set_output_array(a.out, c++);
  compute_encoder.set_bytes(a.K, c++);
  compute_encoder.set_bytes(a.N, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, a.x, a.w, a.scales, a.biases, c++);
  compute_encoder.dispatch_threadgroups(MTLSizeMake(a.M, (a.N + bn - 1) / bn, B), MTLSizeMake(bk, num_simdgroups, 1));
}

void qvm_split_k(CommandEncoder &compute_encoder, const Args &a) {
  int split_k = a.K > 8192 ? 32 : 8;
  int split_D = (a.K + split_k - 1) / split_k;
  int B = a.B();
  B *= split_k;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(a.group_size, 32) * num_simdgroups;

  auto x_shape = a.x.shape();
  auto x_strides = a.x.strides();
  if (x_shape.size() == 1) {
    x_shape.insert(x_shape.begin(), 1);
    x_strides.insert(x_strides.begin(), 0);
  }

  int x_ndim = x_shape.size();
  int x_batch_ndims = x_ndim - 2;
  int w_batch_ndims = a.w.ndim() - 2;
  auto w_shape = a.w.shape();
  auto w_strides = a.w.strides();
  auto s_strides = a.scales.strides();

  // Add split_k dim with reshapes
  x_shape.insert(x_shape.end() - 2, split_k);
  x_shape.back() /= split_k;
  x_strides.insert(x_strides.end() - 2, split_D);
  x_strides[x_ndim - 1] = split_D;
  x_batch_ndims += 1;

  w_shape.insert(w_shape.end() - 2, split_k);
  w_shape[a.w.ndim() - 1] /= split_k;
  w_strides.insert(w_strides.end() - 2, split_D * a.w.shape(-1));
  w_batch_ndims += 1;
  s_strides.insert(s_strides.end() - 2, split_D * a.scales.shape(-1));

  int final_block_size = a.K - (split_k - 1) * split_D;

  auto temp_shape = a.out.shape();
  if (temp_shape.size() == 1) {
    temp_shape.insert(temp_shape.begin(), 1);
  }
  temp_shape.insert(temp_shape.end() - 2, split_k);
  array intermediate = compute_encoder.empty(temp_shape, a.x.dtype());

  compute_encoder.set_compute_pipeline_state(concatenate(a.mode, "_qvm_split_k_", a.type_string, "_gs_", a.group_size,
                                                         "_b_", a.bits, "_spk_", split_k));
  int c = 0;
  set_w_scales_biases(compute_encoder, a, c);
  compute_encoder.set_input_array(a.x, c++);
  compute_encoder.set_output_array(intermediate, c++);
  compute_encoder.set_bytes(split_D, c++);
  compute_encoder.set_bytes(a.N, c++);

  compute_encoder.set_bytes(x_batch_ndims, c++);
  compute_encoder.set_vector_bytes(x_shape, c++);
  compute_encoder.set_vector_bytes(x_strides, c++);
  compute_encoder.set_bytes(w_batch_ndims, c++);
  compute_encoder.set_vector_bytes(w_shape, c++);
  compute_encoder.set_vector_bytes(w_strides, c++);
  compute_encoder.set_vector_bytes(s_strides, c++);
  if (a.biases) {
    auto b_strides = a.biases->strides();
    b_strides.insert(b_strides.end() - 2, split_D * a.biases->shape(-1));
    compute_encoder.set_vector_bytes(b_strides, c++);
  }
  compute_encoder.set_bytes(final_block_size, c++);
  compute_encoder.dispatch_threadgroups(MTLSizeMake(a.M, (a.N + bn - 1) / bn, B), MTLSizeMake(bk, num_simdgroups, 1));

  int axis = intermediate.ndim() - 3;
  int outer = 1;
  for (int i = 0; i < axis; ++i) outer *= intermediate.shape(i);
  strided_sum(compute_encoder, intermediate, a.out, intermediate.shape(axis), intermediate.strides(axis), outer);
}

void qmm_nax(CommandEncoder &compute_encoder, const Args &a) {
  int B = a.B();
  int wm = 2;
  int wn = 2;
  // Use smaller bm when one block covers all of M. Only qmm_t_nax has a 32-row instantiation.
  int bm = (a.transpose && a.M <= 32) ? 32 : 64;
  int bn = 64;
  int bk = 64;
  bool aligned = a.N % 64 == 0;
  bool batched = B > 1;
  compute_encoder.set_compute_pipeline_state(concatenate(
      a.mode, a.transpose ? "_qmm_t_nax_" : "_qmm_n_nax_", a.type_string, "_gs_", a.group_size, "_b_", a.bits, "_bm",
      bm, "_bn", bn, "_bk", bk, "_wm", wm, "_wn", wn, a.transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
      batched ? "_batch_1" : "_batch_0"));
  int c = 0;
  compute_encoder.set_input_array(a.w, c++);
  compute_encoder.set_input_array(a.scales, c++);
  if (a.biases) {
    compute_encoder.set_input_array(*a.biases, c++);
  } else if (a.transpose) {
    c++;
  }
  compute_encoder.set_input_array(a.x, c++);
  compute_encoder.set_output_array(a.out, c++);
  compute_encoder.set_bytes(a.K, c++);
  compute_encoder.set_bytes(a.N, c++);
  compute_encoder.set_bytes(a.M, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, a.x, a.w, a.scales, a.biases, c);
  compute_encoder.dispatch_threadgroups(MTLSizeMake((a.N + bn - 1) / bn, (a.M + bm - 1) / bm, B),
                                        MTLSizeMake(32, wn, wm));
}

void qmm(CommandEncoder &compute_encoder, const Args &a) {
  bool has_nax_kernel = is_nax_available() && (a.transpose || a.mode == "affine");
  bool nax_aligned = (a.K % 64 == 0) && (a.transpose || a.N % 64 == 0);
  if (has_nax_kernel && nax_aligned && (enable_tf32() || a.x.dtype() != Dtype::float32)) {
    qmm_nax(compute_encoder, a);
    return;
  }

  int B = a.B();
  int wm = 2;
  int wn = 2;
  int bm = 32;
  int bn = 32;
  bool aligned = a.N % 32 == 0;
  bool batched = B > 1;
  compute_encoder.set_compute_pipeline_state(concatenate(
      a.mode, a.transpose ? "_qmm_t_" : "_qmm_n_", a.type_string, "_gs_", a.group_size, "_b_", a.bits,
      a.transpose ? (aligned ? "_alN_true" : "_alN_false") : "", batched ? "_batch_1" : "_batch_0"));
  int c = 0;
  compute_encoder.set_input_array(a.w, c++);
  compute_encoder.set_input_array(a.scales, c++);
  if (a.biases) {
    compute_encoder.set_input_array(*a.biases, c++);
  } else if (a.transpose) {
    c++;
  }
  compute_encoder.set_input_array(a.x, c++);
  compute_encoder.set_output_array(a.out, c++);
  compute_encoder.set_bytes(a.K, c++);
  compute_encoder.set_bytes(a.N, c++);
  compute_encoder.set_bytes(a.M, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, a.x, a.w, a.scales, a.biases, c);
  compute_encoder.dispatch_threadgroups(MTLSizeMake((a.N + bn - 1) / bn, (a.M + bm - 1) / bm, B),
                                        MTLSizeMake(32, wn, wm));
}

void qmm_splitk(CommandEncoder &compute_encoder, const Args &a) {
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
    return qmm(compute_encoder, a);
  }

  int k_partition_size = a.K / split_k;
  int split_k_partition_stride = a.M * a.N;

  // Intermediate buffer: split_k at the front so that partition_stride = M * N matches its leading
  // stride.
  auto temp_shape = a.out.shape();
  if (temp_shape.size() == 1) {
    temp_shape.insert(temp_shape.begin(), 1);
  }
  temp_shape.insert(temp_shape.begin(), split_k);
  array intermediate = compute_encoder.empty(temp_shape, a.x.dtype());

  bool aligned = a.N % 32 == 0;
  compute_encoder.set_compute_pipeline_state(concatenate(a.mode, "_qmm_t_splitk_", a.type_string, "_gs_",
                                                         a.group_size, "_b_", a.bits,
                                                         aligned ? "_alN_true" : "_alN_false"));
  int c = 0;
  set_w_scales_biases(compute_encoder, a, c);
  compute_encoder.set_input_array(a.x, c++);
  compute_encoder.set_output_array(intermediate, c++);
  compute_encoder.set_bytes(a.K, c++);
  compute_encoder.set_bytes(a.N, c++);
  compute_encoder.set_bytes(a.M, c++);
  compute_encoder.set_bytes(k_partition_size, c++);
  compute_encoder.set_bytes(split_k_partition_stride, c++);
  compute_encoder.dispatch_threadgroups(MTLSizeMake(n_tiles, m_tiles, split_k), MTLSizeMake(32, 2, 2));

  // Sum across split_k dimension (axis 0)
  strided_sum(compute_encoder, intermediate, a.out, split_k, intermediate.strides(0), 1);
}

// QuantizedMatmul::eval_gpu, from "Extract the matmul shapes" on.
void quantized_matmul_eval(CommandEncoder &compute_encoder, const Args &a) {
  auto d = device();
  int vector_limit = a.transpose ? get_qmv_batch_limit(a.K, a.N, d) : 4;
  // It is a matrix matrix product.
  if (a.M >= vector_limit) {
    // Use split-K qmm for small M with transposed weights (non-batched only)
    if (a.transpose && a.B() == 1) {
      return qmm_splitk(compute_encoder, a);
    }
    return qmm(compute_encoder, a);
  }
  // Run of the mill qmv
  if (a.transpose) {
    return dispatch_qmv(compute_encoder, a, d);
  }
  // Run of the mill qvm
  if (a.K < 1024) {
    return qvm(compute_encoder, a);
  }
  // Qvm with large dimension so route to a split K kernel for more parallelism
  qvm_split_k(compute_encoder, a);
}

// ---------------------------------------------------------------------------------------------------
// GatherQMM (quantized.cpp)
// ---------------------------------------------------------------------------------------------------

struct GatherArgs : Args {
  array lhs_indices, rhs_indices;
  bool right_sorted;
};

void set_gather_inputs(CommandEncoder &compute_encoder, const GatherArgs &a) {
  compute_encoder.set_input_array(a.w, 0);
  compute_encoder.set_input_array(a.scales, 1);
  if (a.biases) {
    compute_encoder.set_input_array(*a.biases, 2);
  } else if (a.global_scale) {
    compute_encoder.set_input_array(*a.global_scale, 2);
  }
}

const char *hgs(const Args &a) { return a.global_scale ? "_hgs" : ""; }

void gather_qmm_nax(CommandEncoder &compute_encoder, const GatherArgs &a) {
  int B = a.B();
  int wm = 2;
  int wn = 2;
  int bm = 64;
  int bn = 64;
  // The gather qmm NAX kernels are instantiated with BK = 64 only.
  int bk = 64;
  bool aligned = a.N % 64 == 0;
  compute_encoder.set_compute_pipeline_state(concatenate(
      a.mode, a.transpose ? "_gather_qmm_t_nax_" : "_gather_qmm_n_nax_", a.type_string, "_gs_", a.group_size, "_b_",
      a.bits, "_bm", bm, "_bn", bn, "_bk", bk, "_wm", wm, "_wn", wn,
      a.transpose ? (aligned ? "_alN_true" : "_alN_false") : "", hgs(a)));
  set_gather_inputs(compute_encoder, a);
  int c = 3;
  compute_encoder.set_input_array(a.x, c++);
  compute_encoder.set_input_array(a.lhs_indices, c++);
  compute_encoder.set_input_array(a.rhs_indices, c++);
  compute_encoder.set_output_array(a.out, c++);
  compute_encoder.set_bytes(a.K, c++);
  compute_encoder.set_bytes(a.N, c++);
  compute_encoder.set_bytes(a.M, c++);
  c = add_strides_and_shapes(compute_encoder, false, a.x, a.w, a.scales, a.biases, c);
  add_gather_strides_and_shapes(compute_encoder, a.lhs_indices, a.rhs_indices, c);
  compute_encoder.dispatch_threadgroups(MTLSizeMake((a.N + bn - 1) / bn, (a.M + bm - 1) / bm, B),
                                        MTLSizeMake(32, wn, wm));
}

void gather_qmm(CommandEncoder &compute_encoder, const GatherArgs &a) {
  if (is_nax_available() && a.transpose && (a.K % 64 == 0) &&
      (enable_tf32() || a.x.dtype() != Dtype::float32)) {
    return gather_qmm_nax(compute_encoder, a);
  }
  int B = a.B();
  int wm = 2;
  int wn = 2;
  int bm = 32;
  int bn = 32;
  bool aligned = a.N % 32 == 0;
  compute_encoder.set_compute_pipeline_state(concatenate(
      a.mode, a.transpose ? "_gather_qmm_t_" : "_gather_qmm_n_", a.type_string, "_gs_", a.group_size, "_b_", a.bits,
      a.transpose ? (aligned ? "_alN_true" : "_alN_false") : "", hgs(a)));
  set_gather_inputs(compute_encoder, a);
  int c = 3;
  compute_encoder.set_input_array(a.x, c++);
  compute_encoder.set_input_array(a.lhs_indices, c++);
  compute_encoder.set_input_array(a.rhs_indices, c++);
  compute_encoder.set_output_array(a.out, c++);
  compute_encoder.set_bytes(a.K, c++);
  compute_encoder.set_bytes(a.N, c++);
  compute_encoder.set_bytes(a.M, c++);
  c = add_strides_and_shapes(compute_encoder, false, a.x, a.w, a.scales, a.biases, c);
  add_gather_strides_and_shapes(compute_encoder, a.lhs_indices, a.rhs_indices, c);
  compute_encoder.dispatch_threadgroups(MTLSizeMake((a.N + bn - 1) / bn, (a.M + bm - 1) / bm, B),
                                        MTLSizeMake(32, wn, wm));
}

void gather_qmv(CommandEncoder &compute_encoder, const GatherArgs &a) {
  int B = a.B();
  int bn = 8;
  int bk = 32;
  bool fast = a.N % bn == 0 && a.K % qmv_fast_k_alignment(a.bits) == 0;
  compute_encoder.set_compute_pipeline_state(concatenate(a.mode, fast ? "_gather_qmv_fast_" : "_gather_qmv_",
                                                         a.type_string, "_gs_", a.group_size, "_b_", a.bits, hgs(a)));
  set_gather_inputs(compute_encoder, a);
  int c = 3;
  compute_encoder.set_input_array(a.x, c++);
  compute_encoder.set_input_array(a.lhs_indices, c++);
  compute_encoder.set_input_array(a.rhs_indices, c++);
  compute_encoder.set_output_array(a.out, c++);
  compute_encoder.set_bytes(a.K, c++);
  compute_encoder.set_bytes(a.N, c++);
  c = add_strides_and_shapes(compute_encoder, false, a.x, a.w, a.scales, a.biases, c);
  add_gather_strides_and_shapes(compute_encoder, a.lhs_indices, a.rhs_indices, c);
  compute_encoder.dispatch_threadgroups(MTLSizeMake(a.M, (a.N + bn - 1) / bn, B), MTLSizeMake(bk, 2, 1));
}

void gather_qvm(CommandEncoder &compute_encoder, const GatherArgs &a) {
  int B = a.B();
  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(a.group_size, 32) * num_simdgroups;
  compute_encoder.set_compute_pipeline_state(
      concatenate(a.mode, "_gather_qvm_", a.type_string, "_gs_", a.group_size, "_b_", a.bits, hgs(a)));
  set_gather_inputs(compute_encoder, a);
  int c = 3;
  compute_encoder.set_input_array(a.x, c++);
  compute_encoder.set_input_array(a.lhs_indices, c++);
  compute_encoder.set_input_array(a.rhs_indices, c++);
  compute_encoder.set_output_array(a.out, c++);
  compute_encoder.set_bytes(a.K, c++);
  compute_encoder.set_bytes(a.N, c++);
  c = add_strides_and_shapes(compute_encoder, false, a.x, a.w, a.scales, a.biases, c);
  add_gather_strides_and_shapes(compute_encoder, a.lhs_indices, a.rhs_indices, c);
  compute_encoder.dispatch_threadgroups(MTLSizeMake(a.M, (a.N + bn - 1) / bn, B), MTLSizeMake(bk, num_simdgroups, 1));
}

// gather_mm_offsets (matmul.cpp)
array gather_mm_offsets(CommandEncoder &compute_encoder, const array &indices, int num_groups, int M) {
  array offsets = compute_encoder.empty({num_groups}, Dtype::int32);
  compute_encoder.set_compute_pipeline_state("gather_mm_offsets");
  compute_encoder.set_input_array(indices, 0);
  compute_encoder.set_output_array(offsets, 1);
  compute_encoder.set_bytes(M, 2);
  size_t group_size = std::min<size_t>(num_groups, compute_encoder.max_threads());
  compute_encoder.dispatch_threads(MTLSizeMake(num_groups, 1, 1), MTLSizeMake(group_size, 1, 1));
  return offsets;
}

// gather_qmm_rhs and gather_qmm_rhs_nax, which differ only in their tiles and kernel name.
void gather_qmm_rhs(CommandEncoder &compute_encoder, const GatherArgs &a, int M) {
  bool nax = is_nax_available() && a.transpose && (enable_tf32() || a.x.dtype() != Dtype::float32);

  // Start by normalizing the indices
  array indices = compute_encoder.contiguous(a.rhs_indices);

  // Broadcast x with indices. If we are here that means lhs_indices were not provided so the
  // lhs_indices are implied to be the shape of x broadcasted with rhs_indices.
  array x = a.x;
  if (a.x.size() / a.x.shape(-2) / a.x.shape(-1) == indices.size()) {
    x = compute_encoder.contiguous(a.x);
  } else {
    auto x_shape = indices.shape();
    x_shape.push_back(a.x.shape(-2));
    x_shape.push_back(a.x.shape(-1));
    x = compute_encoder.contiguous(a.x, x_shape);
  }
  array w = compute_encoder.contiguous(a.w);
  array scales = compute_encoder.contiguous(a.scales);
  std::optional<array> biases, gs;
  if (a.biases) biases = compute_encoder.contiguous(*a.biases);
  if (a.global_scale) gs = compute_encoder.contiguous(*a.global_scale);

  int E = w.size() / w.shape(-1) / w.shape(-2);
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

  array offsets = gather_mm_offsets(compute_encoder, indices, E, M);

  const char *func = nax ? (a.transpose ? "_gather_qmm_rhs_nax_nt_" : "_gather_qmm_rhs_nax_nn_")
                         : (a.transpose ? "_gather_qmm_rhs_nt_" : "_gather_qmm_rhs_nn_");
  compute_encoder.set_compute_pipeline_state(concatenate(a.mode, func, get_type_string(x.dtype()), "_gs_",
                                                         a.group_size, "_b_", a.bits, "_bm_", bm, "_bn_", bn, "_bk_",
                                                         bk, "_wm_", wm, "_wn_", wn, hgs(a)),
                                             std::make_pair(align_N, align_K));
  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(w, 1);
  compute_encoder.set_input_array(scales, 2);
  if (biases) {
    compute_encoder.set_input_array(*biases, 3);
  } else if (gs) {
    compute_encoder.set_input_array(*gs, 3);
  }
  int c = 4;
  compute_encoder.set_input_array(offsets, c++);
  compute_encoder.set_output_array(a.out, c++);
  compute_encoder.set_bytes(M, c++);
  compute_encoder.set_bytes(a.N, c++);
  compute_encoder.set_bytes(a.K, c++);
  compute_encoder.set_bytes(E, c++);
  compute_encoder.dispatch_threadgroups(MTLSizeMake((a.N + bn - 1) / bn, std::min(M, (M + bm - 1) / bm + E - 1), 1),
                                        MTLSizeMake(32, wn, wm));
}

// GatherQMM::eval_gpu, after its ensure_row_contiguous_matrix calls.
void gather_qmm_eval(CommandEncoder &compute_encoder, const GatherArgs &a) {
  auto d = device();
  int B = a.B();
  int E = a.w.size() / a.w.shape(-1) / a.w.shape(-2);
  int vector_limit = a.transpose ? get_qmv_batch_limit(a.K, a.N, d) : 4;

  // We are walking x in order and w is also in order so we can batch up the matmuls and reuse
  // reading x and w.
  if (a.M == 1 && B >= 16 && a.right_sorted == true && B / E >= 4) {
    return gather_qmm_rhs(compute_encoder, a, a.x.size() / a.K);
  }
  // It is a matrix matrix product
  if (a.M >= vector_limit) {
    return gather_qmm(compute_encoder, a);
  }
  if (a.transpose) {
    return gather_qmv(compute_encoder, a);
  }
  gather_qvm(compute_encoder, a);
}

}  // namespace

// ---------------------------------------------------------------------------------------------------
// The boundary (common.h)
// ---------------------------------------------------------------------------------------------------

namespace mlxq {

std::vector<std::string> quantized_matmul(const array &x, const array &w, const array &scales,
                                          const std::optional<array> &biases, const array &out, bool transpose,
                                          const Quantization &q, Workspace &ws, bool encode) {
  // Extract the matmul shapes
  bool non_batched = w.ndim() == 2 && row_contiguous(x);
  Args a{x, w, scales, biases, std::nullopt, out, q.group_size, q.bits, 0, 0, x.shape(-1),
         q.mode, get_type_string(x.dtype()), transpose};
  a.M = non_batched ? x.size() / a.K : x.shape(-2);
  a.N = out.shape(-1);
  return run(ws, encode, [&](CommandEncoder &e) { quantized_matmul_eval(e, a); });
}

std::vector<std::string> gather_qmm(const array &x, const array &w, const array &scales,
                                    const std::optional<array> &biases, const std::optional<array> &global_scale,
                                    const array &lhs_indices, const array &rhs_indices, const array &out,
                                    bool transpose, bool right_sorted, const Quantization &q, Workspace &ws,
                                    bool encode) {
  GatherArgs a;
  static_cast<Args &>(a) = Args{x, w, scales, biases, global_scale, out, q.group_size, q.bits, x.shape(-2),
                                out.shape(-1), x.shape(-1), q.mode, get_type_string(x.dtype()), transpose};
  a.lhs_indices = lhs_indices;
  a.rhs_indices = rhs_indices;
  a.right_sorted = right_sorted;
  return run(ws, encode, [&](CommandEncoder &e) { gather_qmm_eval(e, a); });
}

// quantize_impl and get_quantize_kernel_dims
std::vector<std::string> quantize(const array &w, const array &out, const array &scales,
                                  const std::optional<array> &biases, const std::optional<array> &global_scale,
                                  bool dequantize, const Quantization &q, Workspace &ws, bool encode) {
  return run(ws, encode, [&](CommandEncoder &compute_encoder) {
    bool has_biases = q.mode == "affine";
    bool has_global_scale = !has_biases && global_scale.has_value();
    std::string kname = concatenate(q.mode, dequantize ? "_dequantize" : "_quantize", "_",
                                    get_type_string(dequantize ? out.dtype() : w.dtype()), "_gs_", q.group_size,
                                    "_b_", q.bits);
    if (!has_biases) {
      kname += concatenate("_hgs_", has_global_scale ? "true" : "false");
    }
    compute_encoder.set_compute_pipeline_state(kname);
    if (dequantize) {
      if (has_biases) {
        compute_encoder.set_input_array(*biases, 2);
      } else if (has_global_scale) {
        compute_encoder.set_input_array(*global_scale, 2);
      }
      compute_encoder.set_input_array(w, 0);
      compute_encoder.set_input_array(scales, 1);
      compute_encoder.set_output_array(out, 3);
    } else {
      if (has_biases) {
        compute_encoder.set_output_array(*biases, 3);
      } else if (has_global_scale) {
        compute_encoder.set_input_array(*global_scale, 3);
      }
      compute_encoder.set_input_array(w, 0);
      compute_encoder.set_output_array(out, 1);
      compute_encoder.set_output_array(scales, 2);
    }

    // get_quantize_kernel_dims. Treat uint32 as uint8 in kernel
    constexpr int simd_size = 32;
    int packs_per_int = (q.bits == 3 || q.bits == 5) ? 8 : q.bits == 6 ? 4 : 8 / q.bits;
    int per_thread = dequantize ? packs_per_int : std::max(q.group_size / simd_size, 1);
    size_t nthreads = dequantize ? out.size() / packs_per_int : w.size() / per_thread;
    MLXQ_CHECK(nthreads <= UINT_MAX, "tensor too large for a 1D grid");
    size_t thread_group_size = std::min<size_t>(compute_encoder.max_threads(), nthreads);
    compute_encoder.dispatch_threads(MTLSizeMake(nthreads, 1, 1), MTLSizeMake(thread_group_size, 1, 1));
  });
}

}  // namespace mlxq
