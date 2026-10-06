/* Metal side: load MLX's quantized metallib and encode its kernels on torch's own stream.
 *
 * The kernels are upstream's, compiled as they ship (vendor/UPSTREAM pins the revision). What lives
 * here is the host side, which cannot be vendored: MLX's `quantized.cpp` is written against its own
 * array and device types. So this file transcribes the part of it a linear layer reaches --
 * `QuantizedMatmul::eval_gpu` with `transpose=true` and a 2D weight -- keeping upstream's choices:
 *
 *   - below `get_qmv_batch_limit` rows, a matrix-vector kernel (`qmv_quad` for K of 64/128,
 *     `qmv_wide` for 2+ rows on gen-15+ GPUs, else `qmv_fast` when aligned, else `qmv`);
 *   - above it, `qmm_t_splitk`, which falls back to `qmm_t` once there is enough parallelism.
 *
 * Not transcribed: the NAX path (upstream's M5 qmm, which needs the Metal 4 toolchain), batched
 * weights, the fp modes (mxfp4/nvfp4/mxfp8) and gather_qmm. `tests/test_vendor_drift.py` checks
 * the transcribed pieces against vendor/mlx/backend/metal/quantized.cpp.
 */

#import <Metal/Metal.h>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <torch/torch.h>

#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>

#include "torch_binding.h"

#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#endif

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

id<MTLLibrary> library() {
  static id<MTLLibrary> lib = [] {
    id<MTLDevice> dev = at::mps::MPSDevice::getInstance()->device();
    NSError *error = nil;
#ifdef EMBEDDED_METALLIB_HEADER
    id<MTLLibrary> l = EMBEDDED_METALLIB_NAMESPACE::createLibrary(dev, &error);
#else
    id<MTLLibrary> l = [dev newDefaultLibrary];
#endif
    TORCH_CHECK(l != nil, "mlx-quantization-metal-kernels: failed to load the metallib: ",
                error ? error.localizedDescription.UTF8String : "unknown error");
    return l;
  }();
  return lib;
}

id<MTLComputePipelineState> pipeline(const std::string &name) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(name);
  if (it != cache.end()) return it->second;

  id<MTLFunction> fn = [library() newFunctionWithName:@(name.c_str())];
  TORCH_CHECK(fn != nil, "mlx-quantization-metal-kernels: no kernel named ", name, " in the metallib");
  NSError *error = nil;
  id<MTLComputePipelineState> state =
      [at::mps::MPSDevice::getInstance()->device() newComputePipelineStateWithFunction:fn error:&error];
  TORCH_CHECK(state != nil, "mlx-quantization-metal-kernels: failed to build pipeline ", name, ": ",
              error ? error.localizedDescription.UTF8String : "unknown error");
  cache[name] = state;
  return state;
}

// Upstream's get_type_string, for the three dtypes the kernels are instantiated for.
std::string type_string(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return "float";
    case at::kHalf: return "float16_t";
    case at::kBFloat16: return "bfloat16_t";
    default: TORCH_CHECK(false, "mlx-quantization-metal-kernels: unsupported dtype ", t);
  }
}

// ---------------------------------------------------------------------------------------------------
// The plan: which kernel, on which grid. Kept apart from the encoding so `kernel_for` can report it.
// ---------------------------------------------------------------------------------------------------

enum class Kind { QmvQuad, QmvWide, Qmv, QmmSplitK, Qmm };

struct Plan {
  Kind kind;
  std::string name;
  MTLSize grid;
  MTLSize group;
  int split_k = 1;  // QmmSplitK only
};

std::string prefix(const char *func, const std::string &type, int group_size, int bits) {
  return std::string("affine_") + func + "_" + type + "_gs_" + std::to_string(group_size) + "_b_" +
         std::to_string(bits);
}

// QuantizedMatmul::eval_gpu with transpose=true, B=1 (a 2D weight and a row-contiguous x).
Plan plan(int M, int N, int K, int group_size, int bits, const std::string &type) {
  metal::Device d = device_info();
  const std::string mode = "affine";

  int vector_limit = get_qmv_batch_limit(K, N, d);
  if (M >= vector_limit) {
    // qmm_splitk: choose split_k to target ~512 threadgroups
    int bm = 32, bn = 32;
    int n_tiles = (N + bn - 1) / bn;
    int m_tiles = (M + bm - 1) / bm;
    int current_tgs = n_tiles * m_tiles;
    int split_k = std::max(1, 512 / current_tgs);
    int k_align = group_size > 32 ? group_size : 32;
    split_k = std::min(split_k, K / k_align);
    while (split_k > 1 && (K % (split_k * k_align) != 0)) {
      split_k--;
    }
    bool aligned = N % 32 == 0;
    const char *al = aligned ? "_alN_true" : "_alN_false";
    if (split_k > 1) {
      return {Kind::QmmSplitK, prefix("qmm_t_splitk", type, group_size, bits) + al,
              MTLSizeMake(n_tiles, m_tiles, split_k), MTLSizeMake(32, 2, 2), split_k};
    }
    // qmm (upstream's non-NAX path)
    return {Kind::Qmm, prefix("qmm_t", type, group_size, bits) + al + "_batch_0",
            MTLSizeMake(n_tiles, m_tiles, 1), MTLSizeMake(32, 2, 2)};
  }

  // dispatch_qmv
  if ((K == 128 || K == 64) && is_power_of_2(bits)) {
    // qmv_quad
    int bn = 8 * 8;  // quads_per_simd * results_per_quadgroup
    return {Kind::QmvQuad,
            prefix("qmv_quad", type, group_size, bits) + "_d_" + std::to_string(K) + "_batch_0",
            MTLSizeMake(M, (N + bn - 1) / bn, 1), MTLSizeMake(32, 1, 1)};
  }
  if (M >= 2 && use_qmv_wide(mode, d)) {
    int n_tiles = (M + 4) / 5;
    int vecs_per_tg = (M + n_tiles - 1) / n_tiles;
    int k_lanes = 8;
    int rows_per_tg = (32 / k_lanes) * 2;
    return {Kind::QmvWide,
            prefix("qmv_wide", type, group_size, bits) + "_nv_" + std::to_string(vecs_per_tg) +
                "_kl_" + std::to_string(k_lanes) + "_batch_0",
            MTLSizeMake((M + vecs_per_tg - 1) / vecs_per_tg, (N + rows_per_tg - 1) / rows_per_tg, 1),
            MTLSizeMake(32, 2, 1)};
  }
  // qmv (results_per_simdgroup is 4 for affine; upstream narrows it only for nvfp4)
  int bn = 8;
  bool fast = N % bn == 0 && K % qmv_fast_k_alignment(bits) == 0;
  return {Kind::Qmv, prefix(fast ? "qmv_fast" : "qmv", type, group_size, bits) + "_batch_0",
          MTLSizeMake(M, (N + bn - 1) / bn, 1), MTLSizeMake(32, 2, 1)};
}

void set(id<MTLComputeCommandEncoder> enc, const at::Tensor &t, int index) {
  // an MPS tensor's storage is a whole MTLBuffer that the tensor may be a view into
  [enc setBuffer:(__bridge id<MTLBuffer>)t.storage().data()
          offset:t.storage_offset() * t.element_size()
         atIndex:index];
}

// ---------------------------------------------------------------------------------------------------
// Split-K's sum: upstream's strided_reduce_general_dispatch (reduce.cpp), as qmm_splitk calls it.
//
// That call reduces axis 0 of a row-contiguous [split_k, M, N] intermediate, for which upstream's
// ColReduceArgs comes out as reduction_size = split_k, reduction_stride = M * N, no outer dims
// (ndim 0) and non_col_reductions 1; each kernel then appends (reduction_size, reduction_stride) as
// its single reduce dim. output_grid_for_col_reduce is (1, 1, 1): every output stride is below
// M * N. Sum keeps the input dtype (remap_reduce_types), so fp16 partials are summed in fp16.
// ---------------------------------------------------------------------------------------------------

enum class Reduce { Small, Looped, TwoPass };

struct ReducePlan {
  Reduce kind;
  bool large;  // in.size() > INT32_MAX selects the int64-indexed instantiations
};

ReducePlan reduce_plan(int64_t split_k, int64_t MN) {
  const size_t total = split_k;  // reduction_size * non_col_reductions
  if (total < 32) return {Reduce::Small, split_k * MN > INT32_MAX};
  TORCH_CHECK(!(MN < 32 && total >= 1024),
              "mlx-quantization-metal-kernels: split_k=", split_k, " with M*N=", MN,
              " needs upstream's col_reduce_longcolumn, which is not transcribed");
  if (total > 256 && MN / 32 < 1024) return {Reduce::TwoPass, split_k * MN > INT32_MAX};
  return {Reduce::Looped, split_k * MN > INT32_MAX};
}

// Upstream's type_to_name, for the reduce kernels' names.
std::string reduce_type_name(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return "float32";
    case at::kHalf: return "float16";
    case at::kBFloat16: return "bfloat16";
    default: TORCH_CHECK(false, "mlx-quantization-metal-kernels: unsupported dtype ", t);
  }
}

std::string reduce_kernel_name(const char *func, bool large, const char *tile, at::ScalarType t) {
  return std::string(func) + (large ? "_large" : "") + "_1" + tile + "_reduce_sum" + reduce_type_name(t);
}

// ColReduceArgs::encode, with the one reduce dim the kernel appended.
void encode_col_reduce_args(id<MTLComputeCommandEncoder> enc, size_t reduction_size,
                            int64_t reduction_stride, int reduce_dim, int64_t reduce_dim_stride) {
  const int zero_i = 0, ndim = 0, reduce_ndim = 1;
  const int64_t zero_l = 0;
  const size_t non_col_reductions = 1;
  [enc setBytes:&reduction_size length:sizeof(size_t) atIndex:2];
  [enc setBytes:&reduction_stride length:sizeof(int64_t) atIndex:3];
  [enc setBytes:&zero_i length:sizeof(int) atIndex:4];  // shape: empty, pushed as {0}
  [enc setBytes:&zero_l length:sizeof(int64_t) atIndex:5];  // strides: empty, pushed as {0}
  [enc setBytes:&ndim length:sizeof(int) atIndex:6];
  [enc setBytes:&reduce_dim length:sizeof(int) atIndex:7];
  [enc setBytes:&reduce_dim_stride length:sizeof(int64_t) atIndex:8];
  [enc setBytes:&reduce_ndim length:sizeof(int) atIndex:9];
  [enc setBytes:&non_col_reductions length:sizeof(size_t) atIndex:10];
}

// Sums `partials` [split_k, M, N] over axis 0 into `out` [M, N]. `scratch` is the [32, M, N]
// accumulator strided_reduce_2pass allocates, and is only read for that path.
void encode_split_k_sum(id<MTLComputeCommandEncoder> enc, const ReducePlan &rp,
                        const at::Tensor &partials, const at::Tensor &out, const at::Tensor &scratch,
                        int64_t split_k, int64_t MN) {
  const at::ScalarType t = partials.scalar_type();
  const int BN = 32;
  const int threadgroup_size = 8 * 32;
  switch (rp.kind) {
    case Reduce::Small: {
      id<MTLComputePipelineState> pso = pipeline(reduce_kernel_name("col_reduce_small", rp.large, "", t));
      [enc setComputePipelineState:pso];
      set(enc, partials, 0);
      set(enc, out, 1);
      encode_col_reduce_args(enc, split_k, MN, split_k, MN);
      const int n_reads = 4;
      const size_t reduction_stride_blocks = (MN + n_reads - 1) / n_reads;
      const size_t total = split_k;
      const size_t threadgroup_x = std::min<size_t>(reduction_stride_blocks, 32);
      const size_t threadgroup_y =
          std::min<size_t>(8, std::min<size_t>(pso.maxTotalThreadsPerThreadgroup / threadgroup_x, total));
      [enc dispatchThreadgroups:MTLSizeMake((reduction_stride_blocks + threadgroup_x - 1) / threadgroup_x, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threadgroup_x, threadgroup_y, 1)];
      break;
    }
    case Reduce::Looped: {
      [enc setComputePipelineState:pipeline(reduce_kernel_name("col_reduce_looped", rp.large, "_32_32", t))];
      set(enc, partials, 0);
      set(enc, out, 1);
      encode_col_reduce_args(enc, split_k, MN, split_k, MN);
      [enc dispatchThreads:MTLSizeMake(threadgroup_size * ((MN + BN - 1) / BN), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
      break;
    }
    case Reduce::TwoPass: {
      const int outer_blocks = 32;
      [enc setComputePipelineState:pipeline(reduce_kernel_name("col_reduce_2pass", rp.large, "_32_32", t))];
      set(enc, partials, 0);
      set(enc, scratch, 1);
      encode_col_reduce_args(enc, split_k, MN, split_k, MN);
      const size_t out_size = 1;  // out.size() / reduction_stride
      [enc setBytes:&out_size length:sizeof(size_t) atIndex:11];
      [enc dispatchThreads:MTLSizeMake(threadgroup_size * ((MN + BN - 1) / BN), outer_blocks, 1)
          threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
      // second pass: ColReduceArgs(intermediate), plus the outer_blocks reduce dim
      const bool large = int64_t(outer_blocks) * MN > INT32_MAX;
      [enc setComputePipelineState:pipeline(reduce_kernel_name("col_reduce_looped", large, "_32_32", t))];
      set(enc, scratch, 0);
      set(enc, out, 1);
      encode_col_reduce_args(enc, outer_blocks, MN, outer_blocks, MN);
      [enc dispatchThreads:MTLSizeMake(threadgroup_size * ((MN + BN - 1) / BN), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
      break;
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------------------------------

at::Tensor affine_qmm_t(const at::Tensor &x, const at::Tensor &w, const at::Tensor &scales,
                        const at::Tensor &biases, int64_t group_size, int64_t bits) {
  TORCH_CHECK(x.is_mps() && w.is_mps() && scales.is_mps() && biases.is_mps(),
              "mlx-quantization-metal-kernels: all inputs must be on mps");
  TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
              "mlx-quantization-metal-kernels: group_size must be 32, 64 or 128, got ", group_size);
  TORCH_CHECK(bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8,
              "mlx-quantization-metal-kernels: bits must be one of 2, 3, 4, 5, 6, 8, got ", bits);
  TORCH_CHECK(w.dim() == 2 && w.scalar_type() == at::kUInt32,
              "mlx-quantization-metal-kernels: w must be a 2D uint32 tensor [N, K * bits / 32]");
  TORCH_CHECK(x.dim() >= 1, "mlx-quantization-metal-kernels: x must have at least one dimension");
  TORCH_CHECK(scales.scalar_type() == x.scalar_type() && biases.scalar_type() == x.scalar_type(),
              "mlx-quantization-metal-kernels: scales and biases must have x's dtype (", x.scalar_type(),
              "), got ", scales.scalar_type(), " and ", biases.scalar_type());

  const int64_t K = x.size(-1);
  const int64_t N = w.size(0);
  TORCH_CHECK(K % group_size == 0, "mlx-quantization-metal-kernels: K=", K, " is not a multiple of group_size=",
              group_size);
  TORCH_CHECK(w.size(1) * 32 == K * bits, "mlx-quantization-metal-kernels: w is ", w.sizes(), " but K=", K,
              " at ", bits, " bits packs to [", N, ", ", K * bits / 32, "]");
  const std::vector<int64_t> sb_shape = {N, K / group_size};
  TORCH_CHECK(scales.sizes() == sb_shape && biases.sizes() == sb_shape,
              "mlx-quantization-metal-kernels: scales and biases must be [", N, ", ", K / group_size, "], got ",
              scales.sizes(), " and ", biases.sizes());

  // Upstream's non-batched case: a 2D weight and a row-contiguous x, so every leading dim of x
  // folds into M. The kernels assume row-contiguous operands (upstream's
  // ensure_row_contiguous_matrix), so anything else is copied first.
  const auto x2 = x.reshape({-1, K}).contiguous();
  const auto wc = w.contiguous();
  const auto sc = scales.contiguous();
  const auto bc = biases.contiguous();
  const int64_t M = x2.size(0);

  auto out_shape = x.sizes().vec();
  out_shape.back() = N;
  if (M == 0 || N == 0) return at::zeros(out_shape, x.options());
  TORCH_CHECK(M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX,
              "mlx-quantization-metal-kernels: dimensions must fit in int32");

  const Plan p = plan(M, N, K, group_size, bits, type_string(x.scalar_type()));
  auto out = at::empty({M, N}, x.options());
  // split-K writes one partial product per K partition, then sums them the way upstream does
  const bool split = p.kind == Kind::QmmSplitK;
  auto target = split ? at::empty({p.split_k, M, N}, x.options()) : out;
  const ReducePlan rp = split ? reduce_plan(p.split_k, M * N) : ReducePlan{};
  const at::Tensor scratch =
      split && rp.kind == Reduce::TwoPass ? at::empty({32, M, N}, x.options()) : at::Tensor();

  const int Ki = K, Ni = N, Mi = M;
  id<MTLComputePipelineState> pso = pipeline(p.name);
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      set(enc, wc, 0);
      set(enc, sc, 1);
      set(enc, bc, 2);
      set(enc, x2, 3);
      set(enc, target, 4);
      switch (p.kind) {
        case Kind::QmvQuad:
        case Kind::Qmv:
          [enc setBytes:&Ki length:sizeof(int) atIndex:5];
          [enc setBytes:&Ni length:sizeof(int) atIndex:6];
          break;
        case Kind::QmvWide:
        case Kind::Qmm:
          [enc setBytes:&Ki length:sizeof(int) atIndex:5];
          [enc setBytes:&Ni length:sizeof(int) atIndex:6];
          [enc setBytes:&Mi length:sizeof(int) atIndex:7];
          break;
        case Kind::QmmSplitK: {
          const int k_partition_size = Ki / p.split_k;
          const int split_k_partition_stride = Mi * Ni;
          [enc setBytes:&Ki length:sizeof(int) atIndex:5];
          [enc setBytes:&Ni length:sizeof(int) atIndex:6];
          [enc setBytes:&Mi length:sizeof(int) atIndex:7];
          [enc setBytes:&k_partition_size length:sizeof(int) atIndex:8];
          [enc setBytes:&split_k_partition_stride length:sizeof(int) atIndex:9];
          break;
        }
      }
      [enc dispatchThreadgroups:p.grid threadsPerThreadgroup:p.group];
      if (split) encode_split_k_sum(enc, rp, target, out, scratch, p.split_k, M * N);
    }
  });
  return out.view(out_shape);
}

std::string kernel_for(int64_t M, int64_t N, int64_t K, int64_t group_size, int64_t bits,
                       at::ScalarType dtype) {
  const Plan p = plan(M, N, K, group_size, bits, type_string(dtype));
  if (p.kind != Kind::QmmSplitK) return p.name;
  const ReducePlan rp = reduce_plan(p.split_k, M * N);
  const char *reduce = rp.kind == Reduce::Small ? "col_reduce_small"
                       : rp.kind == Reduce::Looped ? "col_reduce_looped"
                                                   : "col_reduce_2pass";
  return p.name + " x split_k=" + std::to_string(p.split_k) + " + " + reduce;
}
