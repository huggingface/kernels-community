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
  // split-K writes one partial product per K partition, summed below
  auto target = p.kind == Kind::QmmSplitK ? at::empty({p.split_k, M, N}, x.options()) : out;

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
    }
  });

  if (p.kind == Kind::QmmSplitK) {
    // upstream sums the partitions with a strided reduce; torch's sum is the same reduction
    at::sum_out(out, target, {0});
  }
  return out.view(out_shape);
}

std::string kernel_for(int64_t M, int64_t N, int64_t K, int64_t group_size, int64_t bits,
                       at::ScalarType dtype) {
  const Plan p = plan(M, N, K, group_size, bits, type_string(dtype));
  return p.kind == Kind::QmmSplitK ? p.name + " x split_k=" + std::to_string(p.split_k) : p.name;
}
