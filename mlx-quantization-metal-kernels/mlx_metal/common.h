#pragma once

/* The boundary between the torch side (mlx_quantization.cpp) and the Metal side (mlx_dispatch.mm).
 *
 * The Metal side transcribes MLX's launchers, which are written against `mlx::core::array`; `array`
 * below is the part of it they read, so the transcription keeps upstream's spelling (`x.shape(-1)`,
 * `w.strides()`). Torch tensors stay on the torch side: a tensor crosses as its MTLBuffer, the byte
 * offset of its view into it, and its shape and strides.
 */

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlxq {

enum class Dtype { float32, float16, bfloat16, uint8, uint32, int32 };

struct array {
  void *buffer = nullptr;  // the MTLBuffer holding the storage (null on the meta device)
  size_t offset = 0;       // in bytes
  std::vector<int> shape_;
  std::vector<int64_t> strides_;  // in elements
  Dtype dtype_ = Dtype::float32;

  int ndim() const { return shape_.size(); }
  int shape(int dim) const { return shape_[dim < 0 ? dim + ndim() : dim]; }
  const std::vector<int> &shape() const { return shape_; }
  int64_t strides(int dim) const { return strides_[dim < 0 ? dim + ndim() : dim]; }
  const std::vector<int64_t> &strides() const { return strides_; }
  Dtype dtype() const { return dtype_; }
  size_t size() const {
    size_t n = 1;
    for (int s : shape_) n *= s;
    return n;
  }
};

// What a launcher needs beyond its inputs: temporaries, and the copies upstream makes with
// ensure_row_contiguous / broadcast. The torch side provides them (from torch's allocator); they
// are only requested while planning, outside torch's stream queue.
struct Workspace {
  virtual ~Workspace() = default;
  virtual array empty(const std::vector<int> &shape, Dtype dtype) = 0;
  // `a` as a row-contiguous copy, broadcast to `shape` first if one is given
  virtual array contiguous(const array &a, const std::optional<std::vector<int>> &shape = std::nullopt) = 0;
};

struct Quantization {
  std::string mode;  // "affine", "mxfp4", "mxfp8" or "nvfp4"
  int group_size;
  int bits;
};

// Each runs upstream's eval_gpu on prepared inputs (the op-level checks, casts and broadcasts are
// the torch side's) and returns the kernels it launches, in order. With `encode` false it only
// plans, which is what the tests trace with.

// QuantizedMatmul::eval_gpu
std::vector<std::string> quantized_matmul(const array &x, const array &w, const array &scales,
                                          const std::optional<array> &biases, const array &out, bool transpose,
                                          const Quantization &q, Workspace &ws, bool encode);

// GatherQMM::eval_gpu
std::vector<std::string> gather_qmm(const array &x, const array &w, const array &scales,
                                    const std::optional<array> &biases, const std::optional<array> &global_scale,
                                    const array &lhs_indices, const array &rhs_indices, const array &out,
                                    bool transpose, bool right_sorted, const Quantization &q, Workspace &ws,
                                    bool encode);

// fast::Quantize::eval_gpu: quantize `w` into (`out`, `scales`, `biases`), or with `dequantize`,
// `w`/`scales`/`biases` back into `out`.
std::vector<std::string> quantize(const array &w, const array &out, const array &scales,
                                  const std::optional<array> &biases, const std::optional<array> &global_scale,
                                  bool dequantize, const Quantization &q, Workspace &ws, bool encode);

}  // namespace mlxq
