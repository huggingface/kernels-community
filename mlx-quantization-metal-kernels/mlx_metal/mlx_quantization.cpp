/* Torch side: the ops as mlx/ops.cpp defines them, on torch tensors.
 *
 * This is upstream's op layer -- argument checks, per-mode defaults, dtype promotion, broadcasting,
 * the default gather indices -- plus each primitive's ensure_row_contiguous_matrix calls. What is
 * left (choosing and launching kernels) crosses the boundary in common.h to mlx_dispatch.mm. Every
 * output and temporary comes from torch's allocator.
 */

#include <torch/torch.h>

#include <optional>
#include <string>
#include <vector>

#include "common.h"
#include "torch_binding.h"

#define MLXQ_CHECK(cond, ...) TORCH_CHECK(cond, "mlx-quantization-metal-kernels: ", __VA_ARGS__)

namespace {

mlxq::Dtype to_dtype(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return mlxq::Dtype::float32;
    case at::kHalf: return mlxq::Dtype::float16;
    case at::kBFloat16: return mlxq::Dtype::bfloat16;
    case at::kByte: return mlxq::Dtype::uint8;
    case at::kUInt32: return mlxq::Dtype::uint32;
    case at::kInt: return mlxq::Dtype::int32;
    default: MLXQ_CHECK(false, "unsupported dtype ", t);
  }
}

at::ScalarType to_scalar_type(mlxq::Dtype t) {
  switch (t) {
    case mlxq::Dtype::float32: return at::kFloat;
    case mlxq::Dtype::float16: return at::kHalf;
    case mlxq::Dtype::bfloat16: return at::kBFloat16;
    case mlxq::Dtype::uint8: return at::kByte;
    case mlxq::Dtype::uint32: return at::kUInt32;
    case mlxq::Dtype::int32: return at::kInt;
  }
  MLXQ_CHECK(false, "unsupported dtype");
}

// A torch MPS tensor's storage is a whole MTLBuffer that the tensor may be a view into, so it
// crosses as (buffer, byte offset) with its own shape and strides.
mlxq::array to_array(const at::Tensor &t) {
  mlxq::array a;
  a.buffer = t.is_meta() ? nullptr : const_cast<void *>(t.storage().data());
  a.offset = t.storage_offset() * t.element_size();
  a.shape_.assign(t.sizes().begin(), t.sizes().end());
  a.strides_ = t.strides().vec();
  a.dtype_ = to_dtype(t.scalar_type());
  return a;
}

std::optional<mlxq::array> to_array(const std::optional<at::Tensor> &t) {
  if (!t) return std::nullopt;
  return to_array(*t);
}

// Temporaries from torch's allocator, kept alive until the op returns. Encoding happens before that
// and torch's stream is serial, so later work cannot reuse them early.
struct TorchWorkspace : mlxq::Workspace {
  at::TensorOptions options;
  std::vector<at::Tensor> owned;
  std::vector<std::pair<mlxq::array, at::Tensor>> sources;  // the inputs `contiguous` may copy

  explicit TorchWorkspace(const at::Tensor &like) : options(like.options()) {}

  mlxq::array track(const at::Tensor &t) {
    sources.emplace_back(to_array(t), t);
    return sources.back().first;
  }

  mlxq::array empty(const std::vector<int> &shape, mlxq::Dtype dtype) override {
    owned.push_back(at::empty(std::vector<int64_t>(shape.begin(), shape.end()), options.dtype(to_scalar_type(dtype))));
    return to_array(owned.back());
  }

  mlxq::array contiguous(const mlxq::array &a, const std::optional<std::vector<int>> &shape) override {
    for (auto &[arr, t] : sources) {
      if (arr.buffer == a.buffer && arr.offset == a.offset && arr.shape_ == a.shape_ && arr.strides_ == a.strides_) {
        at::Tensor src = shape ? t.expand(std::vector<int64_t>(shape->begin(), shape->end())) : t;
        owned.push_back(src.contiguous());
        return to_array(owned.back());
      }
    }
    MLXQ_CHECK(false, "internal error: copy of an array the workspace does not know");
  }
};

// ---------------------------------------------------------------------------------------------------
// ops.cpp's checks and defaults
// ---------------------------------------------------------------------------------------------------

// string_to_quantization_mode + quantization_params_from_mode, then what upstream instantiates.
mlxq::Quantization quantization(const std::string &mode, std::optional<int64_t> group_size,
                                std::optional<int64_t> bits) {
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
    MLXQ_CHECK(false, "unknown quantization mode '", mode, "'; expected affine, mxfp4, mxfp8 or nvfp4");
  }
  mlxq::Quantization q{mode, int(group_size.value_or(default_group_size)), int(bits.value_or(default_bits))};
  if (mode == "affine") {
    MLXQ_CHECK(q.group_size == 32 || q.group_size == 64 || q.group_size == 128,
          "affine group_size must be 32, 64 or 128, got ", q.group_size);
    MLXQ_CHECK(q.bits == 2 || q.bits == 3 || q.bits == 4 || q.bits == 5 || q.bits == 6 || q.bits == 8,
          "affine bits must be one of 2, 3, 4, 5, 6, 8, got ", q.bits);
  } else {
    // fp_quantize's own check, and the only instantiations
    MLXQ_CHECK(q.group_size == default_group_size && q.bits == default_bits, mode, " requires group_size ",
          default_group_size, " and bits ", default_bits, ", got ", q.group_size, " and ", q.bits);
  }
  return q;
}

bool is_float(at::ScalarType t) { return t == at::kFloat || t == at::kHalf || t == at::kBFloat16; }

// validate_mode_with_type: the scales/biases contract, and the dtype it implies.
at::ScalarType validate_mode_with_type(const mlxq::Quantization &q, const at::Tensor &scales,
                                       const std::optional<at::Tensor> &biases,
                                       std::optional<at::ScalarType> out_type) {
  if (q.mode == "affine") {
    MLXQ_CHECK(biases.has_value(), "biases must be provided for affine quantization");
    auto dtype = at::result_type(scales, *biases);
    MLXQ_CHECK(is_float(dtype), "scales and biases must be floating point, got ", scales.scalar_type(), " and ",
          biases->scalar_type());
    return out_type.value_or(dtype);
  }
  MLXQ_CHECK(scales.scalar_type() == at::kByte, "scales must be uint8 for mode '", q.mode, "', got ", scales.scalar_type());
  MLXQ_CHECK(!biases.has_value(), "biases must be None for mode '", q.mode, "'");
  return out_type.value_or(at::kBFloat16);
}

void validate_quantized_input(const at::Tensor &w, const at::Tensor &scales, const mlxq::Quantization &q,
                              const std::optional<at::Tensor> &biases) {
  MLXQ_CHECK(w.scalar_type() == at::kUInt32, "the weight matrix should be uint32, got ", w.scalar_type());
  MLXQ_CHECK(w.dim() >= 2, "the weight matrix must have at least 2 dimensions, got ", w.sizes());
  MLXQ_CHECK(!biases || scales.sizes() == biases->sizes(), "scales and biases should have the same shape, got ",
        scales.sizes(), " and ", biases->sizes());
  MLXQ_CHECK(scales.dim() == w.dim() && std::equal(w.sizes().begin(), w.sizes().end() - 2, scales.sizes().begin()),
        "weight and scales should have the same batch shape, got ", w.sizes(), " and ", scales.sizes());
  MLXQ_CHECK(w.size(-1) * 32 / q.bits == scales.size(-1) * q.group_size, "the shapes of the weight ", w.sizes(),
        " and scales ", scales.sizes(), " are incompatible with group_size=", q.group_size, " and bits=", q.bits);
}

// extract_quantized_matmul_dims: (w_inner_dims, w_outer_dims).
std::pair<int64_t, int64_t> quantized_matmul_dims(const at::Tensor &x, const at::Tensor &w, const at::Tensor &scales,
                                                   const std::optional<at::Tensor> &biases, bool transpose,
                                                   const mlxq::Quantization &q) {
  validate_quantized_input(w, scales, q, biases);
  int64_t w_inner_dims = transpose ? w.size(-1) * 32 / q.bits : w.size(-2);
  int64_t w_outer_dims = transpose ? w.size(-2) : w.size(-1) * 32 / q.bits;
  MLXQ_CHECK(x.dim() >= 1 && x.size(-1) == w_inner_dims, "last dimension of x ", x.sizes(),
        " does not match the expanded quantized matrix (", w_inner_dims, ", ", w_outer_dims, ") computed from w ",
        w.sizes(), " with group_size=", q.group_size, ", bits=", q.bits, " and transpose=", transpose);
  return {w_inner_dims, w_outer_dims};
}

// broadcast_arrays(inputs, {-2, -1}): broadcast every leading dimension, keep the last two.
std::vector<at::Tensor> broadcast_batch(const std::vector<at::Tensor> &inputs) {
  std::vector<int64_t> batch;
  for (const auto &t : inputs) batch = at::infer_size(batch, t.sizes().slice(0, t.dim() - 2));
  std::vector<at::Tensor> out;
  for (const auto &t : inputs) {
    auto shape = batch;
    shape.push_back(t.size(-2));
    shape.push_back(t.size(-1));
    out.push_back(t.expand(shape));
  }
  return out;
}

// ensure_row_contiguous_matrix
at::Tensor row_contiguous_matrix(const at::Tensor &t) {
  if (t.is_contiguous()) return t;
  if (t.dim() < 2) {
    if (t.stride(0) == 1) return t;
  } else if (t.stride(-2) == t.size(-1) && t.stride(-1) == 1) {
    return t;
  }
  return t.contiguous();
}

std::optional<at::Tensor> row_contiguous_matrix(const std::optional<at::Tensor> &t) {
  if (!t) return std::nullopt;
  return row_contiguous_matrix(*t);
}

at::Tensor indices_or_default(const std::optional<at::Tensor> &indices, const at::Tensor &x) {
  if (indices) {
    MLXQ_CHECK(!at::isFloatingType(indices->scalar_type()) && indices->scalar_type() != at::kBool,
          "indices must be integers, got ", indices->scalar_type());
    return indices->to(at::kInt);  // the kernels read uint32: the same bits for every valid index
  }
  std::vector<int64_t> shape(x.sizes().begin(), x.sizes().end() - 2);
  return at::arange(c10::multiply_integers(shape), x.options().dtype(at::kInt)).reshape(shape);
}

void check_global_scale(const mlxq::Quantization &q, const std::optional<at::Tensor> &global_scale) {
  if (!global_scale) return;
  MLXQ_CHECK(q.mode == "nvfp4", "global scale is only supported for 'nvfp4' quantization mode");
  MLXQ_CHECK(global_scale->scalar_type() == at::kFloat, "global scale must be float32, got ",
        global_scale->scalar_type());
}

void check_mps(std::initializer_list<std::optional<at::Tensor>> ts) {
  for (const auto &t : ts) {
    if (t) MLXQ_CHECK(t->is_mps(), "all inputs must be on mps, got one on ", t->device());
  }
}

// ---------------------------------------------------------------------------------------------------
// The ops: prepare as ops.cpp does, then hand over to the Metal side
// ---------------------------------------------------------------------------------------------------

std::pair<at::Tensor, std::vector<std::string>> run_quantized_matmul(
    const at::Tensor &x_in, const at::Tensor &w_in, const at::Tensor &scales_in,
    const std::optional<at::Tensor> &biases_in, bool transpose, std::optional<int64_t> group_size,
    std::optional<int64_t> bits, const std::string &mode, bool encode) {
  auto q = quantization(mode, group_size, bits);
  bool affine = q.mode == "affine";
  at::ScalarType dtype = validate_mode_with_type(q, scales_in, biases_in, std::nullopt);
  auto [w_inner_dims, w_outer_dims] = quantized_matmul_dims(x_in, w_in, scales_in, biases_in, transpose, q);
  dtype = affine ? at::promote_types(x_in.scalar_type(), dtype) : x_in.scalar_type();
  MLXQ_CHECK(is_float(dtype), "x must be float32, float16 or bfloat16, got ", x_in.scalar_type());

  std::vector<at::Tensor> inputs = {x_in.to(dtype), w_in, affine ? scales_in.to(dtype) : scales_in};
  if (affine) inputs.push_back(biases_in->to(dtype));
  if (x_in.dim() > 2 && w_in.dim() > 2) inputs = broadcast_batch(inputs);
  auto out_shape = inputs[0].sizes().vec();
  out_shape.back() = w_outer_dims;

  // QuantizedMatmul::eval_gpu: make sure the last two dims of x and w, s, b are contiguous
  at::Tensor x = row_contiguous_matrix(inputs[0]);
  at::Tensor w = row_contiguous_matrix(inputs[1]);
  at::Tensor scales = row_contiguous_matrix(inputs[2]);
  std::optional<at::Tensor> biases;
  if (affine) biases = row_contiguous_matrix(inputs[3]);
  at::Tensor out = at::empty(out_shape, x.options());
  MLXQ_CHECK(x.dim() >= 2 || w.dim() == 2, "x must have at least 2 dimensions when w is batched");
  if (out.numel() == 0 || x.size(-1) == 0) return {out.zero_(), {}};

  TorchWorkspace ws(x);
  auto names = mlxq::quantized_matmul(to_array(x), to_array(w), to_array(scales), to_array(biases), to_array(out),
                                      transpose, q, ws, encode);
  return {out, names};
}

std::pair<at::Tensor, std::vector<std::string>> run_gather_qmm(
    const at::Tensor &x_in, const at::Tensor &w_in, const at::Tensor &scales_in,
    const std::optional<at::Tensor> &biases_in, const std::optional<at::Tensor> &lhs_in,
    const std::optional<at::Tensor> &rhs_in, bool transpose, std::optional<int64_t> group_size,
    std::optional<int64_t> bits, const std::string &mode, const std::optional<at::Tensor> &global_scale_in,
    bool sorted_indices, bool encode) {
  if (!lhs_in && !rhs_in) {
    MLXQ_CHECK(!global_scale_in, "global scale is not supported without indices");
    return run_quantized_matmul(x_in, w_in, scales_in, biases_in, transpose, group_size, bits, mode, encode);
  }
  auto q = quantization(mode, group_size, bits);
  bool affine = q.mode == "affine";
  at::ScalarType out_type = validate_mode_with_type(q, scales_in, biases_in, std::nullopt);
  auto [w_inner_dims, w_outer_dims] = quantized_matmul_dims(x_in, w_in, scales_in, biases_in, transpose, q);
  MLXQ_CHECK(x_in.dim() >= 2, "x must have at least 2 dimensions, got ", x_in.sizes());
  check_global_scale(q, global_scale_in);
  if (global_scale_in) {
    // one scale per expert, so it matches the batch dimensions of w
    std::vector<int64_t> expected(w_in.sizes().begin(), w_in.sizes().end() - 2);
    MLXQ_CHECK(global_scale_in->sizes().vec() == expected, "global scale must have one entry per expert, shape ",
          expected, ", got ", global_scale_in->sizes());
  }
  out_type = affine ? at::promote_types(x_in.scalar_type(), out_type) : x_in.scalar_type();
  MLXQ_CHECK(is_float(out_type), "x must be float32, float16 or bfloat16, got ", x_in.scalar_type());

  // Extract indices and broadcast them
  auto indices = at::broadcast_tensors({indices_or_default(lhs_in, x_in), indices_or_default(rhs_in, w_in)});
  auto out_shape = indices[0].sizes().vec();
  out_shape.push_back(x_in.size(-2));
  out_shape.push_back(w_outer_dims);

  // GatherQMM::eval_gpu's ensure_row_contiguous_matrix calls
  at::Tensor x = row_contiguous_matrix(x_in.to(out_type));
  at::Tensor w = row_contiguous_matrix(w_in);
  at::Tensor scales = row_contiguous_matrix(affine ? scales_in.to(out_type) : scales_in);
  std::optional<at::Tensor> biases;
  if (affine) biases = row_contiguous_matrix(biases_in->to(out_type));
  std::optional<at::Tensor> global_scale;
  if (global_scale_in) global_scale = global_scale_in->contiguous();
  at::Tensor out = at::empty(out_shape, x.options());
  if (out.numel() == 0 || x.size(-1) == 0) return {out.zero_(), {}};

  TorchWorkspace ws(x);
  mlxq::array xa = ws.track(x), wa = ws.track(w), sa = ws.track(scales), ra = ws.track(indices[1]);
  std::optional<mlxq::array> ba, ga;
  if (biases) ba = ws.track(*biases);
  if (global_scale) ga = ws.track(*global_scale);
  auto names = mlxq::gather_qmm(xa, wa, sa, ba, ga, to_array(indices[0]), ra, to_array(out), transpose,
                                sorted_indices && !lhs_in, q, ws, encode);
  return {out, names};
}

}  // namespace

// ---------------------------------------------------------------------------------------------------
// Entry points (torch_binding.h)
// ---------------------------------------------------------------------------------------------------

at::Tensor quantized_matmul(const at::Tensor &x, const at::Tensor &w, const at::Tensor &scales,
                            const std::optional<at::Tensor> &biases, bool transpose,
                            std::optional<int64_t> group_size, std::optional<int64_t> bits,
                            const std::string &mode) {
  check_mps({x, w, scales, biases});
  return run_quantized_matmul(x, w, scales, biases, transpose, group_size, bits, mode, true).first;
}

at::Tensor gather_qmm(const at::Tensor &x, const at::Tensor &w, const at::Tensor &scales,
                      const std::optional<at::Tensor> &biases, const std::optional<at::Tensor> &lhs_indices,
                      const std::optional<at::Tensor> &rhs_indices, bool transpose,
                      std::optional<int64_t> group_size, std::optional<int64_t> bits, const std::string &mode,
                      const std::optional<at::Tensor> &global_scale, bool sorted_indices) {
  check_mps({x, w, scales, biases, lhs_indices, rhs_indices, global_scale});
  return run_gather_qmm(x, w, scales, biases, lhs_indices, rhs_indices, transpose, group_size, bits, mode,
                        global_scale, sorted_indices, true)
      .first;
}

std::vector<at::Tensor> quantize(const at::Tensor &w_in, std::optional<int64_t> group_size,
                                 std::optional<int64_t> bits, const std::string &mode,
                                 const std::optional<at::Tensor> &global_scale) {
  check_mps({w_in, global_scale});
  auto q = quantization(mode, group_size, bits);
  bool affine = q.mode == "affine";
  MLXQ_CHECK(is_float(w_in.scalar_type()), "only float32, float16 or bfloat16 can be quantized, got ",
        w_in.scalar_type());
  MLXQ_CHECK(w_in.dim() >= 2, "the matrix to be quantized must have at least 2 dimensions, got ", w_in.sizes());
  MLXQ_CHECK(w_in.size(-1) % q.group_size == 0, "the last dimension (", w_in.size(-1),
        ") must be divisible by the group size ", q.group_size);
  check_global_scale(q, global_scale);
  if (global_scale) MLXQ_CHECK(global_scale->numel() == 1, "global scale must be a scalar, got ", global_scale->sizes());

  at::Tensor w = w_in.contiguous();
  auto wq_shape = w.sizes().vec(), scales_shape = w.sizes().vec();
  wq_shape.back() = w.size(-1) * q.bits / 32;
  scales_shape.back() = w.size(-1) / q.group_size;
  at::Tensor wq = at::empty(wq_shape, w.options().dtype(at::kUInt32));
  at::Tensor scales = at::empty(scales_shape, w.options().dtype(affine ? w.scalar_type() : at::kByte));
  std::optional<at::Tensor> biases;
  if (affine) biases = at::empty(scales_shape, w.options());
  std::optional<at::Tensor> gs;
  if (global_scale) gs = global_scale->contiguous();
  if (w.numel() > 0) {
    TorchWorkspace ws(w);
    mlxq::quantize(to_array(w), to_array(wq), to_array(scales), to_array(biases), to_array(gs), false, q, ws, true);
  }
  if (biases) return {wq, scales, *biases};
  return {wq, scales};
}

at::Tensor dequantize(const at::Tensor &w_in, const at::Tensor &scales_in, const std::optional<at::Tensor> &biases_in,
                      std::optional<int64_t> group_size, std::optional<int64_t> bits, const std::string &mode,
                      const std::optional<at::Tensor> &global_scale, std::optional<at::ScalarType> dtype) {
  check_mps({w_in, scales_in, biases_in, global_scale});
  auto q = quantization(mode, group_size, bits);
  bool affine = q.mode == "affine";
  at::ScalarType out_type = validate_mode_with_type(q, scales_in, biases_in, dtype);
  MLXQ_CHECK(is_float(out_type), "dtype must be float32, float16 or bfloat16, got ", out_type);
  MLXQ_CHECK(w_in.scalar_type() == at::kUInt32, "the matrix should be given as uint32, got ", w_in.scalar_type());
  MLXQ_CHECK(w_in.dim() >= 2, "the matrix to be dequantized must have at least 2 dimensions, got ", w_in.sizes());
  check_global_scale(q, global_scale);
  auto w_shape = w_in.sizes().vec(), s_shape = scales_in.sizes().vec();
  w_shape.back() = s_shape.back() = -1;
  MLXQ_CHECK(w_shape == s_shape, "shape of scales ", scales_in.sizes(), " does not match the matrix ", w_in.sizes());
  int64_t out_size = w_in.size(-1) * 32 / q.bits;
  MLXQ_CHECK(out_size == scales_in.size(-1) * q.group_size, "shape of scales ", scales_in.sizes(),
        " does not match the matrix ", w_in.sizes(), " with group_size=", q.group_size, " and bits=", q.bits);

  at::Tensor w = w_in.contiguous();
  at::Tensor scales = scales_in.contiguous();
  std::optional<at::Tensor> biases;
  if (affine) {
    MLXQ_CHECK(biases_in->sizes() == scales_in.sizes(), "shape of biases ", biases_in->sizes(), " does not match scales ",
          scales_in.sizes());
    // affine_dequantize's primitive reads scales, biases and its output as the scales' type
    biases = biases_in->to(scales.scalar_type()).contiguous();
  }
  std::optional<at::Tensor> gs;
  if (global_scale) gs = global_scale->contiguous();
  auto out_shape = w.sizes().vec();
  out_shape.back() = out_size;
  at::Tensor out = at::empty(out_shape, w.options().dtype(affine ? scales.scalar_type() : out_type));
  if (out.numel() > 0) {
    TorchWorkspace ws(w);
    mlxq::quantize(to_array(w), to_array(out), to_array(scales), to_array(biases), to_array(gs), true, q, ws, true);
  }
  return out.scalar_type() == out_type ? out : out.to(out_type);  // then astype, as upstream
}

std::vector<std::string> trace_quantized_matmul(const at::Tensor &x, const at::Tensor &w, const at::Tensor &scales,
                                                const std::optional<at::Tensor> &biases, bool transpose,
                                                std::optional<int64_t> group_size, std::optional<int64_t> bits,
                                                const std::string &mode) {
  return run_quantized_matmul(x, w, scales, biases, transpose, group_size, bits, mode, false).second;
}

std::vector<std::string> trace_gather_qmm(const at::Tensor &x, const at::Tensor &w, const at::Tensor &scales,
                                          const std::optional<at::Tensor> &biases,
                                          const std::optional<at::Tensor> &lhs_indices,
                                          const std::optional<at::Tensor> &rhs_indices, bool transpose,
                                          std::optional<int64_t> group_size, std::optional<int64_t> bits,
                                          const std::string &mode, const std::optional<at::Tensor> &global_scale,
                                          bool sorted_indices) {
  return run_gather_qmm(x, w, scales, biases, lhs_indices, rhs_indices, transpose, group_size, bits, mode,
                        global_scale, sorted_indices, false)
      .second;
}
