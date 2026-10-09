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

// string_to_quantization_mode(mode, "qqmm") + quantization_params_from_mode. qqmm rounds its activation
// through an fp format; mlx.core accepts "affine" here, then crashes building the op, so it is refused.
mlxq::Quantization qq_quantization(const std::string &mode, std::optional<int64_t> group_size,
                                   std::optional<int64_t> bits) {
  MLXQ_CHECK(mode != "affine", "qqmm needs an fp mode (mxfp4, mxfp8 or nvfp4), got 'affine'");
  return quantization(mode, group_size, bits);
}

// validate_global_scale
void validate_global_scale(const mlxq::Quantization &q, const std::optional<at::Tensor> &global_scale) {
  check_global_scale(q, global_scale);
  if (global_scale) {
    MLXQ_CHECK(global_scale->numel() == 1, "global scale must be a scalar, got shape ", global_scale->sizes());
  }
}

// extract_qqmm_dims: (w_inner_dims, w_outer_dims), for a quantized `w` or a plain one.
std::pair<int64_t, int64_t> qqmm_dims(const at::Tensor &x, const at::Tensor &w, const std::optional<at::Tensor> &scales,
                                      const mlxq::Quantization &q) {
  if (w.scalar_type() != at::kUInt32) {
    MLXQ_CHECK(w.dim() >= 2, "w must have at least 2 dimensions, got ", w.sizes());
    MLXQ_CHECK(x.size(-1) == w.size(-1), "last dimension of x ", x.sizes(), " must match last dimension of w ",
               w.sizes());
    return {w.size(-1), w.size(-2)};
  }
  MLXQ_CHECK(scales.has_value(), "scales must be provided if w is quantized");
  return quantized_matmul_dims(x, w, *scales, std::nullopt, true, q);
}

// What QQMatmul and GatherQQMM run before their matmul, in their order: quantize_input on `w` when it
// comes unquantized, then quantize_dequantize_input on `x`. Each is its own pass on torch's stream,
// so the matmul's planning can copy their outputs.
struct QQInputs {
  at::Tensor x, w, scales;
  std::vector<std::string> names;
};

QQInputs quantize_qq_inputs(const at::Tensor &x_in, const at::Tensor &w_in, const std::optional<at::Tensor> &scales_in,
                            const std::optional<at::Tensor> &global_scale_x,
                            const std::optional<at::Tensor> &global_scale_w, const mlxq::Quantization &q,
                            bool encode) {
  QQInputs in;
  if (w_in.scalar_type() != at::kUInt32) {
    // quantize_input
    at::Tensor w = w_in.contiguous();
    auto wq_shape = w.sizes().vec(), scales_shape = w.sizes().vec();
    wq_shape.back() = w.size(-1) * q.bits / 32;
    scales_shape.back() = w.size(-1) / q.group_size;
    in.w = at::empty(wq_shape, w.options().dtype(at::kUInt32));
    in.scales = at::empty(scales_shape, w.options().dtype(at::kByte));
    std::optional<at::Tensor> gs;
    if (global_scale_w) gs = global_scale_w->contiguous();
    TorchWorkspace ws(w);
    in.names = mlxq::quantize(to_array(w), to_array(in.w), to_array(in.scales), std::nullopt, to_array(gs), false, q,
                              ws, encode);
  } else {
    // ensure_row_contiguous_matrix(w_pre), ensure_row_contiguous_matrix(scales)
    in.w = row_contiguous_matrix(w_in);
    in.scales = row_contiguous_matrix(*scales_in);
  }
  // quantize_dequantize_input
  at::Tensor x = x_in.contiguous();
  in.x = at::empty(x.sizes(), x.options());
  std::optional<at::Tensor> gs;
  if (global_scale_x) gs = global_scale_x->contiguous();
  TorchWorkspace ws(x);
  auto names = mlxq::quantize_dequantize(to_array(x), to_array(gs), to_array(in.x), q, ws, encode);
  in.names.insert(in.names.end(), names.begin(), names.end());
  return in;
}

// The group the activation is quantized in has to divide K, as the weight's does. mlx.core does not
// check this for an unquantized `w`, and the last group would then be misread.
void check_groups(const at::Tensor &x, const mlxq::Quantization &q) {
  MLXQ_CHECK(x.size(-1) % q.group_size == 0, "the last dimension of x (", x.size(-1),
             ") must be divisible by the group size ", q.group_size);
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

std::pair<at::Tensor, std::vector<std::string>> run_qqmm(
    const at::Tensor &x_in, const at::Tensor &w, const std::optional<at::Tensor> &scales,
    std::optional<int64_t> group_size, std::optional<int64_t> bits, const std::string &mode,
    const std::optional<at::Tensor> &global_scale_x, const std::optional<at::Tensor> &global_scale_w, bool encode) {
  auto q = qq_quantization(mode, group_size, bits);

  // Allow gemv
  at::Tensor x = x_in;
  if (x.dim() == 1) {
    x = x.unsqueeze(0);
  } else if (w.dim() == 2 && x.dim() > 2) {
    x = x.flatten(0, -2);
  }

  // validate_qqmm_inputs
  MLXQ_CHECK(x.dim() <= 2 && w.dim() <= 2, "qqmm only supports 2D inputs, got x ", x.sizes(), " and w ", w.sizes());
  bool w_quantized = w.scalar_type() == at::kUInt32;
  if (w_quantized) {
    MLXQ_CHECK(scales.has_value(), "scales must be provided if w is quantized");
    validate_quantized_input(w, *scales, q, std::nullopt);
  } else {
    MLXQ_CHECK(is_float(w.scalar_type()), "w must be float32, float16 or bfloat16 (or quantized), got ",
               w.scalar_type());
  }
  MLXQ_CHECK(is_float(x.scalar_type()), "x must be float32, float16 or bfloat16, got ", x.scalar_type());
  validate_global_scale(q, global_scale_x);
  validate_global_scale(q, global_scale_w);
  if (q.mode == "nvfp4") {
    MLXQ_CHECK(global_scale_x.has_value() == global_scale_w.has_value(),
               "for nvfp4, either both global_scale_x and global_scale_w must be provided, or neither");
  }
  auto [w_inner_dims, w_outer_dims] = qqmm_dims(x, w, scales, q);
  check_groups(x, q);
  bool has_global_scales = q.mode == "nvfp4" && global_scale_x && global_scale_w;

  auto out_shape = x.sizes().vec();
  out_shape.back() = w_outer_dims;
  at::Tensor out = at::empty(out_shape, x.options());  // output dtype is the same as x dtype
  std::vector<std::string> names;
  if (out.numel() > 0 && x.size(-1) > 0) {
    auto in = quantize_qq_inputs(x, w, scales, has_global_scales ? global_scale_x : std::nullopt,
                                 has_global_scales ? global_scale_w : std::nullopt, q, encode);
    std::optional<at::Tensor> gs;
    if (has_global_scales) gs = global_scale_w->contiguous();
    TorchWorkspace ws(in.x);
    names = mlxq::qqmm(to_array(in.x), to_array(in.w), to_array(in.scales), to_array(gs), to_array(out),
                       has_global_scales, w_quantized, q, ws, encode);
    names.insert(names.begin(), in.names.begin(), in.names.end());
  } else {
    out.zero_();
  }

  if (x_in.dim() > 2) {
    auto shape = x_in.sizes().vec();
    shape.back() = w_outer_dims;
    out = out.reshape(shape);
  } else if (x_in.dim() == 1) {
    out = out.squeeze(0);
  }
  return {out, names};
}

std::pair<at::Tensor, std::vector<std::string>> run_gather_qqmm(
    const at::Tensor &x_in, const at::Tensor &w, const std::optional<at::Tensor> &scales,
    const std::optional<at::Tensor> &lhs_in, const std::optional<at::Tensor> &rhs_in,
    std::optional<int64_t> group_size, std::optional<int64_t> bits, const std::string &mode,
    const std::optional<at::Tensor> &global_scale_x, const std::optional<at::Tensor> &global_scale_w,
    bool sorted_indices, bool encode) {
  auto q = qq_quantization(mode, group_size, bits);
  MLXQ_CHECK(x_in.dim() >= 2, "x must have at least 2 dimensions, got ", x_in.sizes());
  MLXQ_CHECK(is_float(x_in.scalar_type()), "x must be float32, float16 or bfloat16, got ", x_in.scalar_type());
  bool w_quantized = w.scalar_type() == at::kUInt32;
  MLXQ_CHECK(w_quantized || is_float(w.scalar_type()), "w must be float32, float16 or bfloat16 (or quantized), got ",
             w.scalar_type());

  // Extract indices and broadcast them
  auto indices = at::broadcast_tensors({indices_or_default(lhs_in, x_in), indices_or_default(rhs_in, w)});
  auto [w_inner_dims, w_outer_dims] = qqmm_dims(x_in, w, scales, q);
  check_groups(x_in, q);
  // The global scales only count as a pair, and only for nvfp4 (GatherQQMM's inputs.size() check).
  bool has_global_scales = q.mode == "nvfp4" && global_scale_x && global_scale_w;
  if (has_global_scales) {
    validate_global_scale(q, global_scale_x);
    validate_global_scale(q, global_scale_w);
  }

  auto out_shape = indices[0].sizes().vec();
  out_shape.push_back(x_in.size(-2));
  out_shape.push_back(w_outer_dims);
  at::Tensor out = at::empty(out_shape, x_in.options());
  if (out.numel() == 0 || x_in.size(-1) == 0) return {out.zero_(), {}};

  auto in = quantize_qq_inputs(x_in, w, scales, has_global_scales ? global_scale_x : std::nullopt,
                               has_global_scales ? global_scale_w : std::nullopt, q, encode);
  // temporary, until we add proper scaling for gather qqmm: the scale broadcast to one per expert
  std::optional<at::Tensor> gs;
  if (has_global_scales) {
    int64_t E = in.w.numel() / in.w.size(-1) / in.w.size(-2);
    gs = global_scale_w->reshape({1}).expand({E}).contiguous();
  }
  // ensure_row_contiguous on both index arrays
  at::Tensor lhs = indices[0].contiguous(), rhs = indices[1].contiguous();

  TorchWorkspace ws(in.x);
  mlxq::array xa = ws.track(in.x), wa = ws.track(in.w), sa = ws.track(in.scales), ra = ws.track(rhs);
  std::optional<mlxq::array> ga;
  if (gs) ga = ws.track(*gs);
  auto names = mlxq::gather_qqmm(xa, wa, sa, ga, to_array(lhs), ra, to_array(out), has_global_scales, w_quantized,
                                 sorted_indices && !lhs_in, q, ws, encode);
  names.insert(names.begin(), in.names.begin(), in.names.end());
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

at::Tensor qqmm(const at::Tensor &x, const at::Tensor &w, const std::optional<at::Tensor> &scales,
                std::optional<int64_t> group_size, std::optional<int64_t> bits, const std::string &mode,
                const std::optional<at::Tensor> &global_scale_x, const std::optional<at::Tensor> &global_scale_w) {
  check_mps({x, w, scales, global_scale_x, global_scale_w});
  return run_qqmm(x, w, scales, group_size, bits, mode, global_scale_x, global_scale_w, true).first;
}

at::Tensor gather_qqmm(const at::Tensor &x, const at::Tensor &w, const std::optional<at::Tensor> &scales,
                       const std::optional<at::Tensor> &lhs_indices, const std::optional<at::Tensor> &rhs_indices,
                       std::optional<int64_t> group_size, std::optional<int64_t> bits, const std::string &mode,
                       const std::optional<at::Tensor> &global_scale_x, const std::optional<at::Tensor> &global_scale_w,
                       bool sorted_indices) {
  check_mps({x, w, scales, lhs_indices, rhs_indices, global_scale_x, global_scale_w});
  return run_gather_qqmm(x, w, scales, lhs_indices, rhs_indices, group_size, bits, mode, global_scale_x,
                         global_scale_w, sorted_indices, true)
      .first;
}

std::vector<std::string> trace_qqmm(const at::Tensor &x, const at::Tensor &w, const std::optional<at::Tensor> &scales,
                                    std::optional<int64_t> group_size, std::optional<int64_t> bits,
                                    const std::string &mode, const std::optional<at::Tensor> &global_scale_x,
                                    const std::optional<at::Tensor> &global_scale_w) {
  return run_qqmm(x, w, scales, group_size, bits, mode, global_scale_x, global_scale_w, false).second;
}

std::vector<std::string> trace_gather_qqmm(const at::Tensor &x, const at::Tensor &w,
                                           const std::optional<at::Tensor> &scales,
                                           const std::optional<at::Tensor> &lhs_indices,
                                           const std::optional<at::Tensor> &rhs_indices,
                                           std::optional<int64_t> group_size, std::optional<int64_t> bits,
                                           const std::string &mode, const std::optional<at::Tensor> &global_scale_x,
                                           const std::optional<at::Tensor> &global_scale_w, bool sorted_indices) {
  return run_gather_qqmm(x, w, scales, lhs_indices, rhs_indices, group_size, bits, mode, global_scale_x,
                         global_scale_w, sorted_indices, false)
      .second;
}
