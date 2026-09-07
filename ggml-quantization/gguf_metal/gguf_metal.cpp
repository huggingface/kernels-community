/* Metal implementation of the two entry points.
 *
 * Same shape as gguf_cuda.cu: no ggml header is included here, all ggml and Metal contact sits
 * behind the extern "C" boundary in ggml_dispatch.mm, and every output comes from torch's
 * allocator so the ops compose with the rest of a model's memory.
 *
 * A torch MPS tensor's storage is a whole MTLBuffer that the tensor may be a view into, so each
 * buffer crosses the boundary as a (buffer, byte offset) pair.
 */

#include <torch/torch.h>

#include "common.h"
#include "torch_binding.h"

namespace {

void *mtl_buffer(const at::Tensor &t) {
  return const_cast<void *>(t.storage().data());
}

size_t byte_offset(const at::Tensor &t) {
  return static_cast<size_t>(t.storage_offset()) * t.element_size();
}

}  // namespace

std::vector<int64_t> gemv_types() {
  int ids[64];
  const int n = gguf_metal_gemv_types(ids, 64);
  TORCH_CHECK(n <= 64, "ggml-quantization: the Metal type table outgrew the buffer here");
  return std::vector<int64_t>(ids, ids + n);
}

at::Tensor get_rows(const at::Tensor &blocks, const at::Tensor &indices, int64_t ggml_type,
                    int64_t cols, at::ScalarType dtype) {
  TORCH_CHECK(blocks.is_mps() && blocks.scalar_type() == at::kByte, "blocks must be mps uint8");
  TORCH_CHECK(blocks.is_contiguous(), "blocks must be contiguous");

  auto ids = indices.to(at::kInt).contiguous();
  const int64_t rows = ids.numel();
  auto out = at::empty({rows, cols}, blocks.options().dtype(at::kFloat));

  const int rc = gguf_metal_get_rows(mtl_buffer(blocks), byte_offset(blocks), mtl_buffer(ids),
                                     byte_offset(ids), mtl_buffer(out), byte_offset(out),
                                     static_cast<int>(ggml_type), rows, cols, 0);
  TORCH_CHECK(rc == 0, "ggml-quantization: get_rows has no implementation for ggml type ", ggml_type);
  return dtype == at::kFloat ? out : out.to(dtype);
}

at::Tensor dequantize(const at::Tensor &blocks, int64_t ggml_type, int64_t rows, int64_t cols,
                      at::ScalarType dtype) {
  // A whole weight is every row in order.
  return get_rows(blocks, at::arange(rows, blocks.options().dtype(at::kInt)), ggml_type, cols, dtype);
}

at::Tensor mul_mat_vec(const at::Tensor &blocks, const at::Tensor &x, int64_t ggml_type,
                       int64_t out_features) {
  TORCH_CHECK(blocks.is_mps() && blocks.scalar_type() == at::kByte, "blocks must be mps uint8");
  TORCH_CHECK(x.is_mps() && x.dim() == 2, "x must be a 2D mps tensor");

  const int64_t rows = x.size(0), in_features = x.size(1);
  // ggml's mul_mv/mul_mm kernels for quantized weights all take an f32 activation.
  const auto xc = x.scalar_type() == at::kFloat ? x.contiguous() : x.to(at::kFloat).contiguous();
  auto out = at::empty({rows, out_features}, x.options().dtype(at::kFloat));

  const int rc = gguf_metal_mul_mat(mtl_buffer(blocks), byte_offset(blocks), mtl_buffer(xc),
                                    byte_offset(xc), mtl_buffer(out), byte_offset(out),
                                    static_cast<int>(ggml_type), in_features, out_features, rows);
  TORCH_CHECK(rc == 0, "ggml-quantization: no matmul for ggml type ", ggml_type, " at ", rows, " rows");
  return out;
}

at::Tensor mul_mat_id(const at::Tensor &blocks, const at::Tensor &x, const at::Tensor &ids,
                      int64_t ggml_type, int64_t out_features) {
  TORCH_CHECK(blocks.is_mps() && blocks.scalar_type() == at::kByte, "blocks must be mps uint8");
  TORCH_CHECK(blocks.dim() == 3, "blocks must be (n_experts, out_features, bytes_per_row)");
  TORCH_CHECK(x.is_mps() && x.dim() == 2, "x must be a 2D mps tensor");
  TORCH_CHECK(ids.dim() == 2 && ids.size(0) == x.size(0), "ids must be (n_tokens, n_used)");

  const int64_t experts = blocks.size(0), tokens = x.size(0), in_features = x.size(1);
  const int64_t used = ids.size(1);
  const auto xc = x.scalar_type() == at::kFloat ? x.contiguous() : x.to(at::kFloat).contiguous();
  const auto idc = ids.to(at::kInt).contiguous();
  auto out = at::empty({tokens, used, out_features}, x.options().dtype(at::kFloat));

  const int rc = gguf_metal_mul_mat_id(mtl_buffer(blocks), byte_offset(blocks), mtl_buffer(xc),
                                       byte_offset(xc), mtl_buffer(idc), byte_offset(idc),
                                       mtl_buffer(out), byte_offset(out),
                                       static_cast<int>(ggml_type), in_features, out_features,
                                       experts, tokens, used);
  TORCH_CHECK(rc == 0, "ggml-quantization: no mul_mat_id for ggml type ", ggml_type);
  return out;
}

