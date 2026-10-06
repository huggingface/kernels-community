#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "quantized_matmul(Tensor x, Tensor w, Tensor scales, Tensor? biases, bool transpose, int? group_size, "
      "int? bits, str mode) -> Tensor");
  ops.impl("quantized_matmul", torch::kMPS, &quantized_matmul);
  ops.def(
      "gather_qmm(Tensor x, Tensor w, Tensor scales, Tensor? biases, Tensor? lhs_indices, Tensor? rhs_indices, "
      "bool transpose, int? group_size, int? bits, str mode, Tensor? global_scale, bool sorted_indices) -> Tensor");
  ops.impl("gather_qmm", torch::kMPS, &gather_qmm);
  ops.def("quantize(Tensor w, int? group_size, int? bits, str mode, Tensor? global_scale) -> Tensor[]");
  ops.impl("quantize", torch::kMPS, &quantize);
  ops.def(
      "dequantize(Tensor w, Tensor scales, Tensor? biases, int? group_size, int? bits, str mode, "
      "Tensor? global_scale, ScalarType? dtype) -> Tensor");
  ops.impl("dequantize", torch::kMPS, &dequantize);

  // Not part of the Python API: the tests use these (through `_ops`) to check which kernels the
  // dispatch picks. Registered as catch-alls so they also take meta tensors.
  ops.def(
      "trace_quantized_matmul(Tensor x, Tensor w, Tensor scales, Tensor? biases, bool transpose, "
      "int? group_size, int? bits, str mode) -> str[]");
  ops.impl("trace_quantized_matmul", &trace_quantized_matmul);
  ops.def(
      "trace_gather_qmm(Tensor x, Tensor w, Tensor scales, Tensor? biases, Tensor? lhs_indices, "
      "Tensor? rhs_indices, bool transpose, int? group_size, int? bits, str mode, Tensor? global_scale, "
      "bool sorted_indices) -> str[]");
  ops.impl("trace_gather_qmm", &trace_gather_qmm);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
