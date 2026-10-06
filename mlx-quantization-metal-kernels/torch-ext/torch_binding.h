#pragma once

#include <torch/torch.h>

#include <optional>
#include <string>
#include <vector>

// MLX's quantized ops, on torch tensors: same arguments, defaults and checks as mlx.core's, and the
// same Metal kernels. `group_size`/`bits` default per mode (affine 64/4, mxfp4 32/4, mxfp8 32/8,
// nvfp4 16/4), as in MLX.

at::Tensor quantized_matmul(const at::Tensor &x, const at::Tensor &w, const at::Tensor &scales,
                            const std::optional<at::Tensor> &biases, bool transpose,
                            std::optional<int64_t> group_size, std::optional<int64_t> bits,
                            const std::string &mode);

at::Tensor gather_qmm(const at::Tensor &x, const at::Tensor &w, const at::Tensor &scales,
                      const std::optional<at::Tensor> &biases, const std::optional<at::Tensor> &lhs_indices,
                      const std::optional<at::Tensor> &rhs_indices, bool transpose,
                      std::optional<int64_t> group_size, std::optional<int64_t> bits, const std::string &mode,
                      const std::optional<at::Tensor> &global_scale, bool sorted_indices);

std::vector<at::Tensor> quantize(const at::Tensor &w, std::optional<int64_t> group_size,
                                 std::optional<int64_t> bits, const std::string &mode,
                                 const std::optional<at::Tensor> &global_scale);

at::Tensor dequantize(const at::Tensor &w, const at::Tensor &scales, const std::optional<at::Tensor> &biases,
                      std::optional<int64_t> group_size, std::optional<int64_t> bits, const std::string &mode,
                      const std::optional<at::Tensor> &global_scale, std::optional<at::ScalarType> dtype);

// The kernels the two matmuls would launch for these inputs, in order, without launching them. For
// the tests: inputs may be on the meta device.
std::vector<std::string> trace_quantized_matmul(const at::Tensor &x, const at::Tensor &w, const at::Tensor &scales,
                                                const std::optional<at::Tensor> &biases, bool transpose,
                                                std::optional<int64_t> group_size, std::optional<int64_t> bits,
                                                const std::string &mode);

std::vector<std::string> trace_gather_qmm(const at::Tensor &x, const at::Tensor &w, const at::Tensor &scales,
                                          const std::optional<at::Tensor> &biases,
                                          const std::optional<at::Tensor> &lhs_indices,
                                          const std::optional<at::Tensor> &rhs_indices, bool transpose,
                                          std::optional<int64_t> group_size, std::optional<int64_t> bits,
                                          const std::string &mode, const std::optional<at::Tensor> &global_scale,
                                          bool sorted_indices);
