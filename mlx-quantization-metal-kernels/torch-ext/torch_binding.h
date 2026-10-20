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

// mx.qqmm / mx.gather_qqmm: `x` is quantized on the fly (rounded through the fp format), `w` is used
// as given when quantized (uint32 with uint8 `scales`) and quantized on the fly otherwise. fp modes
// only; the global scales are nvfp4's, and go together.
at::Tensor qqmm(const at::Tensor &x, const at::Tensor &w, const std::optional<at::Tensor> &scales,
                std::optional<int64_t> group_size, std::optional<int64_t> bits, const std::string &mode,
                const std::optional<at::Tensor> &global_scale_x, const std::optional<at::Tensor> &global_scale_w);

at::Tensor gather_qqmm(const at::Tensor &x, const at::Tensor &w, const std::optional<at::Tensor> &scales,
                       const std::optional<at::Tensor> &lhs_indices, const std::optional<at::Tensor> &rhs_indices,
                       std::optional<int64_t> group_size, std::optional<int64_t> bits, const std::string &mode,
                       const std::optional<at::Tensor> &global_scale_x, const std::optional<at::Tensor> &global_scale_w,
                       bool sorted_indices);

// The kernels the matmuls would launch for these inputs, in order, without launching them. For
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

std::vector<std::string> trace_qqmm(const at::Tensor &x, const at::Tensor &w, const std::optional<at::Tensor> &scales,
                                    std::optional<int64_t> group_size, std::optional<int64_t> bits,
                                    const std::string &mode, const std::optional<at::Tensor> &global_scale_x,
                                    const std::optional<at::Tensor> &global_scale_w);

std::vector<std::string> trace_gather_qqmm(const at::Tensor &x, const at::Tensor &w,
                                           const std::optional<at::Tensor> &scales,
                                           const std::optional<at::Tensor> &lhs_indices,
                                           const std::optional<at::Tensor> &rhs_indices,
                                           std::optional<int64_t> group_size, std::optional<int64_t> bits,
                                           const std::string &mode, const std::optional<at::Tensor> &global_scale_x,
                                           const std::optional<at::Tensor> &global_scale_w, bool sorted_indices);
