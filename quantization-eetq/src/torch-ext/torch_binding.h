#pragma once

#include <vector>

#include <torch/torch.h>

std::vector<torch::Tensor>
symmetric_quantize_last_axis_of_tensor(torch::Tensor const &weight,
                                       at::ScalarType quant_type,
                                       bool return_unprocessed_quantized_tensor);

torch::Tensor preprocess_weights_cuda(torch::Tensor const &ori_weight,
                                      bool is_int4);

torch::Tensor w8_a16_gemm_forward_cuda(torch::Tensor const &input,
                                       torch::Tensor const&weight,
                                       torch::Tensor const &scale);

torch::Tensor w8_a16_gemm_forward_cuda_(torch::Tensor const &input,
                                        torch::Tensor const &weight,
                                        torch::Tensor const &scale,
                                        torch::Tensor &output,
                                        const int64_t m,
                                        const int64_t n,
                                        const int64_t k);
