#pragma once

#include <torch/torch.h>

#include <string>

// y = x @ dequantize(w, scales, biases).T, MLX's `quantized_matmul(..., transpose=True)` in affine
// mode. `x` is [..., K] (float32/float16/bfloat16), `w` is [N, K * bits / 32] uint32, `scales` and
// `biases` are [N, K / group_size] in x's dtype. Returns [..., N]. Picks the kernel the way MLX does.
at::Tensor affine_qmm_t(const at::Tensor &x, const at::Tensor &w, const at::Tensor &scales,
                        const at::Tensor &biases, int64_t group_size, int64_t bits);

// The kernel `affine_qmm_t` would launch for an [M, K] x [N, K] product on this GPU, by name.
std::string kernel_for(int64_t M, int64_t N, int64_t K, int64_t group_size, int64_t bits,
                       at::ScalarType dtype);
