#pragma once

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>

#include <optional>
#include <vector>

// FP4 attention forward (sage_blackwell/blackwell/api.cu)
std::vector<at::Tensor>
mha_fwd(at::Tensor &q,               // batch_size x num_heads x seqlen_q x (head_size // 2), packed fp4
        const at::Tensor &k,         // batch_size x num_heads_k x seqlen_k x (head_size // 2), packed fp4
        const at::Tensor &v,         // batch_size x num_heads_k x head_size x (seqlen_k // 2), packed fp4
        const at::Tensor &sfq,
        const at::Tensor &sfk,
        const at::Tensor &sfv,
        const at::Tensor &delta_s,
        int unpadded_k,
        std::optional<at::Tensor> out_,
        const float softmax_scale,
        bool is_causal,
        bool per_block_mean,
        bool is_bf16);

// FP4 quantization (sage_blackwell/quantization/fp4_quantization_4d.cu)
void scaled_fp4_quant(torch::Tensor const &input,
                      torch::Tensor const &output,
                      torch::Tensor const &output_sf,
                      int tensor_layout);

void scaled_fp4_quant_permute(torch::Tensor const &input,
                              torch::Tensor const &output,
                              torch::Tensor const &output_sf,
                              int tensor_layout);

void scaled_fp4_quant_trans(torch::Tensor const &input,
                            torch::Tensor const &output,
                            torch::Tensor const &output_sf,
                            int tensor_layout);

static std::vector<at::Tensor> fwd_wrap(
    at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor sfq, at::Tensor sfk, at::Tensor sfv,
    at::Tensor delta_s, int64_t unpadded_k,
    std::optional<at::Tensor> out_, double softmax_scale,
    bool is_causal, bool per_block_mean, bool is_bf16) {
    return mha_fwd(
        q, k, v, sfq, sfk, sfv, delta_s,
        static_cast<int>(unpadded_k), out_,
        static_cast<float>(softmax_scale), is_causal, per_block_mean, is_bf16);
}

static void scaled_fp4_quant_wrap(
    at::Tensor input, at::Tensor output, at::Tensor output_sf,
    int64_t tensor_layout) {
    scaled_fp4_quant(input, output, output_sf, static_cast<int>(tensor_layout));
}

static void scaled_fp4_quant_permute_wrap(
    at::Tensor input, at::Tensor output, at::Tensor output_sf,
    int64_t tensor_layout) {
    scaled_fp4_quant_permute(input, output, output_sf, static_cast<int>(tensor_layout));
}

static void scaled_fp4_quant_trans_wrap(
    at::Tensor input, at::Tensor output, at::Tensor output_sf,
    int64_t tensor_layout) {
    scaled_fp4_quant_trans(input, output, output_sf, static_cast<int>(tensor_layout));
}
