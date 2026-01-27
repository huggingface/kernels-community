#pragma once

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>

// SM80
torch::Tensor qk_int8_sv_f16_accum_f32_attn(torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse);

torch::Tensor qk_int8_sv_f16_accum_f16_attn(torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse);

torch::Tensor qk_int8_sv_f16_accum_f16_attn_inst_buf(torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse);

torch::Tensor qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    torch::Tensor value_mean,
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse);

//SM89 & 90
torch::Tensor qk_int8_sv_f8_accum_f32_attn(torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse);

torch::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    torch::Tensor value_scale,
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse);

torch::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn(torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    torch::Tensor value_scale,
    torch::Tensor value_mean,
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse);

torch::Tensor qk_int8_sv_f8_accum_f32_attn_inst_buf(torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse);

torch::Tensor qk_int8_sv_f8_accum_f32_attn_inst_buf_sm90(torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse);

torch::Tensor qk_int8_sv_f8_accum_f16_attn_inst_buf(torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse);

torch::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    torch::Tensor value_scale,
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse);

torch::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90(torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    torch::Tensor value_scale,
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse);

torch::Tensor qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor output,
    torch::Tensor query_scale,
    torch::Tensor key_scale,
    torch::Tensor value_scale,
    int tensor_layout,
    int is_causal,
    int qk_quant_gran,
    float sm_scale,
    int return_lse);

// Fused
void quant_per_block_int8_cuda(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    float sm_scale,
    int block_size,
    int tensor_layout);

void quant_per_block_int8_fuse_sub_mean_cuda(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor output,
    torch::Tensor scale,
    int block_size,
    int tensor_layout);

void quant_per_warp_int8_cuda(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    int block_size,
    int warp_block_size,
    int tensor_layout);

void sub_mean_cuda(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor output,
    int tensor_layout);

void transpose_pad_permute_cuda(
    torch::Tensor input,
    torch::Tensor output,
    int tensor_layout);

void scale_fuse_quant_cuda(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    int num_tokens,
    float scale_max,
    int tensor_layout);

void mean_scale_fuse_quant_cuda(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor mean,
    torch::Tensor scale,
    int num_tokens,
    float scale_max,
    int tensor_layout);



void sm_check_89(torch::Tensor x, std::string op_name) {
    int device_index = x.get_device();
    const auto& prop = at::cuda::getDeviceProperties(device_index);
  
    if (prop->major < 8 || (prop->major == 8 && prop->minor < 9)) {
        TORCH_CHECK(false, op_name + " requires compute capability 8.9+");
    }
}

void sm_check_90(torch::Tensor x, std::string op_name) {
    int device_index = x.get_device();
    const auto& prop = at::cuda::getDeviceProperties(device_index);

    if (prop->major < 9) {
        TORCH_CHECK(false, op_name + " requires compute capability 9.0+");
    }
}

void sm_check_80(torch::Tensor x, std::string op_name) {
    int device_index = x.get_device();
    const auto& prop = at::cuda::getDeviceProperties(device_index);
    
    if (prop->major < 8) {
        TORCH_CHECK(false, op_name + " requires compute capability 8.0+");
    }
}

// ##############################################################################
// SM89
// ##############################################################################
static at::Tensor qk_int8_sv_f8_accum_f32_attn_wrap(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
    at::Tensor q_scale, at::Tensor k_scale,
    int64_t tensor_layout, int64_t is_causal, int64_t qk_quant_gran,
    double sm_scale, int64_t return_lse) {
    sm_check_89(q, "qk_int8_sv_f8_accum_f32_attn");
    return qk_int8_sv_f8_accum_f32_attn(
        q, k, v, o, q_scale, k_scale,
        static_cast<int>(tensor_layout), static_cast<int>(is_causal), static_cast<int>(qk_quant_gran),
        static_cast<float>(sm_scale), static_cast<int>(return_lse));
}

static at::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_wrap(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
    at::Tensor q_scale, at::Tensor k_scale, at::Tensor v_scale,
    int64_t tensor_layout, int64_t is_causal, int64_t qk_quant_gran,
    double sm_scale, int64_t return_lse) {
    sm_check_89(q, "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn");
    return qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(
        q, k, v, o, q_scale, k_scale, v_scale,
        static_cast<int>(tensor_layout), static_cast<int>(is_causal), static_cast<int>(qk_quant_gran),
        static_cast<float>(sm_scale), static_cast<int>(return_lse));
}

static at::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn_wrap(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
    at::Tensor q_scale, at::Tensor k_scale, at::Tensor v_scale, at::Tensor v_mean,
    int64_t tensor_layout, int64_t is_causal, int64_t qk_quant_gran,
    double sm_scale, int64_t return_lse) {
    sm_check_89(q, "qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn");
    return qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn(
        q, k, v, o, q_scale, k_scale, v_scale, v_mean,
        static_cast<int>(tensor_layout), static_cast<int>(is_causal), static_cast<int>(qk_quant_gran),
        static_cast<float>(sm_scale), static_cast<int>(return_lse));
}

static at::Tensor qk_int8_sv_f8_accum_f32_attn_inst_buf_wrap(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
    at::Tensor q_scale, at::Tensor k_scale,
    int64_t tensor_layout, int64_t is_causal, int64_t qk_quant_gran,
    double sm_scale, int64_t return_lse) {
    sm_check_89(q, "qk_int8_sv_f8_accum_f32_attn_inst_buf");
    return qk_int8_sv_f8_accum_f32_attn_inst_buf(
        q, k, v, o, q_scale, k_scale,
        static_cast<int>(tensor_layout), static_cast<int>(is_causal), static_cast<int>(qk_quant_gran),
        static_cast<float>(sm_scale), static_cast<int>(return_lse));
}

static at::Tensor qk_int8_sv_f8_accum_f16_attn_inst_buf_wrap(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
    at::Tensor q_scale, at::Tensor k_scale,
    int64_t tensor_layout, int64_t is_causal, int64_t qk_quant_gran,
    double sm_scale, int64_t return_lse) {
    sm_check_89(q, "qk_int8_sv_f8_accum_f16_attn_inst_buf");
    return qk_int8_sv_f8_accum_f16_attn_inst_buf(
        q, k, v, o, q_scale, k_scale,
        static_cast<int>(tensor_layout), static_cast<int>(is_causal), static_cast<int>(qk_quant_gran),
        static_cast<float>(sm_scale), static_cast<int>(return_lse));
}

static at::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_wrap(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
    at::Tensor q_scale, at::Tensor k_scale, at::Tensor v_scale,
    int64_t tensor_layout, int64_t is_causal, int64_t qk_quant_gran,
    double sm_scale, int64_t return_lse) {
    sm_check_89(q, "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf");
    return qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
        q, k, v, o, q_scale, k_scale, v_scale,
        static_cast<int>(tensor_layout), static_cast<int>(is_causal), static_cast<int>(qk_quant_gran),
        static_cast<float>(sm_scale), static_cast<int>(return_lse));
}

static at::Tensor qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf_wrap(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
    at::Tensor q_scale, at::Tensor k_scale, at::Tensor v_scale,
    int64_t tensor_layout, int64_t is_causal, int64_t qk_quant_gran,
    double sm_scale, int64_t return_lse) {
    sm_check_89(q, "qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf");
    return qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
        q, k, v, o, q_scale, k_scale, v_scale,
        static_cast<int>(tensor_layout), static_cast<int>(is_causal), static_cast<int>(qk_quant_gran),
        static_cast<float>(sm_scale), static_cast<int>(return_lse));
}


// ##############################################################################
// SM90
// ##############################################################################

static at::Tensor qk_int8_sv_f8_accum_f32_attn_inst_buf_sm90_wrap(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
    at::Tensor q_scale, at::Tensor k_scale,
    int64_t tensor_layout, int64_t is_causal, int64_t qk_quant_gran,
    double sm_scale, int64_t return_lse) {
    sm_check_90(q, "qk_int8_sv_f8_accum_f32_attn_inst_buf_sm90");
return qk_int8_sv_f8_accum_f32_attn_inst_buf_sm90(
    q, k, v, o, q_scale, k_scale,
    static_cast<int>(tensor_layout), static_cast<int>(is_causal), static_cast<int>(qk_quant_gran),
    static_cast<float>(sm_scale), static_cast<int>(return_lse));
}

static at::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90_wrap(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
    at::Tensor q_scale, at::Tensor k_scale, at::Tensor v_scale,
    int64_t tensor_layout, int64_t is_causal, int64_t qk_quant_gran,
    double sm_scale, int64_t return_lse) {
    sm_check_90(q, "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90");
return qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90(
    q, k, v, o, q_scale, k_scale, v_scale,
    static_cast<int>(tensor_layout), static_cast<int>(is_causal), static_cast<int>(qk_quant_gran),
    static_cast<float>(sm_scale), static_cast<int>(return_lse));
}

// ##############################################################################
// SM80
// ##############################################################################
static at::Tensor qk_int8_sv_f16_accum_f32_attn_wrap(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
    at::Tensor q_scale, at::Tensor k_scale,
    int64_t tensor_layout, int64_t is_causal, int64_t qk_quant_gran,
    double sm_scale, int64_t return_lse) {
    sm_check_80(q, "qk_int8_sv_f16_accum_f32_attn");
    return qk_int8_sv_f16_accum_f32_attn(
        q, k, v, o, q_scale, k_scale,
        static_cast<int>(tensor_layout), static_cast<int>(is_causal), static_cast<int>(qk_quant_gran),
        static_cast<float>(sm_scale), static_cast<int>(return_lse));
}

static at::Tensor qk_int8_sv_f16_accum_f16_attn_wrap(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
    at::Tensor q_scale, at::Tensor k_scale,
    int64_t tensor_layout, int64_t is_causal, int64_t qk_quant_gran,
    double sm_scale, int64_t return_lse) {
    sm_check_80(q, "qk_int8_sv_f16_accum_f16_attn");
    return qk_int8_sv_f16_accum_f16_attn(
        q, k, v, o, q_scale, k_scale,
        static_cast<int>(tensor_layout), static_cast<int>(is_causal), static_cast<int>(qk_quant_gran),
        static_cast<float>(sm_scale), static_cast<int>(return_lse));
}

static at::Tensor qk_int8_sv_f16_accum_f16_attn_inst_buf_wrap(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
    at::Tensor q_scale, at::Tensor k_scale,
    int64_t tensor_layout, int64_t is_causal, int64_t qk_quant_gran,
    double sm_scale, int64_t return_lse) {
    sm_check_80(q, "qk_int8_sv_f16_accum_f16_attn_inst_buf");
    return qk_int8_sv_f16_accum_f16_attn_inst_buf(
        q, k, v, o, q_scale, k_scale,
        static_cast<int>(tensor_layout), static_cast<int>(is_causal), static_cast<int>(qk_quant_gran),
        static_cast<float>(sm_scale), static_cast<int>(return_lse));
}

static at::Tensor qk_int8_sv_f16_accum_f16_fuse_v_mean_attn_wrap(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
    at::Tensor q_scale, at::Tensor k_scale, at::Tensor v_mean,
    int64_t tensor_layout, int64_t is_causal, int64_t qk_quant_gran,
    double sm_scale, int64_t return_lse) {
    sm_check_80(q, "qk_int8_sv_f16_accum_f16_fuse_v_mean_attn");
    return qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(
        q, k, v, o, q_scale, k_scale, v_mean,
        static_cast<int>(tensor_layout), static_cast<int>(is_causal), static_cast<int>(qk_quant_gran),
        static_cast<float>(sm_scale), static_cast<int>(return_lse));
}

// ##############################################################################
// Fused
// ##############################################################################
static void quant_per_block_int8_cuda_wrap(
    at::Tensor input, at::Tensor output, at::Tensor scale,
    double sm_scale, int64_t block_size, int64_t tensor_layout) {
    quant_per_block_int8_cuda(
        input, output, scale,
        static_cast<float>(sm_scale), static_cast<int>(block_size), static_cast<int>(tensor_layout));
}

static void quant_per_block_int8_fuse_sub_mean_cuda_wrap(
    at::Tensor input, at::Tensor mean, at::Tensor output, at::Tensor scale,
    int64_t block_size, int64_t tensor_layout) {
    quant_per_block_int8_fuse_sub_mean_cuda(
        input, mean, output, scale,
        static_cast<int>(block_size), static_cast<int>(tensor_layout));
}

static void quant_per_warp_int8_cuda_wrap(
    at::Tensor input, at::Tensor output, at::Tensor scale,
    int64_t block_size, int64_t warp_block_size, int64_t tensor_layout) {
    quant_per_warp_int8_cuda(
        input, output, scale,
        static_cast<int>(block_size), static_cast<int>(warp_block_size), static_cast<int>(tensor_layout));
}

static void sub_mean_cuda_wrap(
    at::Tensor input, at::Tensor mean, at::Tensor output,
    int64_t tensor_layout) {
    sub_mean_cuda(input, mean, output, static_cast<int>(tensor_layout));
}

static void transpose_pad_permute_cuda_wrap(
    at::Tensor input, at::Tensor output, int64_t tensor_layout) {
    transpose_pad_permute_cuda(input, output, static_cast<int>(tensor_layout));
}

static void scale_fuse_quant_cuda_wrap(
    at::Tensor input, at::Tensor output, at::Tensor scale,
    int64_t num_tokens, double scale_max, int64_t tensor_layout) {
    scale_fuse_quant_cuda(
        input, output, scale,
        static_cast<int>(num_tokens), static_cast<float>(scale_max), static_cast<int>(tensor_layout));
}

static void mean_scale_fuse_quant_cuda_wrap(
    at::Tensor input, at::Tensor output, at::Tensor mean, at::Tensor scale,
    int64_t num_tokens, double scale_max, int64_t tensor_layout) {
    mean_scale_fuse_quant_cuda(
        input, output, mean, scale,
        static_cast<int>(num_tokens), static_cast<float>(scale_max), static_cast<int>(tensor_layout));
}