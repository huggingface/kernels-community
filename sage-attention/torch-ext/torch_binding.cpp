#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"


TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    // SM90
    ops.def("qk_int8_sv_f8_accum_f32_attn_inst_buf_sm90(Tensor q, Tensor k, Tensor v, Tensor! o, Tensor q_scale, Tensor k_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor");
    ops.impl("qk_int8_sv_f8_accum_f32_attn_inst_buf_sm90", torch::kCUDA, &qk_int8_sv_f8_accum_f32_attn_inst_buf_sm90_wrap);

    ops.def("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90(Tensor q, Tensor k, Tensor v, Tensor! o, Tensor q_scale, Tensor k_scale, Tensor v_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor");
    ops.impl("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90", torch::kCUDA, &qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90_wrap);

    ops.def("qk_int8_sv_f8_accum_f32_attn_inst_buf(Tensor q, Tensor k, Tensor v, Tensor! o, Tensor q_scale, Tensor k_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor");
    ops.impl("qk_int8_sv_f8_accum_f32_attn_inst_buf", torch::kCUDA, &qk_int8_sv_f8_accum_f32_attn_inst_buf_wrap);

    ops.def("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(Tensor q, Tensor k, Tensor v, Tensor! o, Tensor q_scale, Tensor k_scale, Tensor v_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor");
    ops.impl("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf", torch::kCUDA, &qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_wrap);
    
    ops.def("qk_int8_sv_f8_accum_f32_attn(Tensor q, Tensor k, Tensor v, Tensor! o, Tensor q_scale, Tensor k_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor");
    ops.impl("qk_int8_sv_f8_accum_f32_attn", torch::kCUDA, &qk_int8_sv_f8_accum_f32_attn_wrap);

    ops.def("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(Tensor q, Tensor k, Tensor v, Tensor! o, Tensor q_scale, Tensor k_scale, Tensor v_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor");
    ops.impl("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn", torch::kCUDA, &qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_wrap);

    ops.def("qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn(Tensor q, Tensor k, Tensor v, Tensor! o, Tensor q_scale, Tensor k_scale, Tensor v_scale, Tensor v_mean, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor");
    ops.impl("qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn", torch::kCUDA, &qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn_wrap);
    
    ops.def("qk_int8_sv_f8_accum_f16_attn_inst_buf(Tensor q, Tensor k, Tensor v, Tensor! o, Tensor q_scale, Tensor k_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor");
    ops.impl("qk_int8_sv_f8_accum_f16_attn_inst_buf", torch::kCUDA, &qk_int8_sv_f8_accum_f16_attn_inst_buf_wrap);
    
    ops.def("qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(Tensor q, Tensor k, Tensor v, Tensor! o, Tensor q_scale, Tensor k_scale, Tensor v_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor");
    ops.impl("qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf", torch::kCUDA, &qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf_wrap);

    ops.def("qk_int8_sv_f16_accum_f32_attn(Tensor q, Tensor k, Tensor v, Tensor! o, Tensor q_scale, Tensor k_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor");
    ops.impl("qk_int8_sv_f16_accum_f32_attn", torch::kCUDA, &qk_int8_sv_f16_accum_f32_attn_wrap);

    ops.def("qk_int8_sv_f16_accum_f16_attn(Tensor q, Tensor k, Tensor v, Tensor! o, Tensor q_scale, Tensor k_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor");
    ops.impl("qk_int8_sv_f16_accum_f16_attn", torch::kCUDA, &qk_int8_sv_f16_accum_f16_attn_wrap);

    ops.def("qk_int8_sv_f16_accum_f16_attn_inst_buf(Tensor q, Tensor k, Tensor v, Tensor! o, Tensor q_scale, Tensor k_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor");
    ops.impl("qk_int8_sv_f16_accum_f16_attn_inst_buf", torch::kCUDA, &qk_int8_sv_f16_accum_f16_attn_inst_buf_wrap);

    ops.def("qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(Tensor q, Tensor k, Tensor v, Tensor! o, Tensor q_scale, Tensor k_scale, Tensor v_mean, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor");
    ops.impl("qk_int8_sv_f16_accum_f16_fuse_v_mean_attn", torch::kCUDA, &qk_int8_sv_f16_accum_f16_fuse_v_mean_attn_wrap);

    //Fused (available across supported archs)
    ops.def("quant_per_block_int8_cuda(Tensor input, Tensor! output, Tensor scale, float sm_scale, int block_size, int tensor_layout) -> ()");
    ops.impl("quant_per_block_int8_cuda", torch::kCUDA, &quant_per_block_int8_cuda_wrap);

    ops.def("quant_per_block_int8_fuse_sub_mean_cuda(Tensor input, Tensor mean, Tensor! output, Tensor scale, int block_size, int tensor_layout) -> ()");
    ops.impl("quant_per_block_int8_fuse_sub_mean_cuda", torch::kCUDA, &quant_per_block_int8_fuse_sub_mean_cuda_wrap);

    ops.def("quant_per_warp_int8_cuda(Tensor input, Tensor! output, Tensor scale, int block_size, int warp_block_size, int tensor_layout) -> ()");
    ops.impl("quant_per_warp_int8_cuda", torch::kCUDA, &quant_per_warp_int8_cuda_wrap);

    ops.def("sub_mean_cuda(Tensor input, Tensor mean, Tensor! output, int tensor_layout) -> ()");
    ops.impl("sub_mean_cuda", torch::kCUDA, &sub_mean_cuda_wrap);

    ops.def("transpose_pad_permute_cuda(Tensor input, Tensor! output, int tensor_layout) -> ()");
    ops.impl("transpose_pad_permute_cuda", torch::kCUDA, &transpose_pad_permute_cuda_wrap);

    ops.def("scale_fuse_quant_cuda(Tensor input, Tensor! output, Tensor scale, int num_tokens, float scale_max, int tensor_layout) -> ()");
    ops.impl("scale_fuse_quant_cuda", torch::kCUDA, &scale_fuse_quant_cuda_wrap);

    ops.def("mean_scale_fuse_quant_cuda(Tensor input, Tensor! output, Tensor mean, Tensor scale, int num_tokens, float scale_max, int tensor_layout) -> ()");
    ops.impl("mean_scale_fuse_quant_cuda", torch::kCUDA, &mean_scale_fuse_quant_cuda_wrap);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME);