#include <torch/library.h>
#include <ATen/Tensor.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    // Runtime ops (no device dispatch - these are host-side)
    ops.def("init(str path, str cuda_home) -> ()");
    ops.impl("init", &deep_gemm_init);

    ops.def("set_num_sms(int num_sms) -> ()");
    ops.impl("set_num_sms", &deep_gemm_set_num_sms);

    ops.def("get_num_sms() -> int");
    ops.impl("get_num_sms", &deep_gemm_get_num_sms);

    ops.def("set_tc_util(int tc_util) -> ()");
    ops.impl("set_tc_util", &deep_gemm_set_tc_util);

    ops.def("get_tc_util() -> int");
    ops.impl("get_tc_util", &deep_gemm_get_tc_util);

    ops.def("get_mk_alignment_for_contiguous_layout() -> int");
    ops.impl("get_mk_alignment_for_contiguous_layout",
             &deep_gemm_get_mk_alignment_for_contiguous_layout);

    // Layout ops (CUDA dispatch)
    ops.def(
        "get_tma_aligned_size(int mn, int element_size) -> Tensor"
    );
    ops.impl("get_tma_aligned_size", torch::kCUDA, &deep_gemm_get_tma_aligned_size);

    ops.def(
        "get_mn_major_tma_aligned_tensor(Tensor sf) -> Tensor"
    );
    ops.impl("get_mn_major_tma_aligned_tensor", torch::kCUDA,
             &deep_gemm_get_mn_major_tma_aligned_tensor);

    ops.def(
        "get_mn_major_tma_aligned_packed_ue8m0_tensor(Tensor sf) -> Tensor"
    );
    ops.impl("get_mn_major_tma_aligned_packed_ue8m0_tensor", torch::kCUDA,
             &deep_gemm_get_mn_major_tma_aligned_packed_ue8m0_tensor);

    ops.def(
        "get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor("
        "Tensor sf, Tensor ks_tensor, Tensor ks_int_tensor) -> Tensor"
    );
    ops.impl("get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor", torch::kCUDA,
             &deep_gemm_get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor);

    ops.def(
        "transform_sf_into_required_layout("
        "Tensor sf, int mn, int k, "
        "int recipe_0, int recipe_1, int recipe_2, bool has_recipe, "
        "int recipe_ab_0, int recipe_ab_1, bool has_recipe_ab, "
        "int num_groups, bool has_num_groups, "
        "bool is_sfa, bool disable_ue8m0_cast) -> Tensor"
    );
    ops.impl("transform_sf_into_required_layout", torch::kCUDA,
             &deep_gemm_transform_sf_into_required_layout);

    // FP8/FP4 GEMM ops (CUDA dispatch)
    ops.def(
        "fp8_fp4_gemm_nt("
        "Tensor a_data, Tensor a_sf, Tensor b_data, Tensor b_sf, "
        "Tensor d, Tensor? c, "
        "int recipe_0, int recipe_1, int recipe_2, bool has_recipe, "
        "int recipe_a_0, int recipe_a_1, bool has_recipe_a, "
        "int recipe_b_0, int recipe_b_1, bool has_recipe_b, "
        "str compiled_dims, bool disable_ue8m0_cast) -> ()"
    );
    ops.impl("fp8_fp4_gemm_nt", torch::kCUDA, &deep_gemm_fp8_fp4_gemm_nt);

    ops.def(
        "fp8_fp4_gemm_nn("
        "Tensor a_data, Tensor a_sf, Tensor b_data, Tensor b_sf, "
        "Tensor d, Tensor? c, "
        "int recipe_0, int recipe_1, int recipe_2, bool has_recipe, "
        "int recipe_a_0, int recipe_a_1, bool has_recipe_a, "
        "int recipe_b_0, int recipe_b_1, bool has_recipe_b, "
        "str compiled_dims, bool disable_ue8m0_cast) -> ()"
    );
    ops.impl("fp8_fp4_gemm_nn", torch::kCUDA, &deep_gemm_fp8_fp4_gemm_nn);

    ops.def(
        "fp8_fp4_gemm_tn("
        "Tensor a_data, Tensor a_sf, Tensor b_data, Tensor b_sf, "
        "Tensor d, Tensor? c, "
        "int recipe_0, int recipe_1, int recipe_2, bool has_recipe, "
        "int recipe_a_0, int recipe_a_1, bool has_recipe_a, "
        "int recipe_b_0, int recipe_b_1, bool has_recipe_b, "
        "str compiled_dims, bool disable_ue8m0_cast) -> ()"
    );
    ops.impl("fp8_fp4_gemm_tn", torch::kCUDA, &deep_gemm_fp8_fp4_gemm_tn);

    ops.def(
        "fp8_fp4_gemm_tt("
        "Tensor a_data, Tensor a_sf, Tensor b_data, Tensor b_sf, "
        "Tensor d, Tensor? c, "
        "int recipe_0, int recipe_1, int recipe_2, bool has_recipe, "
        "int recipe_a_0, int recipe_a_1, bool has_recipe_a, "
        "int recipe_b_0, int recipe_b_1, bool has_recipe_b, "
        "str compiled_dims, bool disable_ue8m0_cast) -> ()"
    );
    ops.impl("fp8_fp4_gemm_tt", torch::kCUDA, &deep_gemm_fp8_fp4_gemm_tt);

    // M-grouped FP8/FP4 GEMM ops (CUDA dispatch)
    ops.def(
        "m_grouped_fp8_fp4_gemm_nt_contiguous("
        "Tensor a_data, Tensor a_sf, Tensor b_data, Tensor b_sf, "
        "Tensor d, Tensor grouped_layout, "
        "int recipe_0, int recipe_1, int recipe_2, bool has_recipe, "
        "int recipe_a_0, int recipe_a_1, bool has_recipe_a, "
        "int recipe_b_0, int recipe_b_1, bool has_recipe_b, "
        "str compiled_dims, bool disable_ue8m0_cast, "
        "bool use_psum_layout, int expected_m_for_psum_layout, "
        "bool has_expected_m_for_psum_layout) -> ()"
    );
    ops.impl("m_grouped_fp8_fp4_gemm_nt_contiguous", torch::kCUDA,
             &deep_gemm_m_grouped_fp8_fp4_gemm_nt_contiguous);

    ops.def(
        "m_grouped_fp8_fp4_gemm_nn_contiguous("
        "Tensor a_data, Tensor a_sf, Tensor b_data, Tensor b_sf, "
        "Tensor d, Tensor grouped_layout, "
        "int recipe_0, int recipe_1, int recipe_2, bool has_recipe, "
        "int recipe_a_0, int recipe_a_1, bool has_recipe_a, "
        "int recipe_b_0, int recipe_b_1, bool has_recipe_b, "
        "str compiled_dims, bool disable_ue8m0_cast, "
        "bool use_psum_layout) -> ()"
    );
    ops.impl("m_grouped_fp8_fp4_gemm_nn_contiguous", torch::kCUDA,
             &deep_gemm_m_grouped_fp8_fp4_gemm_nn_contiguous);

    ops.def(
        "m_grouped_fp8_fp4_gemm_nt_masked("
        "Tensor a_data, Tensor a_sf, Tensor b_data, Tensor b_sf, "
        "Tensor d, Tensor masked_m, int expected_m, "
        "int recipe_0, int recipe_1, int recipe_2, bool has_recipe, "
        "int recipe_a_0, int recipe_a_1, bool has_recipe_a, "
        "int recipe_b_0, int recipe_b_1, bool has_recipe_b, "
        "str compiled_dims, bool disable_ue8m0_cast) -> ()"
    );
    ops.impl("m_grouped_fp8_fp4_gemm_nt_masked", torch::kCUDA,
             &deep_gemm_m_grouped_fp8_fp4_gemm_nt_masked);

    // K-grouped FP8 GEMM ops (CUDA dispatch)
    ops.def(
        "k_grouped_fp8_gemm_tn_contiguous("
        "Tensor a_data, Tensor a_sf, Tensor b_data, Tensor b_sf, "
        "Tensor d, Tensor ks_tensor, Tensor? c, "
        "int recipe_0, int recipe_1, int recipe_2, "
        "str compiled_dims) -> ()"
    );
    ops.impl("k_grouped_fp8_gemm_tn_contiguous", torch::kCUDA,
             &deep_gemm_k_grouped_fp8_gemm_tn_contiguous);

    ops.def(
        "k_grouped_fp8_gemm_nt_contiguous("
        "Tensor a_data, Tensor a_sf, Tensor b_data, Tensor b_sf, "
        "Tensor d, Tensor ks_tensor, Tensor? c, "
        "int recipe_0, int recipe_1, int recipe_2, "
        "str compiled_dims) -> ()"
    );
    ops.impl("k_grouped_fp8_gemm_nt_contiguous", torch::kCUDA,
             &deep_gemm_k_grouped_fp8_gemm_nt_contiguous);

    // BF16 GEMM ops (CUDA dispatch)
    ops.def(
        "bf16_gemm_nt(Tensor a, Tensor b, Tensor d, Tensor? c, "
        "str compiled_dims) -> ()"
    );
    ops.impl("bf16_gemm_nt", torch::kCUDA, &deep_gemm_bf16_gemm_nt);

    ops.def(
        "bf16_gemm_nn(Tensor a, Tensor b, Tensor d, Tensor? c, "
        "str compiled_dims) -> ()"
    );
    ops.impl("bf16_gemm_nn", torch::kCUDA, &deep_gemm_bf16_gemm_nn);

    ops.def(
        "bf16_gemm_tn(Tensor a, Tensor b, Tensor d, Tensor? c, "
        "str compiled_dims) -> ()"
    );
    ops.impl("bf16_gemm_tn", torch::kCUDA, &deep_gemm_bf16_gemm_tn);

    ops.def(
        "bf16_gemm_tt(Tensor a, Tensor b, Tensor d, Tensor? c, "
        "str compiled_dims) -> ()"
    );
    ops.impl("bf16_gemm_tt", torch::kCUDA, &deep_gemm_bf16_gemm_tt);

    // M-grouped BF16 GEMM ops (CUDA dispatch)
    ops.def(
        "m_grouped_bf16_gemm_nt_contiguous("
        "Tensor a, Tensor b, Tensor d, Tensor grouped_layout, "
        "str compiled_dims, bool use_psum_layout, "
        "int expected_m_for_psum_layout, bool has_expected_m_for_psum_layout) -> ()"
    );
    ops.impl("m_grouped_bf16_gemm_nt_contiguous", torch::kCUDA,
             &deep_gemm_m_grouped_bf16_gemm_nt_contiguous);

    ops.def(
        "m_grouped_bf16_gemm_nn_contiguous("
        "Tensor a, Tensor b, Tensor d, Tensor grouped_layout, "
        "str compiled_dims, bool use_psum_layout) -> ()"
    );
    ops.impl("m_grouped_bf16_gemm_nn_contiguous", torch::kCUDA,
             &deep_gemm_m_grouped_bf16_gemm_nn_contiguous);

    ops.def(
        "m_grouped_bf16_gemm_nt_masked("
        "Tensor a, Tensor b, Tensor d, Tensor masked_m, "
        "int expected_m, str compiled_dims) -> ()"
    );
    ops.impl("m_grouped_bf16_gemm_nt_masked", torch::kCUDA,
             &deep_gemm_m_grouped_bf16_gemm_nt_masked);

    // K-grouped BF16 GEMM ops (CUDA dispatch)
    ops.def(
        "k_grouped_bf16_gemm_tn_contiguous("
        "Tensor a, Tensor b, Tensor d, Tensor ks_tensor, Tensor? c, "
        "str compiled_dims) -> ()"
    );
    ops.impl("k_grouped_bf16_gemm_tn_contiguous", torch::kCUDA,
             &deep_gemm_k_grouped_bf16_gemm_tn_contiguous);

    // cuBLASLt GEMM ops (CUDA dispatch)
    ops.def(
        "cublaslt_gemm_nt(Tensor a, Tensor b, Tensor d, Tensor? c) -> ()"
    );
    ops.impl("cublaslt_gemm_nt", torch::kCUDA, &deep_gemm_cublaslt_gemm_nt);

    ops.def(
        "cublaslt_gemm_nn(Tensor a, Tensor b, Tensor d, Tensor? c) -> ()"
    );
    ops.impl("cublaslt_gemm_nn", torch::kCUDA, &deep_gemm_cublaslt_gemm_nn);

    ops.def(
        "cublaslt_gemm_tn(Tensor a, Tensor b, Tensor d, Tensor? c) -> ()"
    );
    ops.impl("cublaslt_gemm_tn", torch::kCUDA, &deep_gemm_cublaslt_gemm_tn);

    ops.def(
        "cublaslt_gemm_tt(Tensor a, Tensor b, Tensor d, Tensor? c) -> ()"
    );
    ops.impl("cublaslt_gemm_tt", torch::kCUDA, &deep_gemm_cublaslt_gemm_tt);

    // Attention ops (CUDA dispatch)
    ops.def(
        "fp8_gemm_nt_skip_head_mid("
        "Tensor a_data, Tensor a_sf, Tensor b_data, Tensor b_sf, "
        "Tensor d, "
        "int head_split_left, int head_split_mid, int head_split_right, "
        "int recipe_0, int recipe_1, int recipe_2, bool has_recipe, "
        "str compiled_dims, bool disable_ue8m0_cast) -> ()"
    );
    ops.impl("fp8_gemm_nt_skip_head_mid", torch::kCUDA,
             &deep_gemm_fp8_gemm_nt_skip_head_mid);

    ops.def(
        "fp8_mqa_logits("
        "Tensor q, Tensor kv_data, Tensor kv_sf, "
        "Tensor weights, Tensor cu_seq_len_k_start, Tensor cu_seq_len_k_end, "
        "bool clean_logits, int max_seqlen_k) -> Tensor"
    );
    ops.impl("fp8_mqa_logits", torch::kCUDA, &deep_gemm_fp8_mqa_logits);

    ops.def(
        "get_paged_mqa_logits_metadata("
        "Tensor context_lens, int block_kv, int num_sms) -> Tensor"
    );
    ops.impl("get_paged_mqa_logits_metadata", torch::kCUDA,
             &deep_gemm_get_paged_mqa_logits_metadata);

    ops.def(
        "fp8_paged_mqa_logits("
        "Tensor q, Tensor fused_kv_cache, "
        "Tensor weights, Tensor context_lens, "
        "Tensor block_table, Tensor schedule_meta, "
        "int max_context_len, bool clean_logits) -> Tensor"
    );
    ops.impl("fp8_paged_mqa_logits", torch::kCUDA,
             &deep_gemm_fp8_paged_mqa_logits);

    // Einsum ops (CUDA dispatch)
    ops.def(
        "einsum(str expr, Tensor a, Tensor b, Tensor d, "
        "Tensor? c, bool use_cublaslt) -> ()"
    );
    ops.impl("einsum", torch::kCUDA, &deep_gemm_einsum);

    ops.def(
        "fp8_einsum(str expr, "
        "Tensor a_data, Tensor a_sf, Tensor b_data, Tensor b_sf, "
        "Tensor d, Tensor? c, "
        "int recipe_0, int recipe_1, int recipe_2) -> ()"
    );
    ops.impl("fp8_einsum", torch::kCUDA, &deep_gemm_fp8_einsum);

    // Hyperconnection ops (CUDA dispatch)
    ops.def(
        "tf32_hc_prenorm_gemm("
        "Tensor a, Tensor b, Tensor d, Tensor sqr_sum, "
        "int num_splits, bool has_num_splits) -> ()"
    );
    ops.impl("tf32_hc_prenorm_gemm", torch::kCUDA,
             &deep_gemm_tf32_hc_prenorm_gemm);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
