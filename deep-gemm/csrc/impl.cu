#include <torch/torch.h>
#include <numeric>
#include <vector>

#include "../torch-ext/torch_binding.h"
#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/runtime.hpp"

using Tensor = at::Tensor;

// Helper: convert 1D int tensor to std::vector<int>
static std::vector<int> tensor_to_vec_int(const Tensor& t) {
    auto cpu_t = t.cpu().contiguous();
    auto ptr = cpu_t.data_ptr<int>();
    return std::vector<int>(ptr, ptr + cpu_t.numel());
}

// Helper: reconstruct optional recipe tuple
static std::optional<std::tuple<int, int, int>> make_recipe3(
    int64_t r0, int64_t r1, int64_t r2, bool has) {
    if (!has) return std::nullopt;
    return std::make_tuple(static_cast<int>(r0), static_cast<int>(r1), static_cast<int>(r2));
}

static std::optional<std::tuple<int, int>> make_recipe2(
    int64_t r0, int64_t r1, bool has) {
    if (!has) return std::nullopt;
    return std::make_tuple(static_cast<int>(r0), static_cast<int>(r1));
}

// Runtime ops

void deep_gemm_init(const std::string& path, const std::string& cuda_home) {
    deep_gemm::runtime::deep_gemm_init(path, cuda_home);
}

void deep_gemm_set_num_sms(int64_t num_sms) {
    deep_gemm::runtime::deep_gemm_set_num_sms(num_sms);
}

int64_t deep_gemm_get_num_sms() {
    return deep_gemm::runtime::deep_gemm_get_num_sms();
}

void deep_gemm_set_tc_util(int64_t tc_util) {
    deep_gemm::runtime::deep_gemm_set_tc_util(tc_util);
}

int64_t deep_gemm_get_tc_util() {
    return deep_gemm::runtime::deep_gemm_get_tc_util();
}

// Layout ops

int64_t deep_gemm_get_mk_alignment_for_contiguous_layout() {
    return deep_gemm::get_mk_alignment_for_contiguous_layout();
}

Tensor deep_gemm_get_tma_aligned_size(int64_t mn, int64_t element_size) {
    // Returns scalar tensor to satisfy TORCH_LIBRARY (can't return plain int)
    auto result = deep_gemm::get_tma_aligned_size(static_cast<int>(mn), static_cast<int>(element_size));
    return torch::tensor(result, torch::kInt64);
}

Tensor deep_gemm_get_mn_major_tma_aligned_tensor(const Tensor& sf) {
    return deep_gemm::get_mn_major_tma_aligned_tensor(sf);
}

Tensor deep_gemm_get_mn_major_tma_aligned_packed_ue8m0_tensor(const Tensor& sf) {
    return deep_gemm::get_mn_major_tma_aligned_packed_ue8m0_tensor(sf);
}

Tensor deep_gemm_get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
    const Tensor& sf, const Tensor& ks_tensor, const Tensor& ks_int_tensor) {
    auto ks = tensor_to_vec_int(ks_int_tensor);
    return deep_gemm::get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks);
}

Tensor deep_gemm_transform_sf_into_required_layout(
    const Tensor& sf, int64_t mn, int64_t k,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2, bool has_recipe,
    int64_t recipe_ab_0, int64_t recipe_ab_1, bool has_recipe_ab,
    int64_t num_groups, bool has_num_groups,
    bool is_sfa, bool disable_ue8m0_cast) {
    auto recipe = make_recipe3(recipe_0, recipe_1, recipe_2, has_recipe);
    auto recipe_ab = make_recipe2(recipe_ab_0, recipe_ab_1, has_recipe_ab);
    auto ng = has_num_groups ? std::make_optional(static_cast<int>(num_groups)) : std::nullopt;
    return deep_gemm::layout::transform_sf_into_required_layout(
        sf, static_cast<int>(mn), static_cast<int>(k),
        recipe, recipe_ab, ng, is_sfa, disable_ue8m0_cast);
}

// GEMM ops - FP8/FP4

void deep_gemm_fp8_fp4_gemm_nt(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const std::optional<Tensor>& c,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2, bool has_recipe,
    int64_t recipe_a_0, int64_t recipe_a_1, bool has_recipe_a,
    int64_t recipe_b_0, int64_t recipe_b_1, bool has_recipe_b,
    const std::string& compiled_dims, bool disable_ue8m0_cast) {
    auto a = std::make_pair(a_data, a_sf);
    auto b = std::make_pair(b_data, b_sf);
    auto recipe = make_recipe3(recipe_0, recipe_1, recipe_2, has_recipe);
    auto recipe_a = make_recipe2(recipe_a_0, recipe_a_1, has_recipe_a);
    auto recipe_b = make_recipe2(recipe_b_0, recipe_b_1, has_recipe_b);
    deep_gemm::gemm::fp8_fp4_gemm_nt(a, b, d, c, recipe, recipe_a, recipe_b,
                                       compiled_dims, disable_ue8m0_cast);
}

void deep_gemm_fp8_fp4_gemm_nn(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const std::optional<Tensor>& c,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2, bool has_recipe,
    int64_t recipe_a_0, int64_t recipe_a_1, bool has_recipe_a,
    int64_t recipe_b_0, int64_t recipe_b_1, bool has_recipe_b,
    const std::string& compiled_dims, bool disable_ue8m0_cast) {
    auto a = std::make_pair(a_data, a_sf);
    auto b = std::make_pair(b_data, b_sf);
    auto recipe = make_recipe3(recipe_0, recipe_1, recipe_2, has_recipe);
    auto recipe_a = make_recipe2(recipe_a_0, recipe_a_1, has_recipe_a);
    auto recipe_b = make_recipe2(recipe_b_0, recipe_b_1, has_recipe_b);
    deep_gemm::gemm::fp8_fp4_gemm_nn(a, b, d, c, recipe, recipe_a, recipe_b,
                                       compiled_dims, disable_ue8m0_cast);
}

void deep_gemm_fp8_fp4_gemm_tn(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const std::optional<Tensor>& c,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2, bool has_recipe,
    int64_t recipe_a_0, int64_t recipe_a_1, bool has_recipe_a,
    int64_t recipe_b_0, int64_t recipe_b_1, bool has_recipe_b,
    const std::string& compiled_dims, bool disable_ue8m0_cast) {
    auto a = std::make_pair(a_data, a_sf);
    auto b = std::make_pair(b_data, b_sf);
    auto recipe = make_recipe3(recipe_0, recipe_1, recipe_2, has_recipe);
    auto recipe_a = make_recipe2(recipe_a_0, recipe_a_1, has_recipe_a);
    auto recipe_b = make_recipe2(recipe_b_0, recipe_b_1, has_recipe_b);
    deep_gemm::gemm::fp8_fp4_gemm_tn(a, b, d, c, recipe, recipe_a, recipe_b,
                                       compiled_dims, disable_ue8m0_cast);
}

void deep_gemm_fp8_fp4_gemm_tt(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const std::optional<Tensor>& c,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2, bool has_recipe,
    int64_t recipe_a_0, int64_t recipe_a_1, bool has_recipe_a,
    int64_t recipe_b_0, int64_t recipe_b_1, bool has_recipe_b,
    const std::string& compiled_dims, bool disable_ue8m0_cast) {
    auto a = std::make_pair(a_data, a_sf);
    auto b = std::make_pair(b_data, b_sf);
    auto recipe = make_recipe3(recipe_0, recipe_1, recipe_2, has_recipe);
    auto recipe_a = make_recipe2(recipe_a_0, recipe_a_1, has_recipe_a);
    auto recipe_b = make_recipe2(recipe_b_0, recipe_b_1, has_recipe_b);
    deep_gemm::gemm::fp8_fp4_gemm_tt(a, b, d, c, recipe, recipe_a, recipe_b,
                                       compiled_dims, disable_ue8m0_cast);
}

// GEMM ops - M-grouped FP8/FP4

void deep_gemm_m_grouped_fp8_fp4_gemm_nt_contiguous(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const Tensor& grouped_layout,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2, bool has_recipe,
    int64_t recipe_a_0, int64_t recipe_a_1, bool has_recipe_a,
    int64_t recipe_b_0, int64_t recipe_b_1, bool has_recipe_b,
    const std::string& compiled_dims, bool disable_ue8m0_cast,
    bool use_psum_layout, int64_t expected_m_for_psum_layout,
    bool has_expected_m_for_psum_layout) {
    auto a = std::make_pair(a_data, a_sf);
    auto b = std::make_pair(b_data, b_sf);
    auto recipe = make_recipe3(recipe_0, recipe_1, recipe_2, has_recipe);
    auto recipe_a = make_recipe2(recipe_a_0, recipe_a_1, has_recipe_a);
    auto recipe_b = make_recipe2(recipe_b_0, recipe_b_1, has_recipe_b);
    auto em = has_expected_m_for_psum_layout ?
        std::make_optional(static_cast<int>(expected_m_for_psum_layout)) : std::nullopt;
    deep_gemm::gemm::m_grouped_fp8_fp4_gemm_nt_contiguous(
        a, b, d, grouped_layout, recipe, recipe_a, recipe_b,
        compiled_dims, disable_ue8m0_cast, use_psum_layout, em);
}

void deep_gemm_m_grouped_fp8_fp4_gemm_nn_contiguous(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const Tensor& grouped_layout,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2, bool has_recipe,
    int64_t recipe_a_0, int64_t recipe_a_1, bool has_recipe_a,
    int64_t recipe_b_0, int64_t recipe_b_1, bool has_recipe_b,
    const std::string& compiled_dims, bool disable_ue8m0_cast,
    bool use_psum_layout) {
    auto a = std::make_pair(a_data, a_sf);
    auto b = std::make_pair(b_data, b_sf);
    auto recipe = make_recipe3(recipe_0, recipe_1, recipe_2, has_recipe);
    auto recipe_a = make_recipe2(recipe_a_0, recipe_a_1, has_recipe_a);
    auto recipe_b = make_recipe2(recipe_b_0, recipe_b_1, has_recipe_b);
    deep_gemm::gemm::m_grouped_fp8_fp4_gemm_nn_contiguous(
        a, b, d, grouped_layout, recipe, recipe_a, recipe_b,
        compiled_dims, disable_ue8m0_cast, use_psum_layout);
}

void deep_gemm_m_grouped_fp8_fp4_gemm_nt_masked(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const Tensor& masked_m, int64_t expected_m,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2, bool has_recipe,
    int64_t recipe_a_0, int64_t recipe_a_1, bool has_recipe_a,
    int64_t recipe_b_0, int64_t recipe_b_1, bool has_recipe_b,
    const std::string& compiled_dims, bool disable_ue8m0_cast) {
    auto a = std::make_pair(a_data, a_sf);
    auto b = std::make_pair(b_data, b_sf);
    auto recipe = make_recipe3(recipe_0, recipe_1, recipe_2, has_recipe);
    auto recipe_a = make_recipe2(recipe_a_0, recipe_a_1, has_recipe_a);
    auto recipe_b = make_recipe2(recipe_b_0, recipe_b_1, has_recipe_b);
    deep_gemm::gemm::m_grouped_fp8_fp4_gemm_nt_masked(
        a, b, d, masked_m, static_cast<int>(expected_m),
        recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast);
}

// GEMM ops - K-grouped FP8

void deep_gemm_k_grouped_fp8_gemm_tn_contiguous(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const Tensor& ks_tensor,
    const std::optional<Tensor>& c,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2,
    const std::string& compiled_dims) {
    auto a = std::make_pair(a_data, a_sf);
    auto b = std::make_pair(b_data, b_sf);
    auto ks = tensor_to_vec_int(ks_tensor);
    auto recipe = std::make_tuple(static_cast<int>(recipe_0),
                                  static_cast<int>(recipe_1),
                                  static_cast<int>(recipe_2));
    deep_gemm::gemm::k_grouped_fp8_gemm_tn_contiguous(
        a, b, d, ks, ks_tensor, c, recipe, compiled_dims);
}

void deep_gemm_k_grouped_fp8_gemm_nt_contiguous(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const Tensor& ks_tensor,
    const std::optional<Tensor>& c,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2,
    const std::string& compiled_dims) {
    auto a = std::make_pair(a_data, a_sf);
    auto b = std::make_pair(b_data, b_sf);
    auto ks = tensor_to_vec_int(ks_tensor);
    auto recipe = std::make_tuple(static_cast<int>(recipe_0),
                                  static_cast<int>(recipe_1),
                                  static_cast<int>(recipe_2));
    deep_gemm::gemm::k_grouped_fp8_gemm_nt_contiguous(
        a, b, d, ks, ks_tensor, c, recipe, compiled_dims);
}

// GEMM ops - BF16

void deep_gemm_bf16_gemm_nt(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const std::optional<Tensor>& c, const std::string& compiled_dims) {
    deep_gemm::gemm::bf16_gemm_nt(a, b, d, c, compiled_dims);
}

void deep_gemm_bf16_gemm_nn(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const std::optional<Tensor>& c, const std::string& compiled_dims) {
    deep_gemm::gemm::bf16_gemm_nn(a, b, d, c, compiled_dims);
}

void deep_gemm_bf16_gemm_tn(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const std::optional<Tensor>& c, const std::string& compiled_dims) {
    deep_gemm::gemm::bf16_gemm_tn(a, b, d, c, compiled_dims);
}

void deep_gemm_bf16_gemm_tt(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const std::optional<Tensor>& c, const std::string& compiled_dims) {
    deep_gemm::gemm::bf16_gemm_tt(a, b, d, c, compiled_dims);
}

// GEMM ops - M-grouped BF16

void deep_gemm_m_grouped_bf16_gemm_nt_contiguous(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const Tensor& grouped_layout, const std::string& compiled_dims,
    bool use_psum_layout, int64_t expected_m_for_psum_layout,
    bool has_expected_m_for_psum_layout) {
    auto em = has_expected_m_for_psum_layout ?
        std::make_optional(static_cast<int>(expected_m_for_psum_layout)) : std::nullopt;
    deep_gemm::gemm::m_grouped_bf16_gemm_nt_contiguous(
        a, b, d, grouped_layout, compiled_dims, use_psum_layout, em);
}

void deep_gemm_m_grouped_bf16_gemm_nn_contiguous(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const Tensor& grouped_layout, const std::string& compiled_dims,
    bool use_psum_layout) {
    deep_gemm::gemm::m_grouped_bf16_gemm_nn_contiguous(
        a, b, d, grouped_layout, compiled_dims, use_psum_layout);
}

void deep_gemm_m_grouped_bf16_gemm_nt_masked(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const Tensor& masked_m, int64_t expected_m,
    const std::string& compiled_dims) {
    deep_gemm::gemm::m_grouped_bf16_gemm_nt_masked(
        a, b, d, masked_m, static_cast<int>(expected_m), compiled_dims);
}

// GEMM ops - K-grouped BF16

void deep_gemm_k_grouped_bf16_gemm_tn_contiguous(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const Tensor& ks_tensor, const std::optional<Tensor>& c,
    const std::string& compiled_dims) {
    auto ks = tensor_to_vec_int(ks_tensor);
    deep_gemm::gemm::k_grouped_bf16_gemm_tn_contiguous(
        a, b, d, ks, ks_tensor, c, compiled_dims);
}

// GEMM ops - cuBLASLt

void deep_gemm_cublaslt_gemm_nt(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const std::optional<Tensor>& c) {
    deep_gemm::gemm::cublaslt_gemm_nt(a, b, d, c);
}

void deep_gemm_cublaslt_gemm_nn(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const std::optional<Tensor>& c) {
    deep_gemm::gemm::cublaslt_gemm_nn(a, b, d, c);
}

void deep_gemm_cublaslt_gemm_tn(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const std::optional<Tensor>& c) {
    deep_gemm::gemm::cublaslt_gemm_tn(a, b, d, c);
}

void deep_gemm_cublaslt_gemm_tt(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const std::optional<Tensor>& c) {
    deep_gemm::gemm::cublaslt_gemm_tt(a, b, d, c);
}

// Attention ops

void deep_gemm_fp8_gemm_nt_skip_head_mid(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d,
    int64_t head_split_left, int64_t head_split_mid, int64_t head_split_right,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2, bool has_recipe,
    const std::string& compiled_dims, bool disable_ue8m0_cast) {
    auto a = std::make_pair(a_data, a_sf);
    auto b = std::make_pair(b_data, b_sf);
    auto head_splits = std::make_tuple(static_cast<int>(head_split_left),
                                       static_cast<int>(head_split_mid),
                                       static_cast<int>(head_split_right));
    auto recipe = make_recipe3(recipe_0, recipe_1, recipe_2, has_recipe);
    deep_gemm::attention::fp8_gemm_nt_skip_head_mid(
        a, b, d, head_splits, recipe, compiled_dims, disable_ue8m0_cast);
}

Tensor deep_gemm_fp8_mqa_logits(
    const Tensor& q,
    const Tensor& kv_data, const Tensor& kv_sf,
    const Tensor& weights,
    const Tensor& cu_seq_len_k_start, const Tensor& cu_seq_len_k_end,
    bool clean_logits, int64_t max_seqlen_k) {
    auto kv = std::make_pair(kv_data, kv_sf);
    return deep_gemm::attention::fp8_mqa_logits(
        q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end,
        clean_logits, static_cast<int>(max_seqlen_k));
}

Tensor deep_gemm_get_paged_mqa_logits_metadata(
    const Tensor& context_lens, int64_t block_kv, int64_t num_sms) {
    return deep_gemm::attention::get_paged_mqa_logits_metadata(
        context_lens, static_cast<int>(block_kv), static_cast<int>(num_sms));
}

Tensor deep_gemm_fp8_paged_mqa_logits(
    const Tensor& q, const Tensor& fused_kv_cache,
    const Tensor& weights, const Tensor& context_lens,
    const Tensor& block_table, const Tensor& schedule_meta,
    int64_t max_context_len, bool clean_logits) {
    return deep_gemm::attention::fp8_paged_mqa_logits(
        q, fused_kv_cache, weights, context_lens, block_table, schedule_meta,
        static_cast<int>(max_context_len), clean_logits);
}

// Einsum ops

void deep_gemm_einsum(
    const std::string& expr,
    const Tensor& a, const Tensor& b, const Tensor& d,
    const std::optional<Tensor>& c, bool use_cublaslt) {
    deep_gemm::einsum::einsum(expr, a, b, d, c, use_cublaslt);
}

void deep_gemm_fp8_einsum(
    const std::string& expr,
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const std::optional<Tensor>& c,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2) {
    auto a = std::make_pair(a_data, a_sf);
    auto b = std::make_pair(b_data, b_sf);
    auto recipe = std::make_tuple(static_cast<int>(recipe_0),
                                  static_cast<int>(recipe_1),
                                  static_cast<int>(recipe_2));
    deep_gemm::einsum::fp8_einsum(expr, a, b, d, c, recipe);
}

// Hyperconnection ops

void deep_gemm_tf32_hc_prenorm_gemm(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const Tensor& sqr_sum, int64_t num_splits, bool has_num_splits) {
    auto ns = has_num_splits ? std::make_optional(static_cast<int>(num_splits)) : std::nullopt;
    deep_gemm::hyperconnection::tf32_hc_prenorm_gemm(a, b, d, sqr_sum, ns);
}
