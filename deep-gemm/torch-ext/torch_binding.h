#pragma once

#include <torch/torch.h>
#include <optional>
#include <string>

using Tensor = at::Tensor;

// ============================================================================
// Runtime ops
// ============================================================================

void deep_gemm_init(const std::string& path, const std::string& cuda_home);

void deep_gemm_set_num_sms(int64_t num_sms);
int64_t deep_gemm_get_num_sms();

void deep_gemm_set_tc_util(int64_t tc_util);
int64_t deep_gemm_get_tc_util();

// ============================================================================
// Layout ops
// ============================================================================

int64_t deep_gemm_get_mk_alignment_for_contiguous_layout();

Tensor deep_gemm_get_tma_aligned_size(int64_t mn, int64_t element_size);

Tensor deep_gemm_get_mn_major_tma_aligned_tensor(const Tensor& sf);

Tensor deep_gemm_get_mn_major_tma_aligned_packed_ue8m0_tensor(const Tensor& sf);

Tensor deep_gemm_get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
    const Tensor& sf, const Tensor& ks_tensor, const Tensor& ks_int_tensor);

Tensor deep_gemm_transform_sf_into_required_layout(
    const Tensor& sf, int64_t mn, int64_t k,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2, bool has_recipe,
    int64_t recipe_ab_0, int64_t recipe_ab_1, bool has_recipe_ab,
    int64_t num_groups, bool has_num_groups,
    bool is_sfa, bool disable_ue8m0_cast);

// ============================================================================
// GEMM ops - FP8/FP4
// ============================================================================

void deep_gemm_fp8_fp4_gemm_nt(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const std::optional<Tensor>& c,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2, bool has_recipe,
    int64_t recipe_a_0, int64_t recipe_a_1, bool has_recipe_a,
    int64_t recipe_b_0, int64_t recipe_b_1, bool has_recipe_b,
    const std::string& compiled_dims, bool disable_ue8m0_cast);

void deep_gemm_fp8_fp4_gemm_nn(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const std::optional<Tensor>& c,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2, bool has_recipe,
    int64_t recipe_a_0, int64_t recipe_a_1, bool has_recipe_a,
    int64_t recipe_b_0, int64_t recipe_b_1, bool has_recipe_b,
    const std::string& compiled_dims, bool disable_ue8m0_cast);

void deep_gemm_fp8_fp4_gemm_tn(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const std::optional<Tensor>& c,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2, bool has_recipe,
    int64_t recipe_a_0, int64_t recipe_a_1, bool has_recipe_a,
    int64_t recipe_b_0, int64_t recipe_b_1, bool has_recipe_b,
    const std::string& compiled_dims, bool disable_ue8m0_cast);

void deep_gemm_fp8_fp4_gemm_tt(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const std::optional<Tensor>& c,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2, bool has_recipe,
    int64_t recipe_a_0, int64_t recipe_a_1, bool has_recipe_a,
    int64_t recipe_b_0, int64_t recipe_b_1, bool has_recipe_b,
    const std::string& compiled_dims, bool disable_ue8m0_cast);

// ============================================================================
// GEMM ops - M-grouped FP8/FP4
// ============================================================================

void deep_gemm_m_grouped_fp8_fp4_gemm_nt_contiguous(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const Tensor& grouped_layout,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2, bool has_recipe,
    int64_t recipe_a_0, int64_t recipe_a_1, bool has_recipe_a,
    int64_t recipe_b_0, int64_t recipe_b_1, bool has_recipe_b,
    const std::string& compiled_dims, bool disable_ue8m0_cast,
    bool use_psum_layout, int64_t expected_m_for_psum_layout,
    bool has_expected_m_for_psum_layout);

void deep_gemm_m_grouped_fp8_fp4_gemm_nn_contiguous(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const Tensor& grouped_layout,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2, bool has_recipe,
    int64_t recipe_a_0, int64_t recipe_a_1, bool has_recipe_a,
    int64_t recipe_b_0, int64_t recipe_b_1, bool has_recipe_b,
    const std::string& compiled_dims, bool disable_ue8m0_cast,
    bool use_psum_layout);

void deep_gemm_m_grouped_fp8_fp4_gemm_nt_masked(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const Tensor& masked_m, int64_t expected_m,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2, bool has_recipe,
    int64_t recipe_a_0, int64_t recipe_a_1, bool has_recipe_a,
    int64_t recipe_b_0, int64_t recipe_b_1, bool has_recipe_b,
    const std::string& compiled_dims, bool disable_ue8m0_cast);

// ============================================================================
// GEMM ops - K-grouped FP8
// ============================================================================

void deep_gemm_k_grouped_fp8_gemm_tn_contiguous(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const Tensor& ks_tensor,
    const std::optional<Tensor>& c,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2,
    const std::string& compiled_dims);

void deep_gemm_k_grouped_fp8_gemm_nt_contiguous(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const Tensor& ks_tensor,
    const std::optional<Tensor>& c,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2,
    const std::string& compiled_dims);

// ============================================================================
// GEMM ops - BF16
// ============================================================================

void deep_gemm_bf16_gemm_nt(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const std::optional<Tensor>& c, const std::string& compiled_dims);

void deep_gemm_bf16_gemm_nn(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const std::optional<Tensor>& c, const std::string& compiled_dims);

void deep_gemm_bf16_gemm_tn(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const std::optional<Tensor>& c, const std::string& compiled_dims);

void deep_gemm_bf16_gemm_tt(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const std::optional<Tensor>& c, const std::string& compiled_dims);

// ============================================================================
// GEMM ops - M-grouped BF16
// ============================================================================

void deep_gemm_m_grouped_bf16_gemm_nt_contiguous(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const Tensor& grouped_layout, const std::string& compiled_dims,
    bool use_psum_layout, int64_t expected_m_for_psum_layout,
    bool has_expected_m_for_psum_layout);

void deep_gemm_m_grouped_bf16_gemm_nn_contiguous(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const Tensor& grouped_layout, const std::string& compiled_dims,
    bool use_psum_layout);

void deep_gemm_m_grouped_bf16_gemm_nt_masked(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const Tensor& masked_m, int64_t expected_m,
    const std::string& compiled_dims);

// ============================================================================
// GEMM ops - K-grouped BF16
// ============================================================================

void deep_gemm_k_grouped_bf16_gemm_tn_contiguous(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const Tensor& ks_tensor, const std::optional<Tensor>& c,
    const std::string& compiled_dims);

// ============================================================================
// GEMM ops - cuBLASLt
// ============================================================================

void deep_gemm_cublaslt_gemm_nt(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const std::optional<Tensor>& c);

void deep_gemm_cublaslt_gemm_nn(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const std::optional<Tensor>& c);

void deep_gemm_cublaslt_gemm_tn(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const std::optional<Tensor>& c);

void deep_gemm_cublaslt_gemm_tt(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const std::optional<Tensor>& c);

// ============================================================================
// Attention ops
// ============================================================================

void deep_gemm_fp8_gemm_nt_skip_head_mid(
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d,
    int64_t head_split_left, int64_t head_split_mid, int64_t head_split_right,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2, bool has_recipe,
    const std::string& compiled_dims, bool disable_ue8m0_cast);

Tensor deep_gemm_fp8_mqa_logits(
    const Tensor& q,
    const Tensor& kv_data, const Tensor& kv_sf,
    const Tensor& weights,
    const Tensor& cu_seq_len_k_start, const Tensor& cu_seq_len_k_end,
    bool clean_logits, int64_t max_seqlen_k);

Tensor deep_gemm_get_paged_mqa_logits_metadata(
    const Tensor& context_lens, int64_t block_kv, int64_t num_sms);

Tensor deep_gemm_fp8_paged_mqa_logits(
    const Tensor& q, const Tensor& fused_kv_cache,
    const Tensor& weights, const Tensor& context_lens,
    const Tensor& block_table, const Tensor& schedule_meta,
    int64_t max_context_len, bool clean_logits);

// ============================================================================
// Einsum ops
// ============================================================================

void deep_gemm_einsum(
    const std::string& expr,
    const Tensor& a, const Tensor& b, const Tensor& d,
    const std::optional<Tensor>& c, bool use_cublaslt);

void deep_gemm_fp8_einsum(
    const std::string& expr,
    const Tensor& a_data, const Tensor& a_sf,
    const Tensor& b_data, const Tensor& b_sf,
    const Tensor& d, const std::optional<Tensor>& c,
    int64_t recipe_0, int64_t recipe_1, int64_t recipe_2);

// ============================================================================
// Hyperconnection ops
// ============================================================================

void deep_gemm_tf32_hc_prenorm_gemm(
    const Tensor& a, const Tensor& b, const Tensor& d,
    const Tensor& sqr_sum, int64_t num_splits, bool has_num_splits);
