#pragma once

#include <torch/torch.h>

#include <core/scalar_type.hpp>

bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability);

bool cutlass_scaled_mm_supports_block_fp8(int64_t cuda_device_capability);

void cutlass_scaled_mm(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       c10::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_azp(torch::Tensor& out, torch::Tensor const& a,
                           torch::Tensor const& b,
                           torch::Tensor const& a_scales,
                           torch::Tensor const& b_scales,
                           torch::Tensor const& azp_adj,
                           c10::optional<torch::Tensor> const& azp,
                           c10::optional<torch::Tensor> const& bias);

void static_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor const& scale,
                              c10::optional<torch::Tensor> const& azp);

void dynamic_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                               torch::Tensor& scales,
                               c10::optional<torch::Tensor> const& azp);

torch::Tensor gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_gptq_qzeros,
                        torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                        bool use_exllama, int64_t bit);

void gptq_shuffle(torch::Tensor q_weight, torch::Tensor q_perm, int64_t bit);

void static_scaled_fp8_quant(torch::Tensor& out, torch::Tensor const& input,
                             torch::Tensor const& scale);

void dynamic_scaled_fp8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor& scale);

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scale,
    c10::optional<torch::Tensor> const& scale_ub);

// GPTQ-Marlin

torch::Tensor awq_marlin_repack(torch::Tensor& b_q_weight, int64_t size_k,
                                int64_t size_n, int64_t num_bits);

torch::Tensor awq_marlin_repack_meta(torch::Tensor& b_q_weight,
                                     c10::SymInt size_k, c10::SymInt size_n,
                                     int64_t num_bits);

torch::Tensor gptq_marlin_gemm(
    torch::Tensor& a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor& b_q_weight, torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& global_scale_or_none,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none, torch::Tensor& workspace,
    vllm::ScalarTypeId const& b_q_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float);

torch::Tensor gptq_marlin_repack(torch::Tensor& b_q_weight, torch::Tensor& perm,
                                 int64_t size_k, int64_t size_n,
                                 int64_t num_bits);

torch::Tensor gptq_marlin_repack_meta(torch::Tensor& b_q_weight,
                                      torch::Tensor& perm, c10::SymInt size_k,
                                      c10::SymInt size_n, int64_t num_bits);


// Marlin

torch::Tensor marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                          torch::Tensor& b_scales, torch::Tensor& workspace,
                          int64_t size_m, int64_t size_n, int64_t size_k);

torch::Tensor gptq_marlin_24_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                                  torch::Tensor& b_meta,
                                  torch::Tensor& b_scales,
                                  torch::Tensor& workspace,
                                  vllm::ScalarTypeId const b_q_type_id,
                                  int64_t size_m, int64_t size_n,
                                  int64_t size_k);

torch::Tensor marlin_qqq_gemm(torch::Tensor const& a,
                              torch::Tensor const& b_q_weight,
                              torch::Tensor const& s_tok,
                              torch::Tensor const& s_ch,
                              torch::Tensor const& s_group,
                              torch::Tensor& workspace, int64_t size_m,
                              int64_t size_n, int64_t size_k);
