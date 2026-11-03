#pragma once

#include <torch/torch.h>
#include <vector>

torch::Tensor givens_forward(
    torch::Tensor x,
    torch::Tensor angles,
    torch::Tensor alpha,
    torch::Tensor scales,
    torch::Tensor shifts,
    int64_t K);

std::vector<torch::Tensor> givens_backward(
    torch::Tensor grad_out,
    torch::Tensor x_in,
    torch::Tensor angles,
    torch::Tensor alpha,
    torch::Tensor scales,
    torch::Tensor shifts,
    int64_t K);

torch::Tensor givens_coupling_forward(
    torch::Tensor x,
    torch::Tensor angles,
    torch::Tensor coupling,
    torch::Tensor scales,
    torch::Tensor shifts,
    int64_t K,
    int64_t B);

std::vector<torch::Tensor> givens_coupling_backward(
    torch::Tensor grad_out,
    torch::Tensor x_in,
    torch::Tensor angles,
    torch::Tensor coupling,
    torch::Tensor scales,
    torch::Tensor shifts,
    int64_t K,
    int64_t B);

torch::Tensor givens_mor_forward(
    torch::Tensor x,
    torch::Tensor expert_angles,
    torch::Tensor expert_scales,
    torch::Tensor router_weights,
    torch::Tensor shifts,
    int64_t K,
    int64_t E);

std::vector<torch::Tensor> givens_mor_backward(
    torch::Tensor grad_out,
    torch::Tensor x_in,
    torch::Tensor expert_angles,
    torch::Tensor expert_scales,
    torch::Tensor router_weights,
    torch::Tensor shifts,
    int64_t K,
    int64_t E);

torch::Tensor clifford_coupling_forward(torch::Tensor x_pairs);

torch::Tensor clifford_coupling_backward(
    torch::Tensor x_pairs,
    torch::Tensor grad_features);

torch::Tensor fused_embedding_forward(
    torch::Tensor phase_angles,
    torch::Tensor log_magnitude,
    torch::Tensor token_ids);

std::vector<torch::Tensor> fused_embedding_backward(
    torch::Tensor phase_angles,
    torch::Tensor log_magnitude,
    torch::Tensor token_ids,
    torch::Tensor grad_output);

std::vector<torch::Tensor> fused_norm_forward(
    torch::Tensor x,
    torch::Tensor scale,
    double eps);

std::vector<torch::Tensor> fused_norm_backward(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor rms_inv,
    double eps);

torch::Tensor fused_gelu_forward(torch::Tensor x, double eps);

torch::Tensor fused_gelu_backward(
    torch::Tensor grad_out,
    torch::Tensor x,
    double eps);

std::vector<torch::Tensor> fused_mixer_forward(
    torch::Tensor x,
    torch::Tensor angles,
    torch::Tensor coupling,
    torch::Tensor raw_scales,
    double eps);

std::vector<torch::Tensor> fused_mixer_backward(
    torch::Tensor grad_out,
    torch::Tensor flat_x,
    torch::Tensor angles,
    torch::Tensor coupling,
    torch::Tensor raw_scales,
    torch::Tensor save_feat_norm,
    torch::Tensor save_inv_norm,
    torch::Tensor save_theta,
    double eps,
    std::vector<int64_t> out_shape);

std::vector<torch::Tensor> fused_zloss_forward(
    torch::Tensor logits,
    torch::Tensor mask);

torch::Tensor fused_zloss_backward(
    torch::Tensor grad_z_loss,
    torch::Tensor flat_logits,
    torch::Tensor lse,
    torch::Tensor flat_mask,
    torch::Tensor n_valid,
    std::vector<int64_t> out_shape);
