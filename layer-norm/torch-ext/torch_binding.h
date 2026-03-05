#pragma once

#include <torch/torch.h>

// Declarations for implementations defined in layer_norm/ln_api.cpp
std::vector<at::Tensor> dropout_add_ln_fwd(
    const at::Tensor &x0,
    c10::optional<const at::Tensor> &residual,
    const at::Tensor &gamma,
    c10::optional<const at::Tensor> &beta,
    c10::optional<const at::Tensor> &rowscale,
    c10::optional<const at::Tensor> &colscale,
    c10::optional<const at::Tensor> &x0_subset,
    c10::optional<const at::Tensor> &z_subset,
    const float dropout_p,
    const float epsilon,
    const float rowscale_const,
    const int64_t z_numrows,
    c10::optional<at::Generator> gen,
    bool residual_in_fp32,
    bool is_rms_norm);

std::vector<at::Tensor> dropout_add_ln_bwd(
    const at::Tensor &dz,
    c10::optional<const at::Tensor> &dx,
    const at::Tensor &x,
    c10::optional<const at::Tensor> &x0,
    c10::optional<const at::Tensor> &dmask,
    const at::Tensor &mu,
    const at::Tensor &rsigma,
    const at::Tensor &gamma,
    c10::optional<const at::Tensor> &rowscale,
    c10::optional<const at::Tensor> &colscale,
    c10::optional<const at::Tensor> &x0_subset,
    c10::optional<const at::Tensor> &z_subset,
    const float dropout_p,
    const float rowscale_const,
    const int64_t x0_numrows,
    const bool has_residual,
    bool is_rms_norm);

std::vector<at::Tensor> dropout_add_ln_parallel_residual_fwd(
    const at::Tensor &x0,
    c10::optional<const at::Tensor> &x1,
    c10::optional<const at::Tensor> &residual,
    const at::Tensor &gamma0,
    c10::optional<const at::Tensor> &beta0,
    c10::optional<const at::Tensor> &gamma1,
    c10::optional<const at::Tensor> &beta1,
    const float dropout_p,
    const float epsilon,
    c10::optional<at::Generator> gen,
    bool residual_in_fp32,
    bool is_rms_norm);

std::vector<at::Tensor> dropout_add_ln_parallel_residual_bwd(
    const at::Tensor &dz0,
    c10::optional<const at::Tensor> &dz1,
    c10::optional<const at::Tensor> &dx,
    const at::Tensor &x,
    c10::optional<const at::Tensor> &dmask0,
    c10::optional<const at::Tensor> &dmask1,
    const at::Tensor &mu,
    const at::Tensor &rsigma,
    const at::Tensor &gamma0,
    c10::optional<const at::Tensor> &gamma1,
    const float dropout_p,
    const bool has_x1,
    const bool has_residual,
    bool is_rms_norm);