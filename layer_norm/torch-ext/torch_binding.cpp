#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

// Helper to turn Tensor? from schema (optional by value) into optional<const Tensor>& args
template <typename T>
static c10::optional<const at::Tensor> as_const_opt(const c10::optional<T>& v) {
  if (v.has_value()) return c10::optional<const at::Tensor>(v.value());
  return c10::optional<const at::Tensor>();
}

// Wrappers with dispatcher-friendly types (double scalars, optional Generator)
// Forward
static std::vector<at::Tensor> dropout_add_ln_fwd_wrap(
    const at::Tensor& input,
    const at::Tensor& gamma,
    c10::optional<at::Tensor> beta,
    c10::optional<at::Tensor> rowscale,
    c10::optional<at::Tensor> colscale,
    c10::optional<at::Tensor> x0_subset,
    c10::optional<at::Tensor> z_subset,
    double dropout_p,
    double epsilon,
    double rowscale_const,
    int64_t z_numrows,
    c10::optional<at::Generator> gen,
    bool residual_in_fp32,
    bool is_rms_norm) {

  // residual is not exposed in this schema (None)
  auto residual_c = c10::optional<const at::Tensor>();
  auto beta_c = as_const_opt(beta);
  auto rowscale_c = as_const_opt(rowscale);
  auto colscale_c = as_const_opt(colscale);
  auto x0_subset_c = as_const_opt(x0_subset);
  auto z_subset_c = as_const_opt(z_subset);

  return dropout_add_ln_fwd(
      input, residual_c, gamma, beta_c, rowscale_c, colscale_c, x0_subset_c, z_subset_c,
      static_cast<float>(dropout_p),
      static_cast<float>(epsilon),
      static_cast<float>(rowscale_const),
      z_numrows, gen, residual_in_fp32, is_rms_norm);
}

// Backward
static std::vector<at::Tensor> dropout_add_ln_bwd_wrap(
    const at::Tensor& dz,
    c10::optional<at::Tensor> dx,
    const at::Tensor& x,
    c10::optional<at::Tensor> x0,
    c10::optional<at::Tensor> dmask,
    const at::Tensor& mu,
    const at::Tensor& rsigma,
    const at::Tensor& gamma,
    c10::optional<at::Tensor> rowscale,
    c10::optional<at::Tensor> colscale,
    c10::optional<at::Tensor> x0_subset,
    c10::optional<at::Tensor> z_subset,
    double dropout_p,
    double rowscale_const,
    int64_t x0_numrows,
    bool has_residual,
    bool is_rms_norm) {

  auto dx_c = as_const_opt(dx);
  auto x0_c = as_const_opt(x0);
  auto dmask_c = as_const_opt(dmask);
  auto rowscale_c = as_const_opt(rowscale);
  auto colscale_c = as_const_opt(colscale);
  auto x0_subset_c = as_const_opt(x0_subset);
  auto z_subset_c = as_const_opt(z_subset);

  return dropout_add_ln_bwd(
      dz, dx_c, x, x0_c, dmask_c, mu, rsigma, gamma,
      rowscale_c, colscale_c, x0_subset_c, z_subset_c,
      static_cast<float>(dropout_p),
      static_cast<float>(rowscale_const),
      x0_numrows, has_residual, is_rms_norm);
}

// Parallel forward
static std::vector<at::Tensor> dropout_add_ln_parallel_residual_fwd_wrap(
    const at::Tensor& input,
    c10::optional<at::Tensor> x1,
    c10::optional<at::Tensor> residual,
    const at::Tensor& gamma0,
    c10::optional<at::Tensor> beta0,
    c10::optional<at::Tensor> gamma1,
    c10::optional<at::Tensor> beta1,
    double dropout_p,
    double epsilon,
    c10::optional<at::Generator> gen,
    bool residual_in_fp32,
    bool is_rms_norm) {

  auto x1_c = as_const_opt(x1);
  auto residual_c = as_const_opt(residual);
  auto beta0_c = as_const_opt(beta0);
  auto gamma1_c = as_const_opt(gamma1);
  auto beta1_c = as_const_opt(beta1);

  return dropout_add_ln_parallel_residual_fwd(
      input, x1_c, residual_c, gamma0, beta0_c, gamma1_c, beta1_c,
      static_cast<float>(dropout_p),
      static_cast<float>(epsilon),
      gen, residual_in_fp32, is_rms_norm);
}

// Parallel backward
static std::vector<at::Tensor> dropout_add_ln_parallel_residual_bwd_wrap(
    const at::Tensor& dz0,
    c10::optional<at::Tensor> dz1,
    c10::optional<at::Tensor> dx,
    const at::Tensor& x,
    c10::optional<at::Tensor> dmask0,
    c10::optional<at::Tensor> dmask1,
    const at::Tensor& mu,
    const at::Tensor& rsigma,
    const at::Tensor& gamma0,
    c10::optional<at::Tensor> gamma1,
    double dropout_p,
    bool has_x1,
    bool has_residual,
    bool is_rms_norm) {

  auto dz1_c = as_const_opt(dz1);
  auto dx_c = as_const_opt(dx);
  auto dmask0_c = as_const_opt(dmask0);
  auto dmask1_c = as_const_opt(dmask1);
  auto gamma1_c = as_const_opt(gamma1);

  return dropout_add_ln_parallel_residual_bwd(
      dz0, dz1_c, dx_c, x, dmask0_c, dmask1_c, mu, rsigma, gamma0, gamma1_c,
      static_cast<float>(dropout_p), has_x1, has_residual, is_rms_norm);
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // Return lists to match std::vector<at::Tensor> from implementations
  ops.def("dropout_add_ln_fwd(Tensor input, Tensor gamma, Tensor? beta, Tensor? rowscale, Tensor? colscale, Tensor? x0_subset, Tensor? z_subset, float dropout_p, float epsilon, float rowscale_const, int z_numrows, Generator? gen, bool residual_in_fp32, bool is_rms_norm) -> Tensor[]");
  ops.impl("dropout_add_ln_fwd", torch::kCUDA, &dropout_add_ln_fwd_wrap);

  ops.def("dropout_add_ln_bwd(Tensor dz, Tensor? dx, Tensor x, Tensor? x0, Tensor? dmask, Tensor mu, Tensor rsigma, Tensor gamma, Tensor? rowscale, Tensor? colscale, Tensor? x0_subset, Tensor? z_subset, float dropout_p, float rowscale_const, int x0_numrows, bool has_residual, bool is_rms_norm) -> Tensor[]");
  ops.impl("dropout_add_ln_bwd", torch::kCUDA, &dropout_add_ln_bwd_wrap);

  ops.def("dropout_add_ln_parallel_residual_fwd(Tensor input, Tensor? x1, Tensor? residual, Tensor gamma0, Tensor? beta0, Tensor? gamma1, Tensor? beta1, float dropout_p, float epsilon, Generator? gen, bool residual_in_fp32, bool is_rms_norm) -> Tensor[]");
  ops.impl("dropout_add_ln_parallel_residual_fwd", torch::kCUDA, &dropout_add_ln_parallel_residual_fwd_wrap);

  ops.def("dropout_add_ln_parallel_residual_bwd(Tensor dz0, Tensor? dz1, Tensor? dx, Tensor x, Tensor? dmask0, Tensor? dmask1, Tensor mu, Tensor rsigma, Tensor gamma0, Tensor? gamma1, float dropout_p, bool has_x1, bool has_residual, bool is_rms_norm) -> Tensor[]");
  ops.impl("dropout_add_ln_parallel_residual_bwd", torch::kCUDA, &dropout_add_ln_parallel_residual_bwd_wrap);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)