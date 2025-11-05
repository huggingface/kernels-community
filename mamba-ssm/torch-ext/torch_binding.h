#pragma once

#include <torch/torch.h>

std::vector<at::Tensor>
selective_scan_fwd(const at::Tensor &u, const at::Tensor &delta,
                  const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                  const c10::optional<at::Tensor> &D_,
                  const c10::optional<at::Tensor> &z_,
                  const c10::optional<at::Tensor> &delta_bias_,
                  bool delta_softplus);

std::vector<at::Tensor>
selective_scan_bwd(const at::Tensor &u, const at::Tensor &delta,
                  const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                  const c10::optional<at::Tensor> &D_,
                  const c10::optional<at::Tensor> &z_,
                  const c10::optional<at::Tensor> &delta_bias_,
                  const at::Tensor &dout,
                  const c10::optional<at::Tensor> &x_,
                  const c10::optional<at::Tensor> &out_,
                  c10::optional<at::Tensor> dz_,
                  bool delta_softplus,
                  bool recompute_out_z);
