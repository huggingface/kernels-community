#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "tdt_logprobs_fwd(Tensor token_logits, Tensor duration_logits, "
      "Tensor targets, Tensor source_lengths, Tensor target_lengths, "
      "int blank_id, float sigma, Tensor! blank_lp, Tensor! label_lp, "
      "Tensor! dur_lp, Tensor! token_lse) -> ()");
  ops.impl("tdt_logprobs_fwd", torch::kCUDA, &tdt_logprobs_fwd);

  ops.def(
      "tdt_loss_fwd(Tensor blank_lp, Tensor label_lp, Tensor dur_lp, "
      "Tensor source_lengths, Tensor target_lengths, Tensor durations, "
      "Tensor! alphas, Tensor! log_ll) -> ()");
  ops.impl("tdt_loss_fwd", torch::kCUDA, &tdt_loss_fwd);

  ops.def(
      "tdt_loss_bwd(Tensor blank_lp, Tensor label_lp, Tensor dur_lp, "
      "Tensor source_lengths, Tensor target_lengths, Tensor durations, "
      "Tensor! betas) -> ()");
  ops.impl("tdt_loss_bwd", torch::kCUDA, &tdt_loss_bwd);

  ops.def(
      "tdt_logits_grad(Tensor token_logits, Tensor targets, "
      "Tensor source_lengths, Tensor target_lengths, Tensor durations, "
      "Tensor blank_lp, Tensor label_lp, Tensor dur_lp, Tensor token_lse, "
      "Tensor alphas, Tensor betas, Tensor log_ll, Tensor grad_loss, "
      "int blank_id, Tensor! grad_token_logits, "
      "Tensor! grad_duration_logits) -> ()");
  ops.impl("tdt_logits_grad", torch::kCUDA, &tdt_logits_grad);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
