#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "tdt_logprobs_fwd(Tensor token_logits, Tensor duration_logits, "
      "Tensor targets, Tensor source_lengths, Tensor target_lengths, "
      "int blank_id, float sigma, "
      "Tensor! blank_lp, Tensor! label_lp, Tensor! dur_lp) -> ()");
  ops.impl("tdt_logprobs_fwd", torch::kCUDA, &tdt_logprobs_fwd);

  ops.def(
      "tdt_loss_fwd(Tensor blank_lp, Tensor label_lp, Tensor dur_lp, "
      "Tensor source_lengths, Tensor target_lengths, Tensor durations, "
      "Tensor! alphas, Tensor! log_ll) -> ()");
  ops.impl("tdt_loss_fwd", torch::kCUDA, &tdt_loss_fwd);

  ops.def(
      "tdt_loss_bwd(Tensor blank_lp, Tensor label_lp, Tensor dur_lp, "
      "Tensor source_lengths, Tensor target_lengths, Tensor durations, "
      "Tensor! betas, Tensor! ll_bwd) -> ()");
  ops.impl("tdt_loss_bwd", torch::kCUDA, &tdt_loss_bwd);

  ops.def(
      "tdt_loss_grad(Tensor alphas, Tensor betas, Tensor blank_lp, "
      "Tensor label_lp, Tensor dur_lp, Tensor log_ll, "
      "Tensor source_lengths, Tensor target_lengths, Tensor durations, "
      "Tensor! grad_blank, Tensor! grad_label, Tensor! grad_dur) -> ()");
  ops.impl("tdt_loss_grad", torch::kCUDA, &tdt_loss_grad);

  ops.def(
      "tdt_logprobs_bwd(Tensor token_logits, Tensor duration_logits, "
      "Tensor targets, Tensor source_lengths, Tensor target_lengths, "
      "int blank_id, "
      "Tensor grad_blank, Tensor grad_label, Tensor grad_dur, "
      "Tensor! grad_token_logits, Tensor! grad_duration_logits) -> ()");
  ops.impl("tdt_logprobs_bwd", torch::kCUDA, &tdt_logprobs_bwd);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
