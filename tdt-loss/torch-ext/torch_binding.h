#pragma once

#include <torch/torch.h>

// Forward: compute log-probabilities for blank, label, and duration
void tdt_logprobs_fwd(torch::Tensor const &token_logits,
                      torch::Tensor const &duration_logits,
                      torch::Tensor const &targets,
                      torch::Tensor const &source_lengths,
                      torch::Tensor const &target_lengths, int64_t blank_id,
                      double sigma, torch::Tensor &blank_lp,
                      torch::Tensor &label_lp, torch::Tensor &dur_lp);

// Forward pass of TDT loss (alpha computation)
void tdt_loss_fwd(torch::Tensor const &blank_lp,
                  torch::Tensor const &label_lp,
                  torch::Tensor const &dur_lp,
                  torch::Tensor const &source_lengths,
                  torch::Tensor const &target_lengths,
                  torch::Tensor const &durations, torch::Tensor &alphas,
                  torch::Tensor &log_ll);

// Backward pass of TDT loss (beta computation)
void tdt_loss_bwd(torch::Tensor const &blank_lp,
                  torch::Tensor const &label_lp,
                  torch::Tensor const &dur_lp,
                  torch::Tensor const &source_lengths,
                  torch::Tensor const &target_lengths,
                  torch::Tensor const &durations, torch::Tensor &betas,
                  torch::Tensor &ll_bwd);

// Gradient computation
void tdt_loss_grad(torch::Tensor const &alphas, torch::Tensor const &betas,
                   torch::Tensor const &blank_lp,
                   torch::Tensor const &label_lp,
                   torch::Tensor const &dur_lp, torch::Tensor const &log_ll,
                   torch::Tensor const &source_lengths,
                   torch::Tensor const &target_lengths,
                   torch::Tensor const &durations,
                   torch::Tensor &grad_blank, torch::Tensor &grad_label,
                   torch::Tensor &grad_dur);

// Backward: propagate gradients through log-probabilities to logits
void tdt_logprobs_bwd(torch::Tensor const &token_logits,
                      torch::Tensor const &duration_logits,
                      torch::Tensor const &targets,
                      torch::Tensor const &source_lengths,
                      torch::Tensor const &target_lengths, int64_t blank_id,
                      torch::Tensor const &grad_blank,
                      torch::Tensor const &grad_label,
                      torch::Tensor const &grad_dur,
                      torch::Tensor &grad_token_logits,
                      torch::Tensor &grad_duration_logits);
