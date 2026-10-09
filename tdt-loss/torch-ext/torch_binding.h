#pragma once

#include <torch/csrc/stable/tensor.h>

// Log-softmax gather over the vocabulary and duration heads. Writes the blank
// and label log-probabilities of every lattice node, the duration
// log-probabilities and the token log-normalizer (reused by the backward).
void tdt_logprobs_fwd(torch::stable::Tensor const &token_logits,
                      torch::stable::Tensor const &duration_logits,
                      torch::stable::Tensor const &targets,
                      torch::stable::Tensor const &logit_lengths,
                      torch::stable::Tensor const &target_lengths,
                      int64_t blank_id, double sigma,
                      torch::stable::Tensor &blank_lp,
                      torch::stable::Tensor &label_lp,
                      torch::stable::Tensor &dur_lp,
                      torch::stable::Tensor &token_lse);

// Forward recursion over the lattice (alphas) and per-sample log-likelihood.
void tdt_loss_fwd(torch::stable::Tensor const &blank_lp,
                  torch::stable::Tensor const &label_lp,
                  torch::stable::Tensor const &dur_lp,
                  torch::stable::Tensor const &logit_lengths,
                  torch::stable::Tensor const &target_lengths,
                  torch::stable::Tensor const &durations,
                  torch::stable::Tensor &alphas, torch::stable::Tensor &log_ll);

// Backward recursion over the lattice (betas).
void tdt_loss_bwd(torch::stable::Tensor const &blank_lp,
                  torch::stable::Tensor const &label_lp,
                  torch::stable::Tensor const &dur_lp,
                  torch::stable::Tensor const &logit_lengths,
                  torch::stable::Tensor const &target_lengths,
                  torch::stable::Tensor const &durations,
                  torch::stable::Tensor &betas);

// Gradient of the per-sample losses (scaled by grad_loss) w.r.t. the token
// and duration logits.
void tdt_logits_grad(torch::stable::Tensor const &token_logits,
                     torch::stable::Tensor const &targets,
                     torch::stable::Tensor const &logit_lengths,
                     torch::stable::Tensor const &target_lengths,
                     torch::stable::Tensor const &durations,
                     torch::stable::Tensor const &blank_lp,
                     torch::stable::Tensor const &label_lp,
                     torch::stable::Tensor const &dur_lp,
                     torch::stable::Tensor const &token_lse,
                     torch::stable::Tensor const &alphas,
                     torch::stable::Tensor const &betas,
                     torch::stable::Tensor const &log_ll,
                     torch::stable::Tensor const &grad_loss, int64_t blank_id,
                     torch::stable::Tensor &grad_token_logits,
                     torch::stable::Tensor &grad_duration_logits);
