#pragma once

#include <torch/torch.h>

// Log-softmax gather over the vocabulary and duration heads. Writes the blank
// and label log-probabilities of every lattice node, the duration
// log-probabilities and the token log-normalizer (reused by the backward).
void tdt_logprobs_fwd(torch::Tensor const &token_logits,
                      torch::Tensor const &duration_logits,
                      torch::Tensor const &targets,
                      torch::Tensor const &logit_lengths,
                      torch::Tensor const &target_lengths, int64_t blank_id,
                      double sigma, torch::Tensor &blank_lp,
                      torch::Tensor &label_lp, torch::Tensor &dur_lp,
                      torch::Tensor &token_lse);

// Forward recursion over the lattice (alphas) and per-sample log-likelihood.
void tdt_loss_fwd(torch::Tensor const &blank_lp, torch::Tensor const &label_lp,
                  torch::Tensor const &dur_lp,
                  torch::Tensor const &logit_lengths,
                  torch::Tensor const &target_lengths,
                  torch::Tensor const &durations, torch::Tensor &alphas,
                  torch::Tensor &log_ll);

// Backward recursion over the lattice (betas).
void tdt_loss_bwd(torch::Tensor const &blank_lp, torch::Tensor const &label_lp,
                  torch::Tensor const &dur_lp,
                  torch::Tensor const &logit_lengths,
                  torch::Tensor const &target_lengths,
                  torch::Tensor const &durations, torch::Tensor &betas);

// Gradient of the per-sample losses (scaled by grad_loss) w.r.t. the token
// and duration logits.
void tdt_logits_grad(
    torch::Tensor const &token_logits, torch::Tensor const &targets,
    torch::Tensor const &logit_lengths, torch::Tensor const &target_lengths,
    torch::Tensor const &durations, torch::Tensor const &blank_lp,
    torch::Tensor const &label_lp, torch::Tensor const &dur_lp,
    torch::Tensor const &token_lse, torch::Tensor const &alphas,
    torch::Tensor const &betas, torch::Tensor const &log_ll,
    torch::Tensor const &grad_loss, int64_t blank_id,
    torch::Tensor &grad_token_logits, torch::Tensor &grad_duration_logits);
