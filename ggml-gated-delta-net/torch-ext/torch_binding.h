#pragma once

#include <torch/torch.h>

#include <vector>

// One step (or a run of steps) of the gated delta rule, as one kernel.
//
// Shapes, torch order:
//   q, k, v  (n_seqs, n_tokens, n_heads, head_dim)  f32
//   g, beta  (n_seqs, n_tokens, n_heads)            f32
//   state    (n_seqs, n_heads, head_dim, head_dim)  f32, indexed [value_index][key_index]
//
// Returns one flat f32 tensor: the outputs, `n_seqs*n_tokens*n_heads*head_dim` of them, followed by
// the final state. The python wrapper takes the two views; returning them from here would make two
// returns of a custom op alias each other, which is not allowed.
//
// `q` and `k` must already carry one head per value head; the caller expands them, so its own head
// order is what applies. The output carries the kernel's own 1/sqrt(head_dim) scaling.
at::Tensor gated_delta_net(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                           const at::Tensor &g, const at::Tensor &beta, const at::Tensor &state);

// Whether this build has a gated-delta-rule kernel for `head_dim`. Asked rather than assumed: a
// second backend will cover a different set.
bool supports_gated_delta_net(int64_t head_dim);

// `x / max(|x|, eps)` along the last dimension, one dispatch where the expression is five. This is
// ggml's (and `F.normalize`'s) epsilon placement; `x * rsqrt(sum + eps)` differs by ~3e-8.
std::vector<at::Tensor> delta_gates(const at::Tensor &b, const at::Tensor &a,
                                   const at::Tensor &a_log, const at::Tensor &dt_bias);
at::Tensor rms_norm_gate(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &gate,
                         double eps);
at::Tensor l2_norm(const at::Tensor &x, double eps);

// `causal_conv1d_update` for a single token, in one dispatch instead of five. `state` is
// `(channels, state_width)` and is rolled forward in place; the result is `(channels,)`.
at::Tensor causal_conv_update(const at::Tensor &state, const at::Tensor &x, const at::Tensor &weight,
                              const std::optional<at::Tensor> &bias, bool silu);
