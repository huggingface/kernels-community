/* Metal implementation of the entry points.
 *
 * No ggml header is included here: all ggml and Metal contact sits behind the extern "C" boundary in
 * ggml_dispatch.mm, and every output comes from torch's allocator so the ops compose with the rest of
 * a model's memory.
 *
 * A torch MPS tensor's storage is a whole MTLBuffer that the tensor may be a view into, so each
 * buffer crosses the boundary as a (buffer, byte offset) pair.
 */

#include <torch/torch.h>

#include <optional>

#include "common.h"
#include "torch_binding.h"

namespace {

void *mtl_buffer(const at::Tensor &t) { return const_cast<void *>(t.storage().data()); }

size_t byte_offset(const at::Tensor &t) {
  return static_cast<size_t>(t.storage_offset()) * t.element_size();
}

at::Tensor as_f32(const at::Tensor &t) {
  return t.scalar_type() == at::kFloat ? t.contiguous() : t.to(at::kFloat).contiguous();
}

}  // namespace

bool supports_gated_delta_net(int64_t head_dim) {
  return ggml_gdn_metal_supports_gated_delta_net(head_dim) != 0;
}

std::vector<at::Tensor> delta_gates(const at::Tensor &b, const at::Tensor &a, const at::Tensor &a_log,
                                   const at::Tensor &dt_bias) {
  at::Tensor bc = as_f32(b), ac = as_f32(a), lc = as_f32(a_log), dc = as_f32(dt_bias);
  const int64_t n_heads = lc.numel();
  TORCH_CHECK(bc.numel() == n_heads && ac.numel() == n_heads && dc.numel() == n_heads,
              "delta_gates: every input must have one value per head");
  at::Tensor beta = at::empty_like(lc);
  at::Tensor g = at::empty_like(lc);
  const int status = ggml_gdn_metal_delta_gates(
      mtl_buffer(bc), byte_offset(bc), mtl_buffer(ac), byte_offset(ac), mtl_buffer(lc), byte_offset(lc),
      mtl_buffer(dc), byte_offset(dc), mtl_buffer(beta), byte_offset(beta), mtl_buffer(g), byte_offset(g),
      n_heads);
  TORCH_CHECK(status == 0, "delta_gates: no kernel for this build (", status, ")");
  return {beta, g};
}

at::Tensor rms_norm_gate(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &gate,
                         double eps) {
  at::Tensor xc = as_f32(x), wc = as_f32(weight), gc = as_f32(gate);
  const int64_t n_cols = xc.size(-1);
  TORCH_CHECK(wc.numel() == n_cols && gc.numel() == xc.numel(), "rms_norm_gate: shape mismatch");
  at::Tensor out = at::empty_like(xc);
  const int status = ggml_gdn_metal_rms_norm_gate(
      mtl_buffer(xc), byte_offset(xc), mtl_buffer(wc), byte_offset(wc), mtl_buffer(gc), byte_offset(gc),
      mtl_buffer(out), byte_offset(out), xc.numel() / n_cols, n_cols, (float)eps);
  TORCH_CHECK(status == 0, "rms_norm_gate: no kernel for this build (", status, ")");
  return out;
}

at::Tensor l2_norm(const at::Tensor &x, double eps) {
  TORCH_CHECK(x.is_mps() && x.dim() >= 1, "l2_norm: expected an mps tensor");
  const auto xc = as_f32(x);
  const int64_t n_cols = xc.size(-1), n_rows = xc.numel() / n_cols;
  auto out = at::empty_like(xc);
  const int rc = ggml_gdn_metal_l2_norm(mtl_buffer(xc), byte_offset(xc), mtl_buffer(out),
                                        byte_offset(out), n_rows, n_cols,
                                        static_cast<float>(eps));
  TORCH_CHECK(rc == 0, "ggml-gated-delta-net: l2_norm failed (rc ", rc, ")");
  return out;
}

at::Tensor gated_delta_net(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                           const at::Tensor &g, const at::Tensor &beta, const at::Tensor &state) {
  TORCH_CHECK(q.is_mps(), "gated_delta_net: expected mps tensors");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "gated_delta_net: q, k and v must be (n_seqs, n_tokens, n_heads, head_dim)");
  TORCH_CHECK(g.dim() == 3 && beta.dim() == 3,
              "gated_delta_net: g and beta must be (n_seqs, n_tokens, n_heads)");
  TORCH_CHECK(state.dim() == 4,
              "gated_delta_net: state must be (n_seqs, n_heads, head_dim, head_dim)");

  const int64_t n_seqs = v.size(0), n_tokens = v.size(1), n_heads = v.size(2),
                head_dim = v.size(3);
  TORCH_CHECK(q.size(2) == n_heads && k.size(2) == n_heads,
              "gated_delta_net: q and k must already carry one head per value head, got ",
              q.size(2), " against ", n_heads);
  TORCH_CHECK(q.size(3) == head_dim && k.size(3) == head_dim,
              "gated_delta_net: this kernel needs head_dim equal across q, k and v");
  TORCH_CHECK(state.size(0) == n_seqs && state.size(1) == n_heads &&
                  state.size(2) == head_dim && state.size(3) == head_dim,
              "gated_delta_net: state does not match (n_seqs, n_heads, head_dim, head_dim)");

  const auto qc = as_f32(q), kc = as_f32(k), vc = as_f32(v);
  const auto gc = as_f32(g), bc = as_f32(beta), sc = as_f32(state);

  // The kernel writes the outputs and then the final state through one destination pointer, so one
  // allocation is returned flat and the python wrapper takes the two views. Returning the views from
  // here instead would make them alias each other, which a custom op may not do -- torch warns and is
  // turning that into an error. Views taken outside the op are ordinary tensors.
  const int64_t n_out = n_seqs * n_tokens * n_heads * head_dim;
  const int64_t n_state = n_seqs * n_heads * head_dim * head_dim;
  auto dst = at::empty({n_out + n_state}, vc.options().dtype(at::kFloat));

  const int rc = ggml_gdn_metal_gated_delta_net(
      mtl_buffer(qc), byte_offset(qc), mtl_buffer(kc), byte_offset(kc), mtl_buffer(vc),
      byte_offset(vc), mtl_buffer(gc), byte_offset(gc), mtl_buffer(bc), byte_offset(bc),
      mtl_buffer(sc), byte_offset(sc), mtl_buffer(dst), byte_offset(dst), n_seqs, n_tokens, n_heads,
      head_dim);
  TORCH_CHECK(rc == 0, "ggml-gated-delta-net: no gated-delta-rule kernel for head_dim ", head_dim);

  return dst;
}

at::Tensor causal_conv_update(const at::Tensor &state, const at::Tensor &x, const at::Tensor &weight,
                              const std::optional<at::Tensor> &bias, bool silu) {
  TORCH_CHECK(state.is_mps() && x.is_mps(), "causal_conv_update expects mps tensors");
  TORCH_CHECK(state.is_contiguous(), "the conv state is updated in place and must be contiguous");
  TORCH_CHECK(state.scalar_type() == at::kFloat, "causal_conv_update is f32 only");
  const at::Tensor xc = as_f32(x).view({-1}), wc = as_f32(weight);
  const int64_t channels = state.size(-2), swidth = state.size(-1), k = wc.size(-1);
  TORCH_CHECK(xc.numel() == channels, "x must carry one value per channel (a single token)");
  TORCH_CHECK(wc.size(0) == channels && swidth + 1 >= k, "weight does not match the state");
  at::Tensor out = at::empty({channels}, state.options());
  at::Tensor bc = bias.has_value() ? as_f32(*bias) : at::Tensor();
  const int status = ggml_gdn_metal_causal_conv_update(
      mtl_buffer(state), byte_offset(state), mtl_buffer(xc), byte_offset(xc), mtl_buffer(wc),
      byte_offset(wc), bias.has_value() ? mtl_buffer(bc) : nullptr,
      bias.has_value() ? byte_offset(bc) : 0, mtl_buffer(out), byte_offset(out), channels, swidth, k,
      bias.has_value() ? 1 : 0, silu ? 1 : 0);
  TORCH_CHECK(status == 0, "causal_conv_update: no kernel for this build (", status, ")");
  return out;
}
