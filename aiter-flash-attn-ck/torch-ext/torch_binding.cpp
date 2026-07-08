#include <torch/library.h>

#include <optional>
#include <tuple>
#include <vector>

#include "registration.h"
#include "torch_binding.h"

// Forward declarations of the vendored aiter Torch interfaces (defined in the
// Composable-Kernel FMHA sources). Declared here rather than via
// "torch/mha_fwd.h" so this host translation unit does not depend on the
// kernel-section include directories.
namespace aiter {
namespace torch_itfs {
std::vector<at::Tensor>
mha_fwd(at::Tensor &q, const at::Tensor &k, const at::Tensor &v, float p_dropout,
        float softmax_scale, bool is_causal, int window_size_left,
        int window_size_right, int sink_size, bool return_softmax_lse,
        bool return_dropout_randval, std::optional<at::Tensor> cu_seqlens_q,
        std::optional<at::Tensor> cu_seqlens_kv, std::optional<at::Tensor> out,
        std::optional<const at::Tensor> bias,
        std::optional<const at::Tensor> alibi_slopes,
        std::optional<const at::Tensor> q_descale,
        std::optional<const at::Tensor> k_descale,
        std::optional<const at::Tensor> v_descale,
        std::optional<const at::Tensor> sink_ptr,
        std::optional<at::Generator> gen);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_fwd(at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
               const at::Tensor &cu_seqlens_q,
               std::optional<const at::Tensor> &cu_seqlens_k, int max_seqlen_q,
               int max_seqlen_k, int min_seqlen_q, float p_dropout,
               float softmax_scale, float logits_soft_cap, bool zero_tensors,
               bool is_causal, int window_size_left, int window_size_right,
               int sink_size, bool return_softmax_lse,
               bool return_dropout_randval, std::optional<at::Tensor> out,
               std::optional<const at::Tensor> block_table,
               std::optional<const at::Tensor> bias,
               std::optional<const at::Tensor> alibi_slopes,
               std::optional<const at::Tensor> q_descale,
               std::optional<const at::Tensor> k_descale,
               std::optional<const at::Tensor> v_descale,
               std::optional<at::Generator> gen,
               std::optional<const at::Tensor> cu_seqlens_q_padded,
               std::optional<const at::Tensor> cu_seqlens_k_padded,
               std::optional<const at::Tensor> sink_ptr);
} // namespace torch_itfs
} // namespace aiter

namespace {

// aiter's interfaces take `std::optional<const at::Tensor>` for read-only
// optional tensors, which is not a boxable Torch-library type. Convert from the
// boxable `std::optional<at::Tensor>` we accept at the op boundary.
inline std::optional<const at::Tensor>
to_const_opt(const std::optional<at::Tensor> &t) {
  if (t.has_value()) {
    return std::optional<const at::Tensor>(t.value());
  }
  return std::nullopt;
}

} // namespace

std::vector<at::Tensor>
mha_fwd(at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
        double dropout_p, double softmax_scale, bool is_causal,
        int64_t window_size_left, int64_t window_size_right, int64_t sink_size,
        bool return_softmax_lse, bool return_dropout_randval,
        std::optional<at::Tensor> cu_seqlens_q,
        std::optional<at::Tensor> cu_seqlens_kv, std::optional<at::Tensor> out,
        std::optional<at::Tensor> bias, std::optional<at::Tensor> alibi_slopes,
        std::optional<at::Tensor> q_descale, std::optional<at::Tensor> k_descale,
        std::optional<at::Tensor> v_descale, std::optional<at::Tensor> s_aux,
        std::optional<at::Generator> gen) {
  return aiter::torch_itfs::mha_fwd(
      q, k, v, static_cast<float>(dropout_p),
      static_cast<float>(softmax_scale), is_causal,
      static_cast<int>(window_size_left), static_cast<int>(window_size_right),
      static_cast<int>(sink_size), return_softmax_lse, return_dropout_randval,
      cu_seqlens_q, cu_seqlens_kv, out, to_const_opt(bias),
      to_const_opt(alibi_slopes), to_const_opt(q_descale),
      to_const_opt(k_descale), to_const_opt(v_descale), to_const_opt(s_aux),
      gen);
}

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
               const at::Tensor &cu_seqlens_q,
               std::optional<at::Tensor> cu_seqlens_k, int64_t max_seqlen_q,
               int64_t max_seqlen_k, int64_t min_seqlen_q, double dropout_p,
               double softmax_scale, double logits_soft_cap, bool zero_tensors,
               bool is_causal, int64_t window_size_left,
               int64_t window_size_right, int64_t sink_size,
               bool return_softmax_lse, bool return_dropout_randval,
               std::optional<at::Tensor> out,
               std::optional<at::Tensor> block_table,
               std::optional<at::Tensor> bias,
               std::optional<at::Tensor> alibi_slopes,
               std::optional<at::Tensor> q_descale,
               std::optional<at::Tensor> k_descale,
               std::optional<at::Tensor> v_descale,
               std::optional<at::Generator> gen,
               std::optional<at::Tensor> cu_seqlens_q_padded,
               std::optional<at::Tensor> cu_seqlens_k_padded,
               std::optional<at::Tensor> s_aux) {
  std::optional<const at::Tensor> cu_seqlens_k_const = to_const_opt(cu_seqlens_k);
  auto result = aiter::torch_itfs::mha_varlen_fwd(
      q, k, v, cu_seqlens_q, cu_seqlens_k_const, static_cast<int>(max_seqlen_q),
      static_cast<int>(max_seqlen_k), static_cast<int>(min_seqlen_q),
      static_cast<float>(dropout_p), static_cast<float>(softmax_scale),
      static_cast<float>(logits_soft_cap), zero_tensors, is_causal,
      static_cast<int>(window_size_left), static_cast<int>(window_size_right),
      static_cast<int>(sink_size), return_softmax_lse, return_dropout_randval,
      out, to_const_opt(block_table), to_const_opt(bias),
      to_const_opt(alibi_slopes), to_const_opt(q_descale),
      to_const_opt(k_descale), to_const_opt(v_descale), gen,
      to_const_opt(cu_seqlens_q_padded), to_const_opt(cu_seqlens_k_padded),
      to_const_opt(s_aux));
  return {std::get<0>(result), std::get<1>(result), std::get<2>(result),
          std::get<3>(result)};
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("mha_fwd("
          "Tensor! q, "
          "Tensor k, "
          "Tensor v, "
          "float dropout_p, "
          "float softmax_scale, "
          "bool is_causal, "
          "int window_size_left, "
          "int window_size_right, "
          "int sink_size, "
          "bool return_softmax_lse, "
          "bool return_dropout_randval, "
          "Tensor? cu_seqlens_q, "
          "Tensor? cu_seqlens_kv, "
          "Tensor(out!)? out, "
          "Tensor? bias, "
          "Tensor? alibi_slopes, "
          "Tensor? q_descale, "
          "Tensor? k_descale, "
          "Tensor? v_descale, "
          "Tensor? s_aux, "
          "Generator? gen) -> Tensor[]");
  ops.impl("mha_fwd", torch::kCUDA, &mha_fwd);

  ops.def("mha_varlen_fwd("
          "Tensor! q, "
          "Tensor k, "
          "Tensor v, "
          "Tensor cu_seqlens_q, "
          "Tensor? cu_seqlens_k, "
          "int max_seqlen_q, "
          "int max_seqlen_k, "
          "int min_seqlen_q, "
          "float dropout_p, "
          "float softmax_scale, "
          "float logits_soft_cap, "
          "bool zero_tensors, "
          "bool is_causal, "
          "int window_size_left, "
          "int window_size_right, "
          "int sink_size, "
          "bool return_softmax_lse, "
          "bool return_dropout_randval, "
          "Tensor(out!)? out, "
          "Tensor? block_table, "
          "Tensor? bias, "
          "Tensor? alibi_slopes, "
          "Tensor? q_descale, "
          "Tensor? k_descale, "
          "Tensor? v_descale, "
          "Generator? gen, "
          "Tensor? cu_seqlens_q_padded, "
          "Tensor? cu_seqlens_k_padded, "
          "Tensor? s_aux) -> Tensor[]");
  ops.impl("mha_varlen_fwd", torch::kCUDA, &mha_varlen_fwd);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
