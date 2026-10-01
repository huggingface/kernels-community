/***************************************************************************************************
 * Copyright (c) 2022 - 2026 Ali Hassani.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 **************************************************************************************************/
// kernel-builder port: replaces upstream csrc/natten.cpp (pybind11) with
// TORCH_LIBRARY registrations against the Python limited API. The op schemas
// are out-variant and use `int[]` for NATTEN's dimension tuples (including
// 0/1 encoded causal masks); the adapters below convert to the
// std::tuple<int32_t, ...> signatures of the NATTEN C++ entry points.

#include <torch/library.h>
#include <torch/torch.h>

#include <optional>
#include <tuple>
#include <vector>

#include "natten/blackwell_fmha.h"
#include "natten/blackwell_fna.h"
#include "natten/compute_delta.h"
#include "natten/fmha.h"
#include "natten/fna.h"
#include "natten/hopper_fmha.h"
#include "natten/hopper_fna.h"
#include "natten/reference.h"
#include "natten/token_permute.h"

#include "registration.h"

namespace {

using IntVec = std::vector<int64_t>;
using OptTensor = std::optional<at::Tensor>;

template <size_t N>
auto to_int32_tuple(const IntVec &v) {
  TORCH_CHECK(
      v.size() == N,
      "NATTEN: expected an int list of length ",
      N,
      ", got ",
      v.size());
  if constexpr (N == 1) {
    return std::make_tuple(static_cast<int32_t>(v[0]));
  } else if constexpr (N == 2) {
    return std::make_tuple(
        static_cast<int32_t>(v[0]), static_cast<int32_t>(v[1]));
  } else {
    static_assert(N == 3);
    return std::make_tuple(
        static_cast<int32_t>(v[0]),
        static_cast<int32_t>(v[1]),
        static_cast<int32_t>(v[2]));
  }
}

template <size_t N>
auto to_bool_tuple(const IntVec &v) {
  TORCH_CHECK(
      v.size() == N,
      "NATTEN: expected a causal mask list of length ",
      N,
      ", got ",
      v.size());
  if constexpr (N == 1) {
    return std::make_tuple(v[0] != 0);
  } else if constexpr (N == 2) {
    return std::make_tuple(v[0] != 0, v[1] != 0);
  } else {
    static_assert(N == 3);
    return std::make_tuple(v[0] != 0, v[1] != 0, v[2] != 0);
  }
}

// Legacy (CUTLASS 2.X) fused neighborhood attention.

#define DEFINE_FNA_ADAPTERS(DIM)                                            \
  void na##DIM##d_forward(                                                  \
      at::Tensor &out,                                                      \
      const at::Tensor &query,                                              \
      const at::Tensor &key,                                                \
      const at::Tensor &value,                                              \
      at::Tensor &logsumexp,                                                \
      IntVec kernel_size,                                                   \
      IntVec stride,                                                        \
      IntVec dilation,                                                      \
      IntVec is_causal,                                                     \
      double scale,                                                         \
      IntVec q_tile_shape,                                                  \
      IntVec kv_tile_shape) {                                               \
    natten::na##DIM##d_forward(                                             \
        out,                                                                \
        query,                                                              \
        key,                                                                \
        value,                                                              \
        at::optional<at::Tensor>(logsumexp),                                \
        to_int32_tuple<DIM>(kernel_size),                                   \
        to_int32_tuple<DIM>(stride),                                        \
        to_int32_tuple<DIM>(dilation),                                      \
        to_bool_tuple<DIM>(is_causal),                                      \
        static_cast<float>(scale),                                          \
        to_int32_tuple<DIM>(q_tile_shape),                                  \
        to_int32_tuple<DIM>(kv_tile_shape));                                \
  }                                                                         \
                                                                            \
  void na##DIM##d_backward(                                                 \
      at::Tensor &grad_query,                                               \
      at::Tensor &grad_key,                                                 \
      at::Tensor &grad_value,                                               \
      const at::Tensor &query,                                              \
      const at::Tensor &key,                                                \
      const at::Tensor &value,                                              \
      const at::Tensor &out,                                                \
      const at::Tensor &grad_out,                                           \
      const at::Tensor &logsumexp,                                          \
      IntVec kernel_size,                                                   \
      IntVec stride,                                                        \
      IntVec dilation,                                                      \
      IntVec is_causal,                                                     \
      double scale,                                                         \
      IntVec q_tile_shape,                                                  \
      IntVec kv_tile_shape,                                                 \
      IntVec num_kv_splits,                                                 \
      bool compute_delta_with_torch) {                                      \
    natten::na##DIM##d_backward(                                            \
        grad_query,                                                         \
        grad_key,                                                           \
        grad_value,                                                         \
        query,                                                              \
        key,                                                                \
        value,                                                              \
        out,                                                                \
        grad_out,                                                           \
        logsumexp,                                                          \
        to_int32_tuple<DIM>(kernel_size),                                   \
        to_int32_tuple<DIM>(stride),                                        \
        to_int32_tuple<DIM>(dilation),                                      \
        to_bool_tuple<DIM>(is_causal),                                      \
        static_cast<float>(scale),                                          \
        to_int32_tuple<DIM>(q_tile_shape),                                  \
        to_int32_tuple<DIM>(kv_tile_shape),                                 \
        to_int32_tuple<DIM>(num_kv_splits),                                 \
        compute_delta_with_torch);                                          \
  }

DEFINE_FNA_ADAPTERS(1)
DEFINE_FNA_ADAPTERS(2)
DEFINE_FNA_ADAPTERS(3)

#undef DEFINE_FNA_ADAPTERS

// Hopper (SM90) fused neighborhood attention.

#define DEFINE_HOPPER_FNA_ADAPTERS(DIM)                                     \
  void hopper_na##DIM##d_forward(                                           \
      at::Tensor &out,                                                      \
      const at::Tensor &query,                                              \
      const at::Tensor &key,                                                \
      const at::Tensor &value,                                              \
      at::Tensor &logsumexp,                                                \
      IntVec kernel_size,                                                   \
      IntVec stride,                                                        \
      IntVec dilation,                                                      \
      IntVec is_causal,                                                     \
      double scale,                                                         \
      IntVec q_shape,                                                       \
      IntVec kv_shape,                                                      \
      IntVec qkv_shape,                                                     \
      IntVec q_tile_shape,                                                  \
      IntVec kv_tile_shape,                                                 \
      int64_t kernel_type) {                                                \
    natten::hopper_na##DIM##d_forward(                                      \
        out,                                                                \
        query,                                                              \
        key,                                                                \
        value,                                                              \
        at::optional<at::Tensor>(logsumexp),                                \
        to_int32_tuple<DIM>(kernel_size),                                   \
        to_int32_tuple<DIM>(stride),                                        \
        to_int32_tuple<DIM>(dilation),                                      \
        to_bool_tuple<DIM>(is_causal),                                      \
        static_cast<float>(scale),                                          \
        to_int32_tuple<DIM>(q_shape),                                       \
        to_int32_tuple<DIM>(kv_shape),                                      \
        to_int32_tuple<DIM>(qkv_shape),                                     \
        to_int32_tuple<DIM>(q_tile_shape),                                  \
        to_int32_tuple<DIM>(kv_tile_shape),                                 \
        static_cast<int>(kernel_type));                                     \
  }                                                                         \
                                                                            \
  void hopper_na##DIM##d_backward(                                          \
      at::Tensor &grad_query,                                               \
      at::Tensor &grad_key,                                                 \
      at::Tensor &grad_value,                                               \
      const at::Tensor &query,                                              \
      const at::Tensor &key,                                                \
      const at::Tensor &value,                                              \
      const at::Tensor &out,                                                \
      const at::Tensor &grad_out,                                           \
      const at::Tensor &logsumexp,                                          \
      IntVec kernel_size,                                                   \
      IntVec stride,                                                        \
      IntVec dilation,                                                      \
      IntVec is_causal,                                                     \
      double scale,                                                         \
      IntVec q_shape,                                                       \
      IntVec kv_shape,                                                      \
      IntVec qkv_shape,                                                     \
      IntVec q_tile_shape,                                                  \
      IntVec kv_tile_shape) {                                               \
    natten::hopper_na##DIM##d_backward(                                     \
        grad_query,                                                         \
        grad_key,                                                           \
        grad_value,                                                         \
        query,                                                              \
        key,                                                                \
        value,                                                              \
        out,                                                                \
        grad_out,                                                           \
        logsumexp,                                                          \
        to_int32_tuple<DIM>(kernel_size),                                   \
        to_int32_tuple<DIM>(stride),                                        \
        to_int32_tuple<DIM>(dilation),                                      \
        to_bool_tuple<DIM>(is_causal),                                      \
        static_cast<float>(scale),                                          \
        to_int32_tuple<DIM>(q_shape),                                       \
        to_int32_tuple<DIM>(kv_shape),                                      \
        to_int32_tuple<DIM>(qkv_shape),                                     \
        to_int32_tuple<DIM>(q_tile_shape),                                  \
        to_int32_tuple<DIM>(kv_tile_shape));                                \
  }

DEFINE_HOPPER_FNA_ADAPTERS(1)
DEFINE_HOPPER_FNA_ADAPTERS(2)
DEFINE_HOPPER_FNA_ADAPTERS(3)

#undef DEFINE_HOPPER_FNA_ADAPTERS

// Blackwell (SM100) fused neighborhood attention.

#define DEFINE_BLACKWELL_FNA_ADAPTERS(DIM)                                  \
  void blackwell_na##DIM##d_forward(                                        \
      at::Tensor &out,                                                      \
      const at::Tensor &query,                                              \
      const at::Tensor &key,                                                \
      const at::Tensor &value,                                              \
      at::Tensor &logsumexp,                                                \
      IntVec kernel_size,                                                   \
      IntVec stride,                                                        \
      IntVec dilation,                                                      \
      IntVec is_causal,                                                     \
      double scale,                                                         \
      IntVec q_shape,                                                       \
      IntVec kv_shape,                                                      \
      IntVec qkv_shape,                                                     \
      IntVec q_tile_shape,                                                  \
      IntVec kv_tile_shape,                                                 \
      bool run_persistent) {                                                \
    natten::blackwell_na##DIM##d_forward(                                   \
        out,                                                                \
        query,                                                              \
        key,                                                                \
        value,                                                              \
        at::optional<at::Tensor>(logsumexp),                                \
        to_int32_tuple<DIM>(kernel_size),                                   \
        to_int32_tuple<DIM>(stride),                                        \
        to_int32_tuple<DIM>(dilation),                                      \
        to_bool_tuple<DIM>(is_causal),                                      \
        static_cast<float>(scale),                                          \
        to_int32_tuple<DIM>(q_shape),                                       \
        to_int32_tuple<DIM>(kv_shape),                                      \
        to_int32_tuple<DIM>(qkv_shape),                                     \
        to_int32_tuple<DIM>(q_tile_shape),                                  \
        to_int32_tuple<DIM>(kv_tile_shape),                                 \
        run_persistent);                                                    \
  }                                                                         \
                                                                            \
  void blackwell_na##DIM##d_backward(                                       \
      at::Tensor &grad_query,                                               \
      at::Tensor &grad_key,                                                 \
      at::Tensor &grad_value,                                               \
      const at::Tensor &query,                                              \
      const at::Tensor &key,                                                \
      const at::Tensor &value,                                              \
      const at::Tensor &out,                                                \
      const at::Tensor &grad_out,                                           \
      const at::Tensor &logsumexp,                                          \
      IntVec kernel_size,                                                   \
      IntVec stride,                                                        \
      IntVec dilation,                                                      \
      IntVec is_causal,                                                     \
      double scale,                                                         \
      IntVec q_shape,                                                       \
      IntVec kv_shape,                                                      \
      IntVec qkv_shape,                                                     \
      IntVec q_tile_shape,                                                  \
      IntVec kv_tile_shape) {                                               \
    natten::blackwell_na##DIM##d_backward(                                  \
        grad_query,                                                         \
        grad_key,                                                           \
        grad_value,                                                         \
        query,                                                              \
        key,                                                                \
        value,                                                              \
        out,                                                                \
        grad_out,                                                           \
        logsumexp,                                                          \
        to_int32_tuple<DIM>(kernel_size),                                   \
        to_int32_tuple<DIM>(stride),                                        \
        to_int32_tuple<DIM>(dilation),                                      \
        to_bool_tuple<DIM>(is_causal),                                      \
        static_cast<float>(scale),                                          \
        to_int32_tuple<DIM>(q_shape),                                       \
        to_int32_tuple<DIM>(kv_shape),                                      \
        to_int32_tuple<DIM>(qkv_shape),                                     \
        to_int32_tuple<DIM>(q_tile_shape),                                  \
        to_int32_tuple<DIM>(kv_tile_shape));                                \
  }

DEFINE_BLACKWELL_FNA_ADAPTERS(1)
DEFINE_BLACKWELL_FNA_ADAPTERS(2)
DEFINE_BLACKWELL_FNA_ADAPTERS(3)

#undef DEFINE_BLACKWELL_FNA_ADAPTERS

// Reference kernels.

#define DEFINE_REFERENCE_FNA_ADAPTERS(DIM)                                  \
  void reference_na##DIM##d_forward(                                        \
      at::Tensor &out,                                                      \
      const at::Tensor &query,                                              \
      const at::Tensor &key,                                                \
      const at::Tensor &value,                                              \
      at::Tensor &logsumexp,                                                \
      IntVec kernel_size,                                                   \
      IntVec stride,                                                        \
      IntVec dilation,                                                      \
      IntVec is_causal,                                                     \
      double scale,                                                         \
      IntVec qkv_shape,                                                     \
      int64_t num_extra_kv) {                                               \
    natten::reference_na##DIM##d_forward(                                   \
        out,                                                                \
        query,                                                              \
        key,                                                                \
        value,                                                              \
        logsumexp,                                                          \
        to_int32_tuple<DIM>(kernel_size),                                   \
        to_int32_tuple<DIM>(stride),                                        \
        to_int32_tuple<DIM>(dilation),                                      \
        to_bool_tuple<DIM>(is_causal),                                      \
        static_cast<float>(scale),                                          \
        to_int32_tuple<DIM>(qkv_shape),                                     \
        static_cast<int>(num_extra_kv));                                    \
  }                                                                         \
                                                                            \
  void reference_na##DIM##d_backward(                                       \
      at::Tensor &grad_query,                                               \
      at::Tensor &grad_key,                                                 \
      at::Tensor &grad_value,                                               \
      const at::Tensor &query,                                              \
      const at::Tensor &key,                                                \
      const at::Tensor &value,                                              \
      const at::Tensor &out,                                                \
      const at::Tensor &grad_out,                                           \
      const at::Tensor &logsumexp,                                          \
      IntVec kernel_size,                                                   \
      IntVec stride,                                                        \
      IntVec dilation,                                                      \
      IntVec is_causal,                                                     \
      double scale,                                                         \
      IntVec qkv_shape,                                                     \
      int64_t num_extra_kv) {                                               \
    natten::reference_na##DIM##d_backward(                                  \
        grad_query,                                                         \
        grad_key,                                                           \
        grad_value,                                                         \
        query,                                                              \
        key,                                                                \
        value,                                                              \
        out,                                                                \
        grad_out,                                                           \
        logsumexp,                                                          \
        to_int32_tuple<DIM>(kernel_size),                                   \
        to_int32_tuple<DIM>(stride),                                        \
        to_int32_tuple<DIM>(dilation),                                      \
        to_bool_tuple<DIM>(is_causal),                                      \
        static_cast<float>(scale),                                          \
        to_int32_tuple<DIM>(qkv_shape),                                     \
        static_cast<int>(num_extra_kv));                                    \
  }

DEFINE_REFERENCE_FNA_ADAPTERS(1)
DEFINE_REFERENCE_FNA_ADAPTERS(2)
DEFINE_REFERENCE_FNA_ADAPTERS(3)

#undef DEFINE_REFERENCE_FNA_ADAPTERS

// Token permute kernels.

#define DEFINE_TOKEN_PERMUTE_ADAPTERS(DIM)                                  \
  void token_permute_##DIM##d(                                              \
      at::Tensor &out,                                                      \
      const at::Tensor &in,                                                 \
      IntVec tile_shape,                                                    \
      IntVec dilation,                                                      \
      bool flip_tiled_dims) {                                               \
    natten::token_permute_##DIM##d(                                         \
        out,                                                                \
        in,                                                                 \
        to_int32_tuple<DIM>(tile_shape),                                    \
        to_int32_tuple<DIM>(dilation),                                      \
        flip_tiled_dims);                                                   \
  }                                                                         \
                                                                            \
  void token_unpermute_##DIM##d(                                            \
      at::Tensor &out,                                                      \
      const at::Tensor &in,                                                 \
      IntVec tile_shape,                                                    \
      IntVec dilation,                                                      \
      bool flip_tiled_dims) {                                               \
    natten::token_unpermute_##DIM##d(                                       \
        out,                                                                \
        in,                                                                 \
        to_int32_tuple<DIM>(tile_shape),                                    \
        to_int32_tuple<DIM>(dilation),                                      \
        flip_tiled_dims);                                                   \
  }

DEFINE_TOKEN_PERMUTE_ADAPTERS(1)
DEFINE_TOKEN_PERMUTE_ADAPTERS(2)
DEFINE_TOKEN_PERMUTE_ADAPTERS(3)

#undef DEFINE_TOKEN_PERMUTE_ADAPTERS

// FMHA (CUTLASS 2.X).

void fmha_forward(
    at::Tensor &out,
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    at::Tensor &logsumexp,
    bool is_causal,
    double scale,
    int64_t q_tile_size,
    int64_t kv_tile_size,
    const OptTensor &cumulative_seqlen_Q,
    const OptTensor &cumulative_seqlen_KV,
    int64_t max_seqlen_Q,
    int64_t max_seqlen_KV) {
  natten::fmha_forward(
      out,
      query,
      key,
      value,
      at::optional<at::Tensor>(logsumexp),
      is_causal,
      static_cast<float>(scale),
      static_cast<int32_t>(q_tile_size),
      static_cast<int32_t>(kv_tile_size),
      cumulative_seqlen_Q,
      cumulative_seqlen_KV,
      static_cast<int>(max_seqlen_Q),
      static_cast<int>(max_seqlen_KV));
}

void fmha_backward(
    at::Tensor &grad_query,
    at::Tensor &grad_key,
    at::Tensor &grad_value,
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &out,
    const at::Tensor &grad_out,
    const at::Tensor &logsumexp,
    bool is_causal,
    double scale,
    int64_t q_tile_size,
    int64_t kv_tile_size,
    int64_t num_kv_splits,
    bool compute_delta_with_torch,
    const OptTensor &cumulative_seqlen_Q,
    const OptTensor &cumulative_seqlen_KV,
    int64_t max_seqlen_Q,
    int64_t max_seqlen_KV) {
  natten::fmha_backward(
      grad_query,
      grad_key,
      grad_value,
      query,
      key,
      value,
      out,
      grad_out,
      logsumexp,
      is_causal,
      static_cast<float>(scale),
      static_cast<int32_t>(q_tile_size),
      static_cast<int32_t>(kv_tile_size),
      static_cast<int32_t>(num_kv_splits),
      compute_delta_with_torch,
      cumulative_seqlen_Q,
      cumulative_seqlen_KV,
      static_cast<int>(max_seqlen_Q),
      static_cast<int>(max_seqlen_KV));
}

// Hopper (SM90) FMHA.

void hopper_fmha_forward(
    at::Tensor &out,
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    at::Tensor &logsumexp,
    bool is_causal,
    double scale,
    int64_t q_tile_size,
    int64_t kv_tile_size,
    int64_t kernel_type,
    const OptTensor &cumulative_seqlen_Q,
    const OptTensor &cumulative_seqlen_KV,
    int64_t max_seqlen_Q,
    int64_t max_seqlen_KV) {
  natten::hopper_fmha_forward(
      out,
      query,
      key,
      value,
      at::optional<at::Tensor>(logsumexp),
      is_causal,
      static_cast<float>(scale),
      static_cast<int>(q_tile_size),
      static_cast<int>(kv_tile_size),
      static_cast<int>(kernel_type),
      cumulative_seqlen_Q,
      cumulative_seqlen_KV,
      static_cast<int>(max_seqlen_Q),
      static_cast<int>(max_seqlen_KV));
}

void hopper_fmha_backward(
    at::Tensor &grad_query,
    at::Tensor &grad_key,
    at::Tensor &grad_value,
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &out,
    const at::Tensor &grad_out,
    const at::Tensor &logsumexp,
    bool is_causal,
    double scale,
    int64_t q_tile_size,
    int64_t kv_tile_size,
    const OptTensor &cumulative_seqlen_Q,
    const OptTensor &cumulative_seqlen_KV,
    int64_t max_seqlen_Q,
    int64_t max_seqlen_KV) {
  natten::hopper_fmha_backward(
      grad_query,
      grad_key,
      grad_value,
      query,
      key,
      value,
      out,
      grad_out,
      logsumexp,
      is_causal,
      static_cast<float>(scale),
      static_cast<int>(q_tile_size),
      static_cast<int>(kv_tile_size),
      cumulative_seqlen_Q,
      cumulative_seqlen_KV,
      static_cast<int>(max_seqlen_Q),
      static_cast<int>(max_seqlen_KV));
}

// Blackwell (SM100) FMHA.

void blackwell_fmha_forward(
    at::Tensor &out,
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    at::Tensor &logsumexp,
    bool is_causal,
    double scale,
    int64_t q_tile_size,
    int64_t kv_tile_size,
    bool run_persistent,
    const OptTensor &cumulative_seqlen_Q,
    const OptTensor &cumulative_seqlen_KV,
    int64_t max_seqlen_Q,
    int64_t max_seqlen_KV) {
  natten::blackwell_fmha_forward(
      out,
      query,
      key,
      value,
      at::optional<at::Tensor>(logsumexp),
      is_causal,
      static_cast<float>(scale),
      static_cast<int>(q_tile_size),
      static_cast<int>(kv_tile_size),
      run_persistent,
      cumulative_seqlen_Q,
      cumulative_seqlen_KV,
      static_cast<int>(max_seqlen_Q),
      static_cast<int>(max_seqlen_KV));
}

void blackwell_fmha_backward(
    at::Tensor &grad_query,
    at::Tensor &grad_key,
    at::Tensor &grad_value,
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &out,
    const at::Tensor &grad_out,
    const at::Tensor &logsumexp,
    bool is_causal,
    double scale,
    int64_t q_tile_size,
    int64_t kv_tile_size,
    const OptTensor &cumulative_seqlen_Q,
    const OptTensor &cumulative_seqlen_KV,
    int64_t max_seqlen_Q,
    int64_t max_seqlen_KV,
    bool deterministic) {
  natten::blackwell_fmha_backward(
      grad_query,
      grad_key,
      grad_value,
      query,
      key,
      value,
      out,
      grad_out,
      logsumexp,
      is_causal,
      static_cast<float>(scale),
      static_cast<int>(q_tile_size),
      static_cast<int>(kv_tile_size),
      cumulative_seqlen_Q,
      cumulative_seqlen_KV,
      static_cast<int>(max_seqlen_Q),
      static_cast<int>(max_seqlen_KV),
      deterministic);
}

// Misc.

void compute_delta(
    const at::Tensor &out,
    const at::Tensor &d_out,
    at::Tensor &delta) {
  natten::compute_delta(out, d_out, delta);
}

} // namespace

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // Legacy (CUTLASS 2.X) FNA
#define REGISTER_FNA_OPS(DIM)                                               \
  ops.def(                                                                  \
      "na" #DIM "d_forward(Tensor! out, Tensor query, Tensor key, "         \
      "Tensor value, Tensor! logsumexp, int[] kernel_size, int[] stride, "  \
      "int[] dilation, int[] is_causal, float scale, "                      \
      "int[] q_tile_shape, int[] kv_tile_shape) -> ()");                    \
  ops.impl("na" #DIM "d_forward", torch::kCUDA, &na##DIM##d_forward);       \
  ops.def(                                                                  \
      "na" #DIM "d_backward(Tensor! grad_query, Tensor! grad_key, "         \
      "Tensor! grad_value, Tensor query, Tensor key, Tensor value, "        \
      "Tensor out, Tensor grad_out, Tensor logsumexp, "                     \
      "int[] kernel_size, int[] stride, int[] dilation, int[] is_causal, "  \
      "float scale, int[] q_tile_shape, int[] kv_tile_shape, "              \
      "int[] num_kv_splits, bool compute_delta_with_torch) -> ()");         \
  ops.impl("na" #DIM "d_backward", torch::kCUDA, &na##DIM##d_backward);

  REGISTER_FNA_OPS(1)
  REGISTER_FNA_OPS(2)
  REGISTER_FNA_OPS(3)
#undef REGISTER_FNA_OPS

  // Hopper (SM90) FNA
#define REGISTER_HOPPER_FNA_OPS(DIM)                                        \
  ops.def(                                                                  \
      "hopper_na" #DIM "d_forward(Tensor! out, Tensor query, Tensor key, "  \
      "Tensor value, Tensor! logsumexp, int[] kernel_size, int[] stride, "  \
      "int[] dilation, int[] is_causal, float scale, int[] q_shape, "       \
      "int[] kv_shape, int[] qkv_shape, int[] q_tile_shape, "               \
      "int[] kv_tile_shape, int kernel_type) -> ()");                       \
  ops.impl(                                                                 \
      "hopper_na" #DIM "d_forward",                                         \
      torch::kCUDA,                                                         \
      &hopper_na##DIM##d_forward);                                          \
  ops.def(                                                                  \
      "hopper_na" #DIM "d_backward(Tensor! grad_query, Tensor! grad_key, "  \
      "Tensor! grad_value, Tensor query, Tensor key, Tensor value, "        \
      "Tensor out, Tensor grad_out, Tensor logsumexp, "                     \
      "int[] kernel_size, int[] stride, int[] dilation, int[] is_causal, "  \
      "float scale, int[] q_shape, int[] kv_shape, int[] qkv_shape, "       \
      "int[] q_tile_shape, int[] kv_tile_shape) -> ()");                    \
  ops.impl(                                                                 \
      "hopper_na" #DIM "d_backward",                                        \
      torch::kCUDA,                                                         \
      &hopper_na##DIM##d_backward);

  REGISTER_HOPPER_FNA_OPS(1)
  REGISTER_HOPPER_FNA_OPS(2)
  REGISTER_HOPPER_FNA_OPS(3)
#undef REGISTER_HOPPER_FNA_OPS

  // Blackwell (SM100) FNA
#define REGISTER_BLACKWELL_FNA_OPS(DIM)                                     \
  ops.def(                                                                  \
      "blackwell_na" #DIM "d_forward(Tensor! out, Tensor query, "           \
      "Tensor key, Tensor value, Tensor! logsumexp, int[] kernel_size, "    \
      "int[] stride, int[] dilation, int[] is_causal, float scale, "        \
      "int[] q_shape, int[] kv_shape, int[] qkv_shape, "                    \
      "int[] q_tile_shape, int[] kv_tile_shape, "                           \
      "bool run_persistent) -> ()");                                        \
  ops.impl(                                                                 \
      "blackwell_na" #DIM "d_forward",                                      \
      torch::kCUDA,                                                         \
      &blackwell_na##DIM##d_forward);                                       \
  ops.def(                                                                  \
      "blackwell_na" #DIM "d_backward(Tensor! grad_query, "                 \
      "Tensor! grad_key, Tensor! grad_value, Tensor query, Tensor key, "    \
      "Tensor value, Tensor out, Tensor grad_out, Tensor logsumexp, "       \
      "int[] kernel_size, int[] stride, int[] dilation, int[] is_causal, "  \
      "float scale, int[] q_shape, int[] kv_shape, int[] qkv_shape, "       \
      "int[] q_tile_shape, int[] kv_tile_shape) -> ()");                    \
  ops.impl(                                                                 \
      "blackwell_na" #DIM "d_backward",                                     \
      torch::kCUDA,                                                         \
      &blackwell_na##DIM##d_backward);

  REGISTER_BLACKWELL_FNA_OPS(1)
  REGISTER_BLACKWELL_FNA_OPS(2)
  REGISTER_BLACKWELL_FNA_OPS(3)
#undef REGISTER_BLACKWELL_FNA_OPS

  // Reference kernels
#define REGISTER_REFERENCE_FNA_OPS(DIM)                                     \
  ops.def(                                                                  \
      "reference_na" #DIM "d_forward(Tensor! out, Tensor query, "           \
      "Tensor key, Tensor value, Tensor! logsumexp, int[] kernel_size, "    \
      "int[] stride, int[] dilation, int[] is_causal, float scale, "        \
      "int[] qkv_shape, int num_extra_kv) -> ()");                          \
  ops.impl(                                                                 \
      "reference_na" #DIM "d_forward",                                      \
      torch::kCUDA,                                                         \
      &reference_na##DIM##d_forward);                                       \
  ops.def(                                                                  \
      "reference_na" #DIM "d_backward(Tensor! grad_query, "                 \
      "Tensor! grad_key, Tensor! grad_value, Tensor query, Tensor key, "    \
      "Tensor value, Tensor out, Tensor grad_out, Tensor logsumexp, "       \
      "int[] kernel_size, int[] stride, int[] dilation, int[] is_causal, "  \
      "float scale, int[] qkv_shape, int num_extra_kv) -> ()");             \
  ops.impl(                                                                 \
      "reference_na" #DIM "d_backward",                                     \
      torch::kCUDA,                                                         \
      &reference_na##DIM##d_backward);

  REGISTER_REFERENCE_FNA_OPS(1)
  REGISTER_REFERENCE_FNA_OPS(2)
  REGISTER_REFERENCE_FNA_OPS(3)
#undef REGISTER_REFERENCE_FNA_OPS

  // Token permute
#define REGISTER_TOKEN_PERMUTE_OPS(DIM)                                     \
  ops.def(                                                                  \
      "token_permute_" #DIM "d(Tensor! out, Tensor input, "                 \
      "int[] tile_shape, int[] dilation, bool flip_tiled_dims) -> ()");     \
  ops.impl(                                                                 \
      "token_permute_" #DIM "d", torch::kCUDA, &token_permute_##DIM##d);    \
  ops.def(                                                                  \
      "token_unpermute_" #DIM "d(Tensor! out, Tensor input, "               \
      "int[] tile_shape, int[] dilation, bool flip_tiled_dims) -> ()");     \
  ops.impl(                                                                 \
      "token_unpermute_" #DIM "d",                                          \
      torch::kCUDA,                                                         \
      &token_unpermute_##DIM##d);

  REGISTER_TOKEN_PERMUTE_OPS(1)
  REGISTER_TOKEN_PERMUTE_OPS(2)
  REGISTER_TOKEN_PERMUTE_OPS(3)
#undef REGISTER_TOKEN_PERMUTE_OPS

  // FMHA
  ops.def(
      "fmha_forward(Tensor! out, Tensor query, Tensor key, Tensor value, "
      "Tensor! logsumexp, bool is_causal, float scale, int q_tile_size, "
      "int kv_tile_size, Tensor? cumulative_seqlen_Q, "
      "Tensor? cumulative_seqlen_KV, int max_seqlen_Q, "
      "int max_seqlen_KV) -> ()");
  ops.impl("fmha_forward", torch::kCUDA, &fmha_forward);
  ops.def(
      "fmha_backward(Tensor! grad_query, Tensor! grad_key, "
      "Tensor! grad_value, Tensor query, Tensor key, Tensor value, "
      "Tensor out, Tensor grad_out, Tensor logsumexp, bool is_causal, "
      "float scale, int q_tile_size, int kv_tile_size, int num_kv_splits, "
      "bool compute_delta_with_torch, Tensor? cumulative_seqlen_Q, "
      "Tensor? cumulative_seqlen_KV, int max_seqlen_Q, "
      "int max_seqlen_KV) -> ()");
  ops.impl("fmha_backward", torch::kCUDA, &fmha_backward);

  // Hopper FMHA
  ops.def(
      "hopper_fmha_forward(Tensor! out, Tensor query, Tensor key, "
      "Tensor value, Tensor! logsumexp, bool is_causal, float scale, "
      "int q_tile_size, int kv_tile_size, int kernel_type, "
      "Tensor? cumulative_seqlen_Q, Tensor? cumulative_seqlen_KV, "
      "int max_seqlen_Q, int max_seqlen_KV) -> ()");
  ops.impl("hopper_fmha_forward", torch::kCUDA, &hopper_fmha_forward);
  ops.def(
      "hopper_fmha_backward(Tensor! grad_query, Tensor! grad_key, "
      "Tensor! grad_value, Tensor query, Tensor key, Tensor value, "
      "Tensor out, Tensor grad_out, Tensor logsumexp, bool is_causal, "
      "float scale, int q_tile_size, int kv_tile_size, "
      "Tensor? cumulative_seqlen_Q, Tensor? cumulative_seqlen_KV, "
      "int max_seqlen_Q, int max_seqlen_KV) -> ()");
  ops.impl("hopper_fmha_backward", torch::kCUDA, &hopper_fmha_backward);

  // Blackwell FMHA
  ops.def(
      "blackwell_fmha_forward(Tensor! out, Tensor query, Tensor key, "
      "Tensor value, Tensor! logsumexp, bool is_causal, float scale, "
      "int q_tile_size, int kv_tile_size, bool run_persistent, "
      "Tensor? cumulative_seqlen_Q, Tensor? cumulative_seqlen_KV, "
      "int max_seqlen_Q, int max_seqlen_KV) -> ()");
  ops.impl("blackwell_fmha_forward", torch::kCUDA, &blackwell_fmha_forward);
  ops.def(
      "blackwell_fmha_backward(Tensor! grad_query, Tensor! grad_key, "
      "Tensor! grad_value, Tensor query, Tensor key, Tensor value, "
      "Tensor out, Tensor grad_out, Tensor logsumexp, bool is_causal, "
      "float scale, int q_tile_size, int kv_tile_size, "
      "Tensor? cumulative_seqlen_Q, Tensor? cumulative_seqlen_KV, "
      "int max_seqlen_Q, int max_seqlen_KV, bool deterministic) -> ()");
  ops.impl("blackwell_fmha_backward", torch::kCUDA, &blackwell_fmha_backward);

  // Misc
  ops.def("compute_delta(Tensor out, Tensor d_out, Tensor! delta) -> ()");
  ops.impl("compute_delta", torch::kCUDA, &compute_delta);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
