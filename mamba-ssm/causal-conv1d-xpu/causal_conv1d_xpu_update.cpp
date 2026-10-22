/******************************************************************************
 * SYCL/XPU single-step update (cached decode) for causal-conv1d.
 ******************************************************************************/

#include "causal_conv1d_xpu.hpp"
#include "causal_conv1d_xpu_common.hpp"

using namespace causal_conv1d_xpu;

namespace {

////////////////////////////////////////////////////////////////////////////////
// Update (single-step / cached decode)
////////////////////////////////////////////////////////////////////////////////

template <int Width, typename input_t, typename weight_t>
struct CausalConv1dUpdateKernel {
  ConvStrides p;
  const input_t *x_ptr;
  const weight_t *weight_ptr;
  const weight_t *bias_ptr;
  input_t *out_ptr;
  input_t *conv_state_ptr;
  StateStrides conv_state_strides;
  int conv_state_len;
  const int32_t *cache_seqlens_ptr;
  const int32_t *conv_state_indices_ptr;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t gid = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t total = static_cast<int64_t>(p.batch) * p.dim;
    if (gid >= total) {
      return;
    }
    const int channel = static_cast<int>(gid % p.dim);
    const int batch = static_cast<int>(gid / p.dim);

    const bool circular = cache_seqlens_ptr != nullptr;
    const int conv_state_batch_coord = conv_state_indices_ptr == nullptr
                                           ? batch
                                           : conv_state_indices_ptr[batch];

    const input_t *x =
        x_ptr + batch * p.x_batch_stride + channel * p.x_c_stride;
    const weight_t *weight = weight_ptr + channel * p.weight_c_stride;
    input_t *out =
        out_ptr + batch * p.out_batch_stride + channel * p.out_c_stride;
    input_t *conv_state =
        conv_state_ptr + conv_state_batch_coord * conv_state_strides.batch_stride +
        channel * conv_state_strides.c_stride;
    const int64_t state_l_stride = conv_state_strides.l_stride;

    const float bias_val =
        bias_ptr == nullptr ? 0.f : static_cast<float>(bias_ptr[channel]);

    float weight_vals[Width];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
      weight_vals[i] = static_cast<float>(weight[i * p.weight_width_stride]);
    }

    const int state_len = conv_state_len;
    const int advance_len = p.seqlen;
    const int cache_seqlen =
        circular ? cache_seqlens_ptr[batch] % state_len : 0;
    int update_idx = cache_seqlen - (Width - 1);
    update_idx = update_idx < 0 ? update_idx + state_len : update_idx;

    float x_vals[Width];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
      x_vals[i] = 0.f;
    }

    if (!circular) {
      // Shift the state left by `advance_len` to make room for the new tokens.
      for (int i = 0; i < state_len - advance_len - (Width - 1); ++i) {
        conv_state[i * state_l_stride] =
            conv_state[(i + advance_len) * state_l_stride];
      }
#pragma unroll
      for (int i = 0; i < Width - 1; ++i) {
        const input_t state_val =
            conv_state[(state_len - (Width - 1) + i) * state_l_stride];
        if (i < advance_len + (Width - 1) &&
            state_len - advance_len - (Width - 1) + i >= 0) {
          conv_state[(state_len - advance_len - (Width - 1) + i) *
                     state_l_stride] = state_val;
        }
        x_vals[i] = static_cast<float>(state_val);
      }
    } else {
#pragma unroll
      for (int i = 0; i < Width - 1; ++i) {
        x_vals[i] = static_cast<float>(conv_state[update_idx * state_l_stride]);
        update_idx = update_idx + 1 >= state_len ? update_idx + 1 - state_len
                                                 : update_idx + 1;
      }
    }

    for (int i = 0; i < p.seqlen; ++i) {
      const input_t x_val = x[i * p.x_l_stride];
      if (!circular) {
        if (i < advance_len && state_len - advance_len + i >= 0) {
          conv_state[(state_len - advance_len + i) * state_l_stride] = x_val;
        }
      } else {
        conv_state[update_idx * state_l_stride] = x_val;
        ++update_idx;
        update_idx = update_idx >= state_len ? update_idx - state_len : update_idx;
      }
      x_vals[Width - 1] = static_cast<float>(x_val);

      float out_val = bias_val;
#pragma unroll
      for (int j = 0; j < Width; ++j) {
        out_val += weight_vals[j] * x_vals[j];
      }
      if (p.silu_activation) {
        out_val = silu(out_val);
      }
      out[i * p.out_l_stride] = static_cast<input_t>(out_val);

#pragma unroll
      for (int j = 0; j < Width - 1; ++j) {
        x_vals[j] = x_vals[j + 1];
      }
    }
  }
};

} // namespace

void causal_conv1d_update(const at::Tensor &x, const at::Tensor &conv_state,
                          const at::Tensor &weight,
                          const c10::optional<at::Tensor> &bias_,
                          at::Tensor &out, bool silu_activation,
                          const c10::optional<at::Tensor> &cache_seqlens_,
                          const c10::optional<at::Tensor> &conv_state_indices_) {
  const auto input_type = x.scalar_type();
  const auto weight_type = weight.scalar_type();
  TORCH_CHECK(input_type == at::ScalarType::Float ||
              input_type == at::ScalarType::Half ||
              input_type == at::ScalarType::BFloat16);
  TORCH_CHECK(weight_type == at::ScalarType::Float ||
              weight_type == at::ScalarType::Half ||
              weight_type == at::ScalarType::BFloat16);
  TORCH_CHECK(conv_state.scalar_type() == input_type);
  TORCH_CHECK(x.is_xpu());
  TORCH_CHECK(conv_state.is_xpu());
  TORCH_CHECK(weight.is_xpu());

  const auto sizes = x.sizes();
  const int batch_size = sizes[0];
  const int dim = sizes[1];
  const int seqlen = sizes[2];
  const int width = weight.size(-1);
  const int conv_state_len = conv_state.size(2);
  TORCH_CHECK(conv_state_len >= width - 1);

  CHECK_SHAPE(x, batch_size, dim, seqlen);
  CHECK_SHAPE(weight, dim, width);
  TORCH_CHECK(width >= 2 && width <= kMaxWidth,
              "causal_conv1d only supports width between 2 and 4");

  if (bias_.has_value()) {
    const auto &bias = bias_.value();
    TORCH_CHECK(bias.scalar_type() == weight_type);
    TORCH_CHECK(bias.is_xpu());
    TORCH_CHECK(bias.stride(-1) == 1);
    CHECK_SHAPE(bias, dim);
  }
  if (conv_state_indices_.has_value()) {
    const auto &conv_state_indices = conv_state_indices_.value();
    TORCH_CHECK(conv_state_indices.scalar_type() == torch::kInt32);
    TORCH_CHECK(conv_state_indices.is_xpu());
    TORCH_CHECK(conv_state_indices.stride(0) == 1);
    CHECK_SHAPE(conv_state_indices, batch_size);
    CHECK_SHAPE(conv_state, conv_state.size(0), dim, conv_state_len);
  } else {
    CHECK_SHAPE(conv_state, batch_size, dim, conv_state_len);
  }
  if (cache_seqlens_.has_value()) {
    const auto &cache_seqlens = cache_seqlens_.value();
    TORCH_CHECK(cache_seqlens.scalar_type() == torch::kInt32);
    TORCH_CHECK(cache_seqlens.is_xpu());
    TORCH_CHECK(cache_seqlens.stride(-1) == 1);
    CHECK_SHAPE(cache_seqlens, batch_size);
  }

  const c10::DeviceGuard device_guard(x.device());
  sycl::queue &q = at::xpu::getCurrentXPUStream().queue();

  const ConvStrides p = make_conv_strides(x, weight, out, batch_size, dim,
                                          seqlen, silu_activation);

  DISPATCH_ITYPE(input_type, "causal_conv1d_update", [&] {
    DISPATCH_WTYPE(weight_type, "causal_conv1d_update", [&] {
      DISPATCH_WIDTH(width, "causal_conv1d_update", [&] {
      CausalConv1dUpdateKernel<kWidth, input_t, weight_t> kernel{};
      kernel.p = p;
      kernel.x_ptr = x.data_ptr<input_t>();
      kernel.weight_ptr = weight.data_ptr<weight_t>();
      kernel.bias_ptr =
          bias_.has_value() ? bias_.value().data_ptr<weight_t>() : nullptr;
      kernel.out_ptr = out.data_ptr<input_t>();
      kernel.conv_state_ptr = conv_state.data_ptr<input_t>();
      kernel.conv_state_strides = make_state_strides(conv_state);
      kernel.conv_state_len = conv_state_len;
      kernel.cache_seqlens_ptr =
          cache_seqlens_.has_value()
              ? cache_seqlens_.value().data_ptr<int32_t>()
              : nullptr;
      kernel.conv_state_indices_ptr =
          conv_state_indices_.has_value()
              ? conv_state_indices_.value().data_ptr<int32_t>()
              : nullptr;
      launch_1d(q, static_cast<int64_t>(batch_size) * dim, kernel);
      });
    });
  });
}
