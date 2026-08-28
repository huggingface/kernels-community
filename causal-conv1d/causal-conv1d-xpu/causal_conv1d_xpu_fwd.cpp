/******************************************************************************
 * SYCL/XPU forward pass for causal-conv1d.
 ******************************************************************************/

#include "causal_conv1d_xpu.hpp"
#include "causal_conv1d_xpu_common.hpp"

using namespace causal_conv1d_xpu;

namespace {

////////////////////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////////////////////

template <int Width, typename input_t, typename weight_t>
struct CausalConv1dFwdKernel {
  ConvStrides p;
  const input_t *x_ptr;
  const weight_t *weight_ptr;
  const weight_t *bias_ptr;
  input_t *out_ptr;
  const int32_t *seq_idx_ptr;
  const input_t *initial_states_ptr;
  StateStrides initial_states_strides;
  int64_t num_chunks;
  int seq_chunk;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t gid = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t total =
        static_cast<int64_t>(p.batch) * p.dim * num_chunks;
    if (gid >= total) {
      return;
    }

    // The host only selects this kernel for channel-last inputs, so `channel`
    // is the contiguous axis and has to vary fastest for neighbouring lanes to
    // read neighbouring addresses.
    const int channel = static_cast<int>(gid % p.dim);
    const int64_t rest = gid / p.dim;
    const int chunk = static_cast<int>(rest % num_chunks);
    const int batch = static_cast<int>(rest / num_chunks);

    const input_t *x = x_ptr + batch * p.x_batch_stride +
                       channel * p.x_c_stride;
    const weight_t *weight = weight_ptr + channel * p.weight_c_stride;
    input_t *out =
        out_ptr + batch * p.out_batch_stride + channel * p.out_c_stride;
    const int32_t *seq_idx =
        seq_idx_ptr == nullptr ? nullptr : seq_idx_ptr + batch * p.seqlen;

    float weight_vals[Width];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
      weight_vals[i] = static_cast<float>(weight[i * p.weight_width_stride]);
    }
    const float bias_val =
        bias_ptr == nullptr ? 0.f : static_cast<float>(bias_ptr[channel]);

    // Reads the (conceptually zero/initial-state padded) input at position `l`,
    // which may be negative for the leading `width - 1` taps.
    const auto load_x = [&](int l, int cur_seq_idx) -> float {
      if (l >= 0) {
        if (seq_idx != nullptr && seq_idx[l] != cur_seq_idx) {
          return 0.f;
        }
        return static_cast<float>(x[l * p.x_l_stride]);
      }
      if (initial_states_ptr != nullptr) {
        const int64_t off = batch * initial_states_strides.batch_stride +
                            channel * initial_states_strides.c_stride +
                            (Width - 1 + l) * initial_states_strides.l_stride;
        return static_cast<float>(initial_states_ptr[off]);
      }
      return 0.f;
    };

    const int l_start = chunk * seq_chunk;
    const int l_end = sycl::min(l_start + seq_chunk, p.seqlen);

    // Sliding window of the last `Width` inputs, x_vals[Width - 1] == x[l].
    float x_vals[Width];
    int prev_seq_idx = -1;
    for (int l = l_start; l < l_end; ++l) {
      const int cur_seq_idx = seq_idx == nullptr ? 0 : seq_idx[l];
      if (l == l_start || (seq_idx != nullptr && cur_seq_idx != prev_seq_idx)) {
        // (Re)prime the window: at a chunk boundary we have no history, and at
        // a sequence boundary the history must be discarded anyway.
#pragma unroll
        for (int i = 0; i < Width; ++i) {
          x_vals[i] = load_x(l - (Width - 1) + i, cur_seq_idx);
        }
      } else {
#pragma unroll
        for (int i = 0; i < Width - 1; ++i) {
          x_vals[i] = x_vals[i + 1];
        }
        x_vals[Width - 1] = load_x(l, cur_seq_idx);
      }
      prev_seq_idx = cur_seq_idx;

      float out_val = bias_val;
#pragma unroll
      for (int i = 0; i < Width; ++i) {
        out_val += weight_vals[i] * x_vals[i];
      }
      if (p.silu_activation) {
        out_val = silu(out_val);
      }
      out[l * p.out_l_stride] = static_cast<input_t>(out_val);
    }
  }
};

// Vector of `Vec` inputs, laid out so that a whole run of channels can be moved
// in one machine load. Making the contiguity a compile-time fact is the point:
// with the channel stride only known at runtime the compiler has to emit one
// narrow access per channel, and a 32-wide sub-group then only covers 64 B of a
// 16-bit tensor.
template <typename input_t, int Vec>
struct alignas(sizeof(input_t) * Vec) InputVec {
  input_t v[Vec];
};

// Channel-last forward kernel for inputs whose channel axis is contiguous and
// suitably aligned. Same schedule as CausalConv1dFwdKernel, except that a lane
// owns `Vec` neighbouring channels and moves them as one vector, which is what
// widens the access rather than narrowing it: handing a lane the same channels
// as separate scalar accesses puts neighbouring lanes `Vec` channels apart and
// was measured at 0.8x.
template <int Width, int Vec, typename input_t, typename weight_t>
struct CausalConv1dFwdChanLastVecKernel {
  using vec_t = InputVec<input_t, Vec>;

  ConvStrides p;
  const input_t *x_ptr;
  const weight_t *weight_ptr;
  const weight_t *bias_ptr;
  input_t *out_ptr;
  const int32_t *seq_idx_ptr;
  const input_t *initial_states_ptr;
  StateStrides initial_states_strides;
  int64_t num_chunks;
  // p.dim / Vec, exact: the host only selects this kernel when Vec divides dim.
  int64_t dim_groups;
  int seq_chunk;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t gid = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t total =
        static_cast<int64_t>(p.batch) * dim_groups * num_chunks;
    if (gid >= total) {
      return;
    }

    const int cgroup = static_cast<int>(gid % dim_groups);
    const int64_t rest = gid / dim_groups;
    const int chunk = static_cast<int>(rest % num_chunks);
    const int batch = static_cast<int>(rest / num_chunks);
    const int channel = cgroup * Vec;

    const input_t *x = x_ptr + batch * p.x_batch_stride + channel;
    input_t *out = out_ptr + batch * p.out_batch_stride + channel;
    const int32_t *seq_idx =
        seq_idx_ptr == nullptr ? nullptr : seq_idx_ptr + batch * p.seqlen;

    float weight_vals[Vec][Width];
    float bias_val[Vec];
#pragma unroll
    for (int c = 0; c < Vec; ++c) {
      const weight_t *w = weight_ptr + (channel + c) * p.weight_c_stride;
#pragma unroll
      for (int i = 0; i < Width; ++i) {
        weight_vals[c][i] = static_cast<float>(w[i * p.weight_width_stride]);
      }
      bias_val[c] =
          bias_ptr == nullptr ? 0.f : static_cast<float>(bias_ptr[channel + c]);
    }

    // Sliding window of the last `Width` inputs, x_vals[c][Width - 1] == x[l].
    // Re-reading each input once per tap instead, as a tiled implementation
    // would, frees Vec * Width registers but cost 13% on BMG-G21: the extra
    // traffic is not absorbed by L1 as cheaply as the registers are worth.
    float x_vals[Vec][Width];

    // Fills window slot `slot` from position `l`, which may be negative for the
    // leading taps.
    const auto load_slot = [&](int l, int cur_seq_idx, int slot) {
      if (l >= 0 && (seq_idx == nullptr || seq_idx[l] == cur_seq_idx)) {
        const vec_t xv =
            *reinterpret_cast<const vec_t *>(x + l * p.x_l_stride);
#pragma unroll
        for (int c = 0; c < Vec; ++c) {
          x_vals[c][slot] = static_cast<float>(xv.v[c]);
        }
        return;
      }
      if (l < 0 && initial_states_ptr != nullptr) {
        // Only the first `Width - 1` positions of a chunk can reach here, so
        // this stays a scalar path.
#pragma unroll
        for (int c = 0; c < Vec; ++c) {
          const int64_t off = batch * initial_states_strides.batch_stride +
                              (channel + c) * initial_states_strides.c_stride +
                              (Width - 1 + l) * initial_states_strides.l_stride;
          x_vals[c][slot] = static_cast<float>(initial_states_ptr[off]);
        }
        return;
      }
#pragma unroll
      for (int c = 0; c < Vec; ++c) {
        x_vals[c][slot] = 0.f;
      }
    };

    const int l_start = chunk * seq_chunk;
    const int l_end = sycl::min(l_start + seq_chunk, p.seqlen);

    int prev_seq_idx = -1;
    for (int l = l_start; l < l_end; ++l) {
      const int cur_seq_idx = seq_idx == nullptr ? 0 : seq_idx[l];
      // seq_idx does not depend on the channel, so the whole vector reprimes
      // together.
      if (l == l_start || (seq_idx != nullptr && cur_seq_idx != prev_seq_idx)) {
#pragma unroll
        for (int i = 0; i < Width; ++i) {
          load_slot(l - (Width - 1) + i, cur_seq_idx, i);
        }
      } else {
#pragma unroll
        for (int c = 0; c < Vec; ++c) {
#pragma unroll
          for (int i = 0; i < Width - 1; ++i) {
            x_vals[c][i] = x_vals[c][i + 1];
          }
        }
        load_slot(l, cur_seq_idx, Width - 1);
      }
      prev_seq_idx = cur_seq_idx;

      vec_t ov;
#pragma unroll
      for (int c = 0; c < Vec; ++c) {
        float out_val = bias_val[c];
#pragma unroll
        for (int i = 0; i < Width; ++i) {
          out_val += weight_vals[c][i] * x_vals[c][i];
        }
        if (p.silu_activation) {
          out_val = silu(out_val);
        }
        ov.v[c] = static_cast<input_t>(out_val);
      }
      *reinterpret_cast<vec_t *>(out + l * p.out_l_stride) = ov;
    }
  }
};

// One work item per run of kFwdItemsPerLane output elements, with the sequence
// position as the fastest varying index. Used when the sequence axis is the
// contiguous one: giving each work item a long run of `l` (as the chunked
// kernel above does) makes neighbouring lanes read addresses
// `kSeqChunk * sizeof(T)` apart, which for fp32 is a 256 B stride and thrashes
// the cache sets.
//
// Neighbouring outputs overlap in all but one input, so a short run per lane
// needs kFwdItemsPerLane + Width - 1 loads instead of kFwdItemsPerLane * Width.
//
// `seq_idx` / `initial_states` / `final_states` all require the channel-last
// layout (see the checks in causal_conv1d_fwd), so this path never sees them.
template <int Width, typename input_t, typename weight_t>
struct CausalConv1dFwdContigKernel {
  ConvStrides p;
  const input_t *x_ptr;
  const weight_t *weight_ptr;
  const weight_t *bias_ptr;
  input_t *out_ptr;
  int64_t l_groups;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t gid = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t total = static_cast<int64_t>(p.batch) * p.dim * l_groups;
    if (gid >= total) {
      return;
    }

    constexpr int kItems = kFwdItemsPerLane;
    const int l0 = static_cast<int>(gid % l_groups) * kItems;
    const int64_t bc = gid / l_groups;
    const int channel = static_cast<int>(bc % p.dim);
    const int batch = static_cast<int>(bc / p.dim);

    const input_t *x =
        x_ptr + batch * p.x_batch_stride + channel * p.x_c_stride;
    const weight_t *weight = weight_ptr + channel * p.weight_c_stride;

    float weight_vals[Width];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
      weight_vals[i] = static_cast<float>(weight[i * p.weight_width_stride]);
    }
    const float bias_val =
        bias_ptr == nullptr ? 0.f : static_cast<float>(bias_ptr[channel]);

    // out[l0 + e] reads x[l0 + e - (Width - 1)] .. x[l0 + e], so the run needs
    // x[l0 - (Width - 1)] .. x[l0 + kItems - 1].
    float xv[kItems + Width - 1];
#pragma unroll
    for (int k = 0; k < kItems + Width - 1; ++k) {
      const int gl = l0 - (Width - 1) + k;
      xv[k] = (gl >= 0 && gl < p.seqlen)
                  ? static_cast<float>(x[gl * p.x_l_stride])
                  : 0.f;
    }

#pragma unroll
    for (int e = 0; e < kItems; ++e) {
      const int l = l0 + e;
      if (l >= p.seqlen) {
        continue;
      }
      float out_val = bias_val;
#pragma unroll
      for (int i = 0; i < Width; ++i) {
        out_val += weight_vals[i] * xv[e + i];
      }
      if (p.silu_activation) {
        out_val = silu(out_val);
      }
      out_ptr[batch * p.out_batch_stride + channel * p.out_c_stride +
              l * p.out_l_stride] = static_cast<input_t>(out_val);
    }
  }
};

// Writes `final_states[b, c, i] = concat(initial_states, x)[b, c, -(width-1)+i]`.

template <typename input_t> struct CausalConv1dFinalStatesKernel {
  int batch;
  int dim;
  int seqlen;
  int width;
  const input_t *x_ptr;
  int64_t x_batch_stride;
  int64_t x_c_stride;
  int64_t x_l_stride;
  const input_t *initial_states_ptr;
  StateStrides initial_states_strides;
  input_t *final_states_ptr;
  StateStrides final_states_strides;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t gid = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t state_len = width - 1;
    const int64_t total = static_cast<int64_t>(batch) * dim * state_len;
    if (gid >= total) {
      return;
    }

    const int i = static_cast<int>(gid % state_len);
    const int64_t bc = gid / state_len;
    const int channel = static_cast<int>(bc % dim);
    const int b = static_cast<int>(bc / dim);

    const int l = seqlen - (width - 1) + i;
    float val = 0.f;
    if (l >= 0) {
      val = static_cast<float>(
          x_ptr[b * x_batch_stride + channel * x_c_stride + l * x_l_stride]);
    } else if (initial_states_ptr != nullptr) {
      // The virtual buffer is concat(initial_states, x), so a negative index
      // lands `width - 1 + l` elements into initial_states.
      val = static_cast<float>(
          initial_states_ptr[b * initial_states_strides.batch_stride +
                             channel * initial_states_strides.c_stride +
                             (width - 1 + l) * initial_states_strides.l_stride]);
    }

    final_states_ptr[b * final_states_strides.batch_stride +
                     channel * final_states_strides.c_stride +
                     i * final_states_strides.l_stride] =
        static_cast<input_t>(val);
  }
};

} // namespace

void causal_conv1d_fwd(const at::Tensor &x, const at::Tensor &weight,
                       const c10::optional<at::Tensor> &bias_,
                       const c10::optional<at::Tensor> &seq_idx_,
                       const c10::optional<at::Tensor> &initial_states_,
                       at::Tensor &out,
                       c10::optional<at::Tensor> &final_states_out_,
                       bool silu_activation) {
  const auto input_type = x.scalar_type();
  const auto weight_type = weight.scalar_type();
  TORCH_CHECK(input_type == at::ScalarType::Float ||
              input_type == at::ScalarType::Half ||
              input_type == at::ScalarType::BFloat16);
  TORCH_CHECK(weight_type == at::ScalarType::Float ||
              weight_type == at::ScalarType::Half ||
              weight_type == at::ScalarType::BFloat16);
  TORCH_CHECK(x.is_xpu());
  TORCH_CHECK(weight.is_xpu());

  const auto sizes = x.sizes();
  const int batch_size = sizes[0];
  const int dim = sizes[1];
  const int seqlen = sizes[2];
  const int width = weight.size(-1);

  CHECK_SHAPE(x, batch_size, dim, seqlen);
  CHECK_SHAPE(weight, dim, width);
  TORCH_CHECK(x.stride(2) == 1 || x.stride(1) == 1);
  TORCH_CHECK(width >= 2 && width <= kMaxWidth,
              "causal_conv1d only supports width between 2 and 4");

  const bool is_channel_last = x.stride(1) == 1 && x.stride(2) > 1;
  // For sequence-contiguous inputs the chunked kernel gives each work item a run
  // of `kSeqChunk` positions, so neighbouring lanes are `kSeqChunk *
  // sizeof(input_t)` bytes apart, a stride that aliases in the cache. The
  // element-wise kernel walks `l` across lanes instead and amortises the input
  // reuse over kFwdItemsPerLane outputs per lane.
  const bool use_contig_elem_kernel = !is_channel_last;

  if (bias_.has_value()) {
    const auto &bias = bias_.value();
    TORCH_CHECK(bias.scalar_type() == weight_type);
    TORCH_CHECK(bias.is_xpu());
    TORCH_CHECK(bias.stride(-1) == 1);
    CHECK_SHAPE(bias, dim);
  }
  if (seq_idx_.has_value()) {
    TORCH_CHECK(is_channel_last, "seq_idx is only supported for channel last layout");
    const auto &seq_idx = seq_idx_.value();
    TORCH_CHECK(seq_idx.scalar_type() == torch::kInt32);
    TORCH_CHECK(seq_idx.is_xpu());
    TORCH_CHECK(seq_idx.is_contiguous());
    CHECK_SHAPE(seq_idx, batch_size, seqlen);
  }
  if (initial_states_.has_value()) {
    TORCH_CHECK(is_channel_last,
                "initial_states is only supported for channel last layout");
    const auto &initial_states = initial_states_.value();
    TORCH_CHECK(initial_states.scalar_type() == input_type);
    TORCH_CHECK(initial_states.is_xpu());
    CHECK_SHAPE(initial_states, batch_size, dim, width - 1);
  }
  if (final_states_out_.has_value()) {
    TORCH_CHECK(is_channel_last,
                "final_states is only supported for channel last layout");
    const auto &final_states = final_states_out_.value();
    TORCH_CHECK(final_states.scalar_type() == input_type);
    TORCH_CHECK(final_states.is_xpu());
    CHECK_SHAPE(final_states, batch_size, dim, width - 1);
  }

  const c10::DeviceGuard device_guard(x.device());
  sycl::queue &q = at::xpu::getCurrentXPUStream().queue();

  const ConvStrides p = make_conv_strides(x, weight, out, batch_size, dim,
                                          seqlen, silu_activation);
  const int64_t min_items = min_work_items(q.get_device());
  const int seq_chunk =
      pick_seq_chunk(static_cast<int64_t>(batch_size) * dim, seqlen, min_items);
  const int64_t num_chunks = ceil_div(seqlen, seq_chunk);

  DISPATCH_ITYPE(input_type, "causal_conv1d_fwd", [&] {
    DISPATCH_WTYPE(weight_type, "causal_conv1d_fwd", [&] {
      DISPATCH_WIDTH(width, "causal_conv1d_fwd", [&] {
      if (use_contig_elem_kernel) {
        CausalConv1dFwdContigKernel<kWidth, input_t, weight_t> kernel{};
        kernel.p = p;
        kernel.x_ptr = x.data_ptr<input_t>();
        kernel.weight_ptr = weight.data_ptr<weight_t>();
        kernel.bias_ptr =
            bias_.has_value() ? bias_.value().data_ptr<weight_t>() : nullptr;
        kernel.out_ptr = out.data_ptr<input_t>();
        const int64_t l_groups = ceil_div(seqlen, kFwdItemsPerLane);
        kernel.l_groups = l_groups;
        launch_1d(q, static_cast<int64_t>(batch_size) * dim * l_groups, kernel);
      } else {
        // A lane holds Vec * Width sliding-window inputs plus as many weights,
        // so widening the access and keeping the window in registers pull in
        // opposite directions. Past a budget of about 12 the window stops
        // fitting and the kernel slows sharply: on BMG-G21, halving Vec at
        // width 4 gained 18% for bf16 and 2% for fp16, but cost 8-13% at
        // widths 2 and 3, where the wide access is what matters. Letting the
        // budget reach for 16 B per lane rather than 8 B where it fits is
        // worth a further 1% on fp32.
        constexpr int kVecBudget = 12;
        constexpr int kWideVec = 16 / static_cast<int>(sizeof(input_t));
        constexpr int kVec =
            kWideVec * kWidth <= kVecBudget
                ? kWideVec
                : (kWideVec / 2 * kWidth <= kVecBudget ? kWideVec / 2
                                                       : kWideVec / 4);
        constexpr int64_t kVecBytes = kVec * sizeof(input_t);
        const auto aligned = [](const void *ptr) {
          return reinterpret_cast<uintptr_t>(ptr) % kVecBytes == 0;
        };
        // Every offset the kernel forms has to keep the vector aligned, so the
        // strides it steps by must be multiples of the width as well.
        const bool use_vec_kernel =
            is_channel_last && out.stride(1) == 1 && dim % kVec == 0 &&
            p.x_l_stride % kVec == 0 && p.x_batch_stride % kVec == 0 &&
            p.out_l_stride % kVec == 0 && p.out_batch_stride % kVec == 0 &&
            aligned(x.data_ptr<input_t>()) && aligned(out.data_ptr<input_t>());

        if (use_vec_kernel) {
          const int64_t dim_groups = dim / kVec;
          // A lane now covers kVec channels, so the chunk has to be shortened
          // against the reduced lane count to keep the device fed.
          const int vec_seq_chunk = pick_seq_chunk(
              static_cast<int64_t>(batch_size) * dim_groups, seqlen, min_items);
          const int64_t vec_num_chunks = ceil_div(seqlen, vec_seq_chunk);
          CausalConv1dFwdChanLastVecKernel<kWidth, kVec, input_t, weight_t>
              kernel{};
          kernel.p = p;
          kernel.x_ptr = x.data_ptr<input_t>();
          kernel.weight_ptr = weight.data_ptr<weight_t>();
          kernel.bias_ptr =
              bias_.has_value() ? bias_.value().data_ptr<weight_t>() : nullptr;
          kernel.out_ptr = out.data_ptr<input_t>();
          kernel.seq_idx_ptr = seq_idx_.has_value()
                                   ? seq_idx_.value().data_ptr<int32_t>()
                                   : nullptr;
          kernel.initial_states_ptr =
              initial_states_.has_value()
                  ? initial_states_.value().data_ptr<input_t>()
                  : nullptr;
          kernel.initial_states_strides =
              initial_states_.has_value()
                  ? make_state_strides(initial_states_.value())
                  : StateStrides{0, 0, 0};
          kernel.num_chunks = vec_num_chunks;
          kernel.dim_groups = dim_groups;
          kernel.seq_chunk = vec_seq_chunk;
          launch_1d(
              q, static_cast<int64_t>(batch_size) * dim_groups * vec_num_chunks,
              kernel);
        } else {
          CausalConv1dFwdKernel<kWidth, input_t, weight_t> kernel{};
          kernel.p = p;
          kernel.x_ptr = x.data_ptr<input_t>();
          kernel.weight_ptr = weight.data_ptr<weight_t>();
          kernel.bias_ptr =
              bias_.has_value() ? bias_.value().data_ptr<weight_t>() : nullptr;
          kernel.out_ptr = out.data_ptr<input_t>();
          kernel.seq_idx_ptr = seq_idx_.has_value()
                                   ? seq_idx_.value().data_ptr<int32_t>()
                                   : nullptr;
          kernel.initial_states_ptr =
              initial_states_.has_value()
                  ? initial_states_.value().data_ptr<input_t>()
                  : nullptr;
          kernel.initial_states_strides =
              initial_states_.has_value()
                  ? make_state_strides(initial_states_.value())
                  : StateStrides{0, 0, 0};
          kernel.num_chunks = num_chunks;
          kernel.seq_chunk = seq_chunk;
          launch_1d(q, static_cast<int64_t>(batch_size) * dim * num_chunks,
                    kernel);
        }
      }

      if (final_states_out_.has_value()) {
        const at::Tensor &final_states = final_states_out_.value();
        CausalConv1dFinalStatesKernel<input_t> fs{};
        fs.batch = batch_size;
        fs.dim = dim;
        fs.seqlen = seqlen;
        fs.width = width;
        fs.x_ptr = x.data_ptr<input_t>();
        fs.x_batch_stride = x.stride(0);
        fs.x_c_stride = x.stride(1);
        fs.x_l_stride = x.stride(2);
        fs.initial_states_ptr =
            initial_states_.has_value()
                ? initial_states_.value().data_ptr<input_t>()
                : nullptr;
        fs.initial_states_strides =
            initial_states_.has_value()
                ? make_state_strides(initial_states_.value())
                : StateStrides{0, 0, 0};
        fs.final_states_ptr = final_states.data_ptr<input_t>();
        fs.final_states_strides = make_state_strides(final_states);
        launch_1d(q, static_cast<int64_t>(batch_size) * dim * (width - 1), fs);
      }
      });
    });
  });
}
