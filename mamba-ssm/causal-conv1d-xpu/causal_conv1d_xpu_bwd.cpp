/******************************************************************************
 * SYCL/XPU backward pass for causal-conv1d.
 ******************************************************************************/

#include "causal_conv1d_xpu.hpp"
#include "causal_conv1d_xpu_common.hpp"

using namespace causal_conv1d_xpu;

namespace {

////////////////////////////////////////////////////////////////////////////////
// Backward
////////////////////////////////////////////////////////////////////////////////

// Shared helpers for the backward kernels: both need the "padded" input read
// and the pre-activation gradient at an output position.
template <int Width, typename input_t, typename weight_t> struct BwdCommon {
  ConvStrides p;
  const input_t *x_ptr;
  const weight_t *weight_ptr;
  const weight_t *bias_ptr;
  const input_t *dout_ptr;
  int64_t dout_batch_stride;
  int64_t dout_c_stride;
  int64_t dout_l_stride;
  const int32_t *seq_idx_ptr;
  const input_t *initial_states_ptr;
  StateStrides initial_states_strides;

  float load_x(int batch, int channel, int l, int cur_seq_idx,
               const int32_t *seq_idx) const {
    if (l >= 0) {
      if (seq_idx != nullptr && seq_idx[l] != cur_seq_idx) {
        return 0.f;
      }
      return static_cast<float>(
          x_ptr[batch * p.x_batch_stride + channel * p.x_c_stride +
                l * p.x_l_stride]);
    }
    if (initial_states_ptr != nullptr) {
      return static_cast<float>(
          initial_states_ptr[batch * initial_states_strides.batch_stride +
                             channel * initial_states_strides.c_stride +
                             (Width - 1 + l) * initial_states_strides.l_stride]);
    }
    return 0.f;
  }

  // Gradient w.r.t. the pre-activation output at position `m`.
  float dpre(int batch, int channel, int m, const float *weight_vals,
             float bias_val, const int32_t *seq_idx) const {
    float g = static_cast<float>(
        dout_ptr[batch * dout_batch_stride + channel * dout_c_stride +
                 m * dout_l_stride]);
    if (!p.silu_activation) {
      return g;
    }
    const int cur_seq_idx = seq_idx == nullptr ? 0 : seq_idx[m];
    float pre = bias_val;
#pragma unroll
    for (int i = 0; i < Width; ++i) {
      pre += weight_vals[i] *
             load_x(batch, channel, m - (Width - 1) + i, cur_seq_idx, seq_idx);
    }
    return g * dsilu(pre);
  }
};

// Computes dx (plus the dfinal_states contribution) and accumulates
// dweight/dbias.
template <int Width, typename input_t, typename weight_t>
struct CausalConv1dBwdKernel {
  BwdCommon<Width, input_t, weight_t> common;
  input_t *dx_ptr;
  int64_t dx_batch_stride;
  int64_t dx_c_stride;
  int64_t dx_l_stride;
  float *dweight_ptr;
  int64_t dweight_c_stride;
  int64_t dweight_width_stride;
  float *dbias_ptr;
  const input_t *dfinal_states_ptr;
  StateStrides dfinal_states_strides;
  int64_t num_chunks;
  bool channel_major;
  int seq_chunk;

  void operator()(sycl::nd_item<1> item) const {
    const ConvStrides &p = common.p;
    const int64_t gid = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t total = static_cast<int64_t>(p.batch) * p.dim * num_chunks;
    if (gid >= total) {
      return;
    }

    int chunk, channel, batch;
    if (channel_major) {
      channel = static_cast<int>(gid % p.dim);
      const int64_t rest = gid / p.dim;
      chunk = static_cast<int>(rest % num_chunks);
      batch = static_cast<int>(rest / num_chunks);
    } else {
      chunk = static_cast<int>(gid % num_chunks);
      const int64_t bc = gid / num_chunks;
      channel = static_cast<int>(bc % p.dim);
      batch = static_cast<int>(bc / p.dim);
    }

    const int32_t *seq_idx = common.seq_idx_ptr == nullptr
                                 ? nullptr
                                 : common.seq_idx_ptr + batch * p.seqlen;

    float weight_vals[Width];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
      weight_vals[i] =
          static_cast<float>(common.weight_ptr[channel * p.weight_c_stride +
                                               i * p.weight_width_stride]);
    }
    const float bias_val = common.bias_ptr == nullptr
                               ? 0.f
                               : static_cast<float>(common.bias_ptr[channel]);

    float dweight_acc[Width];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
      dweight_acc[i] = 0.f;
    }
    float dbias_acc = 0.f;

    const int l_start = chunk * seq_chunk;
    const int l_end = sycl::min(l_start + seq_chunk, p.seqlen);
    if (l_start >= l_end) {
      return;
    }

    // dx[l] needs the pre-activation gradient at l .. l + Width - 1, and the
    // dweight accumulation needs it at l. Keeping a sliding window means dpre
    // is evaluated once per position instead of Width + 1 times, which matters
    // because each dpre itself reads Width inputs when silu is enabled.
    float dpre_win[Width];
#pragma unroll
    for (int j = 0; j < Width; ++j) {
      const int m = l_start + j;
      dpre_win[j] = m < p.seqlen ? common.dpre(batch, channel, m, weight_vals,
                                               bias_val, seq_idx)
                                 : 0.f;
    }

    for (int l = l_start; l < l_end; ++l) {
      const int cur_seq_idx = seq_idx == nullptr ? 0 : seq_idx[l];

      // out[m] uses x[m - (width - 1) + i] with weight_vals[i], so x[l] feeds
      // out[l + (width - 1) - i].
      float dx_val = 0.f;
#pragma unroll
      for (int i = 0; i < Width; ++i) {
        const int j = Width - 1 - i;
        const int m = l + j;
        if (m >= p.seqlen) {
          continue;
        }
        if (seq_idx != nullptr && seq_idx[m] != cur_seq_idx) {
          continue;
        }
        dx_val += weight_vals[i] * dpre_win[j];
      }

      // final_states[i] == x[seqlen - (width - 1) + i], so the last width - 1
      // positions also receive gradient from dfinal_states.
      if (dfinal_states_ptr != nullptr) {
        const int i = l - (p.seqlen - (Width - 1));
        if (i >= 0 && i < Width - 1) {
          dx_val += static_cast<float>(
              dfinal_states_ptr[batch * dfinal_states_strides.batch_stride +
                                channel * dfinal_states_strides.c_stride +
                                i * dfinal_states_strides.l_stride]);
        }
      }

      dx_ptr[batch * dx_batch_stride + channel * dx_c_stride +
             l * dx_l_stride] = static_cast<input_t>(dx_val);

      // dweight[i] += dpre[l] * x[l - (width - 1) + i], dbias += dpre[l].
      const float g = dpre_win[0];
      dbias_acc += g;
#pragma unroll
      for (int i = 0; i < Width; ++i) {
        dweight_acc[i] +=
            g * common.load_x(batch, channel, l - (Width - 1) + i, cur_seq_idx,
                              seq_idx);
      }

#pragma unroll
      for (int j = 0; j < Width - 1; ++j) {
        dpre_win[j] = dpre_win[j + 1];
      }
      const int m_next = l + Width;
      dpre_win[Width - 1] =
          m_next < p.seqlen
              ? common.dpre(batch, channel, m_next, weight_vals, bias_val,
                            seq_idx)
              : 0.f;
    }

#pragma unroll
    for (int i = 0; i < Width; ++i) {
      sycl::atomic_ref<float, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          ref(dweight_ptr[channel * dweight_c_stride + i * dweight_width_stride]);
      ref.fetch_add(dweight_acc[i]);
    }
    if (dbias_ptr != nullptr) {
      sycl::atomic_ref<float, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          ref(dbias_ptr[channel]);
      ref.fetch_add(dbias_acc);
    }
  }
};

// Backward counterpart of CausalConv1dFwdContigKernel: one work item per output
// element with the sequence position as the fastest varying index, so that
// neighbouring lanes touch neighbouring addresses.
//
// dweight/dbias are per-channel reductions, and here a whole work group belongs
// to a single channel, so the partials are folded with reduce_over_group and the
// group issues a single atomic each instead of one per work item.
//
// `seq_idx` / `initial_states` require the channel-last layout upstream, so this
// path never sees them; that lets the taps be loaded once into registers.
template <int Width, typename input_t, typename weight_t>
struct CausalConv1dBwdContigKernel {
  BwdCommon<Width, input_t, weight_t> common;
  input_t *dx_ptr;
  int64_t dx_batch_stride;
  int64_t dx_c_stride;
  int64_t dx_l_stride;
  float *dweight_ptr;
  int64_t dweight_c_stride;
  int64_t dweight_width_stride;
  float *dbias_ptr;
  const input_t *dfinal_states_ptr;
  StateStrides dfinal_states_strides;
  int64_t num_tiles;

  void operator()(sycl::nd_item<1> item) const {
    const ConvStrides &p = common.p;
    const auto grp = item.get_group();
    const int64_t gid = static_cast<int64_t>(item.get_group_linear_id());
    const int tile = static_cast<int>(gid % num_tiles);
    const int64_t bc = gid / num_tiles;
    const int channel = static_cast<int>(bc % p.dim);
    const int batch = static_cast<int>(bc / p.dim);
    const int lid = static_cast<int>(item.get_local_linear_id());
    // Each lane owns a run of neighbouring positions. Sharing the sliding
    // window across the run cuts the x reads from kBwdItemsPerLane *
    // (2 * Width - 1) to kBwdItemsPerLane + 2 * Width - 2 and the dout reads
    // from kBwdItemsPerLane * Width to kBwdItemsPerLane + Width - 1, and for
    // 16-bit input the neighbours come back in one wider access.
    constexpr int kItems = kBwdItemsPerLane;
    constexpr int kXWin = kItems + 2 * Width - 2;
    constexpr int kPreWin = kItems + Width - 1;
    const int l0 =
        tile * kItems * static_cast<int>(item.get_local_range(0)) + kItems * lid;

    float weight_vals[Width];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
      weight_vals[i] =
          static_cast<float>(common.weight_ptr[channel * p.weight_c_stride +
                                               i * p.weight_width_stride]);
    }
    const float bias_val = common.bias_ptr == nullptr
                               ? 0.f
                               : static_cast<float>(common.bias_ptr[channel]);

    float dweight_acc[Width];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
      dweight_acc[i] = 0.f;
    }
    float dbias_acc = 0.f;

    if (l0 < p.seqlen) {
      // dpre at l0 .. l0 + kPreWin - 1 collectively reads x[l0 - (Width - 1)]
      // .. x[l0 + kXWin - Width].
      float xv[kXWin];
#pragma unroll
      for (int k = 0; k < kXWin; ++k) {
        const int gl = l0 - (Width - 1) + k;
        // The tail entries are never used, but the load still has to stay
        // inside the buffer.
        xv[k] = gl < p.seqlen ? common.load_x(batch, channel, gl, 0, nullptr)
                              : 0.f;
      }

      float dpre_win[kPreWin];
#pragma unroll
      for (int q = 0; q < kPreWin; ++q) {
        const int m = l0 + q;
        if (m >= p.seqlen) {
          dpre_win[q] = 0.f;
          continue;
        }
        float g = static_cast<float>(
            common.dout_ptr[batch * common.dout_batch_stride +
                            channel * common.dout_c_stride +
                            m * common.dout_l_stride]);
        if (p.silu_activation) {
          float pre = bias_val;
#pragma unroll
          for (int i = 0; i < Width; ++i) {
            pre += weight_vals[i] * xv[q + i];
          }
          g *= dsilu(pre);
        }
        dpre_win[q] = g;
      }

#pragma unroll
      for (int e = 0; e < kItems; ++e) {
        const int l = l0 + e;
        if (l >= p.seqlen) {
          continue;
        }

        // x[l] feeds out[l + (Width - 1) - i] through weight_vals[i].
        float dx_val = 0.f;
#pragma unroll
        for (int i = 0; i < Width; ++i) {
          dx_val += weight_vals[i] * dpre_win[e + Width - 1 - i];
        }

        // The last Width - 1 positions are also final_states entries.
        if (dfinal_states_ptr != nullptr) {
          const int i = l - (p.seqlen - (Width - 1));
          if (i >= 0 && i < Width - 1) {
            dx_val += static_cast<float>(
                dfinal_states_ptr[batch * dfinal_states_strides.batch_stride +
                                  channel * dfinal_states_strides.c_stride +
                                  i * dfinal_states_strides.l_stride]);
          }
        }

        dx_ptr[batch * dx_batch_stride + channel * dx_c_stride +
               l * dx_l_stride] = static_cast<input_t>(dx_val);

        const float g = dpre_win[e];
        dbias_acc += g;
#pragma unroll
        for (int i = 0; i < Width; ++i) {
          dweight_acc[i] += g * xv[e + i];
        }
      }
    }

    // Collective, so it has to run for every work item in the group.
#pragma unroll
    for (int i = 0; i < Width; ++i) {
      const float sum =
          sycl::reduce_over_group(grp, dweight_acc[i], sycl::plus<float>());
      if (lid == 0) {
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ref(dweight_ptr[channel * dweight_c_stride +
                            i * dweight_width_stride]);
        ref.fetch_add(sum);
      }
    }
    const float dbias_sum =
        sycl::reduce_over_group(grp, dbias_acc, sycl::plus<float>());
    if (lid == 0 && dbias_ptr != nullptr) {
      sycl::atomic_ref<float, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          ref(dbias_ptr[channel]);
      ref.fetch_add(dbias_sum);
    }
  }
};

// Gradient w.r.t. initial_states, i.e. the virtual positions -(width-1)..-1.
template <int Width, typename input_t, typename weight_t>
struct CausalConv1dBwdInitialStatesKernel {
  BwdCommon<Width, input_t, weight_t> common;
  input_t *dinitial_states_ptr;
  StateStrides dinitial_states_strides;
  const input_t *dfinal_states_ptr;
  StateStrides dfinal_states_strides;

  void operator()(sycl::nd_item<1> item) const {
    const ConvStrides &p = common.p;
    const int64_t state_len = Width - 1;
    const int64_t gid = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t total = static_cast<int64_t>(p.batch) * p.dim * state_len;
    if (gid >= total) {
      return;
    }

    const int s = static_cast<int>(gid % state_len);
    const int64_t bc = gid / state_len;
    const int channel = static_cast<int>(bc % p.dim);
    const int batch = static_cast<int>(bc / p.dim);

    // initial_states[s] is the input at position l = s - (Width - 1).
    const int l = s - (Width - 1);

    float weight_vals[Width];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
      weight_vals[i] =
          static_cast<float>(common.weight_ptr[channel * p.weight_c_stride +
                                               i * p.weight_width_stride]);
    }
    const float bias_val = common.bias_ptr == nullptr
                               ? 0.f
                               : static_cast<float>(common.bias_ptr[channel]);

    float dval = 0.f;
#pragma unroll
    for (int i = 0; i < Width; ++i) {
      const int m = l + (Width - 1) - i;
      if (m < 0 || m >= p.seqlen) {
        continue;
      }
      dval += weight_vals[i] *
              common.dpre(batch, channel, m, weight_vals, bias_val, nullptr);
    }

    // If seqlen < width - 1 the tail of final_states still lives in
    // initial_states, so route that gradient here as well.
    if (dfinal_states_ptr != nullptr) {
      const int i = l - (p.seqlen - (Width - 1));
      if (i >= 0 && i < Width - 1) {
        dval += static_cast<float>(
            dfinal_states_ptr[batch * dfinal_states_strides.batch_stride +
                              channel * dfinal_states_strides.c_stride +
                              i * dfinal_states_strides.l_stride]);
      }
    }

    dinitial_states_ptr[batch * dinitial_states_strides.batch_stride +
                        channel * dinitial_states_strides.c_stride +
                        s * dinitial_states_strides.l_stride] =
        static_cast<input_t>(dval);
  }
};

} // namespace

void causal_conv1d_bwd(const at::Tensor &x, const at::Tensor &weight,
                       const c10::optional<at::Tensor> &bias_, at::Tensor &dout,
                       const c10::optional<at::Tensor> &seq_idx_,
                       const c10::optional<at::Tensor> &initial_states_,
                       const c10::optional<at::Tensor> &dfinal_states_,
                       at::Tensor &dx, at::Tensor &dweight,
                       c10::optional<at::Tensor> &dbias_,
                       c10::optional<at::Tensor> &dinitial_states_,
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
  TORCH_CHECK(dout.is_xpu());
  TORCH_CHECK(bias_.has_value() == dbias_.has_value());

  const auto sizes = x.sizes();
  const int batch_size = sizes[0];
  const int dim = sizes[1];
  const int seqlen = sizes[2];
  const int width = weight.size(-1);

  TORCH_CHECK(width >= 2 && width <= kMaxWidth,
              "causal_conv1d only supports width between 2 and 4");
  CHECK_SHAPE(x, batch_size, dim, seqlen);
  CHECK_SHAPE(weight, dim, width);
  CHECK_SHAPE(dout, batch_size, dim, seqlen);
  CHECK_SHAPE(dx, batch_size, dim, seqlen);

  // dweight/dbias are accumulated with atomics, so they must be fp32 and
  // zero-initialised by the caller (the Python wrapper does this).
  TORCH_CHECK(dweight.scalar_type() == at::ScalarType::Float,
              "dweight must be float32 on XPU");
  TORCH_CHECK(dweight.is_xpu());
  if (dbias_.has_value()) {
    TORCH_CHECK(dbias_.value().scalar_type() == at::ScalarType::Float,
                "dbias must be float32 on XPU");
    TORCH_CHECK(dbias_.value().stride(-1) == 1);
    CHECK_SHAPE(dbias_.value(), dim);
  }
  if (seq_idx_.has_value()) {
    const auto &seq_idx = seq_idx_.value();
    TORCH_CHECK(seq_idx.scalar_type() == torch::kInt32);
    TORCH_CHECK(seq_idx.is_contiguous());
    CHECK_SHAPE(seq_idx, batch_size, seqlen);
  }
  if (initial_states_.has_value()) {
    CHECK_SHAPE(initial_states_.value(), batch_size, dim, width - 1);
  }
  if (dfinal_states_.has_value()) {
    CHECK_SHAPE(dfinal_states_.value(), batch_size, dim, width - 1);
  }
  if (dinitial_states_.has_value()) {
    CHECK_SHAPE(dinitial_states_.value(), batch_size, dim, width - 1);
  }

  const c10::DeviceGuard device_guard(x.device());
  sycl::queue &q = at::xpu::getCurrentXPUStream().queue();

  // `dout` plays the role of `out` for stride bookkeeping.
  const ConvStrides p = make_conv_strides(x, weight, dout, batch_size, dim,
                                          seqlen, silu_activation);
  const int64_t min_items = min_work_items(q.get_device());
  const int seq_chunk =
      pick_seq_chunk(static_cast<int64_t>(batch_size) * dim, seqlen, min_items);
  const int64_t num_chunks = ceil_div(seqlen, seq_chunk);
  // The group size doubles as the tile width of the element-wise kernel, so a
  // wide group wastes lanes on short sequences but hides more latency on long
  // ones.
  //
  // Every group also pays a fixed cost of Width + 1 group reductions plus the
  // same number of atomics onto the channel's dweight/dbias. At width 4 that
  // cost dominates: the run time came out proportional to the group count
  // rather than to the element count, seqlen 16384 taking 16.7 ms at a 128-wide
  // tile, 7.7 ms at 256 and 4.4 ms at 512. Hence the widest tile that still
  // leaves short sequences a group they can fill. Measured on BMG-G21.
  const int bwd_tile = seqlen >= 512 ? 512 : kWorkGroupSize;
  const int64_t num_tiles = ceil_div(seqlen, bwd_tile);

  // Mirrors the forward path: for sequence-contiguous inputs the chunked kernel
  // gives each lane a run of seq_chunk positions, so neighbouring lanes are
  // seq_chunk * sizeof(input_t) bytes apart and that stride aliases in the
  // cache. The element-wise kernel avoids it by walking `l` across lanes.
  //
  // It only pays off once a sequence fills a whole tile: the per-channel
  // dweight/dbias reduction is a group collective, so a group that is mostly
  // out of range still pays for the collective while most of its lanes idle.
  // At seqlen 4 that was measured at 0.08x of stock PyTorch.
  //
  // seq_idx / initial_states require the channel-last layout upstream, so the
  // element-wise kernel never has to express them.
  const bool use_contig_elem_kernel =
      x.stride(1) != 1 && seqlen >= bwd_tile && !seq_idx_.has_value() &&
      !initial_states_.has_value();

  DISPATCH_ITYPE(input_type, "causal_conv1d_bwd", [&] {
    DISPATCH_WTYPE(weight_type, "causal_conv1d_bwd", [&] {
      DISPATCH_WIDTH(width, "causal_conv1d_bwd", [&] {
      BwdCommon<kWidth, input_t, weight_t> common{};
      common.p = p;
      common.x_ptr = x.data_ptr<input_t>();
      common.weight_ptr = weight.data_ptr<weight_t>();
      common.bias_ptr =
          bias_.has_value() ? bias_.value().data_ptr<weight_t>() : nullptr;
      common.dout_ptr = dout.data_ptr<input_t>();
      common.dout_batch_stride = dout.stride(0);
      common.dout_c_stride = dout.stride(1);
      common.dout_l_stride = dout.stride(2);
      common.seq_idx_ptr = seq_idx_.has_value()
                               ? seq_idx_.value().data_ptr<int32_t>()
                               : nullptr;
      common.initial_states_ptr =
          initial_states_.has_value()
              ? initial_states_.value().data_ptr<input_t>()
              : nullptr;
      common.initial_states_strides =
          initial_states_.has_value()
              ? make_state_strides(initial_states_.value())
              : StateStrides{0, 0, 0};

      const input_t *dfinal_states_ptr =
          dfinal_states_.has_value()
              ? dfinal_states_.value().data_ptr<input_t>()
              : nullptr;
      const StateStrides dfinal_states_strides =
          dfinal_states_.has_value()
              ? make_state_strides(dfinal_states_.value())
              : StateStrides{0, 0, 0};

      CausalConv1dBwdKernel<kWidth, input_t, weight_t> kernel{};
      kernel.common = common;
      kernel.dx_ptr = dx.data_ptr<input_t>();
      kernel.dx_batch_stride = dx.stride(0);
      kernel.dx_c_stride = dx.stride(1);
      kernel.dx_l_stride = dx.stride(2);
      kernel.dweight_ptr = dweight.data_ptr<float>();
      kernel.dweight_c_stride = dweight.stride(0);
      kernel.dweight_width_stride = dweight.stride(1);
      kernel.dbias_ptr =
          dbias_.has_value() ? dbias_.value().data_ptr<float>() : nullptr;
      kernel.dfinal_states_ptr = dfinal_states_ptr;
      kernel.dfinal_states_strides = dfinal_states_strides;

      if (use_contig_elem_kernel) {
        CausalConv1dBwdContigKernel<kWidth, input_t, weight_t> ck{};
        ck.common = common;
        ck.dx_ptr = kernel.dx_ptr;
        ck.dx_batch_stride = kernel.dx_batch_stride;
        ck.dx_c_stride = kernel.dx_c_stride;
        ck.dx_l_stride = kernel.dx_l_stride;
        ck.dweight_ptr = kernel.dweight_ptr;
        ck.dweight_c_stride = kernel.dweight_c_stride;
        ck.dweight_width_stride = kernel.dweight_width_stride;
        ck.dbias_ptr = kernel.dbias_ptr;
        ck.dfinal_states_ptr = dfinal_states_ptr;
        ck.dfinal_states_strides = dfinal_states_strides;
        ck.num_tiles = num_tiles;
        // bwd_tile counts positions, each lane takes kBwdItemsPerLane of them.
        launch_groups(q, static_cast<int64_t>(batch_size) * dim * num_tiles,
                      bwd_tile / kBwdItemsPerLane, ck);
      } else {
        kernel.num_chunks = num_chunks;
        kernel.channel_major = x.stride(1) == 1;
        kernel.seq_chunk = seq_chunk;
        launch_1d(q, static_cast<int64_t>(batch_size) * dim * num_chunks,
                  kernel);
      }

      if (dinitial_states_.has_value()) {
        const at::Tensor &dinitial_states = dinitial_states_.value();
        CausalConv1dBwdInitialStatesKernel<kWidth, input_t, weight_t> init_kernel{};
        init_kernel.common = common;
        init_kernel.dinitial_states_ptr = dinitial_states.data_ptr<input_t>();
        init_kernel.dinitial_states_strides =
            make_state_strides(dinitial_states);
        init_kernel.dfinal_states_ptr = dfinal_states_ptr;
        init_kernel.dfinal_states_strides = dfinal_states_strides;
        launch_1d(q, static_cast<int64_t>(batch_size) * dim * (width - 1),
                  init_kernel);
      }
      });
    });
  });
}
