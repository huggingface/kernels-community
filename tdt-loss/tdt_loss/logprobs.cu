// Per-lattice-node kernels of the TDT loss: the log-softmax gather that feeds
// the alpha/beta recursions, and the fused gradient w.r.t. the logits.
//
// Both kernels use one thread block per lattice node (b, t, u) and stream the
// vocabulary dimension. Nodes outside of a sample's lattice (t >= T or u > U)
// never read the logits.

#include <limits>

#include "common.cuh"
#include "stable.h"

namespace tdt_loss {

namespace {

template <typename scalar_t>
__global__ void __launch_bounds__(kMaxRowThreads) tdt_logprobs_fwd_kernel(
    const scalar_t *__restrict__ token_logits, int64_t tok_sb, int64_t tok_st,
    int64_t tok_su, const scalar_t *__restrict__ duration_logits,
    int64_t dur_sb, int64_t dur_st, int64_t dur_su,
    const int *__restrict__ targets, int64_t targets_stride,
    const int *__restrict__ logit_lengths,
    const int *__restrict__ target_lengths, int max_T, int max_U, int V, int D,
    int blank_id, float sigma, float *__restrict__ blank_lp,
    float *__restrict__ label_lp, float *__restrict__ dur_lp,
    float *__restrict__ token_lse) {
  __shared__ float shm_m[kWarpSize];
  __shared__ float shm_s[kWarpSize];

  const int64_t row = blockIdx.x;
  const int u = row % max_U;
  const int t = (row / max_U) % max_T;
  const int b = row / (static_cast<int64_t>(max_U) * max_T);

  int T, U;
  sample_lengths(logit_lengths, target_lengths, b, max_T, max_U, T, U);

  if (t >= T || u > U) {
    if (threadIdx.x == 0) {
      blank_lp[row] = neg_inf();
      label_lp[row] = neg_inf();
      token_lse[row] = 0.f;
    }
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
      dur_lp[row * D + i] = neg_inf();
    }
    return;
  }

  const scalar_t *x = token_logits + b * tok_sb + t * tok_st + u * tok_su;

  // Online softmax normalizer over the vocabulary.
  float m = neg_inf();
  float s = 0.f;
  for (int v = threadIdx.x; v < V; v += blockDim.x) {
    const float xv = static_cast<float>(x[v]);
    if (xv > m) {
      s = s * expf(m - xv) + 1.f;
      m = xv;
    } else if (xv != -INFINITY) {
      s += expf(xv - m);
    }
  }
  block_reduce_max_sum(m, s, shm_m, shm_s);

  if (threadIdx.x == 0) {
    const float lse = m + logf(s);
    token_lse[row] = lse;
    blank_lp[row] = static_cast<float>(x[blank_id]) - lse - sigma;

    float label = neg_inf();
    if (u < U) {
      const int target = targets[b * targets_stride + u];
      CUDA_KERNEL_ASSERT(target >= 0 && target < V && "target out of range");
      if (target >= 0 && target < V) {
        label = static_cast<float>(x[target]) - lse - sigma;
      }
    }
    label_lp[row] = label;

    // The duration head is tiny (a handful of classes), a serial
    // log-softmax is cheaper than another block reduction.
    const scalar_t *y = duration_logits + b * dur_sb + t * dur_st + u * dur_su;
    float dm = neg_inf();
    for (int i = 0; i < D; ++i)
      dm = fmaxf(dm, static_cast<float>(y[i]));
    float ds = 0.f;
    for (int i = 0; i < D; ++i)
      ds += expf(static_cast<float>(y[i]) - dm);
    const float dlse = dm + logf(ds);
    for (int i = 0; i < D; ++i) {
      dur_lp[row * D + i] = static_cast<float>(y[i]) - dlse;
    }
  }
}

template <typename scalar_t>
__global__ void __launch_bounds__(kMaxRowThreads) tdt_logits_grad_kernel(
    const scalar_t *__restrict__ token_logits, int64_t tok_sb, int64_t tok_st,
    int64_t tok_su, const int *__restrict__ targets, int64_t targets_stride,
    const int *__restrict__ logit_lengths,
    const int *__restrict__ target_lengths, const int *__restrict__ durations,
    const float *__restrict__ blank_lp, const float *__restrict__ label_lp,
    const float *__restrict__ dur_lp, const float *__restrict__ token_lse,
    const double *__restrict__ alphas, const double *__restrict__ betas,
    const double *__restrict__ log_ll, const float *__restrict__ grad_loss,
    int max_T, int max_U, int V, int D, int blank_id,
    scalar_t *__restrict__ grad_token_logits,
    scalar_t *__restrict__ grad_duration_logits) {
  // Posterior occupancy of the blank / label arc leaving this node, for each
  // duration.
  __shared__ float shm_blank[kMaxDurations];
  __shared__ float shm_label[kMaxDurations];

  const int64_t row = blockIdx.x;
  const int u = row % max_U;
  const int t = (row / max_U) % max_T;
  const int b = row / (static_cast<int64_t>(max_U) * max_T);

  int T, U;
  sample_lengths(logit_lengths, target_lengths, b, max_T, max_U, T, U);

  scalar_t *gx = grad_token_logits + row * V;
  scalar_t *gd = grad_duration_logits + row * D;
  const scalar_t zero = static_cast<scalar_t>(0.f);

  const double ll = log_ll[b];
  // Infeasible samples (infinite loss) get a zero gradient instead of NaNs.
  if (t >= T || u > U || !isfinite(ll)) {
    for (int v = threadIdx.x; v < V; v += blockDim.x)
      gx[v] = zero;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
      gd[i] = zero;
    return;
  }

  const int64_t sample = static_cast<int64_t>(b) * max_T * max_U;
  const double alpha = alphas[row];
  const double blank = blank_lp[row];
  const double label = label_lp[row];

  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const int d = durations[i];
    const int t_next = t + d;
    const double lp_dur = dur_lp[row * D + i];

    float p_blank = 0.f;
    if (d > 0) {
      double beta_next = -INFINITY;
      if (t_next < T) {
        beta_next = betas[sample + static_cast<int64_t>(t_next) * max_U + u];
      } else if (t_next == T && u == U) {
        beta_next = 0.0; // terminal arc
      }
      if (beta_next != -INFINITY) {
        p_blank =
            expf(static_cast<float>(alpha + blank + lp_dur + beta_next - ll));
      }
    }

    float p_label = 0.f;
    if (u < U && t_next < T) {
      const double beta_next =
          betas[sample + static_cast<int64_t>(t_next) * max_U + u + 1];
      p_label =
          expf(static_cast<float>(alpha + label + lp_dur + beta_next - ll));
    }

    shm_blank[i] = p_blank;
    shm_label[i] = p_label;
  }
  __syncthreads();

  float occ_blank = 0.f;
  float occ_label = 0.f;
  for (int i = 0; i < D; ++i) {
    occ_blank += shm_blank[i];
    occ_label += shm_label[i];
  }

  // loss = -log_ll, so d loss / d log_prob(arc) = -occupancy(arc).
  const float go = grad_loss[b];
  const float g_blank = -go * occ_blank;
  const float g_label = -go * occ_label;
  const float g_total = g_blank + g_label;

  // Through log_softmax: d/dx_j = g_j - softmax_j * sum_k g_k.
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float g_dur = -go * (shm_blank[i] + shm_label[i]);
    gd[i] = static_cast<scalar_t>(g_dur - expf(dur_lp[row * D + i]) * g_total);
  }

  if (g_blank == 0.f && g_label == 0.f) {
    for (int v = threadIdx.x; v < V; v += blockDim.x)
      gx[v] = zero;
    return;
  }

  const scalar_t *x = token_logits + b * tok_sb + t * tok_st + u * tok_su;
  const float lse = token_lse[row];
  const int target = u < U ? targets[b * targets_stride + u] : -1;
  for (int v = threadIdx.x; v < V; v += blockDim.x) {
    float g = -expf(static_cast<float>(x[v]) - lse) * g_total;
    if (v == blank_id)
      g += g_blank;
    if (v == target)
      g += g_label;
    gx[v] = static_cast<scalar_t>(g);
  }
}

int row_block_size(int64_t V) {
  int threads = 32;
  while (threads < kMaxRowThreads && threads < V)
    threads *= 2;
  return threads;
}

void check_logits(Tensor const &token_logits, Tensor const &duration_logits) {
  STD_TORCH_CHECK(token_logits.is_cuda(), "token_logits must be a CUDA tensor");
  STD_TORCH_CHECK(token_logits.dim() == 4,
                  "token_logits must have shape (batch, T, U+1, vocab_size+1)");
  STD_TORCH_CHECK(
      duration_logits.dim() == 4,
      "duration_logits must have shape (batch, T, U+1, num_durations)");
  STD_TORCH_CHECK(
      same_device(duration_logits, token_logits),
      "token_logits and duration_logits must be on the same device");
  STD_TORCH_CHECK(duration_logits.scalar_type() == token_logits.scalar_type(),
                  "token_logits and duration_logits must have the same dtype");
  for (int i = 0; i < 3; ++i) {
    STD_TORCH_CHECK(duration_logits.size(i) == token_logits.size(i),
                    "token_logits and duration_logits must agree on (batch, T, "
                    "U+1)");
  }
  STD_TORCH_CHECK(token_logits.stride(3) == 1 && duration_logits.stride(3) == 1,
                  "the last dimension of the logits must be contiguous");
  STD_TORCH_CHECK(duration_logits.size(3) <= kMaxDurations, "at most ",
                  kMaxDurations, " durations are supported");
}

void check_int_tensor(Tensor const &x, char const *name, Tensor const &like) {
  STD_TORCH_CHECK(same_device(x, like), name,
                  " must be on the same device as the logits");
  STD_TORCH_CHECK(x.scalar_type() == ScalarType::Int, name, " must be int32");
  STD_TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

void check_float_out(Tensor const &x, char const *name, Tensor const &like) {
  STD_TORCH_CHECK(same_device(x, like), name,
                  " must be on the same device as the logits");
  STD_TORCH_CHECK(x.scalar_type() == ScalarType::Float, name,
                  " must be float32");
  STD_TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

} // namespace

} // namespace tdt_loss

void tdt_logprobs_fwd(Tensor const &token_logits, Tensor const &duration_logits,
                      Tensor const &targets, Tensor const &logit_lengths,
                      Tensor const &target_lengths, int64_t blank_id,
                      double sigma, Tensor &blank_lp, Tensor &label_lp,
                      Tensor &dur_lp, Tensor &token_lse) {
  using namespace tdt_loss;

  check_logits(token_logits, duration_logits);
  const int64_t B = token_logits.size(0);
  const int64_t max_T = token_logits.size(1);
  const int64_t max_U = token_logits.size(2);
  const int64_t V = token_logits.size(3);
  const int64_t D = duration_logits.size(3);

  STD_TORCH_CHECK(blank_id >= 0 && blank_id < V, "blank_id out of range");
  STD_TORCH_CHECK(targets.dim() == 2 && targets.size(0) == B &&
                      targets.size(1) >= max_U - 1,
                  "targets must have shape (batch, U)");
  check_int_tensor(targets, "targets", token_logits);
  check_int_tensor(logit_lengths, "logit_lengths", token_logits);
  check_int_tensor(target_lengths, "target_lengths", token_logits);
  STD_TORCH_CHECK(logit_lengths.numel() == B && target_lengths.numel() == B,
                  "lengths must have shape (batch,)");
  for (auto *out : {&blank_lp, &label_lp, &token_lse}) {
    check_float_out(*out, "log-prob output", token_logits);
    STD_TORCH_CHECK(out->numel() == B * max_T * max_U,
                    "log-prob outputs must have shape (batch, T, U+1)");
  }
  check_float_out(dur_lp, "dur_lp", token_logits);
  STD_TORCH_CHECK(dur_lp.numel() == B * max_T * max_U * D,
                  "dur_lp must have shape (batch, T, U+1, num_durations)");

  const int64_t rows = B * max_T * max_U;
  if (rows == 0)
    return;
  STD_TORCH_CHECK(rows <= std::numeric_limits<int32_t>::max(),
                  "batch * T * (U+1) is too large");

  const torch::stable::accelerator::DeviceGuard device_guard(
      token_logits.get_device_index());
  const cudaStream_t stream = current_stream(token_logits);
  const int threads = row_block_size(V);

  TDT_DISPATCH_FLOATING_TYPES(
      token_logits.scalar_type(), "tdt_logprobs_fwd", [&] {
        tdt_logprobs_fwd_kernel<scalar_t>
            <<<static_cast<unsigned int>(rows), threads, 0, stream>>>(
                ptr<scalar_t>(token_logits), token_logits.stride(0),
                token_logits.stride(1), token_logits.stride(2),
                ptr<scalar_t>(duration_logits), duration_logits.stride(0),
                duration_logits.stride(1), duration_logits.stride(2),
                ptr<int>(targets), targets.stride(0), ptr<int>(logit_lengths),
                ptr<int>(target_lengths), max_T, max_U, V, D, blank_id,
                static_cast<float>(sigma), ptr<float>(blank_lp),
                ptr<float>(label_lp), ptr<float>(dur_lp),
                ptr<float>(token_lse));
      });
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void tdt_logits_grad(Tensor const &token_logits, Tensor const &targets,
                     Tensor const &logit_lengths, Tensor const &target_lengths,
                     Tensor const &durations, Tensor const &blank_lp,
                     Tensor const &label_lp, Tensor const &dur_lp,
                     Tensor const &token_lse, Tensor const &alphas,
                     Tensor const &betas, Tensor const &log_ll,
                     Tensor const &grad_loss, int64_t blank_id,
                     Tensor &grad_token_logits, Tensor &grad_duration_logits) {
  using namespace tdt_loss;

  STD_TORCH_CHECK(token_logits.is_cuda() && token_logits.dim() == 4,
                  "token_logits must be a 4D CUDA tensor");
  STD_TORCH_CHECK(token_logits.stride(3) == 1,
                  "the last dimension of token_logits must be contiguous");
  const int64_t B = token_logits.size(0);
  const int64_t max_T = token_logits.size(1);
  const int64_t max_U = token_logits.size(2);
  const int64_t V = token_logits.size(3);
  const int64_t D = durations.numel();

  STD_TORCH_CHECK(D <= kMaxDurations, "at most ", kMaxDurations,
                  " durations are supported");
  STD_TORCH_CHECK(blank_id >= 0 && blank_id < V, "blank_id out of range");
  check_int_tensor(targets, "targets", token_logits);
  check_int_tensor(logit_lengths, "logit_lengths", token_logits);
  check_int_tensor(target_lengths, "target_lengths", token_logits);
  check_int_tensor(durations, "durations", token_logits);
  for (auto *in : {&blank_lp, &label_lp, &token_lse}) {
    check_float_out(*in, "log-prob tensor", token_logits);
    STD_TORCH_CHECK(in->numel() == B * max_T * max_U,
                    "log-prob tensors must have shape (batch, T, U+1)");
  }
  for (auto *in : {&alphas, &betas}) {
    STD_TORCH_CHECK(
        same_device(*in, token_logits) &&
            in->scalar_type() == ScalarType::Double && in->is_contiguous() &&
            in->numel() == B * max_T * max_U,
        "alphas and betas must be contiguous float64 tensors of shape "
        "(batch, T, U+1)");
  }
  check_float_out(dur_lp, "dur_lp", token_logits);
  STD_TORCH_CHECK(same_device(log_ll, token_logits) &&
                      log_ll.scalar_type() == ScalarType::Double &&
                      log_ll.is_contiguous(),
                  "log_ll must be a contiguous float64 tensor");
  check_float_out(grad_loss, "grad_loss", token_logits);
  STD_TORCH_CHECK(log_ll.numel() == B && grad_loss.numel() == B,
                  "log_ll and grad_loss must have shape (batch,)");
  STD_TORCH_CHECK(
      grad_token_logits.is_contiguous() &&
          grad_token_logits.sizes() == token_logits.sizes() &&
          grad_token_logits.scalar_type() == token_logits.scalar_type(),
      "grad_token_logits must be contiguous and match token_logits");
  STD_TORCH_CHECK(grad_duration_logits.is_contiguous() &&
                      grad_duration_logits.numel() == B * max_T * max_U * D &&
                      grad_duration_logits.scalar_type() ==
                          token_logits.scalar_type(),
                  "grad_duration_logits must be contiguous and match "
                  "duration_logits");

  const int64_t rows = B * max_T * max_U;
  if (rows == 0)
    return;
  STD_TORCH_CHECK(rows <= std::numeric_limits<int32_t>::max(),
                  "batch * T * (U+1) is too large");

  const torch::stable::accelerator::DeviceGuard device_guard(
      token_logits.get_device_index());
  const cudaStream_t stream = current_stream(token_logits);
  const int threads = row_block_size(V);

  TDT_DISPATCH_FLOATING_TYPES(
      token_logits.scalar_type(), "tdt_logits_grad", [&] {
        tdt_logits_grad_kernel<scalar_t>
            <<<static_cast<unsigned int>(rows), threads, 0, stream>>>(
                ptr<scalar_t>(token_logits), token_logits.stride(0),
                token_logits.stride(1), token_logits.stride(2),
                ptr<int>(targets), targets.stride(0), ptr<int>(logit_lengths),
                ptr<int>(target_lengths), ptr<int>(durations),
                ptr<float>(blank_lp), ptr<float>(label_lp), ptr<float>(dur_lp),
                ptr<float>(token_lse), ptr<double>(alphas), ptr<double>(betas),
                ptr<double>(log_ll), ptr<float>(grad_loss), max_T, max_U, V, D,
                blank_id, ptr<scalar_t>(grad_token_logits),
                ptr<scalar_t>(grad_duration_logits));
      });
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}
