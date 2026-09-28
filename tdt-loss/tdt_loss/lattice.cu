// Forward (alpha) and backward (beta) recursions over the TDT lattice.
//
// For a sample with T frames and U labels the lattice nodes are (t, u) with
// t in [0, T) and u in [0, U]. From node (t, u), for every duration d:
//   - a blank arc (only d > 0) goes to (t + d, u),
//   - a label arc goes to (t + d, u + 1).
// The alignment ends with a blank arc from (T - d, U) that exits the lattice.
//
// Every arc strictly increases t + u, so all nodes of an anti-diagonal
// t + u = n are independent. One thread block handles one sample and sweeps
// the anti-diagonals, with the threads striding over u.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "common.cuh"

namespace tdt_loss {

namespace {

__device__ __forceinline__ void load_durations(const int *durations, int D,
                                               int *shm_durations) {
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    shm_durations[i] = durations[i];
  }
  __syncthreads();
}

__global__ void tdt_alpha_kernel(const float *__restrict__ blank_lp,
                                 const float *__restrict__ label_lp,
                                 const float *__restrict__ dur_lp,
                                 const int *__restrict__ source_lengths,
                                 const int *__restrict__ target_lengths,
                                 const int *__restrict__ durations, int max_T,
                                 int max_U, int D, float *__restrict__ alphas,
                                 float *__restrict__ log_ll) {
  __shared__ int shm_durations[kMaxDurations];

  const int b = blockIdx.x;
  int T, U;
  sample_lengths(source_lengths, target_lengths, b, max_T, max_U, T, U);
  load_durations(durations, D, shm_durations);

  const int64_t sample = static_cast<int64_t>(b) * max_T * max_U;
  const float *blank = blank_lp + sample;
  const float *label = label_lp + sample;
  const float *dur = dur_lp + sample * D;
  float *alpha = alphas + sample;

  for (int n = 0; n < T + U; ++n) {
    const int u_min = max(0, n - T + 1);
    const int u_max = min(n, U);
    for (int u = u_min + threadIdx.x; u <= u_max; u += blockDim.x) {
      const int t = n - u;
      float a = n == 0 ? 0.f : neg_inf();
      for (int i = 0; i < D && n > 0; ++i) {
        const int d = shm_durations[i];
        const int t_prev = t - d;
        if (t_prev < 0) continue;
        if (d > 0) {
          const int64_t src = static_cast<int64_t>(t_prev) * max_U + u;
          a = log_add(a, alpha[src] + blank[src] + dur[src * D + i]);
        }
        if (u > 0) {
          const int64_t src = static_cast<int64_t>(t_prev) * max_U + u - 1;
          a = log_add(a, alpha[src] + label[src] + dur[src * D + i]);
        }
      }
      alpha[static_cast<int64_t>(t) * max_U + u] = a;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float ll = neg_inf();
    for (int i = 0; i < D; ++i) {
      const int d = shm_durations[i];
      const int t_prev = T - d;
      if (d == 0 || t_prev < 0) continue;
      const int64_t src = static_cast<int64_t>(t_prev) * max_U + U;
      ll = log_add(ll, alpha[src] + blank[src] + dur[src * D + i]);
    }
    log_ll[b] = ll;
  }
}

__global__ void tdt_beta_kernel(const float *__restrict__ blank_lp,
                                const float *__restrict__ label_lp,
                                const float *__restrict__ dur_lp,
                                const int *__restrict__ source_lengths,
                                const int *__restrict__ target_lengths,
                                const int *__restrict__ durations, int max_T,
                                int max_U, int D, float *__restrict__ betas,
                                float *__restrict__ log_ll) {
  __shared__ int shm_durations[kMaxDurations];

  const int b = blockIdx.x;
  int T, U;
  sample_lengths(source_lengths, target_lengths, b, max_T, max_U, T, U);
  load_durations(durations, D, shm_durations);

  const int64_t sample = static_cast<int64_t>(b) * max_T * max_U;
  const float *blank = blank_lp + sample;
  const float *label = label_lp + sample;
  const float *dur = dur_lp + sample * D;
  float *beta = betas + sample;

  for (int n = T + U - 1; n >= 0; --n) {
    const int u_min = max(0, n - T + 1);
    const int u_max = min(n, U);
    for (int u = u_min + threadIdx.x; u <= u_max; u += blockDim.x) {
      const int t = n - u;
      const int64_t node = static_cast<int64_t>(t) * max_U + u;
      float bt = neg_inf();
      for (int i = 0; i < D; ++i) {
        const int d = shm_durations[i];
        const int t_next = t + d;
        const float lp_dur = dur[node * D + i];
        if (d > 0) {
          if (t_next < T) {
            const int64_t dst = static_cast<int64_t>(t_next) * max_U + u;
            bt = log_add(bt, beta[dst] + blank[node] + lp_dur);
          } else if (t_next == T && u == U) {
            bt = log_add(bt, blank[node] + lp_dur);
          }
        }
        if (u < U && t_next < T) {
          const int64_t dst = static_cast<int64_t>(t_next) * max_U + u + 1;
          bt = log_add(bt, beta[dst] + label[node] + lp_dur);
        }
      }
      beta[node] = bt;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    log_ll[b] = T > 0 ? beta[0] : neg_inf();
  }
}

int lattice_block_size(int64_t max_U) {
  int threads = 32;
  while (threads < 1024 && threads < max_U) threads *= 2;
  return threads;
}

void check_lattice_args(torch::Tensor const &blank_lp,
                        torch::Tensor const &label_lp,
                        torch::Tensor const &dur_lp,
                        torch::Tensor const &source_lengths,
                        torch::Tensor const &target_lengths,
                        torch::Tensor const &durations,
                        torch::Tensor const &out, torch::Tensor const &log_ll) {
  TORCH_CHECK(blank_lp.is_cuda(), "blank_lp must be a CUDA tensor");
  TORCH_CHECK(blank_lp.dim() == 3, "blank_lp must have shape (batch, T, U+1)");
  for (auto const *x : {&blank_lp, &label_lp, &dur_lp, &out, &log_ll}) {
    TORCH_CHECK(x->device() == blank_lp.device(),
                "all tensors must be on the same device");
    TORCH_CHECK(x->scalar_type() == torch::kFloat32,
                "lattice tensors must be float32");
    TORCH_CHECK(x->is_contiguous(), "lattice tensors must be contiguous");
  }
  for (auto const *x : {&source_lengths, &target_lengths, &durations}) {
    TORCH_CHECK(x->device() == blank_lp.device(),
                "all tensors must be on the same device");
    TORCH_CHECK(x->scalar_type() == torch::kInt32,
                "lengths and durations must be int32");
    TORCH_CHECK(x->is_contiguous(), "lengths and durations must be contiguous");
  }
  const int64_t B = blank_lp.size(0);
  const int64_t nodes = blank_lp.numel();
  TORCH_CHECK(label_lp.numel() == nodes && out.numel() == nodes,
              "lattice tensors must have shape (batch, T, U+1)");
  TORCH_CHECK(dur_lp.numel() == nodes * durations.numel(),
              "dur_lp must have shape (batch, T, U+1, num_durations)");
  TORCH_CHECK(source_lengths.numel() == B && target_lengths.numel() == B &&
                  log_ll.numel() == B,
              "lengths and log_ll must have shape (batch,)");
  TORCH_CHECK(durations.numel() <= kMaxDurations, "at most ", kMaxDurations,
              " durations are supported");
}

}  // namespace

}  // namespace tdt_loss

void tdt_loss_fwd(torch::Tensor const &blank_lp, torch::Tensor const &label_lp,
                  torch::Tensor const &dur_lp,
                  torch::Tensor const &source_lengths,
                  torch::Tensor const &target_lengths,
                  torch::Tensor const &durations, torch::Tensor &alphas,
                  torch::Tensor &log_ll) {
  using namespace tdt_loss;
  check_lattice_args(blank_lp, label_lp, dur_lp, source_lengths,
                     target_lengths, durations, alphas, log_ll);
  const int64_t B = blank_lp.size(0);
  if (B == 0) return;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(blank_lp));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  tdt_alpha_kernel<<<B, lattice_block_size(blank_lp.size(2)), 0, stream>>>(
      blank_lp.data_ptr<float>(), label_lp.data_ptr<float>(),
      dur_lp.data_ptr<float>(), source_lengths.data_ptr<int>(),
      target_lengths.data_ptr<int>(), durations.data_ptr<int>(),
      blank_lp.size(1), blank_lp.size(2), durations.numel(),
      alphas.data_ptr<float>(), log_ll.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tdt_loss_bwd(torch::Tensor const &blank_lp, torch::Tensor const &label_lp,
                  torch::Tensor const &dur_lp,
                  torch::Tensor const &source_lengths,
                  torch::Tensor const &target_lengths,
                  torch::Tensor const &durations, torch::Tensor &betas,
                  torch::Tensor &log_ll) {
  using namespace tdt_loss;
  check_lattice_args(blank_lp, label_lp, dur_lp, source_lengths,
                     target_lengths, durations, betas, log_ll);
  const int64_t B = blank_lp.size(0);
  if (B == 0) return;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(blank_lp));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  tdt_beta_kernel<<<B, lattice_block_size(blank_lp.size(2)), 0, stream>>>(
      blank_lp.data_ptr<float>(), label_lp.data_ptr<float>(),
      dur_lp.data_ptr<float>(), source_lengths.data_ptr<int>(),
      target_lengths.data_ptr<int>(), durations.data_ptr<int>(),
      blank_lp.size(1), blank_lp.size(2), durations.numel(),
      betas.data_ptr<float>(), log_ll.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
