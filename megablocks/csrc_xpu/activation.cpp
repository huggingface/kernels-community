#include <sycl/sycl.hpp>
#include <cmath>
#include <algorithm>
#include "utils.h"
#include "dispatch_utils.h"

#define VLLM_LDG(arg) *(arg)

namespace vllm {

template <typename T>
inline T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + sycl::exp((float)-x)));
}

template <typename T>
inline T gelu_fast_kernel(const T& x) {
  const float f = (float)x;
  const T t =
      (T)tanhf(((T)(f * 0.79788456f)) * (((T)1.0) + (T)(0.044715f * f) * x));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
inline T gelu_new_kernel(const T& x) {
  // 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
  const float x3 = (float)(x * x * x);
  const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
inline T gelu_quick_kernel(const T& x) {
  // x * sigmoid(1.702 * x)
  return (T)(((float)x) / (1.0f + (T)sycl::exp(-1.702f * (float)x)));
}

template <typename T>
inline T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + sycl::erf(f * ALPHA)));
}

template <typename T>
inline T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = (float)x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + sycl::tanh(inner)));
}

template <
    typename scalar_t,
    scalar_t (*ACT_FN)(const scalar_t&),
    bool act_first>
inline scalar_t compute(const scalar_t& x, const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
class act_kernel {
 public:
  act_kernel(
      scalar_t* __restrict__ out,          // [..., d]
      const scalar_t* __restrict__ input,  // [..., d]
      const int d)
      : out_(out), input_(input), d_(d) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    const int64_t token_idx = item_ct1.get_group(2);
    for (int64_t idx = item_ct1.get_local_id(2); idx < d_;
         idx += item_ct1.get_local_range(2)) {
      const scalar_t x = input_[token_idx * d_ + idx];
      out_[token_idx * d_ + idx] = ACT_FN(x);
    }
  }

 private:
  scalar_t* __restrict__ out_;          // [..., d]
  const scalar_t* __restrict__ input_;  // [..., d]
  const int d_;
};

template <
    typename scalar_t,
    scalar_t (*ACT_FN)(const scalar_t&),
    bool act_first>
class act_and_mul_kernel {
 public:
  act_and_mul_kernel(
      scalar_t* __restrict__ out,          // [..., d]
      const scalar_t* __restrict__ input,  // [..., 2, d]
      const int d)
      : out_(out), input_(input), d_(d) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    const int64_t token_idx = item_ct1.get_group(2);
    for (int64_t idx = item_ct1.get_local_id(2); idx < d_;
         idx += item_ct1.get_local_range(2)) {
      const scalar_t x = input_[token_idx * 2 * d_ + idx];
      const scalar_t y = input_[token_idx * 2 * d_ + d_ + idx];
      out_[token_idx * d_ + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
    }
  }

 private:
  scalar_t* __restrict__ out_;          // [..., d]
  const scalar_t* __restrict__ input_;  // [..., 2, d]
  const int d_;
};

template <typename T>
[[intel::device_indirectly_callable]] inline __attribute__((always_inline)) T
swigluoai_and_mul(const T& gate, const T& up, float alpha, float limit) {
  // clamp gate: min=None, max=limit
  const float gate_f = (float)gate;
  const float clamped_gate = gate_f > limit ? limit : gate_f;

  // clamp up: min=-limit, max=limit
  const float up_f = (float)up;
  const float clamped_up =
      up_f > limit ? limit : (up_f < -limit ? -limit : up_f);

  // glu = gate * sigmoid(gate * alpha)
  const float sigmoid_val = 1.0f / (1.0f + sycl::exp(-clamped_gate * alpha));
  const float glu = clamped_gate * sigmoid_val;

  // (up + 1) * glu
  return (T)((clamped_up + 1.0f) * glu);
}

template <
    typename scalar_t,
    scalar_t (*ACT_FN)(
        const scalar_t&, const scalar_t&, const float, const float)>
class swigluoai_and_mul_kernel {
 public:
  swigluoai_and_mul_kernel(
      scalar_t* __restrict__ out,          // [..., d]
      const scalar_t* __restrict__ input,  // [..., 2, d]
      const int d,
      const float alpha,
      const float limit)
      : out(out), input(input), d(d), alpha(alpha), limit(limit) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t token_idx = item.get_group(0);
    for (int64_t idx = item.get_local_id(0); idx < d;
         idx += item.get_local_range(0)) {
      // gate = x[..., ::2]  (even indices)
      const scalar_t gate = VLLM_LDG(&input[token_idx * 2 * d + 2 * idx]);
      // up = x[..., 1::2]   (odd indices)
      const scalar_t up = VLLM_LDG(&input[token_idx * 2 * d + 2 * idx + 1]);

      out[token_idx * d + idx] = ACT_FN(gate, up, alpha, limit);
    }
  }

 private:
  scalar_t* out;          // [..., d]
  const scalar_t* input;  // [..., topk, d]
  const int d;
  const float alpha;
  const float limit;
};

}  // namespace vllm

// Launch activation and gating kernel.
// Use ACT_FIRST (bool) indicating whether to apply the activation function
// first.
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)     \
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;   \
  int d = input.size(-1) / 2;                                \
  int64_t num_tokens = input.numel() / input.size(-1);       \
  sycl::range<3> grid(1, 1, num_tokens);                     \
  sycl::range<3> block(1, 1, std::min(d, 1024));             \
  if (num_tokens == 0) {                                     \
    return;                                                  \
  }                                                          \
  auto out_ptr = out.data_ptr<scalar_t>();                   \
  auto input_ptr = input.data_ptr<scalar_t>();               \
  at::DeviceGuard device_guard(input.device());              \
  auto& queue = vllm::xpu::vllmGetQueue();                   \
  queue.submit([&](sycl::handler& cgh) {                     \
    cgh.parallel_for(                                        \
        sycl::nd_range<3>(grid * block, block),              \
        vllm::act_and_mul_kernel<sycl_t, KERNEL, ACT_FIRST>( \
            (sycl_t*)out_ptr, (sycl_t*)input_ptr, d));       \
  });

void silu_and_mul(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input)  // [..., 2 * d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "silu_and_mul", [&] {
    LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, true);
  });
}

void mul_and_silu(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input)  // [..., 2 * d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mul_and_silu", [&] {
    LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, false);
  });
}

void gelu_and_mul(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input)  // [..., 2 * d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_and_mul", [&] {
    LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_kernel, true);
  });
}

void gelu_tanh_and_mul(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input)  // [..., 2 * d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_tanh_and_mul", [&] {
    LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_tanh_kernel, true);
  });
}

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                   \
  using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type; \
  int d = input.size(-1);                                  \
  int64_t num_tokens = input.numel() / input.size(-1);     \
  sycl::range<3> grid(1, 1, num_tokens);                   \
  sycl::range<3> block(1, 1, std::min(d, 1024));           \
  if (num_tokens == 0) {                                   \
    return;                                                \
  }                                                        \
  auto out_ptr = out.data_ptr<scalar_t>();                 \
  auto input_ptr = input.data_ptr<scalar_t>();             \
  at::DeviceGuard device_guard(input.device());            \
  auto& queue = vllm::xpu::vllmGetQueue();                 \
  queue.submit([&](sycl::handler& cgh) {                   \
    cgh.parallel_for(                                      \
        sycl::nd_range<3>(grid * block, block),            \
        vllm::act_kernel<sycl_t, KERNEL>(                  \
            (sycl_t*)out_ptr, (sycl_t*)input_ptr, d));     \
  });

#define LAUNCH_SWIGLUOAI_AND_MUL(KERNEL, ALPHA, LIMIT)                    \
  int d = input.size(-1) / 2;                                             \
  int64_t num_tokens = input.numel() / input.size(-1);                    \
  sycl::range<1> grid(num_tokens);                                        \
  sycl::range<1> block(std::min(d, 1024));                                \
  at::DeviceGuard device_guard(input.device());                           \
  auto& queue = vllm::xpu::vllmGetQueue();                                \
  VLLM_DISPATCH_FLOATING_TYPES(                                           \
      input.scalar_type(), "clamp_swiglu_kernel_with_params", [&] {       \
        queue.submit([&](sycl::handler& cgh) {                            \
          cgh.parallel_for(                                               \
              sycl::nd_range<1>(grid * block, block),                     \
              vllm::swigluoai_and_mul_kernel<scalar_t, KERNEL<scalar_t>>( \
                  out.data_ptr<scalar_t>(),                               \
                  input.data_ptr<scalar_t>(),                             \
                  d,                                                      \
                  ALPHA,                                                  \
                  LIMIT));                                                \
        });                                                               \
      });

void gelu_new(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input)  // [..., d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_new", [&] {
    LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
  });
}

void gelu_fast(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input)  // [..., d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_fast", [&] {
    LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
  });
}

void gelu_quick(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input)  // [..., d]
{
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_quick", [&] {
    LAUNCH_ACTIVATION_KERNEL(vllm::gelu_quick_kernel);
  });
}

void swigluoai_and_mul(
    torch::Tensor& out,    // [..., d]
    torch::Tensor& input,  // [..., 2 * d]
    double alpha,
    double limit) {
  LAUNCH_SWIGLUOAI_AND_MUL(vllm::swigluoai_and_mul, alpha, limit);
}
