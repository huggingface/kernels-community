#include <sycl/sycl.hpp>

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Dispatch.h>
#include <c10/xpu/XPUStream.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>

#include "activation_xpu.hpp"

namespace activation_xpu {

namespace {

// Aligned pack of `vec_size` elements, used to widen global memory accesses so
// that a single work item issues one wide load/store instead of several narrow
// ones. `sycl::vec` cannot be used here because `at::Half` and `at::BFloat16`
// are not valid SYCL vector element types.
template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) AlignedVec {
  scalar_t val[vec_size];
};

// Widest access that is still a natural vector width on Intel GPUs (128 bit).
template <typename scalar_t>
constexpr int kVecSize = 16 / sizeof(scalar_t);

bool is_vec_aligned(const void* ptr, int64_t bytes) {
  return reinterpret_cast<uintptr_t>(ptr) % bytes == 0;
}

// Activation and gating kernel. Each work-group handles one token, the work
// items in the group stride over the `d` elements of that token.
template <typename scalar_t, typename act_fn_t, bool act_first>
struct ActAndMulKernel {
  ActAndMulKernel(scalar_t* out, const scalar_t* input, int64_t d,
                  act_fn_t act_fn)
      : out_(out), input_(input), d_(d), act_fn_(act_fn) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t token_idx = item.get_group(0);
    const int64_t local_size = item.get_local_range(0);
    const scalar_t* input_row = input_ + token_idx * 2 * d_;
    scalar_t* out_row = out_ + token_idx * d_;
    for (int64_t idx = item.get_local_id(0); idx < d_; idx += local_size) {
      const scalar_t x = input_row[idx];
      const scalar_t y = input_row[d_ + idx];
      out_row[idx] = act_first ? act_fn_(x) * y : x * act_fn_(y);
    }
  }

 private:
  scalar_t* out_;
  const scalar_t* input_;
  int64_t d_;
  act_fn_t act_fn_;
};

// Vectorized variant of `ActAndMulKernel`, used when `d` is a multiple of the
// vector width and both halves of the input row are suitably aligned.
template <typename scalar_t, typename act_fn_t, bool act_first, int vec_size>
struct ActAndMulVecKernel {
  using vec_t = AlignedVec<scalar_t, vec_size>;

  ActAndMulVecKernel(scalar_t* out, const scalar_t* input, int64_t d,
                     act_fn_t act_fn)
      : out_(out), input_(input), d_(d), act_fn_(act_fn) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t token_idx = item.get_group(0);
    const int64_t local_size = item.get_local_range(0);
    const int64_t d_vec = d_ / vec_size;
    const vec_t* input_row =
        reinterpret_cast<const vec_t*>(input_ + token_idx * 2 * d_);
    vec_t* out_row = reinterpret_cast<vec_t*>(out_ + token_idx * d_);
    for (int64_t idx = item.get_local_id(0); idx < d_vec; idx += local_size) {
      const vec_t x = input_row[idx];
      const vec_t y = input_row[d_vec + idx];
      vec_t result;
#pragma unroll
      for (int i = 0; i < vec_size; ++i) {
        result.val[i] = act_first ? act_fn_(x.val[i]) * y.val[i]
                                  : x.val[i] * act_fn_(y.val[i]);
      }
      out_row[idx] = result;
    }
  }

 private:
  scalar_t* out_;
  const scalar_t* input_;
  int64_t d_;
  act_fn_t act_fn_;
};

// Element-wise activation kernel. The tensor is treated as flat, work items
// stride over the whole element range.
template <typename scalar_t, typename act_fn_t>
struct ActivationKernel {
  ActivationKernel(scalar_t* out, const scalar_t* input, int64_t numel,
                   act_fn_t act_fn)
      : out_(out), input_(input), numel_(numel), act_fn_(act_fn) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t stride = item.get_global_range(0);
    for (int64_t idx = item.get_global_id(0); idx < numel_; idx += stride) {
      out_[idx] = act_fn_(input_[idx]);
    }
  }

 private:
  scalar_t* out_;
  const scalar_t* input_;
  int64_t numel_;
  act_fn_t act_fn_;
};

// Vectorized variant of `ActivationKernel`.
template <typename scalar_t, typename act_fn_t, int vec_size>
struct ActivationVecKernel {
  using vec_t = AlignedVec<scalar_t, vec_size>;

  ActivationVecKernel(scalar_t* out, const scalar_t* input, int64_t num_vec,
                      act_fn_t act_fn)
      : out_(reinterpret_cast<vec_t*>(out)),
        input_(reinterpret_cast<const vec_t*>(input)),
        num_vec_(num_vec),
        act_fn_(act_fn) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t stride = item.get_global_range(0);
    for (int64_t idx = item.get_global_id(0); idx < num_vec_; idx += stride) {
      const vec_t x = input_[idx];
      vec_t result;
#pragma unroll
      for (int i = 0; i < vec_size; ++i) {
        result.val[i] = act_fn_(x.val[i]);
      }
      out_[idx] = result;
    }
  }

 private:
  vec_t* out_;
  const vec_t* input_;
  int64_t num_vec_;
  act_fn_t act_fn_;
};

// Function objects are used instead of function pointers, since taking the
// address of a function is not supported in SYCL device code.
//
// The functors take and return `scalar_t` so that each of them controls where
// intermediate results are rounded. Most compute in `float` throughout, but
// `gelu_new` and `gelu_fast` deliberately round intermediates to `scalar_t`,
// matching the CUDA implementation so that results stay consistent across
// backends.
template <typename scalar_t>
struct SiluFn {
  scalar_t operator()(scalar_t x) const {
    const float f = static_cast<float>(x);
    return static_cast<scalar_t>(f / (1.0f + sycl::exp(-f)));
  }
};

template <typename scalar_t>
struct GeluFn {
  // Equivalent to PyTorch GELU with 'none' approximation.
  scalar_t operator()(scalar_t x) const {
    constexpr float kAlpha = M_SQRT1_2;
    const float f = static_cast<float>(x);
    return static_cast<scalar_t>(f * 0.5f * (1.0f + sycl::erf(f * kAlpha)));
  }
};

template <typename scalar_t>
struct GeluTanhFn {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  scalar_t operator()(scalar_t x) const {
    constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5f;
    constexpr float kKappa = 0.044715f;
    const float f = static_cast<float>(x);
    const float x_cube = f * f * f;
    const float inner = kBeta * (f + kKappa * x_cube);
    return static_cast<scalar_t>(0.5f * f * (1.0f + sycl::tanh(inner)));
  }
};

template <typename scalar_t>
struct GeluNewFn {
  scalar_t operator()(scalar_t x) const {
    const float x_cube = static_cast<float>(x * x * x);
    const scalar_t inner = static_cast<scalar_t>(
        0.79788456f *
        static_cast<float>(x + static_cast<scalar_t>(0.044715f * x_cube)));
    const scalar_t t =
        static_cast<scalar_t>(sycl::tanh(static_cast<float>(inner)));
    return static_cast<scalar_t>(0.5f) * x * (static_cast<scalar_t>(1.0f) + t);
  }
};

template <typename scalar_t>
struct GeluFastFn {
  scalar_t operator()(scalar_t x) const {
    const float f = static_cast<float>(x);
    const scalar_t inner = static_cast<scalar_t>(f * 0.79788456f) *
                           (static_cast<scalar_t>(1.0f) +
                            static_cast<scalar_t>(0.044715f * f) * x);
    const scalar_t t =
        static_cast<scalar_t>(sycl::tanh(static_cast<float>(inner)));
    return static_cast<scalar_t>(0.5f) * x * (static_cast<scalar_t>(1.0f) + t);
  }
};

template <typename scalar_t>
struct GeluQuickFn {
  // x * sigmoid(1.702 * x)
  scalar_t operator()(scalar_t x) const {
    const float f = static_cast<float>(x);
    return static_cast<scalar_t>(f / (1.0f + sycl::exp(-1.702f * f)));
  }
};

template <typename scalar_t>
struct FatreluFn {
  explicit FatreluFn(float threshold) : threshold_(threshold) {}

  scalar_t operator()(scalar_t x) const {
    const float f = static_cast<float>(x);
    return static_cast<scalar_t>(f > threshold_ ? f : 0.0f);
  }

 private:
  float threshold_;
};

int64_t work_group_size(const sycl::queue& queue, int64_t d) {
  const int64_t max_work_group_size =
      queue.get_device().get_info<sycl::info::device::max_work_group_size>();
  return std::max<int64_t>(1, std::min<int64_t>(d, max_work_group_size));
}

// Number of work-groups that saturates the device, used to size the grid of
// the flat element-wise kernel.
int64_t max_work_groups(sycl::queue& queue) {
  const int64_t compute_units =
      queue.get_device().get_info<sycl::info::device::max_compute_units>();
  return std::max<int64_t>(1, compute_units * 32);
}

void check_act_and_mul_inputs(const torch::Tensor& out,
                              const torch::Tensor& input) {
  TORCH_CHECK(input.device().is_xpu(), "input must be an XPU tensor");
  TORCH_CHECK(out.device() == input.device(),
              "out and input must be on the same device");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(out.scalar_type() == input.scalar_type(),
              "out and input must have the same dtype");
  TORCH_CHECK(input.size(-1) % 2 == 0,
              "the last dimension of input must be even");
  TORCH_CHECK(out.size(-1) == input.size(-1) / 2,
              "the last dimension of out must be half of that of input");
}

void check_activation_inputs(const torch::Tensor& out,
                             const torch::Tensor& input) {
  TORCH_CHECK(input.device().is_xpu(), "input must be an XPU tensor");
  TORCH_CHECK(out.device() == input.device(),
              "out and input must be on the same device");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(out.scalar_type() == input.scalar_type(),
              "out and input must have the same dtype");
  TORCH_CHECK(out.sizes() == input.sizes(),
              "out and input must have the same shape");
}

template <bool act_first, template <typename> class ActFn, typename... Args>
void launch_act_and_mul(torch::Tensor& out, const torch::Tensor& input,
                        Args... args) {
  check_act_and_mul_inputs(out, input);

  const int64_t d = input.size(-1) / 2;
  const int64_t num_tokens = input.numel() / input.size(-1);
  if (num_tokens == 0 || d == 0) {
    return;
  }

  const at::OptionalDeviceGuard device_guard(at::device_of(input));
  sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                  input.scalar_type(), "act_and_mul_kernel_xpu",
                                  [&] {
    using act_fn_t = ActFn<scalar_t>;
    const act_fn_t act_fn(args...);
    constexpr int vec_size = kVecSize<scalar_t>;
    constexpr int64_t vec_bytes = sizeof(scalar_t) * vec_size;
    scalar_t* out_ptr = out.data_ptr<scalar_t>();
    const scalar_t* input_ptr = input.const_data_ptr<scalar_t>();

    const bool vectorize = d % vec_size == 0 &&
                           is_vec_aligned(out_ptr, vec_bytes) &&
                           is_vec_aligned(input_ptr, vec_bytes);

    if (vectorize) {
      const int64_t local_size = work_group_size(queue, d / vec_size);
      queue.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(num_tokens * local_size),
                            sycl::range<1>(local_size)),
          ActAndMulVecKernel<scalar_t, act_fn_t, act_first, vec_size>(
              out_ptr, input_ptr, d, act_fn));
    } else {
      const int64_t local_size = work_group_size(queue, d);
      queue.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(num_tokens * local_size),
                            sycl::range<1>(local_size)),
          ActAndMulKernel<scalar_t, act_fn_t, act_first>(out_ptr, input_ptr, d,
                                                         act_fn));
    }
  });
}

template <template <typename> class ActFn, typename... Args>
void launch_activation(torch::Tensor& out, const torch::Tensor& input,
                       Args... args) {
  check_activation_inputs(out, input);

  const int64_t numel = input.numel();
  if (numel == 0) {
    return;
  }

  const at::OptionalDeviceGuard device_guard(at::device_of(input));
  sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                  input.scalar_type(), "activation_kernel_xpu",
                                  [&] {
    using act_fn_t = ActFn<scalar_t>;
    const act_fn_t act_fn(args...);
    constexpr int vec_size = kVecSize<scalar_t>;
    constexpr int64_t vec_bytes = sizeof(scalar_t) * vec_size;
    scalar_t* out_ptr = out.data_ptr<scalar_t>();
    const scalar_t* input_ptr = input.const_data_ptr<scalar_t>();

    const bool vectorize = numel % vec_size == 0 &&
                           is_vec_aligned(out_ptr, vec_bytes) &&
                           is_vec_aligned(input_ptr, vec_bytes);

    const int64_t work_items = vectorize ? numel / vec_size : numel;
    const int64_t local_size = work_group_size(queue, work_items);
    const int64_t num_groups =
        std::min<int64_t>((work_items + local_size - 1) / local_size,
                          max_work_groups(queue));
    const sycl::nd_range<1> range(sycl::range<1>(num_groups * local_size),
                                  sycl::range<1>(local_size));

    if (vectorize) {
      queue.parallel_for(
          range, ActivationVecKernel<scalar_t, act_fn_t, vec_size>(
                     out_ptr, input_ptr, work_items, act_fn));
    } else {
      queue.parallel_for(range, ActivationKernel<scalar_t, act_fn_t>(
                                    out_ptr, input_ptr, numel, act_fn));
    }
  });
}

}  // namespace

void silu_and_mul(torch::Tensor& out, const torch::Tensor& input) {
  launch_act_and_mul<true, SiluFn>(out, input);
}

void mul_and_silu(torch::Tensor& out, const torch::Tensor& input) {
  launch_act_and_mul<false, SiluFn>(out, input);
}

void gelu_and_mul(torch::Tensor& out, const torch::Tensor& input) {
  launch_act_and_mul<true, GeluFn>(out, input);
}

void gelu_tanh_and_mul(torch::Tensor& out, const torch::Tensor& input) {
  launch_act_and_mul<true, GeluTanhFn>(out, input);
}

void fatrelu_and_mul(torch::Tensor& out, const torch::Tensor& input,
                     double threshold) {
  launch_act_and_mul<true, FatreluFn>(out, input,
                                      static_cast<float>(threshold));
}

void gelu_new(torch::Tensor& out, const torch::Tensor& input) {
  launch_activation<GeluNewFn>(out, input);
}

void gelu_fast(torch::Tensor& out, const torch::Tensor& input) {
  launch_activation<GeluFastFn>(out, input);
}

void gelu_quick(torch::Tensor& out, const torch::Tensor& input) {
  launch_activation<GeluQuickFn>(out, input);
}

void gelu(torch::Tensor& out, const torch::Tensor& input) {
  launch_activation<GeluFn>(out, input);
}

void gelu_tanh(torch::Tensor& out, const torch::Tensor& input) {
  launch_activation<GeluTanhFn>(out, input);
}

void silu(torch::Tensor& out, const torch::Tensor& input) {
  launch_activation<SiluFn>(out, input);
}

}  // namespace activation_xpu
