#include <torch/all.h>

#if defined(CUDA_KERNEL)
#include <c10/cuda/CUDAGuard.h>
#elif defined(XPU_KERNEL)
#include <c10/core/DeviceGuard.h>
#endif

#include "registration.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().type() == torch::kCUDA || x.device().type() == torch::kXPU, #x " must be on CUDA or XPU")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

void _apply_rotary(torch::Tensor const &x1, torch::Tensor const &x2,
                       torch::Tensor const &cos, torch::Tensor const &sin,
                       torch::Tensor &out1, torch::Tensor &out2,
                       bool const conj);

void apply_rotary(torch::Tensor const &x1, torch::Tensor const &x2,
                  torch::Tensor const &cos, torch::Tensor const &sin,
                  torch::Tensor &out1, torch::Tensor &out2,
                  bool const conj) {
    CHECK_DEVICE(x1); CHECK_DEVICE(x2);
    CHECK_DEVICE(cos); CHECK_DEVICE(sin);
    CHECK_DEVICE(out1); CHECK_DEVICE(out1);
    TORCH_CHECK(x1.dtype() == x2.dtype());
    TORCH_CHECK(cos.dtype() == sin.dtype());
    TORCH_CHECK(out1.dtype() == out2.dtype());
    TORCH_CHECK(x1.dtype() == cos.dtype());
    TORCH_CHECK(x1.dtype() == out1.dtype());
    TORCH_CHECK(x1.sizes() == x2.sizes());
    TORCH_CHECK(cos.sizes() == sin.sizes());
    TORCH_CHECK(out1.sizes() == out2.sizes());

#if defined(CUDA_KERNEL)
    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{x1.device()};
#elif defined(XPU_KERNEL)
    c10::DeviceGuard device_guard{x1.device()};
#endif
    _apply_rotary(x1, x2, cos, sin, out1, out2, conj);
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("apply_rotary(Tensor x1, Tensor x2, Tensor cos, Tensor sin,"
          "Tensor! out1, Tensor! out2, bool conj) -> ()");
#if defined(CUDA_KERNEL)
    ops.impl("apply_rotary", torch::kCUDA, &apply_rotary);
#elif defined(XPU_KERNEL)
    ops.impl("apply_rotary", torch::kXPU, &apply_rotary);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
