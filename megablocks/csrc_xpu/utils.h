#pragma once
// Standard library headers that used to be included transitively through
// <torch/all.h>.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/Float8_e4m3fn.h>
#include <torch/headeronly/util/Float8_e5m2.h>
#include <torch/headeronly/util/Half.h>
#include <torch/headeronly/util/complex.h>
#include <sycl/sycl.hpp>

// Define for older Torch versions.
#ifndef TORCH_VERSION_2_13_0
#define TORCH_VERSION_2_13_0 (((0ULL + 2) << 56) | ((0ULL + 13) << 48))
#endif

#if TORCH_FEATURE_VERSION < TORCH_VERSION_2_13_0
// The stable ABI can only return the native XPU stream handle since Torch
// 2.13, so use the regular C++ API on older versions.
#include <c10/xpu/XPUStream.h>
#endif

using torch::headeronly::ScalarType;

#define CHECK_DEVICE(x)                                                   \
  STD_TORCH_CHECK(                                                        \
      x.device().type() == torch::stable::DeviceType::XPU,                \
      #x " must be on XPU")
#define CHECK_CONTIGUOUS(x) \
  STD_TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace vllm {
namespace xpu {

using torch::stable::accelerator::DeviceIndex;

// The SYCL queue backing Torch's current XPU stream on the given device
// (-1 is the current device).
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0
static inline sycl::queue& vllmGetQueue(DeviceIndex device_index = -1) {
  if (device_index == -1) {
    device_index = torch::stable::accelerator::getCurrentDeviceIndex();
  }
  // The queue is owned by Torch's stream pool, so the reference outlives the
  // temporary stream handle.
  void* handle =
      torch::stable::accelerator::getCurrentStream(device_index).nativeHandle();
  STD_TORCH_CHECK(handle != nullptr, "could not get the current XPU queue");
  return *static_cast<sycl::queue*>(handle);
}
#else
static inline sycl::queue& vllmGetQueue(DeviceIndex device_index = -1) {
  return c10::xpu::getCurrentXPUStream(device_index).queue();
}
#endif

namespace syclex = sycl::ext::oneapi::experimental;

static inline syclex::architecture
get_device_architecture(DeviceIndex device_index = -1) {
  return vllmGetQueue(device_index)
      .get_device()
      .get_info<syclex::info::device::architecture>();
}

static inline bool is_bmg(DeviceIndex device_index = -1) {
  return get_device_architecture(device_index) ==
         syclex::architecture::intel_gpu_bmg_g21;
}

static inline bool is_pvc(DeviceIndex device_index = -1) {
  return get_device_architecture(device_index) ==
         syclex::architecture::intel_gpu_pvc;
}

static inline bool is_xe2_arch(DeviceIndex device_index = -1) {
  auto arch = get_device_architecture(device_index);
  return arch == syclex::architecture::intel_gpu_bmg_g21 ||
         arch == syclex::architecture::intel_gpu_pvc;
}

static inline std::optional<std::string> getEnv(const char* name) {
  if (const char* val = std::getenv(name)) return val;
  return std::nullopt;
}

template <typename T>
struct SyclTypeTrait {
  using Type = T;
};

template <>
struct SyclTypeTrait<c10::Half> {
  using Type = sycl::half;
};

template <>
struct SyclTypeTrait<c10::BFloat16> {
  using Type = sycl::ext::oneapi::bfloat16;
};

template <typename T>
struct AccumulateType {
 private:
  static constexpr bool is_narrow_float =
      std::is_same_v<T, c10::Half> || std::is_same_v<T, c10::BFloat16> ||
      std::is_same_v<T, c10::Float8_e4m3fn> ||
      std::is_same_v<T, c10::Float8_e5m2>;

  static constexpr bool is_integer =
      std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> ||
      std::is_same_v<T, char> || std::is_same_v<T, int16_t> ||
      std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>;

  static constexpr bool is_complex = std::is_same_v<T, c10::complex<float>> ||
                                     std::is_same_v<T, c10::complex<double>>;

 public:
  using type = std::conditional_t<
      is_narrow_float,
      float,
      std::conditional_t<
          std::is_floating_point_v<T>,
          T,
          std::conditional_t<
              is_integer,
              int64_t,
              std::conditional_t<is_complex, T, T>>>>;
};

template <typename T>
using acc_type = typename AccumulateType<T>::type;

// aligned vector generates vectorized load/store on XPU
template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vec {
  scalar_t val[vec_size];

  scalar_t& operator[](int index) { return val[index]; }

  scalar_t const& operator[](int index) const { return val[index]; }
};

}  // namespace xpu

}  // namespace vllm
