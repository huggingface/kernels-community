#pragma once

#include <ATen/Config.h>
#include <c10/util/flat_hash_map.h>
#include <c10/xpu/XPUStream.h>
#include "Device.h"
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <vector>

using namespace dnnl;
using namespace kernels::xpu::dpcpp;
using stream_map =
    ska::flat_hash_map<c10::xpu::XPUStream, std::shared_ptr<dnnl::stream>>;

namespace kernels::xpu {
namespace oneDNN {

// Keep non-static and non-inline
bool set_onednn_verbose(int level);

static inline dnnl::memory dpcpp_onednn_memory(
    dnnl::memory::desc md,
    dnnl::engine& engine,
    void* ptr) {
  return dnnl::sycl_interop::make_memory(
      md,
      engine,
      dnnl::sycl_interop::memory_kind::usm,
      ptr == nullptr ? DNNL_MEMORY_ALLOCATE : ptr);
}

// GpuEngineManager singleton
struct GpuEngineManager {
  static GpuEngineManager& Instance(); // Singleton

  engine& get_engine(const at::Device& device) {
    TORCH_INTERNAL_ASSERT(device.type() == at::kXPU);
    TORCH_INTERNAL_ASSERT(device.index() < at::xpu::device_count());
    return *engine_pool[device.index()];
  }

  GpuEngineManager(GpuEngineManager const&) = delete;
  GpuEngineManager& operator=(GpuEngineManager const&) = delete;

 protected:
  GpuEngineManager() {
    int device_count = (int)at::xpu::device_count();
    TORCH_INTERNAL_ASSERT(device_count > 0);
    for (int i = 0; i < device_count; i++) {
      engine_pool.push_back(
          std::make_shared<dnnl::engine>(dnnl::sycl_interop::make_engine(
              at::xpu::get_raw_device(i), at::xpu::get_device_context())));
    }
  }
  ~GpuEngineManager() {}

 private:
  std::vector<std::shared_ptr<dnnl::engine>> engine_pool;
};

// GpuStreamManager singleton
struct GpuStreamManager {
  static GpuStreamManager& Instance(); // Singleton

  dnnl::stream& get_stream() {
    int device_index = at::xpu::current_device();
    TORCH_INTERNAL_ASSERT(device_index < at::xpu::device_count());
    auto stream = c10::xpu::getCurrentXPUStream(device_index);
    auto priority = stream.priority();
    if (stream_pool[device_index][priority].find(stream) ==
        stream_pool[device_index][priority].end()) {
      stream_pool[device_index][priority].emplace(
          stream,
          std::make_shared<dnnl::stream>(dnnl::sycl_interop::make_stream(
              GpuEngineManager::Instance().get_engine({at::kXPU, static_cast<at::DeviceIndex>(device_index)}),
              stream.queue())));
    }
    return *(stream_pool[device_index][priority].at(stream));
  }


  GpuStreamManager(GpuStreamManager const&) = delete;
  GpuStreamManager& operator=(GpuStreamManager const&) = delete;

 protected:
  GpuStreamManager() {
    int deviceCount = at::xpu::device_count();
    TORCH_INTERNAL_ASSERT(deviceCount > 0);
    stream_pool.clear();
    stream_pool.resize(deviceCount);
  }
  ~GpuStreamManager() {}

 private:
  std::vector<
      std::array<stream_map, c10::xpu::max_compile_time_stream_priorities>>
      stream_pool;
};

} // namespace oneDNN
} // namespace kernels::xpu
