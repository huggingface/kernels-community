#include <ATen/xpu/CachingHostAllocator.h>
#include <Memory.h>
#include <Utils.h>

namespace kernels::xpu {
namespace dpcpp {

// For async H2D copy, it wll check the src is allocated by SYCL API or
// system. For system allocated memory, the H2D will firstly copy data
// from the pageable memory to the unpageable memory, then execute H2D.
void memcpyHostToDevice(
    void* dst,
    const void* src,
    size_t n_bytes,
    bool async,
    const void* hctx,
    bool is_pinned = false) {
  if (n_bytes == 0)
    return;

  auto& queue = dpcppGetCurrentQueue();
  sycl::event e;

  if (!async) {
    e = queue.memcpy(dst, src, n_bytes);
    e.wait();
  } else {
    if (is_pinned) {
      e = queue.memcpy(dst, src, n_bytes);
      at::xpu::CachingHostAllocator_recordEvent(
          const_cast<void*>(src),
          const_cast<void*>(hctx),
          at::xpu::getCurrentXPUStream());
    } else {
      // Using stage memory for async copy to avoid incorrect free of src host
      // memory before async H2D copy. E.g. memory allocated by CPU tensor
      // factory won't be cached in CPU allocator. When host memory is freed
      // with CPU tensor dtor at the end of train main loop, but the
      // corresponding H2D copy might not have been executed yet.
      auto stage_mem_dptr = at::xpu::HostAlloc(n_bytes);
      void* stage_mem = stage_mem_dptr.get();
      TORCH_CHECK(
          stage_mem, "Fail to allocate host memory from XPU HostAllocator");
      std::memcpy(stage_mem, src, n_bytes);
      e = queue.memcpy(dst, stage_mem, n_bytes);
      at::xpu::CachingHostAllocator_recordEvent(
          stage_mem,
          stage_mem_dptr.get_context(),
          at::xpu::getCurrentXPUStream());
    }
  }
}

void memcpyDeviceToHost(
    void* dst,
    const void* src,
    size_t n_bytes,
    bool async,
    const void* hctx,
    bool is_pinned = false) {
  if (n_bytes == 0)
    return;

  auto& queue = dpcppGetCurrentQueue();
  auto e = queue.memcpy(dst, src, n_bytes);

  if (!async) {
    e.wait();
  } else if (is_pinned) {
    at::xpu::CachingHostAllocator_recordEvent(
        dst, const_cast<void*>(hctx), at::xpu::getCurrentXPUStream());
  }
}

void memcpyDeviceToDevice(
    void* dst,
    const void* src,
    size_t n_bytes,
    bool async) {
  if (n_bytes == 0)
    return;

  auto& queue = dpcppGetCurrentQueue();
  auto e = queue.memcpy(dst, src, n_bytes);

  if (!async) {
    e.wait();
  }
}

void memsetDevice(void* dst, int value, size_t n_bytes, bool async) {
  auto& queue = dpcppGetCurrentQueue();
  auto e = queue.memset(dst, value, n_bytes);

  if (!async) {
    e.wait();
  }
}


void dpcppMemcpy(
    void* dst,
    const void* src,
    size_t n_bytes,
    dpcppMemcpyKind kind) {
  switch (kind) {
    case HostToDevice:
      // for synchronous copy, the context of host data pointer is unnecessary.
      memcpyHostToDevice(dst, src, n_bytes, false, nullptr);
      break;
    case DeviceToHost:
      // for synchronous copy, the context of host data pointer is unnecessary.
      memcpyDeviceToHost(dst, src, n_bytes, false, nullptr);
      break;
    case DeviceToDevice:
      memcpyDeviceToDevice(dst, src, n_bytes, false);
      break;
    default:
      TORCH_CHECK(false, "Unknown dpcpp memory kind");
  }
}

void dpcppMemcpyAsync(
    void* dst,
    const void* src,
    size_t n_bytes,
    dpcppMemcpyKind kind,
    const void* hctx) {
  switch (kind) {
    case HostToDevice:
      memcpyHostToDevice(
          dst,
          src,
          n_bytes,
          true,
          hctx,
          at::detail::getXPUHooks().isPinnedPtr(src));
      break;
    case DeviceToHost:
      memcpyDeviceToHost(
          dst,
          src,
          n_bytes,
          true,
          hctx,
          at::detail::getXPUHooks().isPinnedPtr(dst));
      break;
    case DeviceToDevice:
      memcpyDeviceToDevice(dst, src, n_bytes, true);
      break;
    default:
      TORCH_CHECK(false, "Unknown dpcpp memory kind");
  }
}

void dpcppMemset(void* data, int value, size_t n_bytes) {
  memsetDevice(data, value, n_bytes, false);
}

void dpcppMemsetAsync(void* data, int value, size_t n_bytes) {
  memsetDevice(data, value, n_bytes, true);
}


} // namespace dpcpp
} // namespace kernels::xpu
