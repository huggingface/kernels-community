#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <mutex>
#include "AllocationInfo.h"

namespace kernels::xpu {
namespace dpcpp {

/// Device Allocator
void emptyCacheInDevAlloc();

DeviceStats getDeviceStatsFromDevAlloc(at::DeviceIndex device_index);

void resetAccumulatedStatsInDevAlloc(at::DeviceIndex device_index);

void resetPeakStatsInDevAlloc(at::DeviceIndex device_index);

std::vector<SegmentInfo> snapshotOfDevAlloc();

at::Allocator* getDeviceAllocator();

void cacheInfoFromDevAlloc(
    at::DeviceIndex deviceIndex,
    size_t* cachedAndFree,
    size_t* largestBlock);

void* getBaseAllocationFromDevAlloc(void* ptr, size_t* size);

void dumpMemoryStatusFromDevAlloc(at::DeviceIndex device_index);

std::mutex* getFreeMutexOfDevAlloc();

} // namespace dpcpp
} // namespace kernels::xpu
