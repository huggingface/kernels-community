/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

// sycl-tla 0.9.1+ (PyTorch 2.13) moved the Xe split-barrier helpers into
// <cute/util/xe_split_barrier.hpp> and switched barrier_arrive/barrier_wait to a
// SPIRVScope enum. Older versions only expose the int-based helpers, so define
// the enums and enum overloads here to keep both versions building.
#if defined(__has_include) && __has_include(<cute/util/xe_split_barrier.hpp>)
#include <cute/util/xe_split_barrier.hpp>
#else
enum SPIRVScope {
  ScopeCrossDevice = 0,
  ScopeDevice = 1,
  ScopeWorkgroup = 2,
  ScopeSubgroup = 3,
  ScopeInvocation = 4,
};

enum SPIRVMemorySemantics {
  SemanticsNone = 0,
  SemanticsAcquire = 0x2,
  SemanticsRelease = 0x4,
  SemanticsAcquireRelease = 0x8,
  SemanticsSGMemory = 0x80,
  SemanticsWGMemory = 0x100,
  SemanticsCrossWGMemory = 0x200,
};

namespace cute {
CUTE_HOST_DEVICE void barrier_arrive(
    SPIRVScope scope, int memory_semantics = SemanticsNone) {
  barrier_arrive(
      static_cast<int>(scope), static_cast<int>(scope), memory_semantics);
}
CUTE_HOST_DEVICE void barrier_wait(
    SPIRVScope scope, int memory_semantics = SemanticsNone) {
  barrier_wait(
      static_cast<int>(scope), static_cast<int>(scope), memory_semantics);
}
}  // namespace cute
#endif

