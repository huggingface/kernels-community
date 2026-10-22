/*****************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 ****************************************************************************************/

#pragma once

// These headers are compiled twice, once per target architecture, and sycl-tla
// expands them into different code each time. Keeping the two expansions in
// separate namespaces stops the linker from merging symbols that share a name
// but not a body.
#if defined(__SYCL_TARGET_INTEL_GPU_CRI__)
#define MEGABLOCKS_XE_TARGET_NS xe35
// Not sycl-tla's SYCL_INTEL_TARGET: only 0.9.2+ rewrites that to 20/35.
#define MEGABLOCKS_XE_TARGET_ARCH 35
#else
#define MEGABLOCKS_XE_TARGET_NS xe20
#define MEGABLOCKS_XE_TARGET_ARCH 20
#endif
