/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
// The mainloops differ per architecture (vllm-xpu-kernels xe_2 / xe_3); the
// rest of the chunk-prefill and paged-decode kernels is shared.
#pragma once

#include "../decode/splitkv_decode.hpp"  // FLASH_XPU_SPLITKV_TARGET_ARCH

#if FLASH_XPU_SPLITKV_TARGET_ARCH >= 35
#include "xe35/chunk_prefill_mainloop.hpp"
#else
#include "xe20/chunk_prefill_mainloop.hpp"
#endif
