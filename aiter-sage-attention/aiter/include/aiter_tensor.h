// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// Subset of aiter's csrc/include/aiter_tensor.h used by the vendored kernels;
// the full header pulls in ck_tile, which these kernels do not need.
#include <cstdint>

struct aiter_tensor_t
{
    void* ptr;
    int ndim;
    int64_t shape[8];

    int64_t size(int i) const { return (i < 0) ? shape[ndim + i] : shape[i]; }
    void* data_ptr() const { return ptr; }
};
