/******************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/
#pragma once

#include <cstdint>

namespace flash_xpu {

// softmax_lse rows [cu_seqlens_q[batch], rows) belong to no batch (padded
// queries after the last sequence). They get LSE 0, as with the generic
// kernel, whose softmax_lse is zero-filled. Run by one work-group of
// `threads` work-items, before or after the others.
inline void zero_padded_lse(float* lse, int64_t head_stride,
                            int64_t row_stride, int num_heads,
                            const int* cu_seqlens_q, int batch, int rows,
                            int thr_id, int threads) {
  const int r0 = cu_seqlens_q[batch];
  const int64_t n = rows - r0;
  for (int64_t i = thr_id; i < n * num_heads; i += threads) {
    const int64_t h = i / n;
    lse[h * head_stride + (r0 + i - h * n) * row_stride] = 0.f;
  }
}

}  // namespace flash_xpu
