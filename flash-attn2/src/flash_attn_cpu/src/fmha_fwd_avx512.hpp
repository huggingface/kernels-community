/*****************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 ****************************************************************************************/

// AVX512 implementation - compile with -mavx512f -mavx512bf16 -mamx-bf16 -mamx-tile

#pragma once

#include <ATen/ATen.h>

namespace flash_attn_cpu {
namespace avx512 {

// AVX512 optimized flash attention varlen implementation
void fmha_fwd_varlen_impl(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& out,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    float softmax_scale,
    bool is_causal);

}  // namespace avx512
}  // namespace flash_attn_cpu
