/*****************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 ****************************************************************************************/

// Fallback implementation for CPUs without AVX512/AMX support

#pragma once

#include <ATen/ATen.h>

namespace flash_attn_cpu {
namespace fallback {

// Fallback flash attention varlen implementation (naive, slower)
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

}  // namespace fallback
}  // namespace flash_attn_cpu
