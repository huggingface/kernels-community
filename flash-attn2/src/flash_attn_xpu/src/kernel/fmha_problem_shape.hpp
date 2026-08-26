/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Standalone definition of FMHAProblemShape, extracted so that the FMHA
 * forward kernel header does not have to drag in the (deleted) full
 * mainloop/epilogue headers.
 **************************************************************************************************/

#pragma once

#include "../collective/fmha_fusion.hpp"
#include "cute/util/type_traits.hpp"

namespace cutlass::fmha::kernel {

template <bool IsVarLen_ = false>
struct FMHAProblemShape {
  using SeqLenType = cute::conditional_t<
      IsVarLen_,
      cutlass::fmha::collective::VariableLength,
      int>;
  int batch;
  int num_heads_q, num_heads_kv;
  SeqLenType seq_len_qo, seq_len_kv;
  int head_size_qk, head_size_vo;
};

}  // namespace cutlass::fmha::kernel
