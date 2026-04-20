/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <sycl/sycl.hpp>
#include "./fmha_fwd_common.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

template <
    class CollectiveMainloop,  // Attention mainloop
    class TileShapeO_,         // Shape of output tile, may be larger than P*V
                               // GEMM
    class TensorO_,            // 2D slice of global output tensor
    class TiledCopyO_ = void>  // Optional TiledCopy for loading O
class FMHAFwdEpilogue {
 public:
  //
  // Type Aliases — delegated to FMHAFwdEpilogueTraits (fmha_fwd_common.hpp)
  //
  using ETraits = FMHAFwdEpilogueTraits<
      CollectiveMainloop, TileShapeO_, TensorO_, TiledCopyO_>;

  using TiledMMAPV   = typename ETraits::TiledMMAPV;
  using TileShapePV  = typename ETraits::TileShapePV;
  using TileShapeO   = typename ETraits::TileShapeO;
  using SGPerWG      = typename ETraits::SGPerWG;
  using TensorO      = typename ETraits::TensorO;
  using TensorO2D    = typename ETraits::TensorO2D;
  using ElementO     = typename ETraits::ElementO;
  using FragA        = typename ETraits::FragA;
  using FragARow     = typename ETraits::FragARow;
  using ElementA     = typename ETraits::ElementA;
  using ReduceK      = typename ETraits::ReduceK;
  using SGTileShapeA = typename ETraits::SGTileShapeA;
  using ReduceSGQ    = typename ETraits::ReduceSGQ;
  using ReduceSGV    = typename ETraits::ReduceSGV;
  using ReduceSGLayout = typename ETraits::ReduceSGLayout;
  using SGTileShapeO   = typename ETraits::SGTileShapeO;
  using ReduceFragA    = typename ETraits::ReduceFragA;
  using ReduceFragARow = typename ETraits::ReduceFragARow;
  using TiledCopyO     = typename ETraits::TiledCopyO;
  using AlignedSGTileA_Q = typename ETraits::AlignedSGTileA_Q;
  using SharedStorage  = typename ETraits::SharedStorage;

  // LSE output type (float for numerical stability)
  using ElementLSE = float;

  struct Arguments {
    ElementLSE* lse_ptr = nullptr;
    int lse_stride_head = 0;
    int lse_stride_batch = 0;
  };
  
  struct Params {
    ElementLSE* lse_ptr = nullptr;
    int lse_stride_head = 0;
    int lse_stride_batch = 0;
  };

 private:
  SharedStorage& shared;
  Params params;

 public:
  static constexpr Params
  to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    return {args.lse_ptr, args.lse_stride_head, args.lse_stride_batch};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  FMHAFwdEpilogue(Params const& params_, SharedStorage& shared_) 
      : shared(shared_), params(params_) {}

  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorO2D const& O,
      FragA& tArA,
      FragARow& tA_max,
      FragARow& tA_sum,
      QVCoord blk_qv,
      int thr_id,
      int head_idx = 0,
      int batch_idx = 0,
      int seq_len_q = 0) {

    using namespace cute;
    using ElementA = typename FragA::element_type;

    auto [rA, rA_sum, rA_max, active] = reduce_A(tArA, tA_max, tA_sum, thr_id);

    if (!active) return;

    if (params.lse_ptr != nullptr) {
      store_lse(rA, rA_max, rA_sum, blk_qv, thr_id, head_idx, batch_idx, seq_len_q);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA_sum.size(); i++) {
      if (rA_sum(i) == ElementA(0))
        rA_sum(i) = ElementA(0);
      else
        rA_sum(i) = ElementA(1) / rA_sum(i);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA.size(); i++)
      rA(i) *= broadcast<0>(rA_sum, rA, i);

    // Write output using shared helper.
    ETraits::write_output(O, rA, blk_qv, thr_id);
  }

  // Store LSE = max + log(sum) for backward pass
  // Use TiledMMA partitioning to determine row indices (same as mainloop)
  template <typename FragA_, typename FragARow, typename QVCoord>
  CUTLASS_DEVICE void store_lse(
      FragA_& rA,           // O accumulator for layout reference
      FragARow& rA_max,     // Row-wise max (reduced over V dimension)
      FragARow& rA_sum,     // Row-wise sum (before normalization)
      QVCoord blk_qv,       // WG tile indices: (q,v)
      int thr_id,           // Work-item ID
      int head_idx,         // Head index
      int batch_idx,        // Batch index
      int seq_len_q) {      // Sequence length for bounds checking

    using namespace cute;
    using namespace sycl::ext::oneapi::this_work_item;
    
    // LSE base offset for this head and batch
    ElementLSE* lse_base = params.lse_ptr + 
                           batch_idx * params.lse_stride_batch + 
                           head_idx * params.lse_stride_head;
    
    constexpr float kLn2 = 0.6931471805599453f;  // ln(2)
    
    int blk_q = get<0>(blk_qv);
    int global_row_offset = blk_q * get<0>(TileShapeO{});
    
    // Use TiledMMAPV to get coordinates within the tile
    // Create identity tensor for a single tile (coordinates are tile-local: [0, TileShapeO))
    Tensor cA = make_identity_tensor(make_shape(
        get<0>(TileShapeO{}), get<1>(TileShapeO{})));

    // Use TiledMMAPV's partitioning to get thread-specific coordinates
    TiledMMAPV mma_pv;
    auto thr_mma = mma_pv.get_slice(thr_id);
    auto cA_thread = thr_mma.partition_C(cA);
    
    // Iterate through elements and write LSE for each unique row
    int prev_row = -1;
    
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA.size(); i++) {
      // Get the tile-local row coordinate and convert to global
      int local_row_idx = get<0>(cA_thread(i));
      int row_idx = global_row_offset + local_row_idx;
      
      // Only write each row once (first V position)
      if (local_row_idx != prev_row && row_idx < seq_len_q) {
        // Use broadcast to get the correct max/sum values for this row
        auto max_val = static_cast<ElementLSE>(broadcast<0>(rA_max, rA, i));
        auto sum_val = static_cast<ElementLSE>(broadcast<0>(rA_sum, rA, i));
        
        // LSE = max + log(sum)
        // max was scaled by log2(e), so: LSE = max * ln(2) + log(sum)
        ElementLSE lse = (sum_val > 0) ? (max_val * kLn2 + sycl::log(sum_val)) : -INFINITY;
        
        // Write LSE value
        lse_base[row_idx] = lse;
        
        prev_row = local_row_idx;
      }
    }
  }

  // Reduce k-blocks of A and A_sum across WG, if needed.
  template <typename FragA, typename FragARow>
  CUTLASS_DEVICE decltype(auto) reduce_A(
      FragA& tArA,
      FragARow& tA_max,
      FragARow& tA_sum,
      int thr_id) {

    if constexpr (ReduceK{} == _1{}) {
      return std::make_tuple(tArA, tA_sum, tA_max, true);
    } else {
      // Delegate to shared ReduceK>1 implementation in ETraits.
      return ETraits::reduce_A_multi_k(tArA, tA_max, tA_sum, thr_id, shared);
    }
  }
};

}  // namespace cutlass::fmha::collective