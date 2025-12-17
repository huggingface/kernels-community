/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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
/*! \file
  \brief Functor performing online softmax.
*/

#pragma once

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/detail/layout.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace flash_attention {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <bool CausalMask_, bool LocalMask_, class DispatchPolicy,
          class... Args>
class FlashChunkPrefillSoftmaxEpilogue {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy>,
                "Could not find an epilogue specialization.");
};

template <bool CausalMask_, bool LocalMask_, class Element_>
class FlashChunkPrefillSoftmaxEpilogue<CausalMask_, LocalMask_,
                                       epilogue::IntelXeXMX16, Element_> {
 public:
  //
  // Type Aliases
  //
  using DispatchPolicy = epilogue::IntelXeXMX16;
  using Element = Element_;

  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool LocalMask = LocalMask_;

  using GmemTiledCopyOut = void;

  // Host side epilogue arguments
  struct Arguments {
    Element const scale;
  };

  // Device side epilogue params
  using Params = Arguments;

  //
  // Methods
  //

  static constexpr Params to_underlying_arguments(Arguments const& args) {
    constexpr double kLog2e = 1.4426950408889634074;  // log_2(e) = M_LOG2E
    Element val = args.scale * static_cast<Element>(kLog2e);
    return Params{val};
  }

  template <class ProblemShape>
  static size_t get_workspace_size() {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace() {
    return Status::kSuccess;
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool can_implement() {
    return true;
  }

  CUTLASS_HOST_DEVICE
  FlashChunkPrefillSoftmaxEpilogue(Params const& params_) : params(params_) {}

  template <int Vec, int FragsM, int FragsN, class FragAcc, class FragMax,
            class FragSum>
  CUTLASS_DEVICE void scale_exp_log2(FragAcc& frag_s, FragMax const& max,
                                     FragSum& sum) {
    auto g = COMPAT::get_nd_item<1>().get_sub_group();
    const auto max_scale = max * params.scale;
    CUTLASS_PRAGMA_UNROLL
    for (int index = 0; index < Vec * FragsM; index++) {
      const auto max_scale_bcast = group_broadcast(g, max_scale, index);
      CUTLASS_PRAGMA_UNROLL
      for (int z = 0; z < FragsN; z++) {
        auto base_index = index + (z * Vec * FragsM);
        if constexpr (LocalMask) {
          if ((std::isinf(max_scale_bcast) && max_scale_bcast < 0) ||
              (std::isinf(frag_s(base_index)) && frag_s(base_index) < 0)) {
            frag_s(base_index) = 0.f;
            // continue;
          } else {
            Element eq = frag_s(base_index) - max_scale_bcast;
            frag_s(base_index) = sycl::native::exp2(eq);
          }
        } else {
          Element eq = frag_s(base_index) - max_scale_bcast;
          frag_s(base_index) = sycl::native::exp2(eq);
        }
        sum(index) += frag_s(base_index);
      }
    }
  }

  template <int Vec, int FragsM, int FragsN, class FragSrc, class FragMax>
  CUTLASS_DEVICE void reduce_max(FragSrc& src, FragMax& max) {
    auto sg = COMPAT::get_nd_item<1>().get_sub_group();
    CUTLASS_PRAGMA_UNROLL
    for (int index = 0; index < Vec * FragsM; index++) {
      auto maxptr = group_broadcast(sg, max, index);
      CUTLASS_PRAGMA_UNROLL
      for (int z = 0; z < FragsN; z++) {
        auto base_index = index + (z * Vec * FragsM);
        maxptr = sycl::max(maxptr, src(base_index));
        src(base_index) *= params.scale;
      }
      maxptr = reduce_over_group(sg, maxptr, sycl::maximum<>());
      if (index == sg.get_local_id()[0]) {
        max = maxptr;
      }
    }
  }

  template <class FragAcc, class FragMax, class FragSum, class FragOut>
  CUTLASS_DEVICE void operator()(bool is_first, FragAcc& frag_s, FragMax& max,
                                 FragSum& sum, FragOut& out) {
    auto max_prev = max;
    using FragAccLayout = typename FragAcc::layout_type;
    using FragOutLayout = typename FragOut::layout_type;
    constexpr int Vec = get<0>(FragAccLayout{}.shape());
    constexpr int FragsM = get<1>(FragAccLayout{}.shape());
    constexpr int FragsNAcc = get<2>(FragAccLayout{}.shape());
    constexpr int FragsNOut = size(select<2, 3>(FragOutLayout{}.shape()));
    reduce_max<Vec, FragsM, FragsNAcc>(frag_s, max);
    // if (max == INFINITY) {
    //   max = 0.f;
    // }
    static_assert(Vec * FragsM % 8 == 0,
                  " No. of attention rows per subgroup should be >= 1 MMA Atom "
                  "worth of rows.");
    if (!is_first) {
      auto sg = COMPAT::get_nd_item<1>().get_sub_group();
      Element max_scale{max * params.scale};
      Element exp_scale;
      if constexpr (LocalMask) {
        if ((std::isinf(max_scale) && max_scale < 0) ||
            (std::isinf(max_prev) && max_prev < 0)) {
          exp_scale = 0.f;
        } else {
          exp_scale = sycl::native::exp2(max_prev * params.scale - max_scale);
        }
      } else {
        exp_scale = sycl::native::exp2(max_prev * params.scale - max_scale);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int index = 0; index < Vec * FragsM; index++) {
        auto max_scale_bcast = group_broadcast(sg, max_scale, index);
        auto exp_scale_bcast = group_broadcast(sg, exp_scale, index);
        sum(index) *= exp_scale_bcast;
        CUTLASS_PRAGMA_UNROLL
        for (int z = 0; z < FragsNAcc; z++) {
          auto base_index = index + (z * Vec * FragsM);
          if constexpr (LocalMask) {
            if ((std::isinf(max_scale_bcast) && max_scale_bcast < 0) ||
                (std::isinf(frag_s(base_index)) && frag_s(base_index) < 0)) {
              frag_s(base_index) = 0.f;
              // continue;
            } else {
              Element eq = frag_s(base_index) - max_scale_bcast;
              frag_s(base_index) = sycl::native::exp2(eq);
            }
          } else {
            Element eq = frag_s(base_index) - max_scale_bcast;
            // eq = eq < -65400.f ? 0.f : eq;
            frag_s(base_index) = sycl::native::exp2(eq);
          }
          sum(index) += frag_s(base_index);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int z = 0; z < FragsNOut; z++) {
          auto base_index = index + (z * Vec * FragsM);
          out(base_index) *= exp_scale_bcast;
        }
      }
    } else {
      scale_exp_log2<Vec, FragsM, FragsNAcc>(frag_s, max, sum);
    }
  }
  Params params;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace collective
}  // namespace flash_attention
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
