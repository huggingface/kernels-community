/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
// Multi-payload block 2D copies with in-place coordinate advance, as used by
// the vllm-xpu-kernels Xe3 chunk-prefill mainloop (FMHAFwdMainloop), which
// builds against SYCL-TLA v0.1.0_next, whose prepare_payloads(copy, coords)
// returns payloads that support `+=`; public sycl-tla 0.9.3 only has
// prepare_payloads(copy, src, dst) without it. This provides the same calls
// (prepare_payloads / copy / prefetch / +=) on top of the public
// Xe2DTraitsBase.
#pragma once

#include "../../decode/splitkv_decode.hpp"  // FLASH_XPU_FMHA_NS
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_xe_2d.hpp>

namespace cutlass::fmha::collective {
inline namespace FLASH_XPU_FMHA_NS {

using namespace cute;

// N block 2D address payloads (one per copy atom) of the copy BaseT.
template <int N, class BaseT>
struct XePreparedPayloads {
  int* payloads[N];

  template <int Field>
  CUTE_DEVICE static void add_payload_field(int* payload, int32_t delta) {
    static_assert(Field == 5 || Field == 6, "only block X/Y can be updated");
#ifdef __SYCL_DEVICE_ONLY__
    asm volatile(
        "{\n"
        ".decl PAYLOAD_D v_type=G type=D num_elts=8 alias=<%0,0>\n"
        ".decl DELTA_D v_type=G type=D num_elts=1 alias=<%1,0>\n"
        "add (M1_NM, 1) PAYLOAD_D(0,%2)<1> PAYLOAD_D(0,%2)<0;1,0> DELTA_D(0,0)<0;1,0>\n"
        "}\n"
        :
        : "rw.u"(payload), "rw.u"(delta), "P"(Field));
#else
    (void)payload;
    (void)delta;
#endif
  }

  // Advance all N payloads by `delta` in the copy's coordinate space; static
  // zero components are skipped.
  template <class Coord>
  CUTE_DEVICE XePreparedPayloads& operator+=(Coord const& delta) {
    using Op = typename BaseT::CopyOp;
    static_assert(!BaseT::nontrivial_tiled_strides,
                  "payload updates need modes handled by block 2D messages");
    auto dx = get<decltype(BaseT::get_x_mode())::value>(delta);
    auto dy = get<decltype(BaseT::get_y_mode())::value>(delta);
    CUTE_UNROLL
    for (int i = 0; i < N; ++i) {
      if constexpr (!is_constant_v<0, decltype(dx)>) {
        add_payload_field<5>(payloads[i],
                             int32_t(dx) * BaseT::ValBits / Op::CopyBits);
      }
      if constexpr (!is_constant_v<0, decltype(dy)>) {
        add_payload_field<6>(payloads[i], int32_t(dy));
      }
    }
    return *this;
  }
};

namespace xe_payload_detail {

// One payload per copy atom of the coordinate tensor `src`, in the order
// cute::copy visits them.
template <int NumValDst, int N, class BaseT, class SEngine, class SLayout>
CUTE_DEVICE void prepare(BaseT const& base, Tensor<SEngine, SLayout> const& src,
                         XePreparedPayloads<N, BaseT>& payloads, int& idx) {
  using Op = typename BaseT::CopyOp;
  constexpr int ValBits = BaseT::ValBits;
  if constexpr (SLayout::rank == 1 &&
                decltype(size(SLayout{}))::value == NumValDst) {
#ifdef __SYCL_DEVICE_ONLY__
    using XMode = decltype(BaseT::get_x_mode());
    using YMode = decltype(BaseT::get_y_mode());
    auto coord = src.data().coord_;
    int32_t x = get<XMode::value>(coord) * ValBits / Op::CopyBits;
    int32_t y = get<YMode::value>(coord);
    uint64_t bp = base.base_ptr;
    if constexpr (BaseT::nontrivial_tiled_strides) {
      auto off = inner_product(coord, base.tiled_strides);
      bp += (off * ValBits) >> 3;
    }
    payloads.payloads[idx] = __builtin_IB_subgroup_createBlock2DAddressPayload(
        bp, base.width - 1, base.height - 1, base.pitch - 1, x, y,
        Op::AtomWidth / Op::BlockCount, Op::AtomHeight, Op::BlockCount);
#endif
    ++idx;
  } else if constexpr (SLayout::rank > 1) {
    constexpr int R = SLayout::rank;
    auto src_v = group_modes<1, R>(src);
    auto src_null = nullspace(cute::layout<1>(src_v));
    auto src_n = zipped_divide(src_v, make_tile(shape<0>(src_v), src_null));
    auto src_c = src_n(make_coord(_, Int<0>{}), make_coord(Int<0>{}, _));
    constexpr int Rest = decltype(size<1>(src_c))::value;
    CUTE_UNROLL
    for (int i = 0; i < Rest; ++i) {
      prepare<NumValDst>(base, src_c(_, i), payloads, idx);
    }
  } else {
    static_assert(is_tuple<decltype(shape<0>(SLayout{}))>::value,
                  "V mode is atomic but its size does not match the atom");
    prepare<NumValDst>(base, tensor<0>(src), payloads, idx);
  }
}

// One load per copy atom into the register fragment `dst`.
template <int NumValDst, int N, class BaseT, class DEngine, class DLayout>
CUTE_DEVICE void load(BaseT const& base, Tensor<DEngine, DLayout>& dst,
                      XePreparedPayloads<N, BaseT> const& payloads, int& idx) {
  using Op = typename BaseT::CopyOp;
  constexpr int ValBits = BaseT::ValBits;
  if constexpr (DLayout::rank == 1 &&
                decltype(size(DLayout{}))::value == NumValDst) {
#ifdef __SYCL_DEVICE_ONLY__
    using ValT = int_byte_t<bits_to_bytes(ValBits)>;
    Op::copy(payloads.payloads[idx],
             const_cast<ValT*>(recast_ptr<ValT>(raw_pointer_cast(dst.data()))));
#endif
    ++idx;
  } else if constexpr (DLayout::rank > 1) {
    constexpr int R = DLayout::rank;
    auto dst_v = group_modes<1, R>(dst);
    auto dst_null = nullspace(cute::layout<1>(dst_v));
    auto dst_n = zipped_divide(dst_v, make_tile(shape<0>(dst_v), dst_null));
    auto dst_c = dst_n(make_coord(_, Int<0>{}), make_coord(Int<0>{}, _));
    constexpr int Rest = decltype(size<1>(dst_c))::value;
    CUTE_UNROLL
    for (int i = 0; i < Rest; ++i) {
      auto d_i = dst_c(_, i);
      load<NumValDst>(base, d_i, payloads, idx);
    }
  } else {
    static_assert(is_tuple<decltype(shape<0>(DLayout{}))>::value,
                  "V mode is atomic but its size does not match the atom");
    auto d_sub = tensor<0>(dst);
    load<NumValDst>(base, d_sub, payloads, idx);
  }
}

}  // namespace xe_payload_detail

template <class TiledCopy, class Engine, class Layout>
CUTE_DEVICE auto prepare_payloads(TiledCopy const& tiled_copy,
                                  Tensor<Engine, Layout> const& coords) {
  auto const& base = cute::detail::as_xe2d_base(tiled_copy);
  using BaseT = remove_cvref_t<decltype(base)>;
  constexpr int NumValDst =
      decltype(size<1>(typename BaseT::Traits::DstLayout{}))::value /
      BaseT::ValBits;
  constexpr int Total = decltype(size(Layout{}))::value;
  static_assert(Total % NumValDst == 0,
                "coordinate tensor is not a multiple of the atom size");
  XePreparedPayloads<Total / NumValDst, BaseT> prepared{};
  int idx = 0;
  xe_payload_detail::prepare<NumValDst>(base, coords, prepared, idx);
  return prepared;
}

template <class TiledCopy, int N, class BaseT, class Engine, class Layout>
CUTE_DEVICE void copy(TiledCopy const& tiled_copy,
                      XePreparedPayloads<N, BaseT> const& prepared,
                      Tensor<Engine, Layout>& dst) {
  auto const& base = cute::detail::as_xe2d_base(tiled_copy);
  constexpr int NumValDst =
      decltype(size<1>(typename BaseT::Traits::DstLayout{}))::value /
      BaseT::ValBits;
  int idx = 0;
  xe_payload_detail::load<NumValDst>(base, dst, prepared, idx);
}

template <class TiledCopy, int N, class BaseT>
CUTE_DEVICE void prefetch(TiledCopy const&,
                          XePreparedPayloads<N, BaseT> const& prepared) {
#ifdef __SYCL_DEVICE_ONLY__
  using Op = typename BaseT::CopyOp;
  CUTE_UNROLL
  for (int i = 0; i < N; ++i) {
    Op::copy(prepared.payloads[i]);
  }
#else
  (void)prepared;
#endif
}

}  // namespace FLASH_XPU_FMHA_NS
}  // namespace cutlass::fmha::collective
