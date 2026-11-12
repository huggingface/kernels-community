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

#include <array>

#include "../splitkv_decode.hpp"  // FLASH_XPU_VLLM_NS

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/util/sycl_vec.hpp"
#include "../../collective/fmha_fusion.hpp"
#include "paged_decode_tuning.h"

#ifndef VLLM_GRF_SIZE
  #define VLLM_GRF_SIZE 256
#endif

#ifdef __SYCL_DEVICE_ONLY__
// SYCL-TLA declares this builtin as ...PayloadHeigth, a spelling IGC does not
// resolve (the call is silently dropped). Declare the name IGC implements.
SYCL_EXTERNAL extern "C" void
__builtin_IB_subgroup_setBlock2DAddressPayloadHeight(
    int* addrPayload, int height_minus_one);
#endif

namespace cutlass::fmha {
// Arch-tagged inline namespace: gives these definitions a mangled name
// distinct from the other Xe architecture's identically named copies,
// while leaving name lookup (cutlass::fmha::...) unchanged.
inline namespace FLASH_XPU_VLLM_NS {

template <int Stages>
class XeDefault {};  // Default FMHA mainloop, P in registers.

}  // namespace vllm_xpu_xe3

};  // namespace cutlass::fmha

namespace cutlass::fmha::collective {
// Arch-tagged inline namespace: gives these definitions a mangled name
// distinct from the other Xe architecture's identically named copies,
// while leaving name lookup (cutlass::fmha::...) unchanged.
inline namespace FLASH_XPU_VLLM_NS {

static inline void sbarrier_wait() { asm volatile("sbarrier.wait\n"); }

static inline void sbarrier_signal() { asm volatile("sbarrier.signal\n"); }

static inline void gfence() { asm volatile("lsc_fence.ugm.none.group\n"); }

static inline void barrier() {
  asm volatile("lsc_fence.ugm.none.group\n");
  asm volatile("barrier\n");
}

using namespace cute;

// The Xe LSC 2D block descriptor stores the surface height in a 24-bit field,
// so one block-load surface addresses at most 2^24 rows. A paged K/V pool can
// be taller; the host then runs the paged mainloops in has_large_surface mode,
// which splits K/V into sub-surfaces of kXeMaxSurfaceRows rows and routes
// each physical K tile to the sub-surface holding it.
inline constexpr int kXeMaxSurfaceRows = 1 << 24;

// Sub-surface holding physical K tile `tile`, and the tile's index within it.
// Both are a shift/mask since tiles are a power of two in rows.
template <int TileK>
CUTLASS_DEVICE int xe_kv_surface(int tile) {
  static_assert(
      kXeMaxSurfaceRows % TileK == 0, "K tiles must not straddle sub-surfaces");
  return static_cast<int>(
      static_cast<unsigned>(tile) / (kXeMaxSurfaceRows / TileK));
}

template <int TileK>
CUTLASS_DEVICE int xe_kv_surface_tile(int tile) {
  return static_cast<int>(
      static_cast<unsigned>(tile) % (kXeMaxSurfaceRows / TileK));
}

// View of sub-surface `surf` of a 2D K or V tensor. RowMode is the token
// mode: 0 for K (k,d), 1 for V (d,k).
template <int RowMode, class Tensor2D>
CUTLASS_DEVICE auto xe_kv_sub_surface(Tensor2D const& t, int surf) {
  int64_t const row0 = int64_t(surf) * kXeMaxSurfaceRows;
  int64_t rows = int64_t(get<RowMode>(t.shape())) - row0;
  rows = rows < kXeMaxSurfaceRows ? rows : kXeMaxSurfaceRows;
  rows = rows < 1 ? 1 : rows;
  return make_tensor(
      t.data() + row0 * get<RowMode>(t.stride()),
      make_layout(replace<RowMode>(t.shape(), int(rows)), t.stride()));
}

// Re-point an existing block 2D payload at sub-surface `surf` of `t` by
// rewriting its base address and height in place, so a sub-surface switch
// needs no new copy atoms.
template <int RowMode, class Tensor2D>
CUTLASS_DEVICE void
xe_kv_retarget_payload(int* payload, Tensor2D const& t, int surf) {
#ifdef __SYCL_DEVICE_ONLY__
  auto const s = xe_kv_sub_surface<RowMode>(t, surf);
  __builtin_IB_subgroup_setBlock2DAddressPayloadBase(
      payload, static_cast<long>((uint64_t)raw_pointer_cast(s.data())));
  __builtin_IB_subgroup_setBlock2DAddressPayloadHeight(
      payload, int(get<RowMode>(s.shape())) - 1);
#endif
}

// Re-point every payload of a prepared K/V copy at sub-surface `surf`.
// (vllm's internal cutlass types this Xe2DPreparedPayloads<N, BaseT>; public
// sycl-tla has Xe2DPreparedPayloads<N>. Only FMHAFwdMainloop, the prefill
// mainloop that kc does not instantiate, calls it.)
template <int RowMode, class Prepared, class Tensor2D>
CUTLASS_DEVICE void
xe_kv_retarget_payloads(Prepared& prepared, Tensor2D const& t, int surf) {
  constexpr int N = sizeof(prepared.payloads) / sizeof(prepared.payloads[0]);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < N; ++i) {
    xe_kv_retarget_payload<RowMode>(prepared.payloads[i], t, surf);
  }
}

// Sub-surface 0 of `t` in large-surface mode, `t` itself otherwise.
template <bool LargeSurface, int RowMode, class Tensor2D>
CUTLASS_DEVICE decltype(auto) xe_kv_first_sub_surface(Tensor2D const& t) {
  if constexpr (LargeSurface) {
    return xe_kv_sub_surface<RowMode>(t, 0);
  } else {
    return (t);
  }
}

#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
CUTE_DEVICE
uint32_t subgroup_any_less(float const& lhs, float const& rhs) {
  uint32_t result;
  asm("{\n"
      ".decl OUT_UD v_type=G type=UD num_elts=16 alias=<%0,0>\n"
      ".decl LHS_F v_type=G type=F num_elts=16 alias=<%1,0>\n"
      ".decl RHS_F v_type=G type=F num_elts=16 alias=<%2,0>\n"
      ".decl ANY_LESS v_type=P num_elts=16\n"
      "cmp.lt (M1_NM, 16) ANY_LESS LHS_F(0,0)<1;1,0> RHS_F(0,0)<1;1,0>\n"
      "(ANY_LESS.any) sel (M1_NM, 16) OUT_UD(0,0)<1> 0x1:ud 0x0:ud\n"
      "}\n"
      : "=rw"(result)
      : "rw"(lhs), "rw"(rhs));
  return result;
}
#else
CUTE_DEVICE
uint32_t subgroup_any_less(float const& /* lhs */, float const& /* rhs */) {
  CUTE_INVALID_CONTROL_PATH(
      "subgroup_any_less requires Intel Xe SYCL device target");
}
#endif

#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
CUTE_DEVICE
void cvt_f32x2_to_bf16x2_bias(
    float const& src0, float const& src1, cute::intel::uint2& tmp) {
  asm("{\n"
      ".decl IN_UD0 v_type=G type=UD num_elts=16 alias=<%1,0>\n"
      ".decl IN_UD1 v_type=G type=UD num_elts=16 alias=<%2,0>\n"
      ".decl TMP_UD v_type=G type=UD num_elts=32 alias=<%0,0>\n"
      "add (M1_NM, 16) TMP_UD(0,0)<1> IN_UD0(0,0)<1;1,0> 0x8000:uw\n"
      "add (M1_NM, 16) TMP_UD(1,0)<1> IN_UD1(0,0)<1;1,0> 0x8000:uw\n"
      "}\n"
      : "=rw"(tmp)
      : "rw"(src0), "rw"(src1));
}

CUTE_DEVICE
void cvt_f32x2_to_bf16x2_pack(
    cute::intel::uint2 const& tmp, cute::intel::ushort2& dst) {
  asm("{\n"
      ".decl TMP_UD v_type=G type=UD num_elts=32 alias=<%1,0>\n"
      ".decl TMP_UW v_type=G type=UW num_elts=64 alias=<TMP_UD,0>\n"
      ".decl OUT_UW v_type=G type=UW num_elts=32 alias=<%0,0>\n"
      "mov (M1_NM, 32) OUT_UW(0,0)<1> TMP_UW(0,1)<2;1,0>\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(tmp));
}

// FP8 packed conversion primitives (ported from
// libraries.ai.cutlass.internal/applications/flash_attention_v2/collective/
// xe_fmha_fwd_mainloop.hpp). Subgroup-vectorized casts used to pack the softmax
// output P into fp8 (e4m3/e5m2) for the full-fp8 P*V GEMM.
CUTE_DEVICE
void cvt_f32x2_to_e5m2x2_pack(
    float const& src0, float const& src1, cute::intel::uchar2& dst) {
  asm("{\n"
      ".decl IN_F0 v_type=G type=F  num_elts=16 alias=<%1,0>\n"
      ".decl IN_F1 v_type=G type=F  num_elts=16 alias=<%2,0>\n"
      ".decl OUT_UB v_type=G type=UB num_elts=32 alias=<%0,0>\n"
      ".decl TMP_HF v_type=G type=HF num_elts=32 align=32\n"
      "mov  (M1, 16) TMP_HF(0,0)<1> IN_F0(0,0)<1;1,0>\n"
      "mov  (M1, 16) TMP_HF(0,16)<1> IN_F1(0,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) OUT_UB(0,0)<1> TMP_HF(0,0)<1;1,0>\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(src0), "rw"(src1));
}

CUTE_DEVICE
void cvt_f32x2_to_e4m3x2_pack(
    float const& src0, float const& src1, cute::intel::uchar2& dst) {
  asm("{\n"
      ".decl IN_F0 v_type=G type=F num_elts=16 alias=<%1,0>\n"
      ".decl IN_F1 v_type=G type=F num_elts=16 alias=<%2,0>\n"
      ".decl OUT_B v_type=G type=B num_elts=32 alias=<%0,0>\n"
      ".decl TMP_HF v_type=G type=HF num_elts=32 align=32\n"
      "mov  (M1, 16) TMP_HF(0,0)<1> IN_F0(0,0)<1;1,0>\n"
      "mov  (M1, 16) TMP_HF(0,16)<1> IN_F1(0,0)<1;1,0>\n"
      "fcvt (M1_NM, 32) OUT_B(0,0)<1> TMP_HF(0,0)<1;1,0>\n"
      "}\n"
      : "=rw"(dst)
      : "rw"(src0), "rw"(src1));
}
#else
CUTE_DEVICE
void cvt_f32x2_to_bf16x2_bias(
    float const& /*src0*/, float const& /*src1*/, cute::intel::uint2& /*tmp*/) {
  CUTE_INVALID_CONTROL_PATH(
      "cvt_f32x2_to_bf16x2_bias requires Intel Xe SYCL device target");
}
CUTE_DEVICE
void cvt_f32x2_to_bf16x2_pack(
    cute::intel::uint2 const& /*tmp*/, cute::intel::ushort2& /*dst*/) {
  CUTE_INVALID_CONTROL_PATH(
      "cvt_f32x2_to_bf16x2_pack requires Intel Xe SYCL device target");
}
CUTE_DEVICE
void cvt_f32x2_to_e5m2x2_pack(
    float const& /*src0*/,
    float const& /*src1*/,
    cute::intel::uchar2& /*dst*/) {
  CUTE_INVALID_CONTROL_PATH(
      "cvt_f32x2_to_e5m2x2_pack requires Intel Xe SYCL device target");
}
CUTE_DEVICE
void cvt_f32x2_to_e4m3x2_pack(
    float const& /*src0*/,
    float const& /*src1*/,
    cute::intel::uchar2& /*dst*/) {
  CUTE_INVALID_CONTROL_PATH(
      "cvt_f32x2_to_e4m3x2_pack requires Intel Xe SYCL device target");
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
// Reorder/convert the f32 softmax output (tSrS) into the fp8 P*V GEMM A operand
// (tArP) for the full-fp8 datapath. When ElementP is fp8 (e4m3/e5m2) and the
// per-work-item reorder layout matches the fused-pack layout, use the
// subgroup-vectorized cvt primitives; otherwise fall back to the generic
// cute::reorder. Non-fp8 ElementP is not handled here (the bf16 P path uses the
// dedicated cvt_f32x2_to_bf16x2_* primitives inline in the mainloop).
/////////////////////////////////////////////////////////////////////////////////////////////////
template <class ElementP, class FragS, class FragP>
CUTLASS_DEVICE void reorder_to_P_fp8(FragS& tSrS, FragP& tArP) {
  using ElementS = cute::remove_cvref_t<decltype(tSrS(0))>;
  auto p_sl0 = cute::detail::subbyte_sg_tv_swizzle<ElementS>(
      project_strides(tSrS.tv_layout()));
  auto p_dl0 = cute::detail::subbyte_sg_tv_swizzle<ElementP>(
      project_strides(tArP.tv_layout()));
  using PReorderLayout =
      decltype(coalesce(composition(right_inverse(p_dl0), p_sl0)));
  constexpr int kNumP = decltype(tArP.size())::value;
  using DecodePackLayout =
      Layout<Shape<_16, C<kNumP / 2>, _2>, Stride<_1, _32, _16>>;
  constexpr bool kCanUseE5M2PackedReorder =
      std::is_same_v<ElementP, cutlass::float_e5m2_t> &&
      (std::is_same_v<
           PReorderLayout,
           Layout<Shape<_16, _16, _2, _2>, Stride<_1, _32, _16, _512>>> ||
       std::is_same_v<
           PReorderLayout,
           Layout<Shape<_16, _16, _2, _4>, Stride<_1, _32, _16, _512>>>);
  constexpr bool kCanUseE4M3PackedReorder =
      std::is_same_v<ElementP, cutlass::float_e4m3_t> &&
      (std::is_same_v<
           PReorderLayout,
           Layout<Shape<_16, _16, _2, _2>, Stride<_1, _32, _16, _512>>> ||
       std::is_same_v<
           PReorderLayout,
           Layout<Shape<_16, _16, _2, _4>, Stride<_1, _32, _16, _512>>>);
  constexpr bool kCanUseE5M2DecodePack =
      std::is_same_v<ElementP, cutlass::float_e5m2_t> &&
      std::is_same_v<PReorderLayout, DecodePackLayout>;
  constexpr bool kCanUseE4M3DecodePack =
      std::is_same_v<ElementP, cutlass::float_e4m3_t> &&
      std::is_same_v<PReorderLayout, DecodePackLayout>;

  if constexpr (kCanUseE5M2PackedReorder) {
    static_assert(
        decltype(tArP.size())::value % 64 == 0,
        "Packed e5m2 reorder expects a multiple of 64 per-WI P elements");
    constexpr int kBlocks = kNumP / 32;
    Tensor tArP_pair = recast<cute::intel::uchar2>(tArP);
    CUTLASS_PRAGMA_UNROLL
    for (int g = 0; g < kBlocks; g++) {
      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < 16; p++) {
        cvt_f32x2_to_e5m2x2_pack(
            tSrS(g * 32 + p), tSrS(g * 32 + p + 16), tArP_pair(g * 16 + p));
      }
    }
  } else if constexpr (kCanUseE4M3PackedReorder) {
    static_assert(
        decltype(tArP.size())::value % 64 == 0,
        "Packed e4m3 reorder expects a multiple of 64 per-WI P elements");
    constexpr int kBlocks = kNumP / 32;
    Tensor tArP_pair = recast<cute::intel::uchar2>(tArP);
    CUTLASS_PRAGMA_UNROLL
    for (int g = 0; g < kBlocks; g++) {
      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < 16; p++) {
        cvt_f32x2_to_e4m3x2_pack(
            tSrS(g * 32 + p), tSrS(g * 32 + p + 16), tArP_pair(g * 16 + p));
      }
    }
  } else if constexpr (kCanUseE5M2DecodePack) {
    constexpr int kHalf = kNumP / 2;
    Tensor tArP_pair = recast<cute::intel::uchar2>(tArP);
    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < kHalf; p++) {
      cvt_f32x2_to_e5m2x2_pack(tSrS(p), tSrS(p + kHalf), tArP_pair(p));
    }
  } else if constexpr (kCanUseE4M3DecodePack) {
    constexpr int kHalf = kNumP / 2;
    Tensor tArP_pair = recast<cute::intel::uchar2>(tArP);
    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < kHalf; p++) {
      cvt_f32x2_to_e4m3x2_pack(tSrS(p), tSrS(p + kHalf), tArP_pair(p));
    }
  } else {
    reorder(tSrS, tArP);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class DispatchPolicy_,
    bool CausalMask_,
    bool LocalMask_,
    bool PagedKV_,
    class TiledMMAQK_,  // Tiling for Q*K GEMM
    class TiledMMAPV_,  // Tiling for P*V GEMM
    int VTiles_,        // # of tiles in V dimension
    class TensorQ_,     // Global Q/K/V tensors
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_ = void,  // Optional TiledCopy for loading Q
    class TiledCopyK_ = void,  // Optional TiledCopy for loading K
    class TiledCopyV_ = void>  // Optional TiledCopy for loading V
struct FMHAFwdMainloop {
  static_assert(
      cutlass::detail::dependent_false<DispatchPolicy_>,
      "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int Stages,
    bool CausalMask_,
    bool LocalMask_,
    bool PagedKV_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_,
    class TiledCopyK_,
    class TiledCopyV_>
struct FMHAFwdMainloop<
    XeDefault<Stages>,
    CausalMask_,
    LocalMask_,
    PagedKV_,
    TiledMMAQK_,
    TiledMMAPV_,
    VTiles_,
    TensorQ_,
    TensorK_,
    TensorV_,
    TiledCopyQ_,
    TiledCopyK_,
    TiledCopyV_> {
  //
  // Type Aliases
  //
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  // Head size (VTiles * PV N-tile) must be divisible by the QK K-tile so that
  // the GEMM1 reduction dimension splits evenly into DTiles register tiles.
  static_assert(
      (VTiles * decltype(get<1>(TileShapePV{}))::value) %
              decltype(get<2>(TileShapeQK{}))::value ==
          0,
      "Head size (VTiles * PV N-tile) must be divisible by the QK "
      "K-tile (BLK_QK_D); check the ShapeQK/ShapePV tile "
      "configuration for this head dimension.");
  static constexpr int DTiles = VTiles *
                                decltype(get<1>(TileShapePV{}))::value /
                                decltype(get<2>(TileShapeQK{}))::value;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(
      take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;

  using ElementQ = typename TensorQ::engine_type::value_type;
  using ElementK = typename TensorK::engine_type::value_type;

  using TensorQ2D =
      decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));
  using TensorK2D =
      decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorV2D =
      decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));

  using TiledCopyQ = conditional_t<
      is_void_v<TiledCopyQ_>,
      decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})),
      TiledCopyQ_>;
  using TiledCopyK = conditional_t<
      is_void_v<TiledCopyK_>,
      decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})),
      TiledCopyK_>;
  using TiledCopyV = conditional_t<
      is_void_v<TiledCopyV_>,
      decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})),
      TiledCopyV_>;

  // TODO: static_asserts on TiledMMAPV here...

  //
  // Accumulator types
  //
  // FragS:    accumulator for Q*K MMA
  // FragO:    accumulator for P*V MMAs.
  //           Note: v mode may be split into multiple pieces
  //             to reduce register pressure.
  // Frag*Row types are reductions of the corresponding Frag* types
  //   over rows.
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  // Per-lane partial row sum: vertical (in-register) reduction only, deferring
  // the cross-lane horizontal hreduce of the softmax denominator to the
  // epilogue.
  using FragSPartialRow =
      decltype(reduce<1, ReduceMode::Vertical>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;  // (atom val,q',v')
  using FragA =
      expand_sg_fragment_t<SingleFragA, 1, VTiles>;  // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool Fp8KV =
      is_any_of_v<ElementK, float_e5m2_t, float_e4m3_t>;
  // bf16 Q consumed by an fp16 MMA (bf16 Q + fp8 KV): Q is range-guarded
  // before the narrowing bf16 -> fp16 conversion.
  static constexpr bool NarrowQToHalf =
      cute::is_same_v<ElementQ, bfloat16_t> &&
      cute::is_same_v<typename TiledMMAQK::ValTypeA, half_t>;
  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool LocalMask = LocalMask_;
  static constexpr bool PagedKV = PagedKV_;

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
    void* const scale_k;
    void* const scale_v;
    void* const scale_q;

    // Paged KV Cache
    const int* ptr_page_table;
    int page_size;
    int max_pages_per_seq;
    int total_seqlen_kv;
    // Local Mask
    int local_left, local_right;
    // Physical stride between paged blocks in seq-position units
    int page_stride_elements;
  };

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data
  struct SharedStorage {};

  Params params;

  //
  // Methods
  //

  FMHAFwdMainloop(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr Params
  to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;  // log_2(e)
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    return Params{
        val,
        args.scale_k,
        args.scale_v,
        args.scale_q,
        args.ptr_page_table,
        args.page_size,
        args.max_pages_per_seq,
        args.total_seqlen_kv,
        args.local_left,
        args.local_right,
        args.page_stride_elements};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  CUTLASS_DEVICE int get_paged_idx(int K, int idx_b) {
    constexpr int kv_tile = get<1>(TileShapeQK{});
    int tiles_per_page = params.page_size / kv_tile;
    int b_offset = idx_b * params.max_pages_per_seq;
    int page_local_idx, tile_in_page;

    // Xe lacks integer divide, so shift/mask when tiles_per_page is a power of
    // two (true for every supported page_size/tile combination).
    if (tiles_per_page > 0 && (tiles_per_page & (tiles_per_page - 1)) == 0) {
      unsigned const shift = static_cast<unsigned>(
          __builtin_ctz(static_cast<unsigned>(tiles_per_page)));
      page_local_idx = K >> shift;
      tile_in_page = K & (tiles_per_page - 1);
    } else {
      page_local_idx = K * kv_tile / params.page_size;
      tile_in_page = K % tiles_per_page;
    }

    // Clamp page_local_idx to the valid range [0, max_pages_per_seq - 1]
    if (page_local_idx >= params.max_pages_per_seq) {
      page_local_idx = params.max_pages_per_seq - 1;
    }

    int block_idx = params.ptr_page_table[b_offset + page_local_idx];
    int page_stride_tiles = params.page_stride_elements / kv_tile;
    return block_idx * page_stride_tiles + tile_in_page;
  }

  template <bool has_large_surface, typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,    // (q,d)
      TensorK2D const& K_2D,    // (k,d)
      TensorV2D const& V_2D,    // (d,k)
      FragA& tArA,              // Output accumulator (q,v)
      FragARow& tA_max,         // Softmax row-wise max accumulator
      FragSPartialRow& tA_sum,  // Softmax row-wise partial sum (per-lane)
      QVCoord blk_qv,           // WG tile indices: (Q,V)
      int const& idx_b,         // WG tile indices: (B)
      int blk_k0,               // K block range: [K0,K1)
      int blk_k1,
      int blk_k1_causal,
      int thr_id,
      int seq_len,
      int full_tile_offset,
      int blk_local_l_safe,
      int blk_local_r_safe) {
    using namespace sycl::ext::oneapi::this_work_item;

    // Short dimension names:
    //    q = sequence len dimension for Q
    //    k = sequence len dimension for K
    //    d = head size dimension for K/Q
    //    v = head size dimension for V
    //   VV = MMA tile indices for V
    // Capital letters (Q, K, ...) refer to WG block indices.
    // Primed letters (q', k', ...) refer to atom block indices.

    constexpr int tile_k = get<1>(TileShapeQK{});

    auto tile_shape_v =
        make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    /* Create proxy coordinate tensors for Q/K/P/V */
    Tensor cQ = make_identity_tensor(Q_2D.shape());               // (q,d)
    Tensor cK = make_identity_tensor(K_2D.shape());               // (k,d)
    Tensor cV = make_identity_tensor(V_2D.shape());               // (v,k)
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));  // (q,k)

    /* Partition global tensors into workgroup tiles */
    Tensor gQ = local_tile(
        cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});  // (q,d,D)
    Tensor gK = local_tile(
        cK,
        TileShapeQK{},
        make_coord(_, _, _),
        Step<X, _1, _1>{});  // (k,d,K,D)
    Tensor gV =
        local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _));  // (v,k,K)
    Tensor gV_split = local_tile(
        gV,
        TileShapePV{},
        make_coord(_, _, 0),
        Step<X, _1, _1>{});  // (v,k,VV,K)
    auto tile_shape_k =
        make_shape(get<1>(TileShapeQK{}), get<2>(TileShapeQK{}) * C<DTiles>{});
    Tensor gK_prefetch =
        local_tile(cK, tile_shape_k, make_coord(_, 0));  // (k,d,K)

    /* Create global -> register copies */
    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{xe_kv_first_sub_surface<has_large_surface, 0>(K_2D)};
    TiledCopyV copy_v{xe_kv_first_sub_surface<has_large_surface, 1>(V_2D)};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    /* Slice TiledCopy/TiledMMA operations down to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tQgQ = thr_copy_q.partition_S(gQ);        // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);        // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);  // (atom_val,v',k',VV,K)

    /* Create register fragments for MMA and copies */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    [[maybe_unused]] auto tSrQ =
        thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));
    std::array<decltype(tSrQ), DTiles> tSrQ_arr;
    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));
    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);
    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));

    /* Create TiledCopy objects for prefetches */
    constexpr int RegularKVPrefetchSGs =
        cute::bit_floor(static_cast<unsigned>(SGPerWG::value));
    constexpr int RegularKVPrefetchThreads =
        RegularKVPrefetchSGs * intel::sg_size;
    constexpr bool kAllSGsPrefetch =
        (RegularKVPrefetchSGs == static_cast<int>(SGPerWG::value));
    int const prefetch_thr_id =
        kAllSGsPrefetch ? thr_id
                        : (thr_id < RegularKVPrefetchThreads ? thr_id : 0);
    const auto subgroup_id = thr_id / intel::sg_size;
    bool const prefetch_sg_active =
        kAllSGsPrefetch || (subgroup_id < RegularKVPrefetchSGs);

    auto prefetch_k = make_block_2d_prefetch<RegularKVPrefetchSGs>(
        tile_shape_k, xe_kv_first_sub_surface<has_large_surface, 0>(K_2D));
    auto prefetch_v = make_block_2d_prefetch<RegularKVPrefetchSGs>(
        tile_shape_v, xe_kv_first_sub_surface<has_large_surface, 1>(V_2D));
    /* Partition global tensors for prefetch */
    auto pKgK = prefetch_k.get_slice(prefetch_thr_id).partition_S(gK_prefetch);
    auto pVgV = prefetch_v.get_slice(prefetch_thr_id).partition_S(gV);

    // ------
    // Kernel
    // ------

    /* Initialization steps for first block: Q/K prefetch, O init */
    using PreparedK_t = decltype(prepare_payloads(copy_k, tKgK(_, _, _, 0, 0)));
    using PreparedV_t = decltype(prepare_payloads(copy_v, tVgV(_, _, _, 0, 0)));
    std::array<PreparedK_t, DTiles> prepared_k;
    std::array<PreparedV_t, VTiles> prepared_v;

    auto k_seq_delta = [](auto n) { return make_coord(n, _0{}); };
    auto v_seq_delta = [](auto n) { return make_coord(_0{}, n); };

    /* Preload + reorder Q once; reused across all K iterations. */
    [[maybe_unused]] float q_amax = 0.f;
    CUTLASS_PRAGMA_UNROLL
    for (int D = 0; D < DTiles; D++) {
      copy(copy_q, tQgQ(_, _, _, D), tQrQ);
      if constexpr (NarrowQToHalf) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tQrQ.size(); i++)
          q_amax = sycl::fmax(q_amax, sycl::fabs(static_cast<float>(tQrQ(i))));
      }
      reorder(tQrQ, tSrQ_arr[D]);
    }

    // bf16 -> fp16 Q range guard. bf16 values >= 2^16 overflow fp16 and very
    // small ones lose precision as fp16 subnormals. If this subgroup's Q rows
    // fall outside [2^-6, 2^16), reload Q scaled by 2^-q_shift (moving max|q|
    // into [2^14, 2^15)) and fold 2^q_shift into the softmax scale.
    // Power-of-two scaling is exact, so S = Q*K^T is unchanged; in-range tiles
    // (the normal case) take the fast path unmodified. Subgroups own whole Q
    // rows, so the per-subgroup shift is consistent for every row-wise softmax
    // reduction.
    [[maybe_unused]] int q_shift = 0;
    if constexpr (NarrowQToHalf) {
      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      q_amax = sycl::reduce_over_group(sg, q_amax, sycl::maximum<float>());
      bool const q_out_of_range = sycl::isfinite(q_amax) && q_amax != 0.f &&
                                  (q_amax >= 65536.f || q_amax < 0.015625f);
      if (q_out_of_range) {
        q_shift = sycl::ilogb(q_amax) - 14;
        float const q_rescale = sycl::ldexp(1.f, -q_shift);
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < DTiles; D++) {
          copy(copy_q, tQgQ(_, _, _, D), tQrQ);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tQrQ.size(); i++)
            tQrQ(i) = ElementQ(static_cast<float>(tQrQ(i)) * q_rescale);
          reorder(tQrQ, tSrQ_arr[D]);
        }
      }
    }

    /* Masking scalars */
    int lane_id = thr_id % intel::sg_size;
    constexpr int sg_tile_q = get<0>(TileShapeQK{}) / SGPerWG::value;
    int row_base = get<0>(blk_qv) * get<0>(TileShapeQK{}) +
                   (thr_id / intel::sg_size) * sg_tile_q;

    // PagedKV
    constexpr int kv_stride = get<1>(TileShapeQK{});
    int page_idx, next_page_idx = blk_k0, page_offset = kv_stride;
    if constexpr (PagedKV) {
      next_page_idx = get_paged_idx(blk_k0, idx_b);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int D = 0; D < DTiles; D++) {
      prepared_k[D] = prepare_payloads(copy_k, tKgK(_, _, _, next_page_idx, D));
    }
    CUTLASS_PRAGMA_UNROLL
    for (int VV = 0; VV < VTiles; VV++) {
      prepared_v[VV] =
          prepare_payloads(copy_v, tVgV(_, _, _, VV, next_page_idx));
    }

    auto prepared_pk =
        prepare_payloads(prefetch_k, pKgK(_, _, _, next_page_idx));
    auto prepared_pv =
        prepare_payloads(prefetch_v, pVgV(_, _, _, next_page_idx));

    // Large-surface mode: payloads address tiles relative to their sub-surface.
    [[maybe_unused]] int kv_surf = 0, last_surf = 0;
    if constexpr (has_large_surface) {
      last_surf = xe_kv_surface<1>(get<0>(K_2D.shape()) - 1);
      kv_surf = cute::min(xe_kv_surface<tile_k>(next_page_idx), last_surf);
      if (kv_surf != 0) {
        int const rows = -kv_surf * kXeMaxSurfaceRows;
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < DTiles; D++) {
          prepared_k[D] += k_seq_delta(rows);
          xe_kv_retarget_payloads<0>(prepared_k[D], K_2D, kv_surf);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          prepared_v[VV] += v_seq_delta(rows);
          xe_kv_retarget_payloads<1>(prepared_v[VV], V_2D, kv_surf);
        }
        prepared_pk += k_seq_delta(rows);
        xe_kv_retarget_payloads<0>(prepared_pk, K_2D, kv_surf);
        prepared_pv += v_seq_delta(rows);
        xe_kv_retarget_payloads<1>(prepared_pv, V_2D, kv_surf);
      }
    }

    if (prefetch_sg_active) {
      prefetch(prefetch_k, prepared_pk);
      prefetch(prefetch_v, prepared_pv);
    }

    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    bool check_remainder_k = (seq_len % tile_k != 0);

    // FP8 KV Scale (per-tensor): fold scale_k into the softmax scale (S = Q·K
    // is linear in K) and defer scale_v to a single post-loop output rescale
    // (O = sum P_i·V_i is linear in V), so no per-element dequantize is needed
    // in the K loop.
    ElementS effective_scale = params.scale;
    float scale_v = 1.f;
    if constexpr (Fp8KV) {
      const float scale_k = *static_cast<const float*>(params.scale_k);
      scale_v = *static_cast<const float*>(params.scale_v);
      effective_scale = params.scale * ElementS(scale_k);
    }
    if constexpr (NarrowQToHalf) {
      if (q_shift != 0)
        effective_scale = effective_scale * ElementS(sycl::ldexp(1.f, q_shift));
    }
    constexpr int kAtomsPerD =
        decltype(get<2>(TileShapeQK{}))::value /
        decltype(get<2>(typename TiledMMAQK::AtomShape_MNK{}))::value;

    /* Main loop, blocked in k. */
    for (int K = blk_k0; K < blk_k1; K++) {
      bool need_causal = false;
      if constexpr (CausalMask) {
        need_causal = K >= blk_k1_causal;
      }

      page_idx = next_page_idx;
      next_page_idx = K + 1;
      // next paged_idx
      if constexpr (PagedKV) {
        next_page_idx = get_paged_idx(next_page_idx, idx_b);
        page_offset = (next_page_idx - page_idx) * kv_stride;
      }

      // Large-surface mode: sub-surface relative delta; prefetches switch here.
      [[maybe_unused]] bool surf_switch = false;
      if constexpr (has_large_surface) {
        int const surf =
            cute::min(xe_kv_surface<tile_k>(next_page_idx), last_surf);
        page_offset -= (surf - kv_surf) * kXeMaxSurfaceRows;
        surf_switch = surf != kv_surf;
        kv_surf = surf;
        if (surf_switch) {
          xe_kv_retarget_payloads<0>(prepared_pk, K_2D, kv_surf);
          xe_kv_retarget_payloads<1>(prepared_pv, V_2D, kv_surf);
        }
      }

      /* V prefetch for next iter */
      if (prefetch_sg_active) {
        prepared_pv += v_seq_delta(page_offset);
        prefetch(prefetch_v, prepared_pv);
      }

      /* GEMM 1: S = Q * K^T */
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < DTiles; D++) {
        copy(copy_k, prepared_k[D], tKrK);
        prepared_k[D] += k_seq_delta(page_offset);
        reorder(tKrK, tSrK);

        auto const& tSrQ_d = tSrQ_arr[D];
        if (D == 0) {
          // Null-src0 DPAS on the first K atom initializes tSrS without a
          // clear(). SYCL-TLA v0.1.0_next selects it through the MMA atom
          // (without_acc) instead of the former gemm<NoAcc> parameter.
          cute::gemm(
              mma_qk.without_acc(), tSrQ_d(_, _, 0), tSrK(_, _, 0), tSrS);
          CUTLASS_PRAGMA_UNROLL
          for (int k = 1; k < kAtomsPerD; k++) {
            cute::gemm(mma_qk, tSrQ_d(_, _, k), tSrK(_, _, k), tSrS);
          }
        } else {
          cute::gemm(mma_qk, tSrQ_d, tSrK, tSrS);
        }
      }

      /* K prefetch for next iteration */
      if (prefetch_sg_active) {
        prepared_pk += k_seq_delta(page_offset);
        prefetch(prefetch_k, prepared_pk);
      }

      /* Early V-load for VV=0: overlap memory latency with softmax scalar work
       */
      copy(copy_v, prepared_v[0], tVrV);
      prepared_v[0] += v_seq_delta(page_offset);

      /* Causal masking and k remainder masking */
      if constexpr (CausalMask) {
        if (need_causal) {
          constexpr int kTileK = tile_k;
          constexpr int n_reps = kTileK / intel::sg_size;
          constexpr int elems_per_n = tSrS.size() / n_reps;
          int k_base = K * kTileK;
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < n_reps; n++) {
            int col = k_base + n * intel::sg_size + lane_id;
            int causal_bound = col - full_tile_offset - row_base;
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < elems_per_n; j++) {
              if (j < causal_bound) {
                tSrS(n * elems_per_n + j) = ElementS(-INFINITY);
              }
            }
          }
        }
      }
      /* k masking for remainder tiles */
      if (check_remainder_k && K == blk_k1 - 1) {
        // Tensor-free remainder masking
        constexpr int n_reps = tile_k / intel::sg_size;
        constexpr int elems_per_n = tSrS.size() / n_reps;
        int k_base = K * tile_k + lane_id;
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < n_reps; n++) {
          if (k_base + n * intel::sg_size >= seq_len) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < elems_per_n; j++) {
              tSrS(n * elems_per_n + j) = ElementS(-INFINITY);
            }
          }
        }
      }
      /* Local masking */
      if constexpr (LocalMask) {
        const bool need_left = (K < blk_local_l_safe);
        const bool need_right = (K > blk_local_r_safe);
        if (need_left || need_right) {
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(
              cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i)) - full_tile_offset;
            bool left_mask = col_idx < row_idx - params.local_left;
            bool right_mask = col_idx > row_idx + params.local_right;
            if (left_mask || right_mask) {
              tSrS(i) = ElementS(-INFINITY);
            }
          }
        }
      }

      /* Apply softmax (deferred row-sum hreduce) */
      auto [rescale, tS_partial_sum, subgroup_needs_rescale] =
          softmax(effective_scale, tSrS, tA_max, tA_sum);
      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      constexpr int kSumSize = decltype(tA_sum.size())::value;
      constexpr bool kSumDivVT = (kSumSize % VTiles == 0);
      constexpr int kSumPerVT = kSumDivVT ? (kSumSize / VTiles) : 0;

      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension.
       * VV=0 pre-loaded before softmax; VV>0 loaded inline. */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        if (VV > 0) {
          copy(copy_v, prepared_v[VV], tVrV);
          prepared_v[VV] += v_seq_delta(page_offset);
        }
        reorder(tVrV, tArV);

        if (subgroup_needs_rescale) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = tArA.size() / VTiles - 1; i >= 0; i--)
            tArA(_, _, _, VV)(i) *= broadcast<0>(rescale, tArA, i);
        }
        /* Rescale + accumulate per-lane partial row sums, fused into V loop. */
        if constexpr (kSumDivVT) {
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < kSumPerVT; j++) {
            int const i = VV * kSumPerVT + j;
            // rescale is subgroup-interleaved: row i lives in lane
            // i % sg_size, register i / sg_size.
            tA_sum(i) =
                tA_sum(i) *
                    group_broadcast(
                        sg, rescale(i / intel::sg_size), i % intel::sg_size) +
                tS_partial_sum(i);
          }
        }
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }

      // Large-surface mode: loads switch once this tile's loads are issued.
      if constexpr (has_large_surface) {
        if (surf_switch) {
          CUTLASS_PRAGMA_UNROLL
          for (int D = 0; D < DTiles; D++) {
            xe_kv_retarget_payloads<0>(prepared_k[D], K_2D, kv_surf);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int VV = 0; VV < VTiles; VV++) {
            xe_kv_retarget_payloads<1>(prepared_v[VV], V_2D, kv_surf);
          }
        }
      }
    }

    // FP8 KV: apply the deferred per-tensor V scale once after the K loop.
    if constexpr (Fp8KV) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tArA.size(); ++i) {
        tArA(i) *= ElementA(scale_v);
      }
    }
  }

  // Single step of blocked softmax.
  // Returns (rescale, per-lane vertical partial sum). The cross-lane horizontal
  // reduction of the row-sum is deferred: tS_sum keeps per-lane partial sums
  // and is collapsed once in the epilogue (reduce<0, Horizontal>).
  CUTLASS_DEVICE
  auto softmax(
      ElementS scale,             // Softmax scale
      FragS& tS,                  // Softmax src/dst block
      FragARow& tA_max,           // Softmax row-wise max accumulator
      FragSPartialRow& tA_sum) {  // Softmax row-wise partial sum (per-lane)

    /* Compute row-wise maxima for this block */
    auto tS_bmax =
        reduce<1, ReduceMode::Full, /*EnableFast64Rows=*/!CausalMask>(
            tS, sycl::maximum<void>{});

    FragARow rescale;
    uint32_t subgroup_needs_rescale = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tA_max.size(); i++) {
      ElementS new_max = sycl::max(tA_max(i), scale * tS_bmax(i));
      subgroup_needs_rescale |= subgroup_any_less(tA_max(i), new_max);
      rescale(i) = sycl::native::exp2(tA_max(i) - new_max);
      tA_max(i) = new_max;
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = scale * tS(i) - broadcast<0>(tA_max, tS, i);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(tS(i));

    /* Per-lane vertical partial sum (deferred horizontal reduction) */
    auto tS_partial_sum =
        reduce<1, ReduceMode::Vertical>(tS, sycl::plus<void>{});

    constexpr int kSumSize = decltype(tA_sum.size())::value;
    constexpr bool kSumDivVT = (kSumSize % VTiles == 0);

    // When tS_sum.size() does not divide VTiles, rescale + accumulate the
    // partial sums once here instead of fusing per VTile in the PV loop.
    if constexpr (!kSumDivVT) {
      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kSumSize; i++) {
        // rescale is subgroup-interleaved: row i lives in lane
        // i % sg_size, register i / sg_size.
        tA_sum(i) =
            tA_sum(i) *
                group_broadcast(
                    sg, rescale(i / intel::sg_size), i % intel::sg_size) +
            tS_partial_sum(i);
      }
    }

    return cute::make_tuple(rescale, tS_partial_sum, subgroup_needs_rescale);
  }
};

template <
    class DispatchPolicy_,
    bool PagedKV_,
    bool CausalMask_,
    class TiledMMAQK_,  // Tiling for Q*K GEMM
    class TiledMMAPV_,  // Tiling for P*V GEMM
    int VTiles_,        // # of tiles in V dimension
    class TensorQ_,     // Global Q/K/V tensors
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_ = void,  // Optional TiledCopy for loading Q
    class TiledCopyK_ = void,  // Optional TiledCopy for loading K
    class TiledCopyV_ = void,  // Optional TiledCopy for loading V
    bool LocalMask_ = false>
struct DecodeFwdMainloop {
  static_assert(
      cutlass::detail::dependent_false<DispatchPolicy_>,
      "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int Stages,
    bool PagedKV_,
    bool CausalMask_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_,
    class TiledCopyK_,
    class TiledCopyV_,
    bool LocalMask_>
struct DecodeFwdMainloop<
    XeDefault<Stages>,
    PagedKV_,
    CausalMask_,
    TiledMMAQK_,
    TiledMMAPV_,
    VTiles_,
    TensorQ_,
    TensorK_,
    TensorV_,
    TiledCopyQ_,
    TiledCopyK_,
    TiledCopyV_,
    LocalMask_> {
  //
  // Type Aliases
  //
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(
      take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;

  using ElementQ = typename TensorQ::engine_type::value_type;
  using ElementK = typename TensorK::engine_type::value_type;

  using TensorQ2D =
      decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));
  using TensorK2D =
      decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorV2D =
      decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));

  using TiledCopyQ = conditional_t<
      is_void_v<TiledCopyQ_>,
      decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})),
      TiledCopyQ_>;
  using TiledCopyK = conditional_t<
      is_void_v<TiledCopyK_>,
      decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})),
      TiledCopyK_>;
  using TiledCopyV = conditional_t<
      is_void_v<TiledCopyV_>,
      decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})),
      TiledCopyV_>;

  // TODO: static_asserts on TiledMMAPV here...

  //
  // Accumulator types
  //
  // FragS:    accumulator for Q*K MMA
  // FragO:    accumulator for P*V MMAs.
  //           Note: v mode may be split into multiple pieces
  //             to reduce register pressure.
  // Frag*Row types are reductions of the corresponding Frag* types
  //   over rows.
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;  // (atom val,q',v')
  using FragA =
      expand_sg_fragment_t<SingleFragA, 1, VTiles>;  // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  // static_assert(is_same_v<decltype(FragSRow{}.shape()), float>, "dtype
  // mismatched");
  using ElementA = typename TiledMMAPV::ValTypeD;
  using ElementP = typename TiledMMAPV::ValTypeA;

  static constexpr bool PagedKV = PagedKV_;
  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool Fp8KV =
      is_any_of_v<ElementK, float_e5m2_t, float_e4m3_t>;
  // True for the full-fp8 path (fp8 Q -> fp8 Q*K and fp8 P*V GEMMs). In that
  // case the softmax applies a +8 exponent offset so the probabilities span a
  // range better represented by fp8 (see softmax()).
  static constexpr bool Fp8Q =
      is_any_of_v<ElementQ, float_e5m2_t, float_e4m3_t>;
  static constexpr bool LocalMask = LocalMask_;

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
    void* const scale_q;
    void* const scale_k;
    void* const scale_v;
    // Paged KV Cache
    int const* ptr_page_table;
    int page_size;
    int max_pages_per_seq;
    int total_seqlen_kv;
    // Local Mask
    int window_size_left;
    int window_size_right;
    // Physical stride between paged blocks in seq-position units
    int page_stride_elements;
    // Packed speculative decode: the packed head-group rows hold spec_q_len
    // consecutive query positions (spec_rows_per_pos rows each, position-
    // major). Row r is query position j = r / spec_rows_per_pos and attends
    // to the first seq_len - (spec_q_len - 1 - j) keys. 1 disables it.
    int spec_q_len = 1;
    int spec_rows_per_pos = 1;
  };

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data
  struct SharedStorage {};

  Params params;

  //
  // Methods
  //

  DecodeFwdMainloop(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr Params
  to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;  // log_2(e)
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    return Params{
        val,
        args.scale_q,
        args.scale_k,
        args.scale_v,
        args.ptr_page_table,
        args.page_size,
        args.max_pages_per_seq,
        args.total_seqlen_kv,
        args.window_size_left,
        args.window_size_right,
        args.page_stride_elements,
        args.spec_q_len,
        args.spec_rows_per_pos};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  template <bool has_large_surface, typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,  // (q,d)
      TensorK2D const& K_2D,  // (k,d)
      TensorV2D const& V_2D,  // (d,k)
      FragA& tArA,            // Output accumulator (q,v)
      FragARow& tA_max,       // Softmax row-wise max accumulator
      FragARow& tA_sum,       // Softmax row-wise sum accumulator
      QVCoord blk_qv,         // WG tile indices: (Q,V)
      int const& idx_b,       // WG tile indices: (B)
      int pair_head,          // KV head of K_2D/V_2D to pair-prefetch, or -1
      int blk_k0,             // K block range: [K0,K1)
      int blk_k1,
      int total_blk,  // Total # of K blocks
      int thr_id,
      int seq_len,
      int full_tile_offset,
      int discard_seq_coord) {
    using namespace sycl::ext::oneapi::this_work_item;

    // Short dimension names:
    //    q = sequence len dimension for Q
    //    k = sequence len dimension for K
    //    d = head size dimension for K/Q
    //    v = head size dimension for V
    //   VV = MMA tile indices for V
    // Capital letters (Q, K, ...) refer to WG block indices.
    // Primed letters (q', k', ...) refer to atom block indices.

    auto tile_shape_v =
        make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    /* Create proxy coordinate tensors for Q/K/P/V */
    Tensor cQ = make_identity_tensor(Q_2D.shape());               // (q,d)
    Tensor cK = make_identity_tensor(K_2D.shape());               // (k,d)
    Tensor cV = make_identity_tensor(V_2D.shape());               // (v,k)
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));  // (q,k)

    /* Partition global tensors into workgroup tiles */
    Tensor gQ = local_tile(
        cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});  // (q,d,D)
    Tensor gK = local_tile(
        cK,
        TileShapeQK{},
        make_coord(_, _, _),
        Step<X, _1, _1>{});  // (k,d,K,D)
    Tensor gV =
        local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _));  // (v,k,K)
    Tensor gV_split = local_tile(
        gV,
        TileShapePV{},
        make_coord(_, _, 0),
        Step<X, _1, _1>{});  // (v,k,VV,K)

    // In has_large_surface mode the K/V copy and prefetch payloads address
    // one sub-surface at a time (see kXeMaxSurfaceRows), starting with the
    // first. Copies use absolute tile coordinates, so a sub-surface switch
    // only re-points base and height of the existing payloads.
    auto K_2D_0 = K_2D;
    auto V_2D_0 = V_2D;
    if constexpr (has_large_surface) {
      static_assert(PagedKV, "only paged K/V is split into sub-surfaces");
      K_2D_0 = xe_kv_sub_surface<0>(K_2D, 0);
      V_2D_0 = xe_kv_sub_surface<1>(V_2D, 0);
    }

    /* Create global -> register copies */
    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{K_2D_0};
    TiledCopyV copy_v{V_2D_0};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    auto copyQ = make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{});

    /* Slice TiledCopy/TiledMMA operations down to to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tQgQ = thr_copy_q.partition_S(gQ);        // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);        // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);  // (atom_val,v',k',VV,K)

    /* Create register fragments for MMA and copies */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));

    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));

    /* Create TiledCopy objects for prefetches */
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_v =
        make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D_0);

    /* Partition global tensors for prefetch */
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV);

    // ------
    // Kernel
    // ------

    // PagedKV
    int tiles_per_page = params.page_size / get<1>(TileShapeQK{});
    int page_stride_tiles = params.page_stride_elements / get<1>(TileShapeQK{});
    int tile_idx = blk_k0;
    int b_offset = idx_b * params.max_pages_per_seq;
    if constexpr (PagedKV) {
      int page_local_idx = tile_idx * get<1>(TileShapeQK{}) / params.page_size;
      int block_idx = params.ptr_page_table[b_offset + page_local_idx];
      tile_idx = block_idx * page_stride_tiles + tile_idx % tiles_per_page;
    }

    // Tuned decode kernels (paged_decode_tuning.h) read the page table one
    // tile ahead, so the next tile's K prefetch at the end of an iteration
    // does not wait for a dependent page-table load. (Issuing the K/V
    // prefetches earlier than they are today was measured slower.)
    constexpr bool kPipelined =
        PagedKV && flash_xpu_splitkv::paged_decode_tuning::is_tuned(
                       get<1>(TileShapePV{}) * VTiles, get<1>(TileShapeQK{}));
    int tile_idx_next = tile_idx;
    if constexpr (kPipelined) {
      const int k1 = blk_k0 + 1;
      if (k1 < blk_k1) {
        int page_local_idx = k1 * get<1>(TileShapeQK{}) / params.page_size;
        tile_idx_next = params.ptr_page_table[b_offset + page_local_idx] *
                            page_stride_tiles +
                        k1 % tiles_per_page;
      }
    }

    // Full fp8 head_dim 256 with a token-major (NHD) cache: a token's KV heads
    // sit side by side, so one head's K/V are 256-byte runs that stream poorly
    // from DRAM. Each work-group prefetches the rows of its head and of the
    // paired head (pair_head ^ 1), so DRAM sees 512 contiguous bytes per token
    // (pairing 128-byte rows at head_dim 128 was slower). The pair prefetch
    // spans the whole K/V pool, so it is off in has_large_surface mode. Its
    // payloads stay live next to the per-head ones, which spills with 256 GRF.
    constexpr int kHeadDim = get<1>(TileShapePV{}) * VTiles;
    constexpr bool kPairPrefetch = kPipelined && Fp8Q && kHeadDim > 128 &&
                                   !has_large_surface && VLLM_GRF_SIZE >= 512;
    const int kv_row = stride<0>(K_2D);  // elements between tokens
    const bool pair_prefetch = kPairPrefetch && pair_head >= 0 &&
                               kv_row >= 2 * kHeadDim &&
                               ((pair_head | 1) + 1) * kHeadDim <= kv_row;
    auto pair = [&]() {
      if constexpr (kPairPrefetch) {
        using KvTile = decltype(get<1>(TileShapeQK{}));
        using PairTileK = Shape<KvTile, Int<2 * kHeadDim>>;
        using PairTileV = Shape<Int<2 * kHeadDim>, KvTile>;
        const int offset = -(pair_head & 1) * kHeadDim;
        auto K_pair = make_tensor(
            K_2D.data() + offset,
            make_layout(
                make_shape(get<0>(K_2D.shape()), Int<2 * kHeadDim>{}),
                make_stride(kv_row, _1{})));
        auto V_pair = make_tensor(
            V_2D.data() + offset,
            make_layout(
                make_shape(Int<2 * kHeadDim>{}, get<1>(V_2D.shape())),
                make_stride(_1{}, stride<1>(V_2D))));
        auto copy_k_pair =
            make_block_2d_prefetch<SGPerWG::value>(PairTileK{}, K_pair);
        auto copy_v_pair =
            make_block_2d_prefetch<SGPerWG::value>(PairTileV{}, V_pair);
        auto pK = copy_k_pair.get_slice(thr_id).partition_S(local_tile(
            make_identity_tensor(K_pair.shape()),
            PairTileK{},
            make_coord(_, 0)));  // (atom_val,k',d',K)
        auto pV = copy_v_pair.get_slice(thr_id).partition_S(local_tile(
            make_identity_tensor(V_pair.shape()),
            PairTileV{},
            make_coord(0, _)));  // (atom_val,v',k',K)
        return cute::make_tuple(copy_k_pair, pK, copy_v_pair, pV);
      } else {
        return 0;
      }
    }();
    auto prefetch_k_tile = [&](int tile) {
      if constexpr (kPairPrefetch) {
        if (pair_prefetch) {
          prefetch(get<0>(pair), get<1>(pair)(_, _, _, tile));
          return;
        }
      }
      for (int D = 0; D < size<4>(pKgK); D++) {
        prefetch(prefetch_k, pKgK(_, _, _, tile, D));
      }
    };
    auto prefetch_v_tile = [&](int tile) {
      if constexpr (kPairPrefetch) {
        if (pair_prefetch) {
          prefetch(get<2>(pair), get<3>(pair)(_, _, _, tile));
          return;
        }
      }
      prefetch(prefetch_v, pVgV(_, _, _, tile));
    };

    /* Initialization steps for first block: Q/K prefetch, O init */
    /* TODO: limit D prefetch for large head size, and reorder K prefetches */
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }

    // Large-surface mode: make a physical tile index local to its sub-surface
    // and re-point the K/V payloads when the sub-surface changes.
    constexpr int tile_k = get<1>(TileShapeQK{});
    [[maybe_unused]] int kv_surf = 0;
    [[maybe_unused]] auto route_tile = [&](int& tile) {
      int const surf = xe_kv_surface<tile_k>(tile);
      tile = xe_kv_surface_tile<tile_k>(tile);
      if (surf != kv_surf) {
        kv_surf = surf;
        xe_kv_retarget_payload<0>(copy_k.payload, K_2D, surf);
        xe_kv_retarget_payload<0>(prefetch_k.payload, K_2D, surf);
        xe_kv_retarget_payload<1>(copy_v.payload, V_2D, surf);
        xe_kv_retarget_payload<1>(prefetch_v.payload, V_2D, surf);
      }
    };

    if constexpr (has_large_surface) {
      route_tile(tile_idx);
    }

    prefetch_k_tile(tile_idx);

    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    /* Check if */
    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    // FP8 KV Scale (per-tensor): fold scale_k into the softmax scale and
    // defer scale_v to a single post-loop output rescale. Mathematically
    // identical to per-element K/V rescaling (S = Q·K is linear in K and
    // O = sum P_i·V_i is linear in V), but removes both rescales from the
    // inner loop. The remaining fp8 -> ElementQ conversion is performed by
    // reorder() using the subgroup-vectorized cast primitive.
    ElementS effective_scale = params.scale;
    float scale_v = 1.f;
    if constexpr (Fp8KV) {
      const float scale_k = *static_cast<const float*>(params.scale_k);
      scale_v = *static_cast<const float*>(params.scale_v);
      effective_scale = params.scale * ElementS(scale_k);
    }
    // FP8 Q Scale (per-tensor): fold the q descale into the softmax scale too,
    // mirroring scale_k. Null pointer means an identity (1.0) descale.
    if constexpr (Fp8Q) {
      if (params.scale_q != nullptr) {
        const float scale_q = *static_cast<const float*>(params.scale_q);
        effective_scale = effective_scale * ElementS(scale_q);
      }
    }

    /* Main loop, blocked in k. */
    int next_tile_idx;
    for (int K = blk_k0; K < blk_k1; K++) {
      /* Split barrier to keep threads together */
      // barrier_arrive(ScopeWorkgroup);

      auto tKgK_cache =
          PagedKV ? tKgK(_, _, _, tile_idx, _) : tKgK(_, _, _, K, _);
      auto tVgV_cache =
          PagedKV ? tVgV(_, _, _, _, tile_idx) : tVgV(_, _, _, _, K);

      // Pipelined path: read the page-table entry of tile K+2 now; it is
      // consumed at the end of this iteration.
      const int k2 = K + 2;
      int block_idx_k2 = 0;
      if constexpr (kPipelined) {
        if (k2 < blk_k1) {
          block_idx_k2 =
              params.ptr_page_table
                  [b_offset + k2 * get<1>(TileShapeQK{}) / params.page_size];
        }
      }

      /* GEMM 1: S = K * Q */
      clear(tSrS); /* TODO: fuse w/ initial gemm call */
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        copy(copy_k, tKgK_cache(_, _, _, D), tKrK);

        reorder(tQrQ, tSrQ);
        // reorder() performs the (vectorized) fp8 -> ElementQ cast; the
        // per-tensor scale_k has been folded into effective_scale above.
        reorder(tKrK, tSrK);

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      /* V prefetch for GEMM 2 */
      prefetch_v_tile(tile_idx);

      /* Causal masking */
      // No Causal masking in decoding
      // if constexpr (CausalMask) {
      //   if (K == blk_k1 - 1) {
      //     // Need to get global col and row indices to mask the elements
      //     Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
      //     Tensor gP = local_tile(cPgP, take<0,2>(TileShapeQK{}),
      //     make_coord(get<0>(blk_qv), K)); auto cS_thread =
      //     thr_mma_qk.partition_C(gP); CUTLASS_PRAGMA_UNROLL for (int i = 0; i
      //     < tSrS.size(); ++i) {
      //       int row_idx = get<0>(cS_thread(i));
      //       int col_idx = get<1>(cS_thread(i));
      //       if (col_idx - full_tile_offset > row_idx - discard_seq_coord) {
      //         tSrS(i) = ElementS(-INFINITY);
      //       }
      //     }
      //   }
      // }

      /* Local/sliding window masking */
      if constexpr (LocalMask) {
        // For decode, all packed GQA heads share the same KV position
        // (seq_len_kv - 1). Use a fixed decode row for all elements.
        int decode_row = seq_len - 1 - full_tile_offset;
        Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
        Tensor gP = local_tile(
            cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
        auto cS_thread = thr_mma_qk.partition_C(gP);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          int col_idx = get<1>(cS_thread(i)) - full_tile_offset;
          bool left_mask = col_idx < decode_row - params.window_size_left;
          bool right_mask = col_idx > decode_row + params.window_size_right;
          if (left_mask || right_mask) {
            tSrS(i) = ElementS(-INFINITY);
          }
        }
      }

      /* Per-position causal masking for packed speculative decode */
      if (params.spec_q_len > 1 &&
          (K + 1) * get<1>(TileShapeQK{}) > seq_len - params.spec_q_len + 1) {
        Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
        Tensor gP = local_tile(
            cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
        auto cS_thread = thr_mma_qk.partition_C(gP);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          int pos = get<0>(cS_thread(i)) / params.spec_rows_per_pos;
          // Rows with no visible key stay fully masked: the epilogue and
          // ReduceSplitK then write O = 0 and LSE = -inf, as FlashAttention.
          int limit = seq_len - (params.spec_q_len - 1 - pos);
          if (get<1>(cS_thread(i)) >= limit) {
            tSrS(i) = ElementS(-INFINITY);
          }
        }
      }

      /* k masking for remainder tiles */
      if (check_remainder_k && K == blk_k1 - 1) {
        FragSCol k_rem_mask;
        int k = get<0>(tKgK(0, 0, 0, K, 0)) + get_sub_group().get_local_id()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
          k_rem_mask(i) =
              (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
        }
      }

      /* Apply softmax and scaling */
      softmax(effective_scale, K == blk_k0, tSrS, tA_max, tA_sum, tArA);
      reorder_to_P_fp8<ElementP>(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v, tVgV_cache(_, _, _, VV), tVrV);
        // reorder() performs the (vectorized) fp8 -> ElementQ cast; the
        // per-tensor scale_v is applied once after the K loop below.
        reorder(tVrV, tArV);
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }

      barrier();

      if constexpr (kPipelined) {
        tile_idx = tile_idx_next;
        if (k2 < blk_k1) {
          tile_idx_next =
              block_idx_k2 * page_stride_tiles + k2 % tiles_per_page;
        }
        /* K prefetch (page index already resolved one tile ahead) */
        if constexpr (has_large_surface) {
          route_tile(tile_idx);
        }
        if (K + 1 < blk_k1) {
          prefetch_k_tile(tile_idx);
        }
        continue;
      }

      // next tile_idx
      next_tile_idx = K + 1;
      if constexpr (PagedKV) {
        int next_page_local_idx =
            next_tile_idx * get<1>(TileShapeQK{}) / params.page_size;
        if (next_page_local_idx < params.max_pages_per_seq) {
          int next_block_idx =
              params.ptr_page_table[b_offset + next_page_local_idx];
          next_tile_idx = next_block_idx * page_stride_tiles +
                          next_tile_idx % tiles_per_page;
        } else {
          // set to last page
          next_tile_idx = params.max_pages_per_seq * tiles_per_page - 1;
        }
      }
      tile_idx = next_tile_idx;

      if constexpr (has_large_surface) {
        route_tile(tile_idx);
      }

      /* K prefetch */
      for (int D = 0; D < size<4>(pKgK); D++) {
        prefetch(prefetch_k, pKgK(_, _, _, tile_idx, D));
      }

      // barrier_wait(ScopeWorkgroup);
    }

    // Apply the per-tensor V scale once after the K loop.
    if constexpr (Fp8KV) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tArA.size(); ++i) {
        tArA(i) *= ElementA(scale_v);
      }
    }
  }

  // Single step of blocked softmax.
  CUTLASS_DEVICE
  void softmax(
      ElementS scale,    // Effective softmax scale (fp8 K scale folded in)
      bool first_block,  // First softmax block?
      FragS& tS,         // Softmax src/dst block
      FragSRow& tS_max,  // Softmax row-wise max accumulator
      FragSRow& tS_sum,  // Softmax row-wise sum accumulator
      FragA& tA) {       // O accumulator (for rescaling)

    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    /* Update (scaled) maxima */
    auto tS_prev_max = tS_max;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      tS_max(i) = sycl::max(tS_max(i), scale * tS_bmax(i));
    }

    /* Scale S and subtract maxima, then exponentiate. For the full-fp8 path
     * (fp8 P*V GEMM) add an exponent offset of +8, i.e. multiply every
     * probability by 2^8 = 256:
     *
     *   P_fp8 = exp2(scale*S - max + 8) = 256 * exp2(scale*S - max)
     *
     * This lifts the softmax probabilities (which would otherwise lie in
     * (0, 1]) into a range better represented by fp8, reducing underflow when
     * P is cast to fp8 for the P*V GEMM. The same 256 factor multiplies the
     * row sums (tS_sum, accumulated below from these values) and the O
     * accumulator, so it cancels exactly in the final reciprocal-sum
     * normalization. The per-block max-rescale (exp2(prev_max - max)) is a
     * difference of maxima and is unaffected by the constant offset. */
    constexpr ElementS kFp8ExpOffset = Fp8Q ? ElementS(8) : ElementS(0);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(
          scale * tS(i) - broadcast<0>(tS_max, tS, i) + kFp8ExpOffset);

    /* Rescale existing S sums and O accumulator */
    if (!first_block) {
      FragSRow rescale;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_max.size(); i++) {
        rescale(i) = sycl::native::exp2(tS_prev_max(i) - tS_max(i));
        tS_sum(i) *= rescale(i);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tA.size(); i++)
        tA(i) *= broadcast<0>(rescale, tA, i);
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);
  }
};

template <typename SGLayoutQK>
CUTLASS_HOST_DEVICE constexpr auto get_sg_layout_pv(SGLayoutQK const&) {
  return make_layout(
      get<0>(SGLayoutQK{}), Layout<_1, _0>{}, get<1>(SGLayoutQK{}));
}

// Get a P*V TiledMMA given K*Q tile size and SG configuration, for mainloops
//   not supporting S data interchange among subgroups (e.g. XeDefault).
template <
    typename MMAOp,
    typename WGTileQK,
    typename SGLayoutQK,
    typename TileV>
CUTLASS_HOST_DEVICE constexpr auto get_tiled_mma_pv(
    MMAOp const&,
    WGTileQK const& wg_tile_qk,
    SGLayoutQK const& sg_layout_qk,
    TileV const&) {
  using TileQ = decltype(get<0>(wg_tile_qk));
  using TileK = decltype(get<1>(wg_tile_qk));

  using WGTilePV = Shape<TileQ, TileV, TileK>;
  using SGLayoutPV = decltype(get_sg_layout_pv(sg_layout_qk));

  static_assert(
      size(SGLayoutPV{}) == size(SGLayoutQK{}),
      "Q*K cannot be parallelized in the head size dimension");

  return TiledMMAHelper<MMAOp, WGTilePV, SGLayoutPV>{};
}

}  // namespace vllm_xpu_xe3

}  // namespace cutlass::fmha::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
