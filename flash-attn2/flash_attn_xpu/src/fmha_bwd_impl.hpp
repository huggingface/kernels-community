#pragma once

#include <sycl/sycl.hpp>
#include <cassert>
#include <cute/util/compat.hpp>
#include <cute/tensor.hpp>

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include "fmha_bwd_types.hpp"
#include "kernel/fmha_bwd_kernel.hpp"

using namespace cute;

// Helper function to convert layout
template <typename Layout>
auto convert_layout_2d_layout(Layout layout) {
    auto l = make_layout(make_layout(get<0>(layout),
                                     get<1>(layout)),
                         get<2>(layout));
    return l;
}

template<typename Layout>
CUTLASS_DEVICE auto convert_layout_acc_layout(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 8);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_1>{});
    auto l2 = make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                          make_layout(get<2>(l)));
    return l2;
}

// Apply causal mask to tensor
template <typename Engine0, typename Layout0,
          typename Engine1, typename Layout1>
CUTLASS_DEVICE void
apply_mask_causal(Tensor<Engine0, Layout0> &tensor,
                  Tensor<Engine1, Layout1> &rC,
                  int m_offset, int n_offset, int diagonal_offset = 0) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    int sg_local_id = sg.get_local_id();
    Tensor rC_2d = make_tensor(
        rC.data(),
        convert_layout_2d_layout(rC.layout()));
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size<1>(tensor); ++n) {
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<0>(tensor); ++m) {
            int y = m_offset + get<1>(rC_2d(m, n)) + sg_local_id + diagonal_offset;
            int x = n_offset + get<0>(rC_2d(m, n));
            if (x > y) {
                tensor(m, n) = -INFINITY;
            }
        }
    }
}

// Apply local (sliding window) mask to tensor
template <typename Engine0, typename Layout0,
          typename Engine1, typename Layout1>
CUTLASS_DEVICE void
apply_mask_local(Tensor<Engine0, Layout0> &tensor,
                 Tensor<Engine1, Layout1> &rC,
                 int m_offset, int n_offset,
                 int window_size_left, int window_size_right,
                 int seqlen_k_minus_seqlen_q = 0) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    int sg_local_id = sg.get_local_id();
    Tensor rC_2d = make_tensor(
        rC.data(),
        convert_layout_2d_layout(rC.layout()));
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size<1>(tensor); ++n) {
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<0>(tensor); ++m) {
            // row_idx is the query position, col_idx is the key position
            int row_idx = m_offset + get<1>(rC_2d(m, n)) + sg_local_id;
            int col_idx = n_offset + get<0>(rC_2d(m, n));
            // col must be in [row + offset - window_left, row + offset + window_right]
            // where offset = seqlen_k - seqlen_q
            int adjusted_row = row_idx + seqlen_k_minus_seqlen_q;
            bool left_mask = (window_size_left >= 0) && (col_idx < adjusted_row - window_size_left);
            bool right_mask = (window_size_right >= 0) && (col_idx > adjusted_row + window_size_right);
            if (left_mask || right_mask) {
                tensor(m, n) = -INFINITY;
            }
        }
    }
}

// Scale and apply exp2 for softmax
template<class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2>
CUTLASS_DEVICE void
scale_apply_exp2(Tensor<Engine0, Layout0> &tensor,
                 Tensor<Engine1, Layout1> &max,
                 Tensor<Engine2, Layout2> &rC,
                 const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    auto sg = compat::get_nd_item<1>().get_sub_group();
    int sg_local_id = sg.get_local_id();
    Tensor rC_2d = make_tensor(
        rC.data(),
        convert_layout_2d_layout(rC.layout()));
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < size<1>(tensor); ++ni)  {
        int n = get<1>(rC_2d(0, ni)) + sg_local_id;
        const float max_scaled = max(n) == -INFINITY ? 0.f : max(n) * M_LOG2E;
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(tensor); ++mi) {
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

// Softmax backward: dS = P * (dP - sum) * scale
template<class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2,
         class Engine3, class Layout3>
CUTLASS_DEVICE void
softmax_backward(Tensor<Engine0, Layout0> &P,
                 Tensor<Engine1, Layout1> &dP_sum,
                 Tensor<Engine2, Layout2> &dP,
                 Tensor<Engine3, Layout3> &rC,
                 const float scale) {
    Tensor rC_2d = make_tensor(
        rC.data(),
        convert_layout_2d_layout(rC.layout()));
    auto sg = compat::get_nd_item<1>().get_sub_group();
    int sg_local_id = sg.get_local_id();
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < size<1>(dP); ++ni) {
        int n = get<1>(rC_2d(0, ni)) + sg_local_id;
        const float dpsum = dP_sum(n);
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(dP); ++mi) {
            dP(mi, ni) = P(mi, ni) * (dP(mi, ni) - dpsum) * scale;
        }
    }
}

// Load a single column vector
template<bool Is_even_M, class Tensor0, class Tensor1, class Tensor2>
CUTLASS_DEVICE void
load_1colvec(Tensor0 &reg, Tensor1 &mT, Tensor2 &coord_row,
             int tail_m = 0) {
    if constexpr(Is_even_M) {
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(reg); ++mi) {
            reg(mi) = mT(get<0>(coord_row(mi)));
        }
    } else {
        for (int mi = 0; mi < size(reg); ++mi) {
            int row = get<0>(coord_row(mi));
            if (row < tail_m) {
                reg(mi) = mT(row);
            }
        }
    }
}

// Create register fragment
template<typename T, class Trait, class MTensor, class TiledMMA>
auto
create_reg(Trait const &trait,
           MTensor const &C,
           TiledMMA const &tiled_mma) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx);

    Tensor cC = make_identity_tensor(C.shape());
    auto tile_mnk = tiled_mma.tile_mnk();
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0, 0));
    auto copy_c = make_block_2d_copy_D(tiled_mma, C);
    auto thr_copy_c = copy_c.get_slice(first_thread_in_sg_idx);
    if constexpr(is_same_v<T, float>) {
        auto r32 = thr_mma.partition_sg_fragment_C(make_identity_tensor(select<0,1>(tile_mnk)));
        return r32;
    } else {
        auto r16 = thr_copy_c.partition_sg_fragment_S(gC);
        return r16;
    }
}

// GEMM kernel with optional accumulator clearing
template<bool clear_acc, class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2, class TVLayout2,
         class TiledMMA>
void
gemm_kernel(Trait &trait,
            Tensor<Engine0, Layout0> const& A,
            Tensor<Engine1, Layout1> const& B,
            SubgroupTensor<Engine2, Layout2, TVLayout2> & acc,
            TiledMMA const & mma,
            const int m_block = 0,
            const int n_block = 0) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * Trait::SubgroupSize;

    Tensor cA = make_identity_tensor(A.shape());
    Tensor cB = make_identity_tensor(B.shape());

    auto tile_mnk = mma.tile_mnk();

    Tensor gA = local_tile(cA, select<0,2>(tile_mnk), make_coord(m_block,_));
    Tensor gB = local_tile(cB, select<1,2>(tile_mnk), make_coord(n_block,_));

    auto copy_a = make_block_2d_copy_A(mma, A);
    auto copy_b = make_block_2d_copy_B(mma, B);

    auto thr_mma    =    mma.get_slice(first_thread_in_sg_idx);
    auto thr_copy_a = copy_a.get_slice(first_thread_in_sg_idx);
    auto thr_copy_b = copy_b.get_slice(first_thread_in_sg_idx);

    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));

    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));

    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);

    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);

    auto thr_prefetch_A = prefetch_a.get_slice(first_thread_in_sg_idx);
    auto thr_prefetch_B = prefetch_b.get_slice(first_thread_in_sg_idx);

    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    const int prefetch_dist = 3;
    constexpr int barrier_scope = 2;

    int k_tile_count = ceil_div(shape<1>(A), get<2>(tile_mnk));
    int k_tile_prefetch = 0;

    if constexpr(clear_acc)
        clear(acc);

    CUTE_UNROLL
    for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
        prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
        prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));
    }

    for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
        barrier_arrive(barrier_scope);

        copy(copy_a, tAgA(_,_,_,k_tile), tArA);
        copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

        prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
        prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));

        reorder(tArA, tCrA);
        reorder(tBrB, tCrB);

        gemm(mma, tCrA, tCrB, acc);

        barrier_wait(barrier_scope);
    }
}

// GEMM for computing S*dP
template<class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2,
         class TVLayout2, class TiledMMA>
void
gemm_SdP(Trait &trait,
         Tensor<Engine0, Layout0> const& A,
         Tensor<Engine1, Layout1> const& B,
         SubgroupTensor<Engine2, Layout2, TVLayout2> & rSdP,
         TiledMMA const & mma) {
    gemm_kernel<true>(trait, A, B, rSdP, mma);
}

// GEMM for computing dK/dV (accumulating)
template<class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2,
         class TVLayout2, class TiledMMA>
void
gemm_dKV(Trait &trait,
         Tensor<Engine0, Layout0> const& A,
         Tensor<Engine1, Layout1> const& B,
         SubgroupTensor<Engine2, Layout2, TVLayout2> & rdKV,
         TiledMMA const & mma) {
    gemm_kernel<false>(trait, A, B, rdKV, mma);
}

// GEMM for computing dQ with atomic add
template<class Trait,
         class Engine0, class Layout0,
         class Engine1, class Layout1,
         class Engine2, class Layout2,
         class TiledMMA>
void
gemm_dQ(Trait &trait,
        Tensor<Engine0, Layout0> const& A,
        Tensor<Engine1, Layout1> const& B,
        Tensor<Engine2, Layout2> const& C,
        TiledMMA const & mma,
        const int m_block = 0,
        const int n_block = 0) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    auto tile_mnk = mma.tile_mnk();
    Tensor cC = make_identity_tensor(C.shape());
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(m_block, n_block));
    auto thr_mma = mma.get_slice(first_thread_in_sg_idx);
    auto tCrC = thr_mma.partition_sg_fragment_C(make_identity_tensor(select<0,1>(tile_mnk)));
    Tensor tCgC = thr_mma.partition_C(gC);
    gemm_kernel<true>(trait, A, B, tCrC, mma, m_block, n_block);
    int local_id = sg.get_local_id();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tCgC); ++i) {
        auto [m, n] = tCgC(i);
        cutlass::atomicAdd(&C(m, n + local_id), tCrC(i));
    }
}

// Copy from register to global memory
template<class Trait, class TiledMma,
         class Engine0, class Layout0, class TVLayout0,
         class Engine1, class Layout1>
void
mha_copy(Trait & trait, TiledMma &tiled_mma,
         SubgroupTensor<Engine0, Layout0, TVLayout0> &r,
         Tensor<Engine1, Layout1> &m,
         int m_block = 0, int n_block = 0) {
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    auto copy_c = make_block_2d_copy_D(tiled_mma, m);
    auto thr_copy_c = copy_c.get_slice(first_thread_in_sg_idx);
    auto tile_mnk = tiled_mma.tile_mnk();
    Tensor cC = make_identity_tensor(m.shape());
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(m_block, n_block));
    Tensor tCgC = thr_copy_c.partition_D(gC);
    copy(copy_c, r, tCgC);
}

// Reorder and copy
template<class Trait, class TiledMma,
         class Engine0, class Layout0, class TVLayout0,
         class Engine1, class Layout1>
void
mha_reorder_copy(Trait & trait, TiledMma &tiled_mma,
                 SubgroupTensor<Engine0, Layout0, TVLayout0> &r,
                 Tensor<Engine1, Layout1> &m){
    auto r16 = create_reg<typename Trait::DType>(trait, m, tiled_mma);
    reorder(r, r16);
    mha_copy(trait, tiled_mma, r16, m);
}

// Helper function to round up to multiple
inline int round_multiple(int x, int m) {
    return (x + m - 1) / m * m;
}

// Main 1-col-block backward computation
template<bool Is_even_N, bool Seq_parallel, class Trait>
void
dq_dk_dv_1colblock(Trait &trait, BwdParam<typename Trait::DType> &param,
                   const int bidb, const int bidh, const int bidhkv, const int n_block,
                   const int tail_n = 0) {
    using T = typename Trait::DType;
    using V = typename Trait::VType;
    constexpr int kHeadDim = Trait::kHeadDim;
    constexpr int kBlockM = Trait::kBlockM;
    constexpr int kBlockN = Trait::kBlockN;
    constexpr int kNSGs = Trait::kNSGs;
    constexpr int SubgroupSize = Trait::SubgroupSize;
    constexpr bool is_causal = Trait::is_causal;
    constexpr bool is_local = Trait::is_local;
    
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto group = compat::get_nd_item<1>().get_group();
    const int local_id = sg.get_local_id();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    auto bofst = BwdOffset(param);

    const index_t q_offset = bofst.q_offset(bidb, bidh, 0);
    const index_t k_offset = bofst.k_offset(bidb, bidhkv, n_block * kBlockN);
    const index_t v_offset = bofst.v_offset(bidb, bidhkv, n_block * kBlockN);
    const index_t dk_offset = bofst.dk_offset(bidb, bidh, n_block * kBlockN);
    const index_t dv_offset = bofst.dv_offset(bidb, bidh, n_block * kBlockN);
    const index_t o_offset = bofst.o_offset(bidb, bidh, 0);
    const index_t dq_offset = bofst.dq_offset(bidb, bidh, 0);
    const index_t lse_offset = bofst.lse_offset(bidb, bidh, 0);
    
    // Buffer offset for intermediate P and dS
    const index_t pb_offset = (bidb * param.num_head_q * param.seq_len_kv_pad * kBlockM
                               + bidh * param.seq_len_kv_pad * kBlockM
                               + n_block * kBlockN * kBlockM) * 2;
    const index_t dsb_offset = pb_offset + kBlockN * kBlockM;

    const auto block_n_dim = tail_n == 0 ? Int<kBlockN>{} : ((tail_n + 1) & ~1);
    auto shapeO = make_shape(kBlockM, Int<kHeadDim>{});
    auto shapeQtOt = make_shape(Int<kHeadDim>{}, kBlockM);
    auto shapeSPt = make_shape(block_n_dim, Int<kBlockM>{});
    auto shapeSP = make_shape(Int<kBlockM>{}, block_n_dim);

    using Shape1 = Shape<
        std::conditional_t<Is_even_N, Int<kBlockN>, int>, Int<kHeadDim>>;
    using Shape2 = Shape<
        Int<kHeadDim>,
        std::conditional_t<Is_even_N, Int<kBlockN>, int>>;
    auto shapeQ = make_shape(kBlockM, Int<kHeadDim>{});
    auto shapedQ = Shape<Int<kBlockM>, Int<kHeadDim>>{};
    Shape1 shapeKtVt;
    Shape2 shapeKV;
    if constexpr(Is_even_N) {
        shapeKtVt = make_shape(Int<kBlockN>{}, Int<kHeadDim>{});
        shapeKV = make_shape(Int<kHeadDim>{}, Int<kBlockN>{});
    } else {
        shapeKtVt = make_shape(tail_n, Int<kHeadDim>{});
        shapeKV = make_shape(Int<kHeadDim>{}, tail_n);
    }

    // Create tensor views
    Tensor mQ = make_tensor(make_gmem_ptr(param.q_ptr + q_offset),
                            make_layout(shapeQ, make_stride(param.q_r_stride, _1{})));
    Tensor mKt = make_tensor(make_gmem_ptr(param.k_ptr + k_offset),
                             make_layout(shapeKtVt, make_stride(param.k_r_stride, _1{})));
    Tensor mdO = make_tensor(make_gmem_ptr(param.do_ptr + o_offset),
                             make_layout(shapeO, make_stride(param.o_r_stride, _1{})));
    Tensor mVt = make_tensor(make_gmem_ptr(param.v_ptr + v_offset),
                             make_layout(shapeKtVt, make_stride(param.v_r_stride, _1{})));
    
    // Intermediate buffer
    Tensor mPt = make_tensor(make_gmem_ptr(param.pb_ptr + pb_offset),
                             make_layout(shapeSPt, make_stride(Int<kBlockM>{}, _1{})));
    Tensor mdOt = make_tensor(make_gmem_ptr(param.do_ptr + o_offset),
                              make_layout(shapeQtOt, make_stride(_1{}, param.o_r_stride)));
    Tensor mK = make_tensor(make_gmem_ptr(param.k_ptr + k_offset),
                            make_layout(shapeKV, make_stride(_1{}, param.k_r_stride)));
    Tensor mdPt = make_tensor(make_gmem_ptr(param.pb_ptr + dsb_offset),
                              make_layout(shapeSPt, make_stride(Int<kBlockM>{}, _1{})));
    Tensor mQt = make_tensor(make_gmem_ptr(param.q_ptr + q_offset),
                              make_layout(shapeQtOt, make_stride(_1{}, param.q_r_stride)));

    Tensor mLSE = make_tensor(make_gmem_ptr(param.lse_ptr + lse_offset),
                              make_layout(Shape<Int<kBlockM>>{}, Stride<_1>{}));
    Tensor mdPsum = make_tensor(make_gmem_ptr(param.odo_ptr + lse_offset),
                                make_layout(Shape<Int<kBlockM>>{}, Stride<_1>{}));

    Tensor mdV = make_tensor(make_gmem_ptr(param.dv_ptr + dv_offset),
                             make_layout(shapeKtVt, make_stride(param.dv_r_stride, _1{})));
    Tensor mdP = make_tensor(make_gmem_ptr(param.pb_ptr + dsb_offset),
                             make_layout(shapeSP, make_stride(_1{}, Int<kBlockM>{})));
    Tensor mdQaccum = make_tensor(make_gmem_ptr(param.dqaccum_ptr + dq_offset),
                                  make_layout(shapedQ, make_stride(param.dq_r_stride, _1{})));
    Tensor mdK = make_tensor(make_gmem_ptr(param.dk_ptr + dk_offset),
                              make_layout(shapeKtVt, make_stride(param.dk_r_stride, _1{})));

    typename Trait::TiledMmaSdP tiled_mma_sdp;
    typename Trait::TiledMmadKV tiled_mma_dkv;
    typename Trait::TiledMmadQ tiled_mma_dq;

    auto thr_mma_sdp = tiled_mma_sdp.get_slice(first_thread_in_sg_idx);

    Tensor caccSt = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{});
    Tensor taccScSt = thr_mma_sdp.partition_C(caccSt);
    Tensor taccScS_rt = logical_divide(taccScSt, Shape<_1>{});

    const int max_m_block = ceil_div(param.seq_len_q, kBlockM);
    const int tail_m = param.seq_len_q % kBlockM;

    auto rdV = create_reg<V>(trait, mdV, tiled_mma_dkv);
    auto rdK = create_reg<V>(trait, mdK, tiled_mma_dkv);
    clear(rdV);
    clear(rdK);
    
    // Main loop over M blocks
    for (int m_block = 0; m_block < max_m_block; ++m_block) {
        const bool Is_even_M = not ((m_block == max_m_block - 1) and (tail_m != 0));
        if (not Is_even_M) {
            mQ = make_tensor(make_gmem_ptr(mQ.data()),
                             make_layout(make_shape(tail_m, Int<kHeadDim>{}),
                                         make_stride(param.q_r_stride, _1{})));
            mdO = make_tensor(make_gmem_ptr(mdO.data()),
                             make_layout(make_shape(tail_m, Int<kHeadDim>{}),
                                         make_stride(param.o_r_stride, _1{})));
            mdOt = make_tensor(make_gmem_ptr(mdOt.data()),
                               make_layout(make_shape(Int<kHeadDim>{}, tail_m),
                                           make_stride(_1{}, param.o_r_stride)));
            mdQaccum = make_tensor(make_gmem_ptr(mdQaccum.data()),
                                   make_layout(shapedQ, make_stride(param.dq_r_stride, _1{})));
            mQt = make_tensor(make_gmem_ptr(mQt.data()),
                              make_layout(make_shape(Int<kHeadDim>{}, tail_m),
                                          make_stride(_1{}, param.q_r_stride)));
        }
        
        {
            auto rS = create_reg<V>(trait, mPt, tiled_mma_sdp);
            // S = Q * K^T
            gemm_SdP(trait, mKt, mQ, rS, tiled_mma_sdp);
            Tensor scores = make_tensor(rS.data(), convert_layout_acc_layout(rS.layout()));
            
            if constexpr(is_causal) {
                apply_mask_causal(scores, taccScS_rt, m_block * kBlockM, n_block * kBlockN, 
                                  param.seq_len_kv - param.seq_len_q);
            }
            
            if constexpr(is_local) {
                apply_mask_local(scores, taccScS_rt, m_block * kBlockM, n_block * kBlockN,
                                 param.window_size_left, param.window_size_right,
                                 param.seq_len_kv - param.seq_len_q);
            }

            // P = softmax(S, lse)
            scale_apply_exp2(scores, mLSE, taccScS_rt, param.scale_softmax_log2);
            
            auto rdP = create_reg<V>(trait, mdP, tiled_mma_sdp);
            // dP = dO * V^T
            gemm_SdP(trait, mVt, mdO, rdP, tiled_mma_sdp);
            Tensor dS = make_tensor(rdP.data(), scores.layout());
            
            // dS = P * (dP - sum) * scale
            softmax_backward(scores, mdPsum, dS, taccScS_rt, param.scale_softmax);
            
            mha_reorder_copy(trait, tiled_mma_sdp, rS, mPt);
            mha_reorder_copy(trait, tiled_mma_sdp, rdP, mdPt);
        }
        sycl::group_barrier(group);
        
        // dV = P^T * dO
        gemm_dKV(trait, mPt, mdOt, rdV, tiled_mma_dkv);
        // dK = dP^T * Q
        gemm_dKV(trait, mdPt, mQt, rdK, tiled_mma_dkv);
        // dQ = dP * K
        gemm_dQ(trait, mdP, mK, mdQaccum, tiled_mma_dq);
        
        // Update pointers
        mQ.data() = mQ.data() + int(kBlockM * param.q_r_stride);
        mdO.data() = mdO.data() + int(kBlockM * param.o_r_stride);
        mdOt.data() = mdOt.data() + int(kBlockM * param.o_r_stride);
        mdQaccum.data() = mdQaccum.data() + int(kBlockM * param.dq_r_stride);
        mQt.data() = mQt.data() + int(kBlockM * param.q_r_stride);
        mLSE.data() = mLSE.data() + int(kBlockM);
        mdPsum.data() = mdPsum.data() + int(kBlockM);
    }
    
    mha_reorder_copy(trait, tiled_mma_dkv, rdV, mdV);
    mha_reorder_copy(trait, tiled_mma_dkv, rdK, mdK);
}

// Compute O * dO (dot product)
template<bool Is_even_M, class T>
void
compute_o_dot_do(T &trait, BwdParam<typename T::DType> &param,
                 const int m_block, const int bidb, const int bidh) {
    constexpr int kBlockM = T::kBlockM;
    constexpr int kHeadDim = T::kHeadDim;
    constexpr int kNSGs = T::kNSGs;
    constexpr int SubgroupSize = T::SubgroupSize;
    using DType = typename T::DType;
    using VType = typename T::VType;

    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto group = compat::get_nd_item<1>().get_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    auto bofst = BwdOffset(param);

    const index_t o_offset = bofst.o_offset(bidb, bidh, m_block * kBlockM);
    const index_t dq_offset = bofst.dq_offset(bidb, bidh, m_block * kBlockM);
    const index_t dpsum_offset = bofst.lse_offset(bidb, bidh, m_block * kBlockM);

    using ShapeO = Shape<std::conditional_t<Is_even_M, Int<kBlockM>, int>, Int<kHeadDim>>;
    using ShapeP = Shape<std::conditional_t<Is_even_M, Int<kBlockM>, int>>;
    ShapeO O_shape;
    ShapeP dP_shape;
    if constexpr(Is_even_M) {
        O_shape = make_shape(Int<kBlockM>{}, Int<kHeadDim>{});
        dP_shape = make_shape(Int<kBlockM>{});
    } else {
        O_shape = make_shape(param.tail_m, Int<kHeadDim>{});
        dP_shape = make_shape(param.tail_m);
    }
    auto dQ_shape = make_shape(Int<kBlockM>{}, Int<kHeadDim>{});

    Tensor mdO = make_tensor(make_gmem_ptr(param.do_ptr + o_offset),
                             make_layout(O_shape, make_stride(param.o_r_stride, _1{})));
    Tensor mO = make_tensor(make_gmem_ptr(param.o_ptr + o_offset),
                            make_layout(O_shape, make_stride(param.o_r_stride, _1{})));
    Tensor mdQaccum = make_tensor(make_gmem_ptr(param.dqaccum_ptr + dq_offset),
                                  make_layout(make_shape(Int<kBlockM>{}, Int<kHeadDim>{}),
                                              make_stride(param.dq_r_stride, _1{})));
    Tensor mdPsum = make_tensor(make_gmem_ptr(param.odo_ptr + dpsum_offset),
                                make_layout(dP_shape, Stride<_1>{}));
    
    using ThreadLayout = Layout<Shape<Int<kNSGs>, Int<SubgroupSize>>,
                                Stride<Int<SubgroupSize>, _1>>;
    // Must ensure AlignedArray size results in power-of-2 alignment
    using ValueLayout = std::conditional_t<
        kHeadDim == 96,
        Layout<Shape<_1, _2>>,
        std::conditional_t<
            kHeadDim == 160,
            Layout<Shape<_1, _2>>,
            std::conditional_t<
                kHeadDim == 192,
                Layout<Shape<_1, _4>>,
                Layout<Shape<_1, Int<kHeadDim / SubgroupSize>>>>>>;
    using OdOType = cutlass::AlignedArray<DType, size(ValueLayout{})>;
    using OdOAtom = Copy_Atom<UniversalCopy<OdOType>, DType>;
    using dQType = cutlass::AlignedArray<VType, size(ValueLayout{})>;
    using dQAtom = Copy_Atom<UniversalCopy<dQType>, VType>;

    auto tileload_odo = make_tiled_copy(OdOAtom{}, ThreadLayout{}, ValueLayout{});
    auto tileload_dq = make_tiled_copy(dQAtom{}, ThreadLayout{}, ValueLayout{});

    auto thr_load_odo = tileload_odo.get_thread_slice(ThreadIdxX());
    auto thr_load_dq = tileload_dq.get_thread_slice(ThreadIdxX());

    Tensor thr_tile_do_S = thr_load_odo.partition_S(mdO);
    Tensor thr_tile_o_S = thr_load_odo.partition_S(mO);
    Tensor thr_tile_dq_D = thr_load_dq.partition_D(mdQaccum);
    Tensor rdQ = make_fragment_like(thr_tile_dq_D);
    Tensor rdO = make_fragment_like<DType>(rdQ);
    Tensor rO = make_fragment_like<DType>(rdQ);
    Tensor cO = make_identity_tensor(dQ_shape);
    Tensor tcO = thr_load_odo.partition_S(cO);
    Tensor tcO_row = logical_divide(tcO, Shape<_1>{})(make_coord(0, 0), _, 0);
    Layout rdO_layout = rdO.layout();
    Tensor rdO_2d = make_tensor(rdO.data(),
                                make_layout(get<1>(rdO_layout),
                                            make_layout(get<0>(rdO_layout), get<2>(rdO_layout))));
    Tensor rO_2d = make_tensor(rO.data(), rdO_2d.layout());

    constexpr int NumValperCol = size<0>(rdO_2d);
    auto smem = compat::local_mem<VType[kNSGs * SubgroupSize * NumValperCol]>();
    auto stensor = make_tensor(make_smem_ptr(smem),
                               make_layout(Shape<Int<NumValperCol>, Int<kNSGs>, Int<SubgroupSize>>{}));
    clear(rdO_2d);
    clear(rO_2d);
    
    if constexpr(Is_even_M) {
        copy(tileload_odo, thr_tile_do_S, rdO);
        copy(tileload_odo, thr_tile_o_S, rO);
    } else {
        for (int mi = 0; mi < size<0>(rdO_2d); ++mi) {
            if (get<0>(tcO_row(mi)) < param.tail_m) {
                copy(tileload_odo, thr_tile_do_S(_, mi, _), rdO(_, mi, _));
                copy(tileload_odo, thr_tile_o_S(_, mi, _), rO(_, mi, _));
            }
        }
    }
    
    int sg_group_id = sg.get_group_id();
    int sg_local_id = sg.get_local_id();
    
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(rdO_2d); ++mi) {
        float accum = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(rdO_2d); ++ni) {
            accum = accum + (float)rdO_2d(mi, ni) * (float)rO_2d(mi, ni);
        }
        stensor(mi, sg_group_id, sg_local_id) = accum;
    }
    
    if (sg_local_id == 0) {
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < NumValperCol; ++mi) {
            float accum = 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < SubgroupSize; ++ni) {
                accum += stensor(mi, sg_group_id, ni);
            }
            if constexpr(Is_even_M) {
                mdPsum(get<0>(tcO_row(mi))) = accum;
            } else {
                if (get<0>(tcO_row(mi)) < param.tail_m) {
                    mdPsum(get<0>(tcO_row(mi))) = accum;
                }
            }
        }
    }
}

// Convert dQ from float accumulator to target type
template<bool Is_even_M, class T>
void
convert_dq(T &trait, BwdParam<typename T::DType> &param, int m_block, int bidb, int bidh) {
    constexpr int kBlockM = T::kBlockM;
    constexpr int kHeadDim = T::kHeadDim;
    using DType = typename T::DType;
    using VType = typename T::VType;
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;

    auto bofst = BwdOffset(param);
    const index_t dq_offset = bofst.dq_offset(bidb, bidh, m_block * kBlockM);
    const index_t q_offset = bofst.q_offset(bidb, bidh, m_block * kBlockM);
    
    using ShapeQ = Shape<std::conditional_t<Is_even_M, Int<kBlockM>, int>, Int<kHeadDim>>;
    ShapeQ shapeQ;
    if constexpr (Is_even_M) {
        shapeQ = make_shape(Int<kBlockM>{}, Int<kHeadDim>{});
    } else {
        shapeQ = make_shape(param.tail_m, Int<kHeadDim>{});
    }

    Tensor mdQaccum = make_tensor(make_gmem_ptr(param.dqaccum_ptr + dq_offset),
                                  make_layout(Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                              make_stride(param.dq_r_stride, _1{})));
    Tensor mdQ = make_tensor(make_gmem_ptr(param.dq_ptr + q_offset),
                            make_layout(shapeQ, make_stride(param.q_r_stride, _1{})));

    typename T::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_slice(first_thread_in_sg_idx);
    auto tile_dq = tiled_mma_dq.tile_mnk();

    auto tileloaddQ = make_block_2d_copy_C(tiled_mma_dq, mdQaccum);
    auto tilesavedQ = make_block_2d_copy_D(tiled_mma_dq, mdQ);

    auto thr_load_dQ = tileloaddQ.get_slice(first_thread_in_sg_idx);
    auto thr_save_dQ = tilesavedQ.get_slice(first_thread_in_sg_idx);

    Tensor gdQaccum = local_tile(make_identity_tensor(mdQaccum.shape()),
                                 select<0, 1>(tile_dq), make_coord(0,0));
    Tensor gdQ = local_tile(make_identity_tensor(mdQ.shape()),
                            select<0, 1>(tile_dq), make_coord(0,0));
    Tensor tdQgdQaccum = thr_load_dQ.partition_S(gdQaccum);
    auto tdQrdQaccum = thr_load_dQ.partition_sg_fragment_D(gdQaccum);
    auto tdQrdQ = thr_save_dQ.partition_sg_fragment_S(gdQ);
    Tensor tdQgdQ = thr_save_dQ.partition_D(gdQ);

    copy(tileloaddQ, tdQgdQaccum, tdQrdQaccum);
    reorder(tdQrdQaccum, tdQrdQ);
    copy(tilesavedQ, tdQrdQ, tdQgdQ);
}

// Kernel entry points
template<class T>
void
mha_dot_do_o(T trait, BwdParam<typename T::DType> param) {
    const int m_block = BlockIdxX();
    const int bidb = BlockIdxZ();
    const int bidh = BlockIdxY();
    if (m_block == param.m_block - 1 and param.tail_m > 0) {
        compute_o_dot_do<false>(trait, param, m_block, bidb, bidh);
    } else {
        compute_o_dot_do<true>(trait, param, m_block, bidb, bidh);
    }
}

template<class T>
void
mha_backward_seq(T trait, BwdParam<typename T::DType> param) {
    const int bidb = BlockIdxZ();
    const int bidhq = BlockIdxY();
    const int bidnblk = BlockIdxX();
    const int bidhkv = bidhq / param.num_qh_per_kvh;
    for (int n_block = bidnblk; n_block < param.n_block; n_block += GridDimX()) {
        if (param.tail_n > 0 and n_block == param.n_block - 1)
            dq_dk_dv_1colblock<false, false>(trait, param, bidb, bidhq, bidhkv, param.n_block - 1, param.tail_n);
        else
            dq_dk_dv_1colblock<true, false>(trait, param, bidb, bidhq, bidhkv, n_block);
    }
}

template<class T>
void
mhd_convert_dq(T trait, BwdParam<typename T::DType> param) {
    const int m_block = BlockIdxX();
    const int bidb = BlockIdxZ();
    const int bidh = BlockIdxY();
    if (param.tail_m > 0 and m_block == param.m_block - 1) {
        convert_dq<false>(trait, param, m_block, bidb, bidh);
    } else {
        convert_dq<true>(trait, param, m_block, bidb, bidh);
    }
}

// Device name templates
template<class...> class mhaodoDeviceNameBwd;
template<class...> class mhabwdDeviceNameBwd;
template<class...> class mhacvtDeviceNameBwd;

// Backward kernel launcher - similar to KernelLauncher in fmha_fwd_impl.hpp
template <class FABwdKernel_>
struct BwdKernelLauncher {
    using FABwdKernel = FABwdKernel_;
    using DType = typename FABwdKernel::DType;
    using VType = typename FABwdKernel::VType;
    
    static constexpr int kBlockM = FABwdKernel::kBlockM;
    static constexpr int kBlockN = FABwdKernel::kBlockN;
    static constexpr int kHeadDim = FABwdKernel::kHeadDim;
    static constexpr int kNSGs = FABwdKernel::kNSGs;
    static constexpr int SubgroupSize = FABwdKernel::SubgroupSize;
    static constexpr int smem_size = FABwdKernel::smem_size;

    cutlass::Status
    run(sycl::queue& queue, const fmha_bwd_args_t& args) {
        auto trait = FABwdKernel{};

        const int BATCH = args.batch_size;
        const int NUM_HEAD_Q = args.num_heads_q;
        const int NUM_HEAD_KV = args.num_heads_k;
        const int SEQ_LEN_Q = args.seqlen_q;
        const int SEQ_LEN_KV = args.seqlen_k;
        const int N_BLOCK = ceil_div(SEQ_LEN_KV, kBlockN);
        const int tail_n = SEQ_LEN_KV % kBlockN;
        const int M_BLOCK = ceil_div(SEQ_LEN_Q, kBlockM);
        const int tail_m = SEQ_LEN_Q % kBlockM;
        
        // Allocate intermediate buffer
        DType* pbuff = compat::malloc<DType>(BATCH * NUM_HEAD_Q * args.seqlen_k_rounded * 2 * kBlockM);
        
        auto param = BwdParam<DType>(
            reinterpret_cast<const DType*>(args.dout),
            reinterpret_cast<const DType*>(args.out),
            reinterpret_cast<const DType*>(args.query),
            reinterpret_cast<const DType*>(args.key),
            reinterpret_cast<const DType*>(args.value),
            reinterpret_cast<const float*>(args.softmax_lse),
            reinterpret_cast<float*>(args.softmax_d),
            reinterpret_cast<float*>(args.dq_accum),
            reinterpret_cast<DType*>(args.dq),
            reinterpret_cast<DType*>(args.dk),
            reinterpret_cast<DType*>(args.dv),
            pbuff,
            args.sm_scale);
        
        param.batch = BATCH;
        param.num_head_q = NUM_HEAD_Q;
        param.num_head_kv = NUM_HEAD_KV;
        param.num_qh_per_kvh = NUM_HEAD_Q / NUM_HEAD_KV;
        param.num_nb_per_blk = std::max(N_BLOCK * NUM_HEAD_Q * BATCH / 1024, 1);
        param.seq_len_q = SEQ_LEN_Q;
        param.seq_len_kv = SEQ_LEN_KV;
        param.head_dim = kHeadDim;
        param.n_block = N_BLOCK;
        param.tail_n = tail_n;
        param.m_block = M_BLOCK;
        param.tail_m = tail_m;
        param.seq_len_kv_pad = args.seqlen_k_rounded;
        param.seq_len_q_pad = args.seqlen_q_rounded;
        param.window_size_left = args.window_size_left;
        param.window_size_right = args.window_size_right;
        param.is_local = args.is_local;
        
        // Setup strides (BSHD layout - batch, seq, head, dim)
        setup_bshd_stride_bwd(param);

        // Phase 1: Compute O * dO
        auto dimGrid0 = compat::dim3(size(M_BLOCK), size(param.num_head_q), size(param.batch));
        auto dimBlock0 = compat::dim3(size(kNSGs * SubgroupSize), size(1), size(1));
        compat::experimental::launch_properties launch_props0{};
        compat::experimental::kernel_properties kernel_props0{
            sycl::ext::oneapi::experimental::sub_group_size<SubgroupSize>};
        compat::experimental::launch_policy policy0{dimGrid0, dimBlock0, launch_props0, kernel_props0};
        auto event0 = compat::experimental::launch<
            mha_dot_do_o<decltype(trait)>,
            mhaodoDeviceNameBwd<decltype(trait)>>(policy0, trait, param);
        EventManager::getInstance().addEvent(event0);
        compat::wait_and_throw();

        // Phase 2: Main backward pass
        auto dimGrid1 = compat::dim3(size(ceil_div(param.n_block, param.num_nb_per_blk)),
                                     size(param.num_head_q), size(param.batch));
        auto dimBlock1 = compat::dim3(size(kNSGs * SubgroupSize), size(1), size(1));
        compat::experimental::launch_properties launch_props1{
            sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
        };
        compat::experimental::kernel_properties kernel_props1{
            sycl::ext::oneapi::experimental::sub_group_size<SubgroupSize>};
        compat::experimental::launch_policy policy1{dimGrid1, dimBlock1, launch_props1, kernel_props1};
        auto event1 = compat::experimental::launch<
            mha_backward_seq<decltype(trait)>,
            mhabwdDeviceNameBwd<decltype(trait)>>(policy1, trait, param);
        EventManager::getInstance().addEvent(event1);
        compat::wait_and_throw();

        // Phase 3: Convert dQ from float to target type
        auto dimGrid2 = compat::dim3(size(M_BLOCK), size(param.num_head_q), size(param.batch));
        auto dimBlock2 = compat::dim3(size(kNSGs * SubgroupSize), size(1), size(1));
        compat::experimental::launch_properties launch_props2{};
        compat::experimental::kernel_properties kernel_props2{
            sycl::ext::oneapi::experimental::sub_group_size<SubgroupSize>};
        compat::experimental::launch_policy policy2{dimGrid2, dimBlock2, launch_props2, kernel_props2};
        auto event2 = compat::experimental::launch<
            mhd_convert_dq<decltype(trait)>,
            mhacvtDeviceNameBwd<decltype(trait)>>(policy2, trait, param);
        EventManager::getInstance().addEvent(event2);
        compat::wait_and_throw();
        
        // Free intermediate buffer
        compat::free(pbuff);
        
        return cutlass::Status::kSuccess;
    }
};

// FMHABwdConfig template - similar to FMHAConfig in fmha_fwd_impl.hpp
template <
    typename bwd_policy,
    typename ElementQ = cute::bfloat16_t,
    typename ElementK = cute::bfloat16_t,
    typename ElementV = cute::bfloat16_t,
    typename ElementO = cute::bfloat16_t>
struct FMHABwdConfig {
    using DType = ElementQ;

    template <bool Causal, bool Local>
    static void run(sycl::queue& queue, const fmha_bwd_args_t& args) {
        using FABwdKernelType = FABwdKernel<
            DType,
            bwd_policy::kHeadDim,
            bwd_policy::kBlockM,
            bwd_policy::kBlockN,
            bwd_policy::kNSGs,
            bwd_policy::AtomLayoutMSdP,
            bwd_policy::AtomLayoutNdKV,
            bwd_policy::AtomLayoutMdQ,
            Causal,
            Local>;

        BwdKernelLauncher<FABwdKernelType> launcher;
        launcher.run(queue, args);
    }

    template <bool... Bs>
    static void kernel_dispatch(sycl::queue& queue, const fmha_bwd_args_t& args) {
        return run<Bs...>(queue, args);
    }

    template <bool... Bs, typename... Ts>
    static void kernel_dispatch(sycl::queue& queue, const fmha_bwd_args_t& args, bool b, Ts... ts) {
        if (b) {
            kernel_dispatch<Bs..., true>(queue, args, ts...);
        } else {
            kernel_dispatch<Bs..., false>(queue, args, ts...);
        }
    }
};

// bwd_policy_dispatch - similar to policy_dispatch in fmha_fwd_impl.hpp
template <typename bwd_policy, int IsCausal = -1, int IsLocal = -1>
void bwd_policy_dispatch(sycl::queue& queue, BwdCutlassType cuType, const fmha_bwd_args_t& args) {
    if (cuType == BwdCutlassType::half) {
        using Config = FMHABwdConfig<
            bwd_policy,
            cute::half_t,
            cute::half_t,
            cute::half_t,
            cute::half_t>;
        if constexpr (IsCausal != -1 && IsLocal != -1) {
            return Config::template kernel_dispatch<IsCausal, IsLocal>(queue, args);
        } else {
            return Config::kernel_dispatch(queue, args, args.is_causal, args.is_local);
        }
    } else {
        using Config = FMHABwdConfig<
            bwd_policy,
            cute::bfloat16_t,
            cute::bfloat16_t,
            cute::bfloat16_t,
            cute::bfloat16_t>;
        if constexpr (IsCausal != -1 && IsLocal != -1) {
            return Config::template kernel_dispatch<IsCausal, IsLocal>(queue, args);
        } else {
            return Config::kernel_dispatch(queue, args, args.is_causal, args.is_local);
        }
    }
}
