/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "utils.h"
#include "skip_list.h"

namespace flash
{

    using namespace cute;

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <bool const zero_init = true, bool const outer_loop_is_rows = false, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
    __device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &summary, Operator &op)
    {
        static_assert(Layout0::rank == 2, "Only support 2D Tensor");
        static_assert(Layout1::rank == 1, "Only support 1D Tensor");
        CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
        
        // Helper lambda to reduce code duplication
        auto reduce_element = [&](int mi, int ni){
            if constexpr (zero_init){
                summary(mi) = ni == 0 ? tensor(mi, ni) : op(summary(mi), tensor(mi, ni));
            }else{
                summary(mi) = op(summary(mi), tensor(mi, ni));
            }
            // summary(mi) = zero_init && ni == 0 ? tensor(mi, ni) : op(summary(mi), tensor(mi, ni));
        };
        
        if constexpr (outer_loop_is_rows) {
            // Outer loop: rows (mi), Inner loop: columns (ni)
#pragma unroll
            for (int mi = 0; mi < size<0>(tensor); mi++)
            {
#pragma unroll
                for (int ni = 0; ni < size<1>(tensor); ni++)
                {
                    reduce_element(mi, ni);
                }
            }
        } else {
            // Outer loop: columns (ni), Inner loop: rows (mi) - original order
#pragma unroll
            for (int ni = 0; ni < size<1>(tensor); ni++)
            {
#pragma unroll
                for (int mi = 0; mi < size<0>(tensor); mi++)
                {
                    reduce_element(mi, ni);
                }
            }
        }
    }

    // Dequantize a 1D tensor (e.g., after reduction) to another 1D tensor with optional max operation
    template <bool const zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
    __device__ __forceinline__ void dequantize_max_1d_(Tensor<Engine0, Layout0> &src, Tensor<Engine1, Layout1> &dst, float const dequan_s)
    {
        MaxOp<float> op;
        static_assert(Layout0::rank == 1, "Only support 1D Tensor for source");
        static_assert(Layout1::rank == 1, "Only support 1D Tensor for destination");
        CUTE_STATIC_ASSERT_V(size(src) == size(dst));
#pragma unroll
        for (int mi = 0; mi < size(src); mi++)
        {
            const float value = src(mi) * dequan_s;
            if constexpr (zero_init){
                dst(mi) = value;
            }else{
                dst(mi) = op(dst(mi), value);
            }
        }
    }

    template <typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
    __device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op)
    {
        CUTE_STATIC_ASSERT_V(size(dst) == size(src));
#pragma unroll
        for (int i = 0; i < size(dst); i++)
        {
            dst(i) = Allreduce<4>::run(src(i), op);
        }
    }

    template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
    __device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &summary, Operator &op)
    {
        thread_reduce_<zero_init>(tensor, summary, op);
        quad_allreduce_(summary, summary, op);
    }

    template <bool const zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
    __device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &max)
    {
        MaxOp<float> max_op;
        reduce_<zero_init>(tensor, max, max_op);
    }

    template <bool const zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
    __device__ __forceinline__ void reduce_max_dequantize(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &max, float const dequan_s)
    {
        MaxOp<int32_t> max_op;
        Tensor max_converted = make_tensor_like<int32_t>(max);
        // thread_reduce_<true, true /*outer_loop_is_rows*/>(tensor, max_converted, max_op);
        thread_reduce_<true, false /*outer_loop_is_rows*/>(tensor, max_converted, max_op);
        quad_allreduce_(max_converted, max_converted, max_op);
        dequantize_max_1d_<zero_init>(max_converted, max, dequan_s);
    }

    template <bool const zero_init = true, bool warp_reduce = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
    __device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &sum)
    {
        SumOp<float> sum_op;
        thread_reduce_<zero_init>(tensor, sum, sum_op);
        if constexpr (warp_reduce)
        {
            quad_allreduce_(sum, sum, sum_op);
        }
    }

    // Apply the exp to all the elements.
    template <bool const Scale_max = true, bool const Check_inf = true, int const Max_offset = 0,
              typename Engine0, typename Layout0, typename Engine1, typename Layout1>
    __forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale)
    {
        // For FP8, we can subtract max by 8.0 so that the value after exp2 is in the range of [0, 256].
        // This lets us use more of the FP8 range (instead of just [0, 1]) to reduce underflow.
        static constexpr float max_offset = float(Max_offset); // We can only template on int, not float
        static_assert(Layout0::rank == 2, "Only support 2D Tensor");
        static_assert(Layout1::rank == 1, "Only support 1D Tensor");
        CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
#pragma unroll
        for (int mi = 0; mi < size<0>(tensor); ++mi)
        {
            // If max is -inf, then all elements must have been -inf (possibly due to masking).
            // We don't want (-inf - (-inf)) since that would give NaN.
            // const float max_scaled = Check_inf
            //                              ? (max(mi) == -INFINITY ? 0.f : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset)
            //                              : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset;
            if constexpr (Check_inf){
                const float max_scaled = max(mi) == -INFINITY ? 0.f : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset;
    #pragma unroll
                for (int ni = 0; ni < size<1>(tensor); ++ni)
                {
                    // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                    // max * log_2(e)). This allows the compiler to use the ffma
                    // instruction instead of fadd and fmul separately.
                    tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
                }
            }else{
                const float max_scaled = (!Scale_max ? max(mi) : max(mi) * scale) - max_offset;
    #pragma unroll
                for (int ni = 0; ni < size<1>(tensor); ++ni)
                {
                    // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                    // max * log_2(e)). This allows the compiler to use the ffma
                    // instruction instead of fadd and fmul separately.
                    tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
                }
            }
        }
    }

    // Apply the exp to all the elements, dequantizing int32 input to float output.
    // tensor: input tensor with int32_t values (from INT8 MMA)
    // tensor_dequantized: output tensor with float values
    // dequan_s: dequantization scale (q_dequant * k_dequant)
    template <bool const Scale_max = true, bool const Check_inf = true, int const Max_offset = 0, bool const outer_loop_is_rows = true,
              typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Engine2, typename Layout2>
    __forceinline__ __device__ void scale_apply_exp2_dequantize(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &max,
                                                                 Tensor<Engine2, Layout2> &tensor_dequantized, const float dequan_s)
    {
        // For FP8, we can subtract max by 8.0 so that the value after exp2 is in the range of [0, 256].
        // This lets us use more of the FP8 range (instead of just [0, 1]) to reduce underflow.
        static constexpr float max_offset = float(Max_offset); // We can only template on int, not float
        static_assert(Layout0::rank == 2, "Only support 2D Tensor");
        static_assert(Layout1::rank == 1, "Only support 1D Tensor");
        static_assert(Layout2::rank == 2, "Only support 2D Tensor for output");
        CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
        CUTE_STATIC_ASSERT_V(size<0>(tensor) == size<0>(tensor_dequantized));
        CUTE_STATIC_ASSERT_V(size<1>(tensor) == size<1>(tensor_dequantized));
        
        // Helper lambda to compute max_scaled for a given row index
        auto get_max_scaled = [&](int mi) -> float {
            if constexpr (Check_inf){
                return max(mi) == -INFINITY ? 0.f : (!Scale_max ? max(mi) : max(mi)) - max_offset;
            }else{
                return (!Scale_max ? max(mi) : max(mi)) - max_offset;
            }
        };
        
        // Helper lambda to process a single element
        auto process_element = [&](int mi, int ni, float max_scaled) {
            // Dequantize int32 to float, then compute exp2(dequantized_value - max_scaled)
            // tensor(mi, ni) is int32_t, multiply by dequan_s to get float
            const float dequantized_value = tensor(mi, ni) * dequan_s - max_scaled;
            tensor_dequantized(mi, ni) = exp2f(dequantized_value);
        };
        
        if constexpr (outer_loop_is_rows) {
            // Outer loop: rows (mi), Inner loop: columns (ni) - original/default order
#pragma unroll
            for (int mi = 0; mi < size<0>(tensor); ++mi)
            {
                const float max_scaled = get_max_scaled(mi);
#pragma unroll
                for (int ni = 0; ni < size<1>(tensor); ++ni)
                {
                    process_element(mi, ni, max_scaled);
                }
            }
        } else {
            // Outer loop: columns (ni), Inner loop: rows (mi)
#pragma unroll
            for (int ni = 0; ni < size<1>(tensor); ++ni)
            {
#pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi)
                {
                    const float max_scaled = get_max_scaled(mi);
                    process_element(mi, ni, max_scaled);
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <int kNRows, int Max_offset = 0, const bool Is_INT8 = false>
    struct Softmax
    {

        using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
        TensorT row_max, row_sum;
        float const softmax_scale_log2; // (log2(e) * 1/sqrt(128)) * q_dequant * k_dequant
        // int const warp_idx_in_warpgroup = (threadIdx.x / 32) % 4;
        int const warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        bool const is_warp_leader = (threadIdx.x % 32) == 0;
        // bool const is_warp_leader = cute::elect_one_sync();
        // int const row_mask;
        // int const local_row_idx;
        int const seqlen_q;
        int const thread_idx;
        // float const dequan_q;
        float dequan_s;

        CUTLASS_DEVICE Softmax(float const softmax_scale_log2_, int const seqlen_q_, int const thread_idx_) 
            : softmax_scale_log2(softmax_scale_log2_), seqlen_q(seqlen_q_), thread_idx(thread_idx_) {};
            // : softmax_scale_log2(softmax_scale_log2_), dequan_q(dequan_q_), seqlen_q(seqlen_q_), thread_idx(thread_idx_) {};

        CUTLASS_DEVICE void set_dequan_s(float const dequan_k)
        {
            // dequan_s = dequan_k * softmax_scale_log2;
            dequan_s = __shfl_sync(0xffffffff, dequan_k * softmax_scale_log2, 0);
        }

        template <int kBlockM, typename TiledMma, bool const Is_first, bool const Check_inf = false, typename Tensor0>
        __forceinline__ __device__ TensorT max_get_scale_detect_qk_skip(
            Tensor0 &acc_s,
            const float threshold,
            auto &skip_reader,
            const int m_block)
        {
            // For INT8: acc_s contains int32 values from INT8 MMA
            // For non-INT8: acc_s contains float values
            // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
            TensorT scores_scale;
            if constexpr (Is_first)
            {
                // consider: seperate into  max scores (over int's) and after it dequantize (multiply by dequan_s) and take the max with row_max
                // flash::template reduce_max</*zero_init=*/true>(scores, row_max);
                if constexpr (!Is_INT8) {
                    flash::template reduce_max</*zero_init=*/true>(scores, row_max);
                } else {
                    flash::template reduce_max_dequantize</*zero_init=*/true>(scores, row_max, dequan_s);
                }
                cute::fill(scores_scale, 1.f);
                if (is_warp_leader)
                {
                    skip_reader.update_skip(false, warp_idx_in_warpgroup);
                }
            }
            else
            {
                Tensor scores_max_prev = make_fragment_like(row_max);
                cute::copy(row_max, scores_max_prev);

                // find the local max for each row
                Tensor scores_max_local = make_fragment_like(row_max);
                /*
                inside reduce_max, each thread reduces over the columns he holds.
                each thread holds a querter of the columns, for example:
                for headdim == 128, we have 176 columns so each thread holds 176 / 4 = 44 columns.
                each 4 consecutive threads hold TOGTHER the full row.
                after the thread level reduction, we reduce across each 4 consecutive threads and get the local max for the row.
                */
                // flash::template reduce_max</*zero_init=*/true>(scores, scores_max_local);
                if constexpr (!Is_INT8) {
                    flash::template reduce_max</*zero_init=*/true>(scores, scores_max_local);
                } else {
                    flash::template reduce_max_dequantize</*zero_init=*/true>(scores, scores_max_local, dequan_s);
                }

                // update row max
                // thread_reduce_<true>(scores_max_local, row_max, MaxOp<float>());
                // flash::template reduce_max</*zero_init=*/false>(scores, row_max);

                // Compute row bounds following the same pattern as mask.h
                // Create identity tensor and partition it to get row coordinates
                auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
                auto thread0_mma = TiledMma{}.get_thread_slice(_0{});
                Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockM>>{});  // Dummy shape, only need row info
                Tensor tScS = thread_mma.partition_C(cS);
                Tensor t0ScS = thread0_mma.partition_C(cS);
                Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol(tScS.layout()));
                Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol(t0ScS.layout()));
                
                // Compute thread_row_offset and seqlenq_row_limit following mask.h pattern
                int const thread_row_offset = get<0>(tScS_rowcol(_0{}, _0{}));
                int const seqlenq_row_limit = seqlen_q - m_block * kBlockM - thread_row_offset;
                
                bool do_qk = false;
#pragma unroll
                for (int mi = 0; mi < size(row_max); ++mi)
                {
                    // Check if this row is out of bounds, following mask.h pattern
                    // Use t0ScS_rowcol to get compile-time known row indices
                    const bool row_not_out_of_bounds = !(int(get<0>(t0ScS_rowcol(mi, _0{}))) >= seqlenq_row_limit);
                    
                    // update row max
                    row_max(mi) = max(row_max(mi), scores_max_local(mi));
                    // float cur = !Check_inf
                    //                 ? row_max(mi)
                    //                 : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                    float cur;
                    if constexpr (Check_inf){
                        cur = row_max(mi) == -INFINITY ? 0.0f : row_max(mi);
                    }else{
                        cur = row_max(mi);
                    }
                    float prev = scores_max_prev(mi);
                    // consider: removing all the uses of softmax_scale_log2 when Is_INT8 is enabled
                    if constexpr (!Is_INT8) {
                        scores_scale(mi) = exp2f((prev - cur) * softmax_scale_log2);
                    } else {
                        scores_scale(mi) = exp2f((prev - cur));
                    }
                    row_sum(mi) *= scores_scale(mi);

                    // do_qk |= (((scores_max_local(mi) - prev) * softmax_scale_log2) > thr) & row_not_out_of_bounds;

                    // do_qk |= ((scores_max_local(mi) * thr) >= prev) & row_not_out_of_bounds;

                    // bool cond1 = (scores_max_local(mi) - prev + abs(scores_max_local(mi)) * 0.5f >= 0); // if the current max is at least 1.5 times the previous max
                    bool cond1 = true;
                    // bool cond2 = ((scores_max_local(mi) - prev) * softmax_scale_log2) > threshold; // if the current max is more than threshold times the previous max
                    bool cond2;
                    if constexpr (!Is_INT8) {
                        cond2 = ((scores_max_local(mi) - prev) * softmax_scale_log2) > threshold; // if the current max is more than threshold times the previous max
                    } else {
                        cond2 = ((scores_max_local(mi) - prev) > threshold); // if the current max is more than threshold times the previous max
                    }
                    do_qk |= cond1 & cond2 & row_not_out_of_bounds; // if both conditions are true and the row is not out of bounds, then set do_qk to true
                }

                // (warp = 32) * 4 = warpgroup, 2 * warpgroup
                const bool skip = !__any_sync(0xffffffffu, do_qk);
                if (is_warp_leader)
                {
                    skip_reader.update_skip(skip, warp_idx_in_warpgroup);
                }
            }

            return scores_scale;
        };

        // TONY: acc_s is Q times K for one tile
        template <bool const Is_first, bool const Check_inf = false, typename Tensor0>
        __forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s)
        { // pass in a bool ref
            // For INT8: acc_s contains int32 values from INT8 MMA
            // For non-INT8: acc_s contains float values
            // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
            TensorT scores_scale;
            if constexpr (Is_first)
            {
                // flash::template reduce_max</*zero_init=*/true>(scores, row_max);
                if constexpr (!Is_INT8) {
                    flash::template reduce_max</*zero_init=*/true>(scores, row_max);
                } else {
                    flash::template reduce_max_dequantize</*zero_init=*/true>(scores, row_max, dequan_s);
                }
                cute::fill(scores_scale, 1.f);
            }
            else
            {
                Tensor scores_max_prev = make_fragment_like(row_max);
                cute::copy(row_max, scores_max_prev);
                
                // For INT8, we need to create a local max tensor and dequantize
                // flash::template reduce_max</*zero_init=*/false>(scores, row_max);
                if constexpr (!Is_INT8) {
                    flash::template reduce_max</*zero_init=*/false>(scores, row_max);
                } else {
                    // For INT8: reduce to local max first, then manually update row_max
                    // Tensor scores_max_local = make_fragment_like(row_max);
                    flash::template reduce_max_dequantize</*zero_init=*/false>(scores, row_max, dequan_s);
                }
                
#pragma unroll
                for (int mi = 0; mi < size(row_max); ++mi)
                {
                    // float scores_max_cur = !Check_inf
                    //                            ? row_max(mi)
                    //                            : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));

                    float scores_max_cur;
                    if constexpr (Check_inf){
                        scores_max_cur = row_max(mi) == -INFINITY ? 0.0f : row_max(mi);
                    }else{
                        scores_max_cur = row_max(mi);
                    }

                    // For INT8: don't multiply by softmax_scale_log2 (dequantization already handles scaling)
                    // scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                    if constexpr (!Is_INT8) {
                        scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                    } else {
                        scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur));
                    }
                    row_sum(mi) *= scores_scale(mi);
                }
            }
            return scores_scale;
        };

        template <bool const Is_first, bool const Check_inf = false, typename Tensor0>
        __forceinline__ __device__ void online_softmax(Tensor0 &acc_s)
        {
            // consider: assume acc_s is int's tensor when Is_INT8 is enabled
            // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
            // consider: scores are Int's when Is_INT8 is enabled, so we need to pass the dequan_s instead of scale_apply_exp2
            //           and also we need to define a new scores tensor for float type (we dequantize and take the exp2 inside this function)
            flash::template scale_apply_exp2</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);

            // consider: here we should reduce_sum with the values of the float exp2 scores (which we need to create a float tensor for when Is_INT8 is enabled)
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            flash::reduce_sum</*zero_init=*/Is_first, /*warp_reduce=*/false>(scores, row_sum);
        };

        template <bool const Is_first, bool const Check_inf = false, typename Tensor0, typename Tensor1>
        __forceinline__ __device__ void online_softmax_dequantize(Tensor0 &acc_s, Tensor1 &acc_s_float)
        {
            // consider: assume acc_s is int's tensor when Is_INT8 is enabled
            // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            Tensor scores_float = make_tensor(acc_s_float.data(), flash::convert_layout_acc_rowcol(acc_s_float.layout()));
            static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
            // consider: scores are Int's when Is_INT8 is enabled, so we need to pass the dequan_s instead of scale_apply_exp2
            //           and also we need to define a new scores tensor for float type (we dequantize and take the exp2 inside this function)
            // flash::template scale_apply_exp2</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);
            flash::template scale_apply_exp2_dequantize</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, scores_float, dequan_s);

            // consider: here we should reduce_sum with the values of the float exp2 scores (which we need to create a float tensor for when Is_INT8 is enabled)
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            flash::reduce_sum</*zero_init=*/Is_first, /*warp_reduce=*/false>(scores_float, row_sum);
        };

        __forceinline__ __device__ TensorT finalize(float const final_scale = 1.f)
        {
            SumOp<float> sum_op;
            quad_allreduce_(row_sum, row_sum, sum_op);
            TensorT scores_scale;
#pragma unroll
            for (int mi = 0; mi < size(row_sum); ++mi)
            {
                float sum = row_sum(mi);
                // float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
                float inv_sum = (sum == 0.f | sum != sum) ? 0.f : 1.f / sum;
                scores_scale(mi) = inv_sum * final_scale;
                // For FP8, we might have scaled the output of exp by 2**8 so we need to divide sum by that amount.
                if constexpr (Max_offset != 0)
                {
                    static constexpr float sum_scale = 1.f / float(1 << Max_offset);
                    sum *= sum_scale;
                }
                // consider: when Is_INT8 is enabled we don't need to multiply by softmax_scale_log2
                // row_sum(mi) = ((sum == 0.f) | (sum != sum)) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
                if constexpr (!Is_INT8) {
                    row_sum(mi) = row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
                } else {
                    row_sum(mi) = row_max(mi) * float(M_LN2) + __logf(sum);
                }
            }
            return scores_scale;
        };

        template <typename Tensor1>
        __forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale)
        {
            // consider: nothing change in this function when Is_INT8 is enabled (it's the same for both cases)
            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(CUTE_STATIC_V(size<0>(acc_o_rowcol)) == kNRows);
#pragma unroll
            for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi)
            {
#pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni)
                {
                    acc_o_rowcol(mi, ni) *= scores_scale(mi);
                }
            }
        };
    };

} // namespace flash
