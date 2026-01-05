#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
static constexpr int WARP_SIZE = 32;

namespace vllm {
namespace moe {
// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing
// the output in the softmax kernel when we extend this module to support
// expert-choice routing.
template <int TPB, typename InputdType>
class moeSoftmax {
 public:
  moeSoftmax(
      sycl::local_accessor<float, 1>& slm,
      const InputdType* input,
      const bool* finished,
      float* output,
      const int num_cols)
      : slm(slm),
        input(input),
        finished(finished),
        output(output),
        num_cols(num_cols) {}

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    void* slm_ptr = static_cast<void*>(
        slm.template get_multi_ptr<sycl::access::decorated::no>().get());

    float* normalizing_factor = reinterpret_cast<float*>(slm_ptr);
    float* float_max = normalizing_factor + 1;

    auto group = item.get_group();
    auto local_id_x = item.get_local_id(0);
    auto group_id_x = item.get_group(0);

    const int thread_row_offset = group_id_x * num_cols;

    float threadData(INFINITY * -1);

    // Don't touch finished rows.
    if ((finished != nullptr) && finished[group_id_x]) {
      return;
    }

    for (int ii = local_id_x; ii < num_cols; ii += TPB) {
      const int idx = thread_row_offset + ii;
      threadData = MAX(static_cast<float>(input[idx]), threadData);
    }

    const float maxElem =
        sycl::reduce_over_group(group, threadData, sycl::maximum<float>());
    if (local_id_x == 0) {
      *float_max = maxElem;
    }
    item.barrier(sycl::access::fence_space::local_space);

    threadData = 0;

    for (int ii = local_id_x; ii < num_cols; ii += TPB) {
      const int idx = thread_row_offset + ii;
      threadData += sycl::exp((static_cast<float>(input[idx]) - *float_max));
    }

    const auto Z = sycl::reduce_over_group(group, threadData, sycl::plus<>());

    if (local_id_x == 0) {
      *normalizing_factor = 1.f / Z;
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (int ii = local_id_x; ii < num_cols; ii += TPB) {
      const int idx = thread_row_offset + ii;
      const float val =
          sycl::exp((static_cast<float>(input[idx]) - (*float_max))) *
          (*normalizing_factor);
      output[idx] = val;
    }
  }

 private:
  sycl::local_accessor<float, 1> slm;
  const InputdType* input;
  const bool* finished;
  float* output;
  const int num_cols;
};

template <int TPB, typename IndType>
class moeTopK {
 public:
  moeTopK(
      const float* inputs_after_softmax,
      const bool* finished,
      float* output,
      IndType* indices,
      int* source_rows,
      const int num_experts,
      const int k,
      const int start_expert,
      const int end_expert,
      const bool renormalize)
      : inputs_after_softmax(inputs_after_softmax),
        finished(finished),
        output(output),
        indices(indices),
        source_rows(source_rows),
        num_experts(num_experts),
        k(k),
        start_expert(start_expert),
        end_expert(end_expert),
        renormalize(renormalize) {}

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    int kIdx;
    float kVal;

    auto group = item.get_group();
    auto local_id_x = item.get_local_id(0);
    auto group_id_x = item.get_group(0);

    const int num_rows = item.get_group_range(0);
    const int block_row = group_id_x;

    const bool row_is_active = finished ? !finished[block_row] : true;
    const int thread_read_offset = group_id_x * num_experts;
    float sum_val = 0.0f;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      kIdx = 0;
      kVal = -1.f;  // This is OK because inputs are probabilities

      int inpIdx;
      float inpVal;
      for (int expert = local_id_x; expert < num_experts; expert += TPB) {
        const int idx = thread_read_offset + expert;
        inpIdx = expert;
        inpVal = inputs_after_softmax[idx];

        for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
          const int prior_winning_expert = indices[k * block_row + prior_k];

          if (prior_winning_expert == expert) {
            inpIdx = kIdx;
            inpVal = kVal;
          }
        }

        if (inpVal > kVal) {
          kIdx = inpIdx;
          kVal = inpVal;
        }
      }

      const float resultVal =
          sycl::reduce_over_group(group, kVal, sycl::maximum<float>());
      const int resultIdx = sycl::reduce_over_group(
          group, resultVal == kVal ? kIdx : 0x7FFFFFFF, sycl::minimum<int>());
      sum_val += resultVal;

      if (local_id_x == 0) {
        // Ignore experts the node isn't responsible for with expert parallelism
        const int expert = resultIdx;
        const bool node_uses_expert =
            expert >= start_expert && expert < end_expert;
        const bool should_process_row = row_is_active && node_uses_expert;

        const int idx = k * block_row + k_idx;
        output[idx] = resultVal;
        indices[idx] =
            should_process_row ? (expert - start_expert) : num_experts;
        assert(indices[idx] >= 0);
        source_rows[idx] = k_idx * num_rows + block_row;
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    if (renormalize) {
      auto local_range_x = item.get_local_range(0);
      for (int k_idx = local_id_x; k_idx < k; k_idx += local_range_x) {
        const int idx = k * block_row + k_idx;
        output[idx] /= sum_val;
      }
    }
  }

 private:
  const float* inputs_after_softmax;
  const bool* finished;
  float* output;
  IndType* indices;
  int* source_rows;
  const int num_experts;
  const int k;
  const int start_expert;
  const int end_expert;
  const bool renormalize;
};

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the
  MoE layers are a small power of 2. This allows us to cleanly share the rows
  among the threads in a single warp and eliminate communication between warps
  (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is optimized for when the number of experts is a small
  power of 2. Additionally it also supports when number of experts is multiple
  of 64 which is still faster than the computing softmax and topK separately. 2)
  This implementation assumes k is small, but will work for any k.
*/

template <
    int VPT,
    int NUM_EXPERTS,
    int WARPS_PER_CTA,
    int BYTES_PER_LDG,
    int WARP_SIZE_PARAM,
    typename InputdType,
    typename IndType>
class topkGatingSoftmax {
 public:
  topkGatingSoftmax(
      const InputdType* input,
      const bool* finished,
      float* output,
      const int num_rows,
      IndType* indices,
      int* source_rows,
      const int k,
      const int start_expert,
      const int end_expert,
      const bool renormalize)
      : input(input),
        finished(finished),
        output(output),
        num_rows(num_rows),
        indices(indices),
        source_rows(source_rows),
        k(k),
        start_expert(start_expert),
        end_expert(end_expert),
        renormalize(renormalize) {}

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<2> item) const {
    auto sg = item.get_sub_group();
    auto local_id_x = item.get_local_id(1);
    auto local_id_y = item.get_local_id(0);
    auto group_id_x = item.get_group(1);
    // We begin by enforcing compile time assertions and setting up compile time
    // constants.
    static_assert(
        BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
        "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    // Number of bytes each thread pulls in per load
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputdType);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    // Restrictions based on previous section.
    static_assert(
        VPT % ELTS_PER_LDG == 0,
        "The elements per thread must be a multiple of the elements per ldg");
    static_assert(
        WARP_SIZE_PARAM % THREADS_PER_ROW == 0,
        "The threads per row must cleanly divide the threads per warp");
    static_assert(
        THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW),
        "THREADS_PER_ROW must be power of 2");
    static_assert(
        THREADS_PER_ROW <= WARP_SIZE_PARAM,
        "THREADS_PER_ROW can be at most warp size");

    // We have NUM_EXPERTS elements per row. We specialize for small #experts
    static constexpr int ELTS_PER_WARP = WARP_SIZE_PARAM * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    // Restrictions for previous section.
    static_assert(
        ELTS_PER_WARP % ELTS_PER_ROW == 0,
        "The elts per row must cleanly divide the total elt per warp");

    // ===================== From this point, we finally start computing
    // run-time variables. ========================

    // Compute CTA and warp rows. We pack multiple rows into a single warp, and
    // a block contains WARPS_PER_CTA warps. This, each block processes a chunk
    // of rows. We start by computing the start row for each block.
    const int cta_base_row = group_id_x * ROWS_PER_CTA;

    // Now, using the base row per thread block, we compute the base row per
    // warp.
    const int warp_base_row = cta_base_row + local_id_y * ROWS_PER_WARP;

    // The threads in a warp are split into sub-groups that will work on a row.
    // We compute row offset for each thread sub-group
    const int thread_row_in_warp = local_id_x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    // Threads with indices out of bounds should early exit here.
    if (thread_row >= num_rows) {
      return;
    }
    const bool row_is_active = finished ? !finished[thread_row] : true;

    // We finally start setting up the read pointers for each thread. First,
    // each thread jumps to the start of the row it will read.
    const InputdType* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    // Now, we compute the group each thread belong to in order to determine the
    // first column to start loads.
    const int thread_group_idx = local_id_x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const InputdType* thread_read_ptr =
        thread_row_ptr + first_elt_read_by_thread;

    // Finally, we pull in the data from global mem
    InputdType row_chunk_load[VPT];
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
#pragma unroll
      for (int jj = 0; jj < ELTS_PER_LDG; ++jj) {
        row_chunk_load[ii * ELTS_PER_LDG + jj] =
            thread_read_ptr[ii * THREADS_PER_ROW * ELTS_PER_LDG + jj];
      }
    }

    float row_chunk[VPT];
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = static_cast<float>(row_chunk_load[ii]);
    }

    // First, we perform a max reduce within the thread. We can do the max in
    // fp16 safely (I think) and just convert to float afterwards for the exp +
    // sum reduction.
    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii) {
      thread_max = MAX(thread_max, row_chunk[ii]);
    }

// Now, we find the max within the thread group and distribute among the
// threads. We use a butterfly reduce.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      auto other_thread_max = sycl::permute_group_by_xor(sg, thread_max, mask);
      thread_max =
          thread_max > other_thread_max ? thread_max : other_thread_max;
    }

    // From this point, thread max in all the threads have the max within the
    // row. Now, we subtract the max from each element in the thread and take
    // the exp. We also compute the thread local sum.
    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = sycl::exp(row_chunk[ii] - thread_max);
      row_sum += row_chunk[ii];
    }

// Now, we perform the sum reduce within each thread group. Similar to the max
// reduce, we use a bufferfly pattern.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      row_sum += sycl::permute_group_by_xor(sg, row_sum, mask);
    }

    // From this point, all threads have the max and the sum for their rows in
    // the thread_max and thread_sum variables respectively. Finally, we can
    // scale the rows for the softmax. Technically, for top-k gating we don't
    // need to compute the entire softmax row. We can likely look at the maxes
    // and only compute for the top-k values in the row. However, this kernel
    // will likely not be a bottle neck and it seems better to closer match
    // torch and find the argmax after computing the softmax.
    const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }

    // Now, softmax_res contains the softmax of the row chunk. Now, I want to
    // find the topk elements in each row, along with the max index.
    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;
    float sum_val = 0.0f;

    for (int k_idx = 0; k_idx < k; ++k_idx) {
      // First, each thread does the local argmax
      float max_val = row_chunk[0];
      int expert_local = start_col;
      int max_val_idx = 0;
#pragma unroll
      for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
           ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
        for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
          float val = row_chunk[ldg * ELTS_PER_LDG + ii];

          // No check on the experts here since columns with the smallest index
          // are processed first and only updated if > (not >=)
          if (val > max_val) {
            max_val = val;
            expert_local = col + ii;
            max_val_idx = ldg * ELTS_PER_LDG + ii;
          }
        }
      }

      // Now, we perform the argmax reduce. We use the butterfly pattern so
      // threads reach consensus about the max. This will be useful for K > 1 so
      // that the threads can agree on "who" had the max value. That thread can
      // then blank out their max with -inf and the warp can run more
      // iterations...
      int expert = expert_local;
#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float other_max = sycl::permute_group_by_xor(sg, max_val, mask);
        int other_expert = sycl::permute_group_by_xor(sg, expert, mask);

        // We want lower indices to "win" in every thread so we break ties this
        // way
        if (other_max > max_val ||
            (other_max == max_val && other_expert < expert)) {
          max_val = other_max;
          expert = other_expert;
        }
      }

      sum_val += max_val;

      // Write the max for this k iteration to global memory.
      if (thread_group_idx == 0) {
        // Add a guard to ignore experts not included by this node
        const bool node_uses_expert =
            expert >= start_expert && expert < end_expert;
        const bool should_process_row = row_is_active && node_uses_expert;

        // The lead thread from each sub-group will write out the final results
        // to global memory. (This will be a single) thread per row of the
        // input/output matrices.
        const int idx = k * thread_row + k_idx;
        output[idx] = max_val;
        indices[idx] =
            should_process_row ? (expert - start_expert) : NUM_EXPERTS;
        source_rows[idx] = k_idx * num_rows + thread_row;
      }

      // Finally, we clear the value in the thread with the current max if there
      // is another iteration to run.
      if (expert == expert_local) {
        row_chunk[max_val_idx] = -10000.f;
      }
    }

    if (renormalize) {
      for (int k_idx = thread_group_idx; k_idx < k; k_idx += THREADS_PER_ROW) {
        const int idx = k * thread_row + k_idx;
        output[idx] /= sum_val;
      }
    }
  }

 private:
  const InputdType* input;
  const bool* finished;
  float* output;
  const int num_rows;
  IndType* indices;
  int* source_rows;
  const int k;
  const int start_expert;
  const int end_expert;
  const bool renormalize;
};

namespace detail {
// Constructs some constants needed to partition the work across threads at
// compile time.
template <
    int EXPERTS,
    int BYTES_PER_LDG,
    int WARP_SIZE_PARAM,
    typename InputdType>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputdType);
  static_assert(
      EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0 ||
          EXPERTS % (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0,
      "");
  static constexpr int VECs_PER_THREAD =
      MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static const int ROWS_PER_WARP = WARP_SIZE_PARAM / THREADS_PER_ROW;
};
}  // namespace detail

template <
    int EXPERTS,
    int WARPS_PER_TB,
    int WARP_SIZE_PARAM,
    int MAX_BYTES_PER_LDG,
    typename InputdType,
    typename IndType>
void topkGatingSoftmaxLauncherHelper(
    const InputdType* input,
    const bool* finished,
    float* output,
    IndType* indices,
    int* source_row,
    const int num_rows,
    const int k,
    const int start_expert,
    const int end_expert,
    bool renormalize,
    sycl::queue& queue) {
  static constexpr int BYTES_PER_LDG =
      MIN(MAX_BYTES_PER_LDG, sizeof(InputdType) * EXPERTS);
  using Constants = detail::
      TopkConstants<EXPERTS, BYTES_PER_LDG, WARP_SIZE_PARAM, InputdType>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  sycl::range<2> grid(1, num_blocks);
  sycl::range<2> block(WARPS_PER_TB, WARP_SIZE_PARAM);
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(grid * block, block),
        topkGatingSoftmax<
            VPT,
            EXPERTS,
            WARPS_PER_TB,
            BYTES_PER_LDG,
            WARP_SIZE_PARAM,
            InputdType,
            IndType>(
            input,
            finished,
            output,
            num_rows,
            indices,
            source_row,
            k,
            start_expert,
            end_expert,
            renormalize));
  });
}

#define LAUNCH_SOFTMAX(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES)                   \
  static_assert(                                                               \
      WARP_SIZE == 32, "Unsupported warp size. Only 32 is supported for XPU"); \
  topkGatingSoftmaxLauncherHelper<                                             \
      NUM_EXPERTS,                                                             \
      WARPS_PER_TB,                                                            \
      WARP_SIZE,                                                               \
      MAX_BYTES>(                                                              \
      gating_output,                                                           \
      nullptr,                                                                 \
      topk_weights,                                                            \
      topk_indices,                                                            \
      token_expert_indices,                                                    \
      num_tokens,                                                              \
      topk,                                                                    \
      0,                                                                       \
      num_experts,                                                             \
      renormalize,                                                             \
      queue);

template <typename InputdType, typename IndType>
void topkGatingSoftmaxKernelLauncher(
    const InputdType* gating_output,
    float* topk_weights,
    IndType* topk_indices,
    int* token_expert_indices,
    float* softmax_workspace,
    const int num_tokens,
    const int num_experts,
    const int topk,
    const bool renormalize,
    sycl::queue& queue) {
  static constexpr int WARPS_PER_TB = 4;
  static constexpr int BYTES_PER_LDG_POWER_OF_2 = 16;
  static constexpr int BYTES_PER_LDG_MULTIPLE_64 = 2 * sizeof(InputdType);

  switch (num_experts) {
    case 1:
      LAUNCH_SOFTMAX(1, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 2:
      LAUNCH_SOFTMAX(2, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 4:
      LAUNCH_SOFTMAX(4, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 8:
      LAUNCH_SOFTMAX(8, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 16:
      LAUNCH_SOFTMAX(16, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 32:
      LAUNCH_SOFTMAX(32, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 64:
      LAUNCH_SOFTMAX(64, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 128:
      LAUNCH_SOFTMAX(128, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 256:
      LAUNCH_SOFTMAX(256, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 512:
      LAUNCH_SOFTMAX(512, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2);
      break;
    case 192:
      LAUNCH_SOFTMAX(192, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 320:
      LAUNCH_SOFTMAX(320, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 384:
      LAUNCH_SOFTMAX(384, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 448:
      LAUNCH_SOFTMAX(448, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    case 576:
      LAUNCH_SOFTMAX(576, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64);
      break;
    default: {
      TORCH_CHECK(
          softmax_workspace != nullptr,
          "softmax_workspace must be provided for num_experts that are "
          "not a power of 2 or multiple of 64.");
      static constexpr int TPB = 256;
      sycl::range<1> grid1(num_tokens);
      sycl::range<1> block1(TPB);
      queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(2), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(grid1 * block1, block1),
            moeSoftmax<TPB, InputdType>(
                slm, gating_output, nullptr, softmax_workspace, num_experts));
      });

      sycl::range<1> grid2(num_tokens);
      sycl::range<1> block2(TPB);
      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(grid2 * block2, block2),
            moeTopK<TPB, IndType>(
                softmax_workspace,
                nullptr,
                topk_weights,
                topk_indices,
                token_expert_indices,
                num_experts,
                topk,
                0,
                num_experts,
                renormalize));
      });
    }
  }
}

}  // namespace moe
}  // namespace vllm

void topk_softmax(
    torch::Tensor& topk_weights,          // [num_tokens, topk]
    torch::Tensor& topk_indices,          // [num_tokens, topk]
    torch::Tensor& token_expert_indices,  // [num_tokens, topk]
    torch::Tensor& gating_output,         // [num_tokens, num_experts]
    const bool renormalize) {
  const int num_experts = gating_output.size(-1);
  const auto num_tokens = gating_output.numel() / num_experts;
  const int topk = topk_weights.size(-1);

  const bool is_pow_2 =
      (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
  const bool needs_workspace = !is_pow_2 || num_experts > 256;
  const int64_t workspace_size = needs_workspace ? num_tokens * num_experts : 0;

  const at::DeviceGuard device_guard(gating_output.device());
  auto& queue = vllm::xpu::vllmGetQueue();
  torch::Tensor softmax_workspace = torch::empty(
      {workspace_size}, gating_output.options().dtype(torch::kFloat));

#define LAUNCH_TOPK_SOFTMAX(INPUTDTYPE, INDTYPE)                       \
  vllm::moe::topkGatingSoftmaxKernelLauncher(                          \
      reinterpret_cast<INPUTDTYPE*>(gating_output.mutable_data_ptr()), \
      topk_weights.data_ptr<float>(),                                  \
      topk_indices.data_ptr<INDTYPE>(),                                \
      token_expert_indices.data_ptr<int>(),                            \
      softmax_workspace.data_ptr<float>(),                             \
      num_tokens,                                                      \
      num_experts,                                                     \
      topk,                                                            \
      renormalize,                                                     \
      queue);

  if (topk_indices.scalar_type() == at::ScalarType::Int) {
    if (gating_output.scalar_type() == at::ScalarType::Float)
      LAUNCH_TOPK_SOFTMAX(float, int)
    else if (gating_output.scalar_type() == at::ScalarType::Half)
      LAUNCH_TOPK_SOFTMAX(sycl::half, int)
    else
      LAUNCH_TOPK_SOFTMAX(sycl::ext::oneapi::bfloat16, int)
  } else if (topk_indices.scalar_type() == at::ScalarType::UInt32) {
    if (gating_output.scalar_type() == at::ScalarType::Float)
      LAUNCH_TOPK_SOFTMAX(float, uint32_t)
    else if (gating_output.scalar_type() == at::ScalarType::Half)
      LAUNCH_TOPK_SOFTMAX(sycl::half, uint32_t)
    else
      LAUNCH_TOPK_SOFTMAX(sycl::ext::oneapi::bfloat16, uint32_t)
  } else {
    TORCH_CHECK(topk_indices.scalar_type() == at::ScalarType::Long);
    if (gating_output.scalar_type() == at::ScalarType::Float)
      LAUNCH_TOPK_SOFTMAX(float, int64_t)
    else if (gating_output.scalar_type() == at::ScalarType::Half)
      LAUNCH_TOPK_SOFTMAX(sycl::half, int64_t)
    else
      LAUNCH_TOPK_SOFTMAX(sycl::ext::oneapi::bfloat16, int64_t)
  }

#undef LAUNCH_TOPK_SOFTMAX
}
