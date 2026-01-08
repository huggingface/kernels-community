#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace vllm {
namespace moe {

constexpr float kNegInfinity = INFINITY * -1;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t BLOCK_SIZE = 512;
constexpr int32_t NUM_WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

namespace warp_topk {

template <int size, typename T>
constexpr T round_up_to_multiple_of(T len) {
  if (len == 0) {
    return 0;
  }
  return ((len - 1) / size + 1) * size;
}

template <typename T>
constexpr bool isPowerOf2(T v) {
  return (v && !(v & (v - 1)));
}

template <bool greater, typename T>
[[intel::device_indirectly_callable]] inline __attribute__((always_inline)) bool
is_better_than(T val, T baseline) {
  return (val > baseline && greater) || (val < baseline && !greater);
}

template <bool greater, typename T, typename idxT>
[[intel::device_indirectly_callable]] inline __attribute__((always_inline)) bool
is_better_than(T val, T baseline, idxT index, idxT baseline_index) {
  bool res = (val > baseline && greater) || (val < baseline && !greater);
  if (val == baseline) {
    res = (index < baseline_index && greater) ||
          (index < baseline_index && !greater);
  }
  return res;
}

template <typename T, typename idxT>
int calc_smem_size_for_block_wide(int num_of_warp, int64_t k) {
  uint64_t cache_topk = (sizeof(T) + sizeof(idxT)) * num_of_warp * k;
  uint64_t n = std::max<int>(num_of_warp / 2 * k, num_of_warp * WARP_SIZE);
  return std::max(
      cache_topk,
      round_up_to_multiple_of<256>(n * sizeof(T)) + n * sizeof(idxT));
}

template <
    int size,
    bool ascending,
    bool reverse,
    typename T,
    typename idxT,
    bool is_stable>
struct BitonicMerge {
  // input should be a bitonic sequence, and sort it to be a monotonic sequence
  [[intel::device_indirectly_callable]] static void merge(
      T* __restrict__ val_arr,
      idxT* __restrict__ idx_arr,
      sycl::sub_group const& sg,
      const int local_id) {
    static_assert(isPowerOf2(size));
    static_assert(size >= 2 * WARP_SIZE);
    constexpr int arr_len = size / WARP_SIZE;

    constexpr int stride = arr_len / 2;
    for (int i = 0; i < stride; ++i) {
      int const other_i = i + stride;
      T& val = val_arr[i];
      T& other_val = val_arr[other_i];
      bool is_better;
      if constexpr (is_stable) {
        is_better = is_better_than<ascending>(
            val, other_val, idx_arr[i], idx_arr[other_i]);
      } else {
        is_better = is_better_than<ascending>(val, other_val);
      }

      if (is_better) {
        T tmp = val;
        val = other_val;
        other_val = tmp;

        idxT tmp2 = idx_arr[i];
        idx_arr[i] = idx_arr[other_i];
        idx_arr[other_i] = tmp2;
      }
    }

    BitonicMerge<size / 2, ascending, reverse, T, idxT, is_stable>::merge(
        val_arr, idx_arr, sg, local_id);
    BitonicMerge<size / 2, ascending, reverse, T, idxT, is_stable>::merge(
        val_arr + arr_len / 2, idx_arr + arr_len / 2, sg, local_id);
  }
};

template <int size, bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort {
  [[intel::device_indirectly_callable]] static void sort(
      T* __restrict__ val_arr,
      idxT* __restrict__ idx_arr,
      sycl::sub_group const& sg,
      const int local_id) {
    static_assert(isPowerOf2(size));
    static_assert(size >= 2 * WARP_SIZE);
    constexpr int arr_len = size / WARP_SIZE;

    BitonicSort<size / 2, true, T, idxT, is_stable>::sort(
        val_arr, idx_arr, sg, local_id);
    BitonicSort<size / 2, false, T, idxT, is_stable>::sort(
        val_arr + arr_len / 2, idx_arr + arr_len / 2, sg, local_id);
    BitonicMerge<size, ascending, ascending, T, idxT, is_stable>::merge(
        val_arr, idx_arr, sg, local_id);
  }
};

template <bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort<32, ascending, T, idxT, is_stable> {
  [[intel::device_indirectly_callable]] static void sort(
      T* __restrict__ val_arr,
      idxT* __restrict__ idx_arr,
      sycl::sub_group const& sg,
      const int local_id) {
    int const lane = local_id % WARP_SIZE;

    // ascending doesn't matter before merging since all we need is a bitonic
    // sequence
    for (int stage = 0; stage < 4; ++stage) {
      for (int stride = (1 << stage); stride > 0; stride /= 2) {
        bool reverse = (lane >> stage) & 2;
        bool is_second = lane & stride;

        T other = sycl::permute_group_by_xor(sg, *val_arr, stride);
        idxT other_idx = sycl::permute_group_by_xor(sg, *idx_arr, stride);

        bool is_better;
        if constexpr (is_stable) {
          if constexpr (ascending) {
            is_better = ((*val_arr > other) ||
                         ((*val_arr == other) && (*idx_arr < other_idx))) !=
                        (reverse != is_second);
          } else {
            is_better = ((*val_arr > other) ||
                         ((*val_arr == other) && (*idx_arr > other_idx))) !=
                        (reverse != is_second);
          }
        } else {
          is_better =
              (*val_arr != other &&
               (*val_arr > other) != (reverse != is_second));
        }
        if (is_better) {
          *val_arr = other;
          *idx_arr = other_idx;
        }
      }
    }

    BitonicMerge<32, ascending, ascending, T, idxT, is_stable>::merge(
        val_arr, idx_arr, sg, local_id);
  }
};

template <
    bool ascending,
    bool reverse,
    typename T,
    typename idxT,
    bool is_stable>
struct BitonicMerge<32, ascending, reverse, T, idxT, is_stable> {
  [[intel::device_indirectly_callable]] static void merge(
      T* __restrict__ val_arr,
      idxT* __restrict__ idx_arr,
      sycl::sub_group const& sg,
      const int local_id) {
    int const lane = local_id % WARP_SIZE;
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
      bool is_second = lane & stride;
      T& val = *val_arr;
      T other = sycl::permute_group_by_xor(sg, val, stride);
      idxT& idx = *idx_arr;
      idxT other_idx = sycl::permute_group_by_xor(sg, idx, stride);

      bool is_better;
      if constexpr (is_stable) {
        if constexpr (ascending) {
          is_better = ((*val_arr > other) ||
                       ((*val_arr == other) && (*idx_arr < other_idx))) ==
                      (reverse != is_second);  // for min
        } else {
          is_better = ((*val_arr > other) ||
                       ((*val_arr == other) && (*idx_arr > other_idx))) ==
                      (reverse != is_second);  // for max
        }
      } else {
        is_better =
            (val != other && ((val > other) == (ascending != is_second)));
      }

      if (is_better) {
        val = other;
        idx = other_idx;
      }
    }
  }
};

template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSort {
 public:
  [[intel::device_indirectly_callable]] WarpSort(
      idxT k, T dummy, const int local_id)
      : lane_(local_id % WARP_SIZE), k_(k), dummy_(dummy) {
    static_assert(capacity >= WARP_SIZE && isPowerOf2(capacity));

    for (int i = 0; i < max_arr_len_; ++i) {
      val_arr_[i] = dummy_;
      idx_arr_[i] = 0;
    }
  }

  [[intel::device_indirectly_callable]] void
  dump(T* __restrict__ out, idxT* __restrict__ out_idx) const {
    for (int i = 0; i < max_arr_len_; ++i) {
      idxT out_i = i * WARP_SIZE + lane_;
      if (out_i < k_) {
        out[out_i] = val_arr_[i];
        out_idx[out_i] = idx_arr_[i];
      }
    }
  }

  [[intel::device_indirectly_callable]] void
  dumpIdx(idxT* __restrict__ out_idx) const {
    for (int i = 0; i < max_arr_len_; ++i) {
      idxT out_i = i * WARP_SIZE + lane_;
      if (out_i < k_) {
        out_idx[out_i] = idx_arr_[i];
      }
    }
  }

 protected:
  static constexpr int max_arr_len_ = capacity / WARP_SIZE;

  T val_arr_[max_arr_len_];
  idxT idx_arr_[max_arr_len_];

  int const lane_;
  idxT const k_;
  T const dummy_;

};  // end class WarpSort

template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSelect : public WarpSort<capacity, greater, T, idxT, is_stable> {
 public:
  [[intel::device_indirectly_callable]] WarpSelect(
      idxT k,
      T dummy,
      char* smem_buf,  // slm_buf
      const int local_id,
      const int local_range)
      : WarpSort<capacity, greater, T, idxT, is_stable>(k, dummy, local_id),
        k_th_(dummy),
        k_th_lane_((k - 1) % WARP_SIZE) {
    int const num_of_warp = local_range / WARP_SIZE;
    int const warp_id = local_id / WARP_SIZE;
    val_smem_ = reinterpret_cast<T*>(smem_buf);
    val_smem_ += warp_id * WARP_SIZE;
    idx_smem_ = reinterpret_cast<idxT*>(
        smem_buf +
        round_up_to_multiple_of<256>(num_of_warp * sizeof(T) * WARP_SIZE));
    idx_smem_ += warp_id * WARP_SIZE;
  }

  [[intel::device_indirectly_callable]] void
  add(T const* in,
      idxT start,
      idxT end,
      sycl::sub_group const& sg,
      const int local_id) {
    idxT const end_for_fullwarp =
        round_up_to_multiple_of<WARP_SIZE>(end - start) + start;
    for (idxT i = start + lane_; i < end_for_fullwarp; i += WARP_SIZE) {
      T val = (i < end) ? in[i] : dummy_;
      add(val, i, sg, local_id);
    }
  }

  [[intel::device_indirectly_callable]] void
  add(T val, idxT idx, sycl::sub_group const& sg, const int local_id) {
    bool do_add;
    if constexpr (is_stable) {
      do_add = is_better_than<greater>(val, k_th_, idx, k_th_idx_);
    } else {
      do_add = is_better_than<greater>(val, k_th_);
    }

    auto mask = sycl::ext::oneapi::group_ballot(sg, do_add);
    if (mask == 0) {
      return;
    }

    int pos = smem_buf_len_ + (mask & ((0x1u << lane_) - 1)).count();
    if (do_add && pos < WARP_SIZE) {
      val_smem_[pos] = val;
      idx_smem_[pos] = idx;
      do_add = false;
    }
    smem_buf_len_ += mask.count();
    if (smem_buf_len_ >= WARP_SIZE) {
      merge_buf_(val_smem_[lane_], idx_smem_[lane_], sg, local_id);
      smem_buf_len_ -= WARP_SIZE;
    }
    if (do_add) {
      pos -= WARP_SIZE;
      val_smem_[pos] = val;
      idx_smem_[pos] = idx;
    }
  }

  [[intel::device_indirectly_callable]] void
  done(sycl::sub_group const& sg, const int local_id) {
    if (smem_buf_len_) {
      T val = (lane_ < smem_buf_len_) ? val_smem_[lane_] : dummy_;
      idxT idx = (lane_ < smem_buf_len_) ? idx_smem_[lane_] : 0;
      merge_buf_(val, idx, sg, local_id);
    }
  }

 private:
  [[intel::device_indirectly_callable]] void
  set_k_th_(sycl::sub_group const& sg) {
    k_th_ = sycl::select_from_group(sg, val_arr_[max_arr_len_ - 1], k_th_lane_);
    if constexpr (is_stable) {
      k_th_idx_ =
          sycl::select_from_group(sg, idx_arr_[max_arr_len_ - 1], k_th_lane_);
    }
  }

  [[intel::device_indirectly_callable]] void
  merge_buf_(T val, idxT idx, sycl::sub_group const& sg, const int local_id) {
    BitonicSort<WARP_SIZE, greater, T, idxT, is_stable>::sort(
        &val, &idx, sg, local_id);

    T& old = val_arr_[max_arr_len_ - 1];

    bool is_better;
    if constexpr (is_stable) {
      is_better =
          is_better_than<greater>(val, old, idx, idx_arr_[max_arr_len_ - 1]);
    } else {
      is_better = is_better_than<greater>(val, old);
    }

    if (is_better) {
      old = val;
      idx_arr_[max_arr_len_ - 1] = idx;
    }

    BitonicMerge<capacity, greater, !greater, T, idxT, is_stable>::merge(
        val_arr_, idx_arr_, sg, local_id);

    set_k_th_(sg);
  }

  using WarpSort<capacity, greater, T, idxT, is_stable>::max_arr_len_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::val_arr_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::idx_arr_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::lane_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::k_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::dummy_;

  T* val_smem_;
  idxT* idx_smem_;
  int smem_buf_len_ = 0;

  T k_th_;
  idxT k_th_idx_;
  int const k_th_lane_;
};  // end class WarpSelect
}  // namespace warp_topk

template <typename T_OUT, typename T_IN>
[[intel::device_indirectly_callable]] inline T_OUT sycl_cast(T_IN val) {
  return val;
}

template <>
[[intel::device_indirectly_callable]] inline float
sycl_cast<float, sycl::ext::oneapi::bfloat16>(sycl::ext::oneapi::bfloat16 val) {
  return static_cast<float>(val);
}

template <typename T>
[[intel::device_indirectly_callable]] inline void topk_with_k2(
    T* output,
    T const* input,
    sycl::sub_group const& sg,
    int32_t const lane_id,
    int const num_experts_per_group) {
  // Get the top2 per thread
  float largest = -INFINITY;
  float second_largest = -INFINITY;

  if (num_experts_per_group > WARP_SIZE) {
    for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE) {
      float value = input[i];
      if (value > largest) {
        second_largest = largest;
        largest = value;
      } else if (value > second_largest) {
        second_largest = value;
      }
    }
  } else {
    for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE) {
      largest = input[i];
    }
  }

  float max1 = sycl::reduce_over_group(sg, largest, sycl::maximum<float>());

  float max2 = max1;
  bool equal_to_max1 = (max1 == largest);

  int count_max1 =
      sycl::reduce_over_group(sg, equal_to_max1 ? 1 : 0, sycl::plus<>());

  if (count_max1 == 1) {
    largest = (largest == max1) ? second_largest : largest;
    max2 = sycl::reduce_over_group(sg, largest, sycl::maximum<float>());
  }

  if (lane_id == 0) {
    *output = max1 + max2;
  }
}

template <typename T>
class topk_with_k2_kernel {
 public:
  topk_with_k2_kernel(
      T* output,
      T* input,
      int64_t const num_tokens,
      int64_t const num_cases,
      int64_t const n_group,
      int64_t const num_experts_per_group)
      : output(output),
        input(input),
        num_tokens(num_tokens),
        num_cases(num_cases),
        n_group(n_group),
        num_experts_per_group(num_experts_per_group) {}

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    auto local_id = item.get_local_id(0);
    auto group_id = item.get_group(0);
    int32_t warp_id = local_id / WARP_SIZE;
    int32_t lane_id = local_id % WARP_SIZE;

    int32_t case_id = group_id * NUM_WARPS_PER_BLOCK + warp_id;
    if (case_id < num_cases) {
      auto current_input = input + case_id * num_experts_per_group;
      auto current_output = output + case_id;
      auto sg = item.get_sub_group();
      topk_with_k2(
          current_output, current_input, sg, lane_id, num_experts_per_group);
    }
  }

 private:
  T* output;
  T* input;
  int64_t const num_tokens;
  int64_t const num_cases;
  int64_t const n_group;
  int64_t const num_experts_per_group;
};

template <typename T, typename IdxT>
class group_idx_and_topk_idx_kernel {
 public:
  group_idx_and_topk_idx_kernel(
      sycl::local_accessor<char, 1>& slm,
      T* scores,
      T const* group_scores,
      T* topk_values,
      IdxT* topk_indices,
      T* scores_with_bias,
      int64_t const num_tokens,
      int64_t const n_group,
      int64_t const topk_group,
      int64_t const topk,
      int64_t const num_experts,
      int64_t const num_experts_per_group,
      bool renormalize,
      double routed_scaling_factor)
      : slm(slm),
        scores(scores),
        group_scores(group_scores),
        topk_values(topk_values),
        topk_indices(topk_indices),
        scores_with_bias(scores_with_bias),
        num_tokens(num_tokens),
        n_group(n_group),
        topk_group(topk_group),
        topk(topk),
        num_experts(num_experts),
        num_experts_per_group(num_experts_per_group),
        renormalize(renormalize),
        routed_scaling_factor(routed_scaling_factor) {}

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    auto local_id = item.get_local_id(0);
    auto group_id = item.get_group(0);
    int32_t warp_id = local_id / WARP_SIZE;
    int32_t lane_id = local_id % WARP_SIZE;
    int32_t case_id =
        group_id * NUM_WARPS_PER_BLOCK + warp_id;  // one per token
    auto current_scores_with_bias = scores_with_bias + case_id * num_experts;
    auto current_scores = scores + case_id * num_experts;
    auto current_group_scores = group_scores + case_id * n_group;
    auto current_topk_values = topk_values + case_id * topk;
    auto current_topk_indices = topk_indices + case_id * topk;

    int32_t align_num_experts_per_group =
        warp_topk::round_up_to_multiple_of<WARP_SIZE>(num_experts_per_group);

    auto sg = item.get_sub_group();

    char* smem_buf =
        slm.template get_multi_ptr<sycl::access::decorated::no>().get();
    int32_t* s_topk_idx = reinterpret_cast<int32_t*>(smem_buf);
    T* s_topk_value =
        reinterpret_cast<T*>(s_topk_idx + NUM_WARPS_PER_BLOCK * topk) +
        warp_id * topk;
    s_topk_idx += warp_id * topk;

    T value = kNegInfinity;
    T topk_group_value = kNegInfinity;
    int32_t num_equalto_topkth_group;

    if (case_id < num_tokens) {
      // calculate group_idx
      int32_t target_num_min = WARP_SIZE - n_group + topk_group;
      if (lane_id < n_group &&
          (std::isfinite(
              sycl_cast<float, T>(
                  current_group_scores[lane_id]))))  // The check is necessary
                                                     // to avoid abnormal input
      {
        value = current_group_scores[lane_id];
      }

      int count_equal_to_top_value = WARP_SIZE - n_group;
      int pre_count_equal_to_top_value = 0;
      // Use loop to find the largset top_group
      while (count_equal_to_top_value < target_num_min) {
        float value_f = sycl_cast<float, T>(value);
        float topk_group_value_f =
            sycl::reduce_over_group(sg, value_f, sycl::maximum<float>());
        topk_group_value = sycl_cast<T, float>(topk_group_value_f);
        if (value_f == topk_group_value_f) {
          value = kNegInfinity;
        }
        pre_count_equal_to_top_value = count_equal_to_top_value;
        count_equal_to_top_value = sycl::reduce_over_group(
            sg, (value == kNegInfinity) ? 1 : 0, sycl::plus<>());
      }
      num_equalto_topkth_group = target_num_min - pre_count_equal_to_top_value;
    }

    item.barrier(sycl::access::fence_space::local_space);

    warp_topk::WarpSelect</*capability*/ WARP_SIZE,
                          /*greater*/ true,
                          T,
                          int32_t,
                          /* is_stable */ true>
        queue(
            (int32_t)topk,
            -INFINITY,
            smem_buf,
            local_id,
            item.get_local_range(0));

    int count_equalto_topkth_group = 0;
    bool if_proceed_next_topk = (topk_group_value != kNegInfinity);
    if (case_id < num_tokens && if_proceed_next_topk) {
      for (int i_group = 0; i_group < n_group; i_group++) {
        if ((current_group_scores[i_group] > topk_group_value) ||
            ((current_group_scores[i_group] == topk_group_value) &&
             (count_equalto_topkth_group < num_equalto_topkth_group))) {
          int32_t offset = i_group * num_experts_per_group;
          for (int32_t i = lane_id; i < align_num_experts_per_group;
               i += WARP_SIZE) {
            T candidates =
                (i < num_experts_per_group) &&
                        std::isfinite(
                            sycl_cast<float, T>(
                                current_scores_with_bias[offset + i]))
                    ? current_scores_with_bias[offset + i]
                    : sycl_cast<T, float>(kNegInfinity);
            queue.add(candidates, offset + i, sg, local_id);
          }
          if (current_group_scores[i_group] == topk_group_value) {
            count_equalto_topkth_group++;
          }
        }
      }
      queue.done(sg, local_id);
    }
    // after done(), smem is used for merging results among warps
    item.barrier(sycl::access::fence_space::local_space);
    if (case_id < num_tokens && if_proceed_next_topk) {
      // Get the topk_idx
      queue.dumpIdx(s_topk_idx);
      ;
    }

    // Load the valid score value
    // Calculate the summation
    float topk_sum = 1e-20;
    if (case_id < num_tokens && if_proceed_next_topk) {
      for (int i = lane_id;
           i < warp_topk::round_up_to_multiple_of<WARP_SIZE>(topk);
           i += WARP_SIZE) {
        T value =
            i < topk
                ? current_scores[s_topk_idx[i]]
                : sycl_cast<T, float>(0.0f);  // Load the valid value of expert
        if (i < topk) {
          s_topk_value[i] = value;
        }
        topk_sum += sycl::reduce_over_group(
            sg, sycl_cast<float, T>(value), sycl::plus<>());
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    if (case_id < num_tokens) {
      if (if_proceed_next_topk) {
        for (int i = lane_id; i < topk; i += WARP_SIZE) {
          float value;
          if (renormalize) {
            value = sycl_cast<float, T>(s_topk_value[i]) / topk_sum *
                    routed_scaling_factor;
          } else {
            value =
                sycl_cast<float, T>(s_topk_value[i]) * routed_scaling_factor;
          }
          current_topk_indices[i] = s_topk_idx[i];
          current_topk_values[i] = sycl_cast<T, float>(value);
        }
      } else {
        for (int i = lane_id; i < topk; i += WARP_SIZE) {
          current_topk_indices[i] = i;
          current_topk_values[i] = sycl_cast<T, float>(1.0f / topk);
        }
      }
      // Note: when if_proceed_next_topk==false, choose the first 8 experts as
      // the default result.
    }
  }

 private:
  sycl::local_accessor<char, 1> slm;
  T* scores;
  T const* group_scores;
  T* topk_values;
  IdxT* topk_indices;
  T* scores_with_bias;
  int64_t const num_tokens;
  int64_t const n_group;
  int64_t const topk_group;
  int64_t const topk;
  int64_t const num_experts;
  int64_t const num_experts_per_group;
  bool renormalize;
  double routed_scaling_factor;
};

template <typename T, typename IdxT>
void invokeNoAuxTc(
    T* scores,
    T* group_scores,
    T* topk_values,
    IdxT* topk_indices,
    T* scores_with_bias,
    int64_t const num_tokens,
    int64_t const num_experts,
    int64_t const n_group,
    int64_t const topk_group,
    int64_t const topk,
    bool const renormalize,
    double const routed_scaling_factor,
    bool enable_pdl,
    sycl::queue& queue) {
  int64_t num_cases = num_tokens * n_group;
  int64_t topk_with_k2_num_blocks = (num_cases - 1) / NUM_WARPS_PER_BLOCK + 1;
  sycl::range<1> grid(topk_with_k2_num_blocks);
  sycl::range<1> block(BLOCK_SIZE);
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(grid * block, block),
        topk_with_k2_kernel<T>(
            group_scores,
            scores_with_bias,
            num_tokens,
            num_cases,
            n_group,
            num_experts / n_group));
  });

  int64_t topk_with_k_group_num_blocks =
      (num_tokens - 1) / NUM_WARPS_PER_BLOCK + 1;
  size_t dynamic_smem_in_bytes =
      warp_topk::calc_smem_size_for_block_wide<T, int32_t>(
          NUM_WARPS_PER_BLOCK, topk);
  sycl::range<1> grid2(topk_with_k_group_num_blocks);
  sycl::range<1> block2(BLOCK_SIZE);
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<char, 1> slm(
        sycl::range<1>(dynamic_smem_in_bytes), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(grid * block, block),
        group_idx_and_topk_idx_kernel<T, IdxT>(
            slm,
            scores,
            group_scores,
            topk_values,
            topk_indices,
            scores_with_bias,
            num_tokens,
            n_group,
            topk_group,
            topk,
            num_experts,
            num_experts / n_group,
            renormalize,
            routed_scaling_factor));
  });
}

#define INSTANTIATE_NOAUX_TC(T, IdxT)     \
  template void invokeNoAuxTc<T, IdxT>(   \
      T * scores,                         \
      T * group_scores,                   \
      T * topk_values,                    \
      IdxT * topk_indices,                \
      T * scores_with_bias,               \
      int64_t const num_tokens,           \
      int64_t const num_experts,          \
      int64_t const n_group,              \
      int64_t const topk_group,           \
      int64_t const topk,                 \
      bool const renormalize,             \
      double const routed_scaling_factor, \
      bool enable_pdl,                    \
      sycl::queue& queue);

INSTANTIATE_NOAUX_TC(float, int32_t);
INSTANTIATE_NOAUX_TC(sycl::half, int32_t);
INSTANTIATE_NOAUX_TC(sycl::ext::oneapi::bfloat16, int32_t);
}  // end namespace moe
}  // namespace vllm

std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    torch::Tensor const& scores,
    torch::Tensor const& scores_with_bias,
    int64_t n_group,
    int64_t topk_group,
    int64_t topk,
    bool renormalize,
    double routed_scaling_factor) {
  auto data_type = scores_with_bias.scalar_type();
  auto input_size = scores_with_bias.sizes();
  int64_t num_tokens = input_size[0];
  int64_t num_experts = input_size[1];
  TORCH_CHECK(input_size.size() == 2, "scores_with_bias must be a 2D Tensor");
  TORCH_CHECK(
      num_experts % n_group == 0, "num_experts should be divisible by n_group");
  TORCH_CHECK(
      n_group <= 32, "n_group should be smaller than or equal to 32 for now");
  TORCH_CHECK(topk <= 32, "topk should be smaller than or equal to 32 for now");

  torch::Tensor group_scores = torch::empty(
      {num_tokens, n_group}, torch::dtype(data_type).device(torch::kXPU));
  torch::Tensor topk_values = torch::empty(
      {num_tokens, topk}, torch::dtype(data_type).device(torch::kXPU));
  torch::Tensor topk_indices = torch::empty(
      {num_tokens, topk}, torch::dtype(torch::kInt32).device(torch::kXPU));

  auto& queue = vllm::xpu::vllmGetQueue();

  switch (data_type) {
    case torch::kFloat16:
      // Handle Float16
      vllm::moe::invokeNoAuxTc<sycl::half, int32_t>(
          reinterpret_cast<sycl::half*>(scores.mutable_data_ptr()),
          reinterpret_cast<sycl::half*>(group_scores.mutable_data_ptr()),
          reinterpret_cast<sycl::half*>(topk_values.mutable_data_ptr()),
          reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()),
          reinterpret_cast<sycl::half*>(scores_with_bias.data_ptr()),
          num_tokens,
          num_experts,
          n_group,
          topk_group,
          topk,
          renormalize,
          routed_scaling_factor,
          false,
          queue);
      break;
    case torch::kFloat32:
      // Handle Float32
      vllm::moe::invokeNoAuxTc<float, int32_t>(
          reinterpret_cast<float*>(scores.mutable_data_ptr()),
          reinterpret_cast<float*>(group_scores.mutable_data_ptr()),
          reinterpret_cast<float*>(topk_values.mutable_data_ptr()),
          reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()),
          reinterpret_cast<float*>(scores_with_bias.data_ptr()),
          num_tokens,
          num_experts,
          n_group,
          topk_group,
          topk,
          renormalize,
          routed_scaling_factor,
          false,
          queue);
      break;
    case torch::kBFloat16:
      // Handle BFloat16
      vllm::moe::invokeNoAuxTc<sycl::ext::oneapi::bfloat16, int32_t>(
          reinterpret_cast<sycl::ext::oneapi::bfloat16*>(
              scores.mutable_data_ptr()),
          reinterpret_cast<sycl::ext::oneapi::bfloat16*>(
              group_scores.mutable_data_ptr()),
          reinterpret_cast<sycl::ext::oneapi::bfloat16*>(
              topk_values.mutable_data_ptr()),
          reinterpret_cast<int32_t*>(topk_indices.mutable_data_ptr()),
          reinterpret_cast<sycl::ext::oneapi::bfloat16*>(
              scores_with_bias.data_ptr()),
          num_tokens,
          num_experts,
          n_group,
          topk_group,
          topk,
          renormalize,
          routed_scaling_factor,
          false,
          queue);
      break;
    default:
      // Handle other data types
      throw std::invalid_argument(
          "Invalid dtype, only supports float16, float32, and bfloat16");
      break;
  }
  return {topk_values, topk_indices};
}