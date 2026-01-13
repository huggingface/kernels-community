#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace vllm {
namespace GroupedTopKImpl {

enum class ScoringFunc {
  DEFAULT = 0,
  SOFTMAX = 1,
  SIGMOID = 2,
};

template <typename T, int MAX_EXPERT_GROUPS>
struct Fused_Grouped_Topk {
  static constexpr int sub_group_size = 32;
  static constexpr int max_group_size = 1024;
  static constexpr int malloc_per_item = MAX_EXPERT_GROUPS;
  static constexpr float kNegInfinity = INFINITY * -1;

  Fused_Grouped_Topk(
      float* topk_weights,
      int* topk_ids,
      const T* gating_output,
      const T* e_score_correction_bias,
      const double routed_scaling_factor,
      const ScoringFunc scoring_mode,
      const bool renormalize,
      const int tokens,
      const int experts,
      const int top_k,
      const int num_expert_group,
      const int topk_group)
      : topk_weights(topk_weights),
        topk_ids(topk_ids),
        gating_output(gating_output),
        e_score_correction_bias(e_score_correction_bias),
        routed_scaling_factor(routed_scaling_factor),
        scoring_mode(scoring_mode),
        renormalize(renormalize),
        tokens(tokens),
        experts(experts),
        top_k(top_k),
        num_expert_group(num_expert_group),
        topk_group(topk_group) {}

  static inline sycl::nd_range<3>
  get_nd_range(const int tokens, const int experts) {
    int calc_per_item = (experts + sub_group_size - 1) / sub_group_size;
    int group_size = (experts + calc_per_item - 1) / calc_per_item;
    group_size = group_size < sub_group_size ? sub_group_size : group_size;
    group_size = group_size < max_group_size ? group_size : max_group_size;
    int sub_groups_per_group =
        (group_size + sub_group_size - 1) / sub_group_size;
    group_size = sub_groups_per_group * sub_group_size;
    int global_size =
        (tokens + sub_groups_per_group - 1) / sub_groups_per_group;

    sycl::range<3> local(1, 1, group_size);
    sycl::range<3> global(1, 1, global_size);
    return sycl::nd_range<3>(global * local, local);
  }

  static inline float Sigmoid(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<3> item) const {
    int group_id = item.get_group_linear_id();
    int local_range = item.get_local_range(2);
    int sub_groups_per_group = local_range / sub_group_size;
    int calc_per_item = (experts + sub_group_size - 1) / sub_group_size;

    int experts_per_group = experts / num_expert_group;

    sycl::sub_group sg = item.get_sub_group();
    int sg_id = sg.get_group_id();
    int sg_local_id = sg.get_local_id();

    int tid = group_id * sub_groups_per_group + sg_id;

    if (tid >= tokens) {
      return;  // Out of bounds
    }

    T load_elems[malloc_per_item];
    int local_idx[malloc_per_item];
    T bias[malloc_per_item];

    int start_offset = sg_local_id * calc_per_item;
    int local_num = calc_per_item;

    if (start_offset + local_num >= experts) {
      local_num = experts - start_offset;
      if (local_num < 0) {
        local_num = 0;  // No elements to process
      }
    }

    for (int e = 0; e < calc_per_item; ++e) {
      load_elems[e] = kNegInfinity;
      local_idx[e] = -1;
      bias[e] = 0.0f;  // Initialize bias to zero
    }

    for (int e = 0; e < local_num; ++e) {
      load_elems[e] = gating_output[tid * experts + start_offset + e];
    }

    T local_elems[malloc_per_item];

    for (int e = 0; e < local_num; ++e) {
      local_elems[e] = load_elems[e];
      local_idx[e] = start_offset + e;
    }

    if (scoring_mode == ScoringFunc::SOFTMAX) {
      float softmax_max = kNegInfinity;
      for (int e = 0; e < local_num; ++e) {
        float s = load_elems[e];
        softmax_max = (softmax_max > s) ? softmax_max : s;
      }
      for (int offset = sub_group_size / 2; offset > 0; offset /= 2) {
        float other_val = sycl::permute_group_by_xor(sg, softmax_max, offset);
        softmax_max = (softmax_max > other_val) ? softmax_max : other_val;
      }
      float softmax_sum = 0.0f;
      for (int e = 0; e < local_num; ++e) {
        float s = local_elems[e];
        softmax_sum += sycl::native::exp(s - softmax_max);
      }
      for (int offset = sub_group_size / 2; offset > 0; offset /= 2) {
        float other_val = sycl::permute_group_by_xor(sg, softmax_sum, offset);
        softmax_sum += other_val;
      }
      for (int e = 0; e < local_num; ++e) {
        float s = local_elems[e];
        local_elems[e] = sycl::native::exp(s - softmax_max) / softmax_sum;
      }
    } else if (scoring_mode == ScoringFunc::SIGMOID) {
      for (int e = 0; e < local_num; ++e) {
        float s = load_elems[e];
        load_elems[e] = Sigmoid(s);
      }
      for (int e = 0; e < local_num; ++e) {
        local_elems[e] = load_elems[e];
      }
    }

    bool has_bias = e_score_correction_bias != nullptr;
    if (has_bias) {
      for (int e = 0; e < local_num; ++e) {
        bias[e] = e_score_correction_bias[start_offset + e];
      }
    }

    // perform topk_group groups
    // 1 calculate each group scores
    float group_scores[malloc_per_item * 2];
    for (int i = 0; i < num_expert_group * 2; ++i) {
      group_scores[i] = kNegInfinity;
    }
    for (int i = 0; i < local_num; ++i) {
      float b = bias[i];
      float score = local_elems[i] + b;
      int i_group = (calc_per_item * sg_local_id + i) / experts_per_group;
      float group_max = group_scores[i_group];
      float group_next_max = group_scores[num_expert_group + i_group];
      if (score > group_max) {
        group_next_max = group_max;
        group_max = score;
      } else if (score > group_next_max) {
        group_next_max = score;
      }
      group_scores[i_group] = group_max;
      group_scores[num_expert_group + i_group] = group_next_max;
    }
    for (int i = 0; i < num_expert_group; ++i) {
      float group_max = group_scores[i];
      float group_next_max = group_scores[num_expert_group + i];

      float max1 = sycl::reduce_over_group(
          sg, sycl::max(group_max, group_next_max), sycl::maximum<>());
      float local_second =
          (group_max < max1 && group_max > -INFINITY) ? group_max : -INFINITY;
      local_second = (group_next_max < max1 && group_next_max > local_second)
                         ? group_next_max
                         : local_second;
      float max2 = sycl::reduce_over_group(sg, local_second, sycl::maximum<>());
      group_scores[i] = max1 + (has_bias ? max2 : 0.0f);
    }

    // 2 find topk_group groups as kNegInfinity
    int group_topk_idx[malloc_per_item];
    for (int k = 0; k < topk_group; ++k) {
      float k_max = group_scores[0];
      int k_max_idx = 0;
      for (int e = 1; e < num_expert_group; ++e) {
        float score = group_scores[e];

        if (score > k_max) {
          k_max = score;
          k_max_idx = e;
        }
      }
      group_scores[k_max_idx] = kNegInfinity;
      group_topk_idx[k] = k_max_idx;
    }

    // 3 mask no-topk_group groups
    for (int i = 0; i < calc_per_item; ++i) {
      bool is_masked = true;
      for (int k = 0; k < topk_group; ++k) {
        if ((local_idx[i] / experts_per_group) == group_topk_idx[k]) {
          is_masked = false;
          break;
        }
      }
      if (is_masked) {
        local_elems[i] = kNegInfinity;
      }
    }

    // Perform top-k selection
    T topk_weights_local[malloc_per_item];
    int topk_ids_local[malloc_per_item];

    for (int k = 0; k < top_k; ++k) {
      float k_max = kNegInfinity;
      int k_max_idx = -1;
      int remove_ix = -1;
      for (int e = 0; e < calc_per_item; ++e) {
        float le = local_elems[e];
        float b = bias[e];
        float my_val = le + b;
        int my_idx = local_idx[e];
        for (int offset = sub_group_size / 2; offset > 0; offset /= 2) {
          float other_val = sycl::permute_group_by_xor(sg, my_val, offset);
          int other_idx = sycl::permute_group_by_xor(sg, my_idx, offset);
          if (other_val > my_val ||
              (other_val == my_val && other_idx < my_idx)) {
            my_val = other_val;
            my_idx = other_idx;
          }
        }
        if (my_val > k_max || (my_val == k_max && my_idx < k_max_idx)) {
          k_max = my_val;
          k_max_idx = my_idx;

          if (k_max_idx == local_idx[e]) {
            remove_ix = e;  // Mark this index for removal
          } else
            remove_ix = -1;
        }
      }

      int select_item = k_max_idx / calc_per_item;
      int select_elem = k_max_idx % calc_per_item;
      k_max = local_elems[select_elem];
      k_max = sycl::group_broadcast(sg, k_max, select_item);
      if (remove_ix != -1) {
        local_elems[remove_ix] =
            kNegInfinity;  // Reset the score to avoid re-selection
        local_idx[remove_ix] = -1;
        remove_ix = -1;
      }

      topk_weights_local[k] = k_max;
      topk_ids_local[k] = k_max_idx < 0 ? k : k_max_idx;
    }

    if (renormalize) {
      // Renormalize the top-k weights
      float sum = 0;
      for (int i = 0; i < top_k; ++i) {
        sum += topk_weights_local[i];
      }
      if (sum > 0) {
        for (int i = 0; i < top_k; ++i) {
          topk_weights_local[i] /= sum;
        }
      }
    }

    if (sg_local_id == 0) {
      int offset = tid * top_k;
      for (int i = 0; i < top_k; ++i) {
        topk_weights[offset + i] =
            topk_weights_local[i] * routed_scaling_factor;
        if (!(topk_ids_local[i] >= 0 && topk_ids_local[i] < experts)) {
          // Ensure valid index
          topk_ids[offset + i] = 0;
          continue;
        }
        topk_ids[offset + i] = topk_ids_local[i];
      }
    }
  }
  float* topk_weights;
  int* topk_ids;
  const T* gating_output;
  const T* e_score_correction_bias;
  const double routed_scaling_factor;
  const ScoringFunc scoring_mode;
  const bool renormalize;
  const int tokens;
  const int experts;
  const int top_k;
  const int num_expert_group;
  const int topk_group;
};

template <typename T, int MAX_EXPERT_GROUPS>
void launch_fused_grouped_topk(
    sycl::queue& queue,
    float* topk_weights,
    int* topk_ids,
    const T* gating_output,
    const T* e_score_correction_bias,
    const double routed_scaling_factor,
    const ScoringFunc scoring_mode,
    const bool renormalize,
    const int tokens,
    const int experts,
    const int top_k,
    const int num_expert_group,
    const int topk_group) {
  using Kernel = Fused_Grouped_Topk<T, MAX_EXPERT_GROUPS>;
  auto range = Kernel::get_nd_range(tokens, experts);

  queue.submit([&](sycl::handler& cgh) {
    Kernel task(
        topk_weights,
        topk_ids,
        gating_output,
        e_score_correction_bias,
        routed_scaling_factor,
        scoring_mode,
        renormalize,
        tokens,
        experts,
        top_k,
        num_expert_group,
        topk_group);
    cgh.parallel_for(range, task);
  });
}

template <typename T>
void fused_grouped_topk(
    float* topk_weights,
    int* topk_ids,
    const T* gating_output,
    const T* e_score_correction_bias,
    const double routed_scaling_factor,
    const ScoringFunc scoring_mode,
    const bool renormalize,
    const int tokens,
    const int experts,
    const int top_k,
    const int num_expert_group,
    const int topk_group) {
  auto& queue = vllm::xpu::vllmGetQueue();

  TORCH_CHECK(
      topk_group <= num_expert_group,
      "topk_group must be less than or equal to num_expert_group");
  TORCH_CHECK(
      experts % num_expert_group == 0,
      "The number of experts (experts=",
      experts,
      ") must be divisible by num_expert_group (",
      num_expert_group,
      ").");

  int max_expert_group = ((num_expert_group + 7) / 8) * 8;
#define CASE_TOPK(K)                 \
  case K:                            \
    launch_fused_grouped_topk<T, K>( \
        queue,                       \
        topk_weights,                \
        topk_ids,                    \
        gating_output,               \
        e_score_correction_bias,     \
        routed_scaling_factor,       \
        scoring_mode,                \
        renormalize,                 \
        tokens,                      \
        experts,                     \
        top_k,                       \
        num_expert_group,            \
        topk_group);                 \
    break;
  switch (max_expert_group) {
    CASE_TOPK(8)
    CASE_TOPK(16)
    default:
      TORCH_CHECK(
          false, "error: not support num_expert_group=%d,\n", num_expert_group);
  }
#undef CASE_TOPK
}

};  // namespace GroupedTopKImpl
}  // namespace vllm

/**
 * @brief Perform grouped topk after sigmoid/addbias on gating_output.
 * @param gating_output The gating output tensor of shape [n_tokens, n_experts].
 * @param n_topk The number of top experts to select.
 * @param n_topk_group The number of top experts to select in the group.
 * @return A tuple of tensors (topk_weights, topk_indices).
 */
std::tuple<torch::Tensor, torch::Tensor> fused_grouped_topk(
    const torch::Tensor& hidden_states,
    const torch::Tensor& gating_output,
    const int64_t n_topk,
    const bool renormalize,
    const int64_t n_expert_group,
    const int64_t n_topk_group,
    const c10::string_view scoring_func,
    const double routed_scaling_factor,
    const c10::optional<torch::Tensor>& bias) {
  auto shape = gating_output.sizes().vec();
  TORCH_CHECK(
      hidden_states.sizes()[0] == gating_output.sizes()[0],
      "Number of tokens mismatch")
  TORCH_CHECK(
      shape.size() == 2,
      "gating_output must be 2D tensor, but got ",
      shape.size(),
      "D");
  if (bias.has_value()) {
    auto shape_bias = bias->sizes().vec();
    TORCH_CHECK(
        shape_bias[0] == shape[1],
        "gating_output and bias must has same innermost dimension, but got ",
        shape,
        " and ",
        shape_bias);
  }
  int n_tokens = shape[0];
  int n_experts = shape[1];

  vllm::GroupedTopKImpl::ScoringFunc scoring_mode;
  if (scoring_func == "sigmoid") {
    scoring_mode = vllm::GroupedTopKImpl::ScoringFunc::SIGMOID;
  } else if (scoring_func == "softmax") {
    scoring_mode = vllm::GroupedTopKImpl::ScoringFunc::SOFTMAX;
  } else {
    scoring_mode = vllm::GroupedTopKImpl::ScoringFunc::DEFAULT;
  }

  auto topk_weights =
      torch::empty({n_tokens, n_topk}, at::dtype(at::kFloat).device(at::kXPU));
  auto topk_indices =
      torch::empty({n_tokens, n_topk}, at::dtype(at::kInt).device(at::kXPU));

  if (gating_output.scalar_type() == at::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    vllm::GroupedTopKImpl::fused_grouped_topk<scalar_t>(
        reinterpret_cast<float*>(topk_weights.data_ptr()),
        reinterpret_cast<int*>(topk_indices.data_ptr()),
        reinterpret_cast<scalar_t*>(gating_output.data_ptr()),
        bias.has_value() ? reinterpret_cast<scalar_t*>(bias->data_ptr())
                         : nullptr,
        routed_scaling_factor,
        scoring_mode,
        renormalize,
        n_tokens,
        n_experts,
        n_topk,
        n_expert_group,
        n_topk_group);
  } else if (gating_output.scalar_type() == at::kHalf) {
    using scalar_t = sycl::half;
    vllm::GroupedTopKImpl::fused_grouped_topk<scalar_t>(
        reinterpret_cast<float*>(topk_weights.data_ptr()),
        reinterpret_cast<int*>(topk_indices.data_ptr()),
        reinterpret_cast<scalar_t*>(gating_output.data_ptr()),
        bias.has_value() ? reinterpret_cast<scalar_t*>(bias->data_ptr())
                         : nullptr,
        routed_scaling_factor,
        scoring_mode,
        renormalize,
        n_tokens,
        n_experts,
        n_topk,
        n_expert_group,
        n_topk_group);
  } else {
    using scalar_t = float;
    vllm::GroupedTopKImpl::fused_grouped_topk<scalar_t>(
        reinterpret_cast<float*>(topk_weights.data_ptr()),
        reinterpret_cast<int*>(topk_indices.data_ptr()),
        reinterpret_cast<scalar_t*>(gating_output.data_ptr()),
        bias.has_value() ? reinterpret_cast<scalar_t*>(bias->data_ptr())
                         : nullptr,
        routed_scaling_factor,
        scoring_mode,
        renormalize,
        n_tokens,
        n_experts,
        n_topk,
        n_expert_group,
        n_topk_group);
  }
  return std::make_tuple(topk_weights, topk_indices);
}