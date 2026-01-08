#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace vllm {
namespace moe {

template <typename T, int TOPK, int ElemsPerItem>
class MoeGather {
 public:
  MoeGather(
      T* output,
      const T* moe_output,
      const float* topk_weights,
      const int* unpermuted_row_to_permuted_row,
      const int num_tokens,
      const int hidden_size)
      : output(output),
        moe_output(moe_output),
        topk_weights(topk_weights),
        unpermuted_row_to_permuted_row(unpermuted_row_to_permuted_row),
        num_tokens(num_tokens),
        hidden_size(hidden_size) {}

  static constexpr int GroupWorkItem = 256;
  static constexpr int WARP_SIZE = 32;
  static constexpr int Stride = GroupWorkItem * ElemsPerItem;

  static inline sycl::nd_range<1> get_nd_range(const int num_tokens) {
    sycl::range<1> local(GroupWorkItem);
    sycl::range<1> group(num_tokens);
    return sycl::nd_range<1>(local * group, local);
  }

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    auto group = item.get_group();
    auto local_id_x = item.get_local_id(0);
    auto group_id_x = item.get_group(0);

    const int token_idx = group_id_x;

    int moe_ids[TOPK];
#pragma unroll
    for (int i = 0; i < TOPK; ++i) {
      moe_ids[i] = unpermuted_row_to_permuted_row[token_idx + i * num_tokens];
    }

    float scores[TOPK];
#pragma unroll
    for (int i = 0; i < TOPK; ++i) {
      scores[i] = topk_weights[token_idx * TOPK + i];
    }

    const int loop_count = (hidden_size + Stride - 1) / Stride;

    for (int i = 0; i < loop_count; ++i) {
      const int hidden_idx = i * Stride + local_id_x * ElemsPerItem;
      if (hidden_idx < hidden_size) {
        float accum[ElemsPerItem];
#pragma unroll
        for (int e = 0; e < ElemsPerItem; ++e) {
          accum[e] = 0.0f;
        }
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
#pragma unroll
          for (int e = 0; e < ElemsPerItem; ++e) {
            accum[e] +=
                static_cast<float>(
                    moe_output[moe_ids[k] * hidden_size + hidden_idx + e]) *
                scores[k];
          }
        }
#pragma unroll
        for (int e = 0; e < ElemsPerItem; ++e) {
          output[token_idx * hidden_size + hidden_idx + e] = accum[e];
        }
      }
    }
  }

 private:
  T* output;
  const T* moe_output;
  const float* topk_weights;
  const int* unpermuted_row_to_permuted_row;
  const int num_tokens;
  const int hidden_size;
};

template <typename T>
void MoeGatherLauncher(
    T* output,
    const T* moe_output,
    const float* topk_weights,
    const int* unpermuted_row_to_permuted_row,
    const int num_tokens,
    const int topk,
    const int hidden_size,
    sycl::queue& queue) {
  int elems_per_item = sizeof(float) * 4 / sizeof(T);
  while (hidden_size % elems_per_item != 0) {
    elems_per_item /= 2;
  }
#define CASE_TOPK(TOPK, ElemsPerItem)                                 \
  case TOPK:                                                          \
    queue.submit([&](sycl::handler& cgh) {                            \
      cgh.parallel_for(                                               \
          MoeGather<T, TOPK, ElemsPerItem>::get_nd_range(num_tokens), \
          MoeGather<T, TOPK, ElemsPerItem>{                           \
              output,                                                 \
              moe_output,                                             \
              topk_weights,                                           \
              unpermuted_row_to_permuted_row,                         \
              num_tokens,                                             \
              hidden_size});                                          \
    });                                                               \
    break;

#define CASE_ElemsPerItem(TOPK, ElemsPerItem)                                  \
  case ElemsPerItem:                                                           \
    switch (TOPK) {                                                            \
      CASE_TOPK(1, ElemsPerItem)                                               \
      CASE_TOPK(2, ElemsPerItem)                                               \
      CASE_TOPK(4, ElemsPerItem)                                               \
      CASE_TOPK(6, ElemsPerItem)                                               \
      CASE_TOPK(8, ElemsPerItem)                                               \
      CASE_TOPK(10, ElemsPerItem)                                              \
      default:                                                                 \
        TORCH_CHECK(false, "error: not support TOPK=" + std::to_string(TOPK)); \
    }                                                                          \
    break;

  switch (elems_per_item) {
    CASE_ElemsPerItem(topk, 1) CASE_ElemsPerItem(topk, 2)
        CASE_ElemsPerItem(topk, 4) CASE_ElemsPerItem(topk, 8) default
        : TORCH_CHECK(
              false,
              "error: not support elems_per_item=" +
                  std::to_string(elems_per_item));
  }
#undef CASE_ElemsPerItem
#undef CASE_TOPK
}

}  // namespace moe
}  // namespace vllm

void moe_gather(
    torch::Tensor& output,              // [num_tokens, hidden_size]
    const torch::Tensor& moe_output,    // [num_tokens * topk, hidden_size]
    const torch::Tensor& topk_weights,  // [num_tokens, topk]
    const torch::Tensor& unpermuted_row_to_permuted_row,  // [num_tokens * topk]
    const int64_t num_experts) {
  // Implementation of the gather operation
  const int num_tokens = topk_weights.size(0);
  const int topk = topk_weights.size(1);
  const int hidden_size = output.size(1);

  TORCH_CHECK(
      topk_weights.scalar_type() == torch::kFloat32,
      "topk_weights must be float32");

  const at::DeviceGuard device_guard(output.device());
  auto& queue = vllm::xpu::vllmGetQueue();

#define LAUNCH_MOE_GATHER(T)                                             \
  vllm::moe::MoeGatherLauncher<T>(                                       \
      reinterpret_cast<T*>(output.data_ptr()),                           \
      reinterpret_cast<T*>(moe_output.data_ptr()),                       \
      reinterpret_cast<float*>(topk_weights.data_ptr()),                 \
      reinterpret_cast<int*>(unpermuted_row_to_permuted_row.data_ptr()), \
      num_tokens,                                                        \
      topk,                                                              \
      hidden_size,                                                       \
      queue);

  if (output.scalar_type() == torch::kFloat16) {
    using scalar_t = sycl::half;
    LAUNCH_MOE_GATHER(scalar_t);
  } else if (output.scalar_type() == torch::kBFloat16) {
    using scalar_t = sycl::ext::oneapi::bfloat16;
    LAUNCH_MOE_GATHER(scalar_t);
  } else if (output.scalar_type() == torch::kFloat32) {
    using scalar_t = float;
    LAUNCH_MOE_GATHER(scalar_t);
  } else {
    throw std::runtime_error("Unsupported data type in moe_gather");
  }
#undef LAUNCH_MOE_GATHER
}