#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace vllm {
namespace moe {
// ceil div helper
inline int32_t div_ceil(int32_t a, int32_t b) { return (a + b - 1) / b; }

template <typename scalar_t, typename token_cnts_t>
struct moe_lora_align_sum_kernel {
  scalar_t* topk_ids;
  int32_t* token_lora_mapping;
  int64_t block_size;
  int32_t num_experts;
  int32_t max_loras;
  size_t numel;
  int32_t max_num_tokens_padded;
  int32_t max_num_m_blocks;
  int32_t* sorted_token_ids;
  int32_t* expert_ids;
  int32_t topk_num;
  int32_t* total_tokens_post_pad;
  int32_t* adapter_enabled;
  int32_t* lora_ids;

  // local accessors (shared mem)
  // size = num_experts + 1
  sycl::local_accessor<int32_t, 1> cumsum_local;
  // size = (num_thread + 1) * num_experts
  sycl::local_accessor<token_cnts_t, 1> tokens_cnts_local;

  moe_lora_align_sum_kernel(
      scalar_t* __restrict__ topk_ids_,
      int32_t* token_lora_mapping_,
      int64_t block_size_,
      int num_experts_,
      int max_loras_,
      size_t numel_,
      int max_num_tokens_padded_,
      int max_num_m_blocks_,
      int32_t* __restrict__ sorted_token_ids_,
      int32_t* __restrict__ expert_ids_,
      int topk_num_,
      int32_t* total_tokens_post_pad_,
      int32_t* adapter_enabled_,
      int32_t* lora_ids_,
      sycl::local_accessor<int32_t, 1> cumsum_local_,
      sycl::local_accessor<token_cnts_t, 1> tokens_cnts_local_)
      : topk_ids(topk_ids_),
        token_lora_mapping(token_lora_mapping_),
        block_size(block_size_),
        num_experts(num_experts_),
        max_loras(max_loras_),
        numel(numel_),
        max_num_tokens_padded(max_num_tokens_padded_),
        max_num_m_blocks(max_num_m_blocks_),
        sorted_token_ids(sorted_token_ids_),
        expert_ids(expert_ids_),
        topk_num(topk_num_),
        total_tokens_post_pad(total_tokens_post_pad_),
        adapter_enabled(adapter_enabled_),
        lora_ids(lora_ids_),
        cumsum_local(cumsum_local_),
        tokens_cnts_local(tokens_cnts_local_) {}

  void operator()(sycl::nd_item<1> it) const {
    // map indices
    const int32_t threadIdx_x = (int32_t)it.get_local_id(0);
    const int32_t blockDim_x = (int32_t)it.get_local_range(0);
    const int32_t blockIdx_x = (int32_t)it.get_group(0);

    const size_t local_idx = threadIdx_x;  // equiv threadIdx.x

    int lora_id = lora_ids[blockIdx_x];
    if (lora_id == -1 || adapter_enabled[lora_id] == 0) {
      return;
    }
    const size_t tokens_per_thread = div_ceil(numel, blockDim_x);
    const size_t start_idx = local_idx * tokens_per_thread;

    // Helpers to compute offsets for shared/local arrays
    auto tokens_cnts_index = [&](int32_t row, int32_t col) -> size_t {
      // rows = num_thread + 1; cols = num_experts
      return (size_t)row * (size_t)num_experts + (size_t)col;
    };

    // Initialize sorted_token_ids with numel
    for (size_t idx = local_idx; idx < (size_t)max_num_tokens_padded;
         idx += it.get_local_range(0)) {
      size_t pos = (size_t)lora_id * (size_t)max_num_tokens_padded + idx;
      sorted_token_ids[pos] = static_cast<int32_t>(numel);
    }

    // Initialize expert_ids with -1
    for (size_t idx = local_idx; idx < (size_t)max_num_m_blocks;
         idx += it.get_local_range(0)) {
      size_t pos = (size_t)lora_id * (size_t)max_num_m_blocks + idx;
      expert_ids[pos] = -1;
    }

    // Initialize total_tokens_post_pad with 0 (one work-item per group)
    if (local_idx == 0) {
      total_tokens_post_pad[lora_id] = 0;
    }

    // Initialize tokens_cnts for this thread row (note: we index row =
    // local_idx + 1)
    for (int i = 0; i < num_experts; ++i) {
      size_t pos = tokens_cnts_index((int32_t)local_idx + 1, i);
      tokens_cnts_local[pos] = (token_cnts_t)0;
    }

    // Count tokens assigned to this lora_id in this thread's shard
    for (size_t i = start_idx; i < numel && i < start_idx + tokens_per_thread;
         ++i) {
      int mask = (token_lora_mapping[i / topk_num] == lora_id) ? 1 : 0;
      size_t idx =
          tokens_cnts_index(local_idx + 1, static_cast<int32_t>(topk_ids[i]));
      tokens_cnts_local[idx] = tokens_cnts_local[idx] + (token_cnts_t)mask;
    }

    it.barrier(sycl::access::fence_space::local_space);

    // For each expert we accumulate the token counts from the different
    // threads.
    if (local_idx < (size_t)num_experts) {
      // prefix-scan over rows for this expert
      // set tokens_cnts[row=0][expert] = 0
      tokens_cnts_local[tokens_cnts_index(0, (int32_t)local_idx)] =
          (token_cnts_t)0;
      // accumulate down the rows
      for (size_t row = 1; row <= it.get_local_range(0); ++row) {
        size_t cur = tokens_cnts_index(row, local_idx);
        size_t prev = tokens_cnts_index(row - 1, local_idx);
        tokens_cnts_local[cur] =
            tokens_cnts_local[cur] + tokens_cnts_local[prev];
      }
    }

    it.barrier(sycl::access::fence_space::local_space);

    // thread 0 accumulates cumsum across experts
    if (local_idx == 0) {
      cumsum_local[0] = 0;
      for (int e = 1; e <= num_experts; ++e) {
        // tokens_cnts[index(num_experts, blockDim.x, e - 1)]
        size_t idx_last_row =
            tokens_cnts_index((int32_t)it.get_local_range(0), (int32_t)(e - 1));
        // div_ceil with block_size
        int32_t raw = static_cast<int32_t>(tokens_cnts_local[idx_last_row]);
        int32_t divceil = div_ceil(raw, (int32_t)block_size);
        int32_t padded = divceil * static_cast<int32_t>(block_size);
        cumsum_local[e] = cumsum_local[e - 1] + padded;
      }
      total_tokens_post_pad[lora_id] = cumsum_local[num_experts];
    }

    it.barrier(sycl::access::fence_space::local_space);

    // For each expert, each thread marks expert_ids for each block it owns.
    if (local_idx < (size_t)num_experts) {
      int32_t expert = (int32_t)local_idx;
      int32_t start = cumsum_local[expert];
      int32_t end = cumsum_local[expert + 1];
      for (int32_t pos = start; pos < end; pos += (int32_t)block_size) {
        int32_t mblock_idx = pos / static_cast<int32_t>(block_size);
        size_t out_pos =
            (size_t)lora_id * (size_t)max_num_m_blocks + (size_t)mblock_idx;
        expert_ids[out_pos] = expert;
      }
    }

    it.barrier(sycl::access::fence_space::local_space);

    for (size_t i = start_idx; i < numel && i < start_idx + tokens_per_thread;
         ++i) {
      int32_t expert_id = static_cast<int32_t>(topk_ids[i]);
      size_t idx_thread_row = tokens_cnts_index((int32_t)local_idx, expert_id);
      token_cnts_t prev_cnt = tokens_cnts_local[idx_thread_row];
      int32_t rank_post_pad =
          static_cast<int32_t>(prev_cnt) + cumsum_local[expert_id];
      int mask = (token_lora_mapping[i / topk_num] == lora_id) ? 1 : 0;

      // atomicAdd on sorted_token_ids[index(max_num_tokens_padded, lora_id,
      // rank_post_pad)], (i - numel) * mask
      // use sycl::atomic_ref for int32
      if (mask) {
        size_t out_idx = (size_t)lora_id * (size_t)max_num_tokens_padded +
                         (size_t)rank_post_pad;
        sycl::atomic_ref<
            int32_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>
            ar(sorted_token_ids[out_idx]);
        ar.fetch_add(static_cast<int32_t>((int64_t)i - (int64_t)numel));

        tokens_cnts_local[idx_thread_row] =
            tokens_cnts_local[idx_thread_row] + (token_cnts_t)1;
      }
    }
  }
};

}  // namespace moe
}  // namespace vllm

template <typename scalar_t, typename token_cnts_t>
void launch_moe_lora_align_sum_kernel(
    sycl::queue& queue,
    int32_t num_thread,
    scalar_t* topk_ids,
    int32_t* token_lora_mapping,
    int64_t block_size,
    int32_t num_experts,
    int32_t max_loras,
    size_t numel,
    int32_t max_num_tokens_padded,
    int32_t max_num_m_blocks,
    int32_t* sorted_token_ids,
    int32_t* expert_ids,
    int32_t topk_num,
    int32_t* total_tokens_post_pad,
    int32_t* adapter_enabled,
    int32_t* lora_ids) {
  sycl::range<1> local_range((size_t)num_thread);
  sycl::range<1> global_range((size_t)max_loras * (size_t)num_thread);
  // Local shared sizes:
  // - cumsum: (num_experts + 1) int32
  // - tokens_cnts: (num_thread + 1) * num_experts of token_cnts_t
  size_t cumsum_elems = (size_t)num_experts + 1;
  size_t tokens_cnts_elems = (size_t)(num_thread + 1) * (size_t)num_experts;

  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);

  // Launch nd_range kernel
  queue.submit([&](sycl::handler& h) {
    // local accessors = shared memory
    sycl::local_accessor<int32_t, 1> cumsum_local(
        sycl::range<1>(cumsum_elems), h);
    sycl::local_accessor<token_cnts_t, 1> tokens_cnts_local(
        sycl::range<1>(tokens_cnts_elems), h);

    vllm::moe::moe_lora_align_sum_kernel<scalar_t, token_cnts_t> kfn(
        topk_ids,
        token_lora_mapping,
        block_size,
        num_experts,
        max_loras,
        numel,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_token_ids,
        expert_ids,
        topk_num,
        total_tokens_post_pad,
        adapter_enabled,
        lora_ids,
        cumsum_local,
        tokens_cnts_local);
    h.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
  });
};

void moe_lora_align_block_size(
    torch::Tensor topk_ids,
    torch::Tensor token_lora_mapping,
    int64_t num_experts,
    int64_t block_size,
    int64_t max_loras,
    int64_t max_num_tokens_padded,
    int64_t max_num_m_blocks,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    torch::Tensor adapter_enabled,
    torch::Tensor lora_ids) {
  const int topk_num = topk_ids.size(1);
  TORCH_CHECK(block_size > 0, "block_size should be greater than 0.");

  const int32_t num_thread = std::max((int32_t)num_experts, 128);
  TORCH_CHECK(
      num_thread <= 1024,
      "num_experts must be less than 1024, "
      "and fallback is not implemented yet.");

  auto& queue = vllm::xpu::vllmGetQueue();
  size_t device_max_shared_mem =
      queue.get_device().get_info<sycl::info::device::local_mem_size>();

  const int32_t shared_mem = (num_thread + 1) * num_experts * sizeof(int32_t) +
                             (num_experts + 1) * sizeof(int32_t);

  if (shared_mem > device_max_shared_mem) {
    TORCH_CHECK(
        false,
        "Shared memory usage exceeds device limit, and global memory "
        "fallback is not implemented yet.");
  }

  VLLM_DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(), "moe_lora_align_sum_kernel", [&] {
        launch_moe_lora_align_sum_kernel<scalar_t, int32_t>(
            queue,
            num_thread,
            topk_ids.data_ptr<scalar_t>(),
            token_lora_mapping.data_ptr<int32_t>(),
            block_size,
            num_experts,
            max_loras,
            topk_ids.numel(),
            max_num_tokens_padded,
            max_num_m_blocks,
            sorted_token_ids.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(),
            topk_num,
            num_tokens_post_pad.data_ptr<int32_t>(),
            adapter_enabled.data_ptr<int32_t>(),
            lora_ids.data_ptr<int32_t>());
      });
}