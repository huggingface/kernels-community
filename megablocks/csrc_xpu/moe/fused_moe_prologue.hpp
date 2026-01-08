#pragma once
#include <torch/all.h>
#include "../utils.h"

#define MAX_SUBGROUP_SIZE 32
constexpr static int EXPAND_THREADS_PER_BLOCK = 256;
typedef at::BFloat16 bfloat16;

template <typename T>
inline T ceilDiv(T a, T b) {
  return (a + b - 1) / b;
}

int64_t computeNumTokensPerBlock(
    int64_t const num_tokens, int64_t const num_experts_per_node) {
  for (int64_t num_tokens_per_block = 32; num_tokens_per_block <= 1024;
       num_tokens_per_block *= 2) {
    int64_t const num_blocks_per_seq =
        ceilDiv(num_tokens, num_tokens_per_block);
    if (num_blocks_per_seq * num_experts_per_node <= num_tokens_per_block) {
      return num_tokens_per_block;
    }
  }
  return 1024;
}

template <typename T>
inline T shuffle_up(sycl::sub_group sg, T value, int delta) {
  int lane_id = sg.get_local_linear_id();
  if (lane_id < delta) {
    // delta = 0;
    return T{0};  // identity for sum
  }
  return sycl::select_from_group(sg, value, lane_id - delta);
}

template <int RANGE_DIM, typename T>
inline T get_subgroup_prefix(
    sycl::group<RANGE_DIM> group,
    sycl::sub_group sg,
    T subgroup_aggregate,
    T& group_aggregate) {
  auto lane_id = sg.get_local_linear_id();
  auto subgroup_id = sg.get_group_linear_id();
  auto subgroup_range = sg.get_group_linear_range();
  auto subgroup_local_range = sg.get_local_linear_range();

  // Use shared memory to store the subgroup aggregate
  auto& subgroup_aggregates =
      *sycl::ext::oneapi::group_local_memory_for_overwrite<
          T[MAX_SUBGROUP_SIZE]>(group);

  if (lane_id == subgroup_local_range - 1) {
    subgroup_aggregates[subgroup_id] = subgroup_aggregate;
  }

  sycl::group_barrier(group);

  group_aggregate = subgroup_aggregates[0];

  T subgroup_prefix;
#pragma unroll
  for (int subgroup_offset = 1; subgroup_offset < subgroup_range;
       ++subgroup_offset) {
    if (subgroup_id == subgroup_offset) {
      subgroup_prefix = group_aggregate;
    }
    group_aggregate = group_aggregate + subgroup_aggregates[subgroup_offset];
  }

  return subgroup_prefix;
}

// implementation of inclusive_scan_stem by shuffle
template <typename T>
inline T
inclusive_scan_over_subgroup_step(sycl::sub_group sg, T input, int offset) {
  int lane_id = sg.get_local_linear_id();
  T temp = shuffle_up(sg, input, offset);

  T output = temp + input;

  if (lane_id < offset) {
    output = input;
  }

  return output;
}

// sub_group scan
template <typename T>
inline void
inclusive_scan_over_subgroup(sycl::sub_group sg, T input, T& inclusive_output) {
  auto sub_group_range = sg.get_local_linear_range();

  int STEPS = sycl::log2(static_cast<float>(sub_group_range));

  inclusive_output = input;
#pragma unroll
  for (int STEP = 0; STEP < STEPS; ++STEP) {
    inclusive_output =
        inclusive_scan_over_subgroup_step(sg, inclusive_output, (1 << STEP));
  }
}

template <int RANGE_DIM, typename T>
inline T exclusive_scan_over_group(
    sycl::nd_item<RANGE_DIM> item, T input, T& group_aggregate) {
  auto sg = item.get_sub_group();

  T inclusive_output;
  inclusive_scan_over_subgroup(sg, input, inclusive_output);

  int lane_id = sg.get_local_linear_id();
  T exclusive_output = shuffle_up(sg, inclusive_output, 1);
  auto group = item.get_group();

  T subgroup_prefix =
      get_subgroup_prefix(group, sg, inclusive_output, group_aggregate);
  auto subgroup_id = sg.get_group_linear_id();

  if (subgroup_id != 0) {
    exclusive_output = subgroup_prefix + exclusive_output;

    if (lane_id == 0) {
      exclusive_output = subgroup_prefix;
    }
  }

  return exclusive_output;
}

template <int kNumTokensPerBlock>
class blockExpertPrefixSumKernel {
 public:
  blockExpertPrefixSumKernel(
      int64_t const* token_selected_experts,
      int* blocked_expert_counts,
      int* blocked_row_to_unpermuted_row,
      int64_t const num_tokens,
      int64_t const num_experts_per_token,
      int const start_expert_id)
      : token_selected_experts(token_selected_experts),
        blocked_expert_counts(blocked_expert_counts),
        blocked_row_to_unpermuted_row(blocked_row_to_unpermuted_row),
        num_tokens(num_tokens),
        num_experts_per_token(num_experts_per_token),
        start_expert_id(start_expert_id) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    // target_expert_id and expert_id are offset by start_expert_id
    int const target_expert_id = item_ct1.get_group(2);
    int const block_id = item_ct1.get_group(1);
    int const num_blocks_per_seq = item_ct1.get_group_range(1);
    int const token_id =
        block_id * kNumTokensPerBlock + item_ct1.get_local_id(2);

    int expanded_token_id = -1;
    if (token_id < num_tokens) {
      for (int i = 0; i < num_experts_per_token; i++) {
        // TODO(enweiz): Fix uncoalesced access with shared memory.
        int const expert_id =
            token_selected_experts[token_id * num_experts_per_token + i] -
            start_expert_id;
        if (expert_id == target_expert_id) {
          expanded_token_id = i * num_tokens + token_id;
          break;
        }
      }
    }

    int const has_matched = expanded_token_id >= 0 ? 1 : 0;
    int index, group_aggregate;
    index = exclusive_scan_over_group(item_ct1, has_matched, group_aggregate);

    if (has_matched) {
      blocked_row_to_unpermuted_row
          [target_expert_id * num_tokens + block_id * kNumTokensPerBlock +
           index] = expanded_token_id;
    }
    if (item_ct1.get_local_id(2) == kNumTokensPerBlock - 1) {
      blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id] =
          index + has_matched;
    }
  }

 private:
  int64_t const* token_selected_experts;
  int* blocked_expert_counts;
  int* blocked_row_to_unpermuted_row;
  int64_t const num_tokens;
  int64_t const num_experts_per_token;
  int const start_expert_id;
};

#define LAUNCH_BLOCK_PREFIXSUM_KERNEL(kNumTokensPerBlock) \
  stream.submit([&](sycl::handler& cgh) {                 \
    cgh.parallel_for(                                     \
        sycl::nd_range<3>(grid * block, block),           \
        blockExpertPrefixSumKernel<kNumTokensPerBlock>(   \
            token_selected_experts,                       \
            blocked_expert_counts,                        \
            blocked_row_to_unpermuted_row,                \
            num_tokens,                                   \
            num_experts_per_token,                        \
            start_expert_id));                            \
  });

void blockExpertPrefixSum(
    int64_t const* token_selected_experts,
    int* blocked_expert_counts,
    int* blocked_row_to_unpermuted_row,
    int64_t const num_tokens,
    int64_t const num_experts_per_node,
    int64_t const num_experts_per_token,
    int64_t const num_tokens_per_block,
    int64_t const num_blocks_per_seq,
    int const start_expert_id,
    sycl::queue& stream) {
  sycl::range<3> grid(1, num_blocks_per_seq, num_experts_per_node);
  sycl::range<3> block(1, 1, num_tokens_per_block);

  if (num_tokens_per_block <= 32) {
    LAUNCH_BLOCK_PREFIXSUM_KERNEL(32);
  } else if (num_tokens_per_block <= 64) {
    LAUNCH_BLOCK_PREFIXSUM_KERNEL(64);
  } else if (num_tokens_per_block <= 128) {
    LAUNCH_BLOCK_PREFIXSUM_KERNEL(128);
  } else if (num_tokens_per_block <= 256) {
    LAUNCH_BLOCK_PREFIXSUM_KERNEL(256);
  } else if (num_tokens_per_block <= 512) {
    LAUNCH_BLOCK_PREFIXSUM_KERNEL(512);
  } else {
    LAUNCH_BLOCK_PREFIXSUM_KERNEL(1024);
  }
}

template <int kNumThreadsPerBlock>
class GlobalExpertPrefixSumLargeKernel {
 public:
  GlobalExpertPrefixSumLargeKernel(
      const int* blocked_expert_counts,
      int* blocked_expert_counts_cumsum,
      int64_t* expert_first_token_offset,
      int64_t num_experts_per_node,
      int64_t num_blocks_per_seq,
      int64_t num_elem_per_thread)
      : blocked_expert_counts(blocked_expert_counts),
        blocked_expert_counts_cumsum(blocked_expert_counts_cumsum),
        expert_first_token_offset(expert_first_token_offset),
        num_experts_per_node(num_experts_per_node),
        num_blocks_per_seq(num_blocks_per_seq),
        num_elem_per_thread(num_elem_per_thread) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (const sycl::nd_item<3>& item) const {
    int offset = item.get_local_id(2) * num_elem_per_thread;
    int cnt = 0;

    // Note: Because of limited registers, cannot store thread-level prefix sum
    // or enable #pragma unroll
    for (int i = 0; i < num_elem_per_thread; i++) {
      // TODO(enweiz): Fix uncoalesced access with shared memory.
      if (offset + i < num_experts_per_node * num_blocks_per_seq) {
        cnt += blocked_expert_counts[offset + i];
      }
    }

    int cumsum, group_aggregate;
    cumsum = exclusive_scan_over_group(item, cnt, group_aggregate);

    for (int i = 0; i < num_elem_per_thread; i++) {
      if (offset + i < num_experts_per_node * num_blocks_per_seq) {
        blocked_expert_counts_cumsum[offset + i] = cumsum;
        if ((offset + i) % num_blocks_per_seq == 0) {
          expert_first_token_offset[(offset + i) / num_blocks_per_seq] = cumsum;
        }
        cumsum += blocked_expert_counts[offset + i];
        if ((offset + i) == num_experts_per_node * num_blocks_per_seq - 1) {
          expert_first_token_offset[num_experts_per_node] = cumsum;
        }
      }
    }
  }

 private:
  const int* blocked_expert_counts;
  int* blocked_expert_counts_cumsum;
  int64_t* expert_first_token_offset;
  const int64_t num_experts_per_node;
  const int64_t num_blocks_per_seq;
  const int64_t num_elem_per_thread;
};

template <int kNumThreadsPerBlock>
class GlobalExpertPrefixSumKernel {
 public:
  GlobalExpertPrefixSumKernel(
      const int* blocked_expert_counts,
      int* blocked_expert_counts_cumsum,
      int64_t* expert_first_token_offset,
      int64_t num_experts_per_node,
      int64_t num_blocks_per_seq)
      : blocked_expert_counts(blocked_expert_counts),
        blocked_expert_counts_cumsum(blocked_expert_counts_cumsum),
        expert_first_token_offset(expert_first_token_offset),
        num_experts_per_node(num_experts_per_node),
        num_blocks_per_seq(num_blocks_per_seq) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (const sycl::nd_item<3>& item) const {
    int const cnt =
        item.get_local_id(2) < num_experts_per_node * num_blocks_per_seq
            ? blocked_expert_counts[item.get_local_id(2)]
            : 0;

    int cumsum, group_aggregate;
    cumsum = exclusive_scan_over_group(item, cnt, group_aggregate);

    if (item.get_local_id(2) < num_experts_per_node * num_blocks_per_seq) {
      blocked_expert_counts_cumsum[item.get_local_id(2)] = cumsum;
      if (item.get_local_id(2) % num_blocks_per_seq == 0) {
        expert_first_token_offset[item.get_local_id(2) / num_blocks_per_seq] =
            cumsum;
      }
      if (item.get_local_id(2) ==
          num_experts_per_node * num_blocks_per_seq - 1) {
        expert_first_token_offset[num_experts_per_node] = cumsum + cnt;
      }
    }
  }

 private:
  const int* blocked_expert_counts;
  int* blocked_expert_counts_cumsum;
  int64_t* expert_first_token_offset;
  const int64_t num_experts_per_node;
  const int64_t num_blocks_per_seq;
};

#define LAUNCH_GLOBAL_PREFIXSUM_KERNEL(kNumThreadsPerBlock) \
  stream.submit([&](sycl::handler& cgh) {                   \
    cgh.parallel_for(                                       \
        sycl::nd_range<3>(grid * block, block),             \
        GlobalExpertPrefixSumKernel<kNumThreadsPerBlock>(   \
            blocked_expert_counts,                          \
            blocked_expert_counts_cumsum,                   \
            expert_first_token_offset,                      \
            num_experts_per_node,                           \
            num_blocks_per_seq));                           \
  });

void globalExpertPrefixSum(
    int const* blocked_expert_counts,
    int* blocked_expert_counts_cumsum,
    int64_t* expert_first_token_offset,
    int64_t const num_experts_per_node,
    int64_t const num_tokens_per_block,
    int64_t const num_blocks_per_seq,
    sycl::queue& stream) {
  int64_t const num_elements = num_experts_per_node * num_blocks_per_seq;
  sycl::range<3> grid(1, 1, 1);

  if (num_elements <= 1024) {
    if (num_elements <= 32) {
      sycl::range<3> block(1, 1, 32);
      LAUNCH_GLOBAL_PREFIXSUM_KERNEL(32);
    } else if (num_elements <= 64) {
      sycl::range<3> block(1, 1, 64);
      LAUNCH_GLOBAL_PREFIXSUM_KERNEL(64);
    } else if (num_elements <= 128) {
      sycl::range<3> block(1, 1, 128);
      LAUNCH_GLOBAL_PREFIXSUM_KERNEL(128);
    } else if (num_elements <= 256) {
      sycl::range<3> block(1, 1, 256);
      LAUNCH_GLOBAL_PREFIXSUM_KERNEL(256);
    } else if (num_elements <= 512) {
      sycl::range<3> block(1, 1, 512);
      LAUNCH_GLOBAL_PREFIXSUM_KERNEL(512);
    } else {
      sycl::range<3> block(1, 1, 1024);
      LAUNCH_GLOBAL_PREFIXSUM_KERNEL(1024);
    }
  } else {
    int64_t const num_elem_per_thread = ceilDiv<int64_t>(num_elements, 1024);
    sycl::range<3> block(1, 1, 1024);
    stream.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          GlobalExpertPrefixSumLargeKernel<1024>(
              blocked_expert_counts,
              blocked_expert_counts_cumsum,
              expert_first_token_offset,
              num_experts_per_node,
              num_blocks_per_seq,
              num_elem_per_thread));
    });
  }
}

class MergeExpertPrefixSumKernel {
 public:
  MergeExpertPrefixSumKernel(
      const int* blocked_expert_counts,
      const int* blocked_expert_counts_cumsum,
      const int* blocked_row_to_unpermuted_row,
      int* permuted_token_selected_experts,
      int* permuted_row_to_unpermuted_row,
      int* unpermuted_row_to_permuted_row,
      int num_tokens)
      : blocked_expert_counts(blocked_expert_counts),
        blocked_expert_counts_cumsum(blocked_expert_counts_cumsum),
        blocked_row_to_unpermuted_row(blocked_row_to_unpermuted_row),
        permuted_token_selected_experts(permuted_token_selected_experts),
        permuted_row_to_unpermuted_row(permuted_row_to_unpermuted_row),
        unpermuted_row_to_permuted_row(unpermuted_row_to_permuted_row),
        num_tokens(num_tokens) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (const sycl::nd_item<3>& item) const {
    int const target_expert_id = item.get_group(2);
    int const block_id = item.get_group(1);
    int const num_blocks_per_seq = item.get_group_range(1);
    int const token_id =
        block_id * item.get_local_range(2) + item.get_local_id(2);

    int const cnt =
        blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id];
    int const offset = blocked_expert_counts_cumsum
        [target_expert_id * num_blocks_per_seq + block_id];
    if (item.get_local_id(2) < cnt) {
      int const unpermuted_row = blocked_row_to_unpermuted_row
          [target_expert_id * num_tokens + token_id];
      int const permuted_row = offset + item.get_local_id(2);

      permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
      permuted_token_selected_experts[permuted_row] = target_expert_id;
      unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
    }
  }

 private:
  const int* blocked_expert_counts;
  const int* blocked_expert_counts_cumsum;
  const int* blocked_row_to_unpermuted_row;
  int* permuted_token_selected_experts;
  int* permuted_row_to_unpermuted_row;
  int* unpermuted_row_to_permuted_row;
  const int num_tokens;
};

void mergeExpertPrefixSum(
    int const* blocked_expert_counts,
    int const* blocked_expert_counts_cumsum,
    int const* blocked_row_to_unpermuted_row,
    int* permuted_token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t const num_tokens,
    int64_t const num_experts_per_node,
    int64_t const num_tokens_per_block,
    int64_t const num_blocks_per_seq,
    sycl::queue& stream) {
  sycl::range<3> grid(1, num_blocks_per_seq, num_experts_per_node);
  sycl::range<3> block(1, 1, num_tokens_per_block);
  stream.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block),
        MergeExpertPrefixSumKernel(
            blocked_expert_counts,
            blocked_expert_counts_cumsum,
            blocked_row_to_unpermuted_row,
            permuted_token_selected_experts,
            permuted_row_to_unpermuted_row,
            unpermuted_row_to_permuted_row,
            num_tokens));
  });
}

void threeStepBuildExpertMapsSortFirstToken(
    int64_t const* token_selected_experts,
    int* permuted_token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset,
    int* blocked_expert_counts,
    int* blocked_expert_counts_cumsum,
    int* blocked_row_to_unpermuted_row,
    int64_t const num_tokens,
    int64_t const num_experts_per_node,
    int64_t const num_experts_per_token,
    int const start_expert_id,
    sycl::queue& stream) {
  int64_t const num_tokens_per_block =
      computeNumTokensPerBlock(num_tokens, num_experts_per_node);
  int64_t const num_blocks_per_seq = ceilDiv(num_tokens, num_tokens_per_block);

  blockExpertPrefixSum(
      token_selected_experts,
      blocked_expert_counts,
      blocked_row_to_unpermuted_row,
      num_tokens,
      num_experts_per_node,
      num_experts_per_token,
      num_tokens_per_block,
      num_blocks_per_seq,
      start_expert_id,
      stream);

  globalExpertPrefixSum(
      blocked_expert_counts,
      blocked_expert_counts_cumsum,
      expert_first_token_offset,
      num_experts_per_node,
      num_tokens_per_block,
      num_blocks_per_seq,
      stream);

  mergeExpertPrefixSum(
      blocked_expert_counts,
      blocked_expert_counts_cumsum,
      blocked_row_to_unpermuted_row,
      permuted_token_selected_experts,
      permuted_row_to_unpermuted_row,
      unpermuted_row_to_permuted_row,
      num_tokens,
      num_experts_per_node,
      num_tokens_per_block,
      num_blocks_per_seq,
      stream);
}

template <class InputActivationsType, class ExpandedActivationsType>
class ExpandInputRowsKernel {
 public:
  ExpandInputRowsKernel(
      InputActivationsType const* unpermuted_input,
      ExpandedActivationsType* permuted_output,
      float const* unpermuted_scales,
      float* permuted_scales,
      int const* permuted_row_to_unpermuted_row,
      int64_t num_tokens,
      int64_t hidden_size,
      int64_t k,
      bool use_per_expert_act_scale,
      int64_t const* expert_first_token_offset,
      int64_t num_experts_per_node,
      InputActivationsType const* prequant_scales = nullptr)
      : unpermuted_input(unpermuted_input),
        permuted_output(permuted_output),
        unpermuted_scales(unpermuted_scales),
        permuted_scales(permuted_scales),
        permuted_row_to_unpermuted_row(permuted_row_to_unpermuted_row),
        num_tokens(num_tokens),
        hidden_size(hidden_size),
        k(k),
        use_per_expert_act_scale(use_per_expert_act_scale),
        expert_first_token_offset(expert_first_token_offset),
        num_experts_per_node(num_experts_per_node),
        prequant_scales(prequant_scales) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (const sycl::nd_item<3>& item) const {
    int64_t const num_valid_tokens =
        expert_first_token_offset[num_experts_per_node];
    for (int64_t permuted_row = item.get_group(2);
         permuted_row < num_valid_tokens;
         permuted_row += item.get_group_range(2)) {
      int64_t const unpermuted_row =
          permuted_row_to_unpermuted_row[permuted_row];

      // Load 128-bits per thread

      int64_t const source_k_rank = unpermuted_row / num_tokens;
      int64_t const source_row = unpermuted_row % num_tokens;

      auto const* source_row_ptr = unpermuted_input + source_row * hidden_size;
      // Cast first to handle when this is FP4
      auto* dest_row_ptr = permuted_output + permuted_row * hidden_size;

      int64_t const start_offset = item.get_local_id(2);
      int64_t const stride = EXPAND_THREADS_PER_BLOCK;
      for (int elem_index = start_offset; elem_index < hidden_size;
           elem_index += stride) {
        dest_row_ptr[elem_index] = source_row_ptr[elem_index];
      }
      if (permuted_scales && item.get_local_id(2) == 0) {
        int64_t const source_k_idx = source_row * k + source_k_rank;
        permuted_scales[permuted_row] =
            unpermuted_scales ? unpermuted_scales[source_k_idx] : 1.0f;
      }
    }
  }

 private:
  InputActivationsType const* unpermuted_input;
  ExpandedActivationsType* permuted_output;
  float const* unpermuted_scales;
  float* permuted_scales;
  int const* permuted_row_to_unpermuted_row;
  const int64_t num_tokens;
  const int64_t hidden_size;
  const int64_t k;
  const bool use_per_expert_act_scale;
  int64_t const* expert_first_token_offset;
  const int64_t num_experts_per_node;
  InputActivationsType const* prequant_scales;
};

template <class InputActivationsType, class ExpandedActivationsType>
void expandInputRowsKernelLauncher(
    InputActivationsType const* unpermuted_input,
    ExpandedActivationsType* permuted_output,
    float const* unpermuted_scales,
    float* permuted_scales,
    int const* permuted_row_to_unpermuted_row,
    int64_t const num_rows,
    int64_t const hidden_size,
    int const k,
    int const num_experts_per_node,
    bool use_per_expert_act_scale,
    int64_t* expert_first_token_offset,
    void const* prequant_scales,
    sycl::queue& stream) {
  int64_t num_padding_tokens = 0;

  int64_t const blocks = 8 * 64;
  sycl::range<3> grid(1, 1, blocks);
  sycl::range<3> block(1, 1, EXPAND_THREADS_PER_BLOCK);

  stream.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block),
        ExpandInputRowsKernel<bfloat16, bfloat16>(
            unpermuted_input,
            permuted_output,
            unpermuted_scales,
            permuted_scales,
            permuted_row_to_unpermuted_row,
            num_rows,
            hidden_size,
            k,
            use_per_expert_act_scale,
            expert_first_token_offset,
            num_experts_per_node,
            reinterpret_cast<InputActivationsType const*>(prequant_scales)));
  });
}
