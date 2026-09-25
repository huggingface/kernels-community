#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/Half.h>

#include <array>
#include <optional>

#include "stable_utils.h"

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

// Header-only equivalent of AT_DISPATCH_FLOATING_TYPES_AND_HALF.
#define DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...) \
  THO_DISPATCH_SWITCH(                                    \
      TYPE,                                               \
      NAME,                                               \
      THO_DISPATCH_CASE(ScalarType::Float, __VA_ARGS__)   \
      THO_DISPATCH_CASE(ScalarType::Double, __VA_ARGS__)  \
      THO_DISPATCH_CASE(ScalarType::Half, __VA_ARGS__))

namespace {

// Replacement for at::acc_type<T, /*is_cuda=*/true>, which is not part of the
// stable ABI, for the types that generic_nms dispatches on.
template <typename T>
struct acc_type {
  using type = T;
};

template <>
struct acc_type<torch::headeronly::Half> {
  using type = float;
};

// There are no torch::stable wrappers for sort and masked_select, so call them
// through the dispatcher, like the wrappers in torch/csrc/stable/ops.h do.

// aten::sort.stable(Tensor self, *, bool? stable, int dim=-1,
//                   bool descending=False) -> (Tensor values, Tensor indices)
// Returns the indices only.
Tensor sort_indices(
    const Tensor& self,
    bool stable,
    int64_t dim,
    bool descending) {
  std::array<StableIValue, 4> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(std::optional<bool>(stable)),
      torch::stable::detail::from(dim),
      torch::stable::detail::from(descending)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::sort", "stable", stack.data(), TORCH_ABI_VERSION));
  // The stack owns both returned handles. Wrap the unused values in a
  // temporary Tensor so that it is released instead of leaked.
  torch::stable::detail::to<Tensor>(stack[0]);
  return torch::stable::detail::to<Tensor>(stack[1]);
}

// aten::masked_select(Tensor self, Tensor mask) -> Tensor
Tensor masked_select(const Tensor& self, const Tensor& mask) {
  std::array<StableIValue, 2> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(mask)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::masked_select", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename T>
__device__ inline bool
devIoU(T const* const a, T const* const b, const float threshold) {
  T left = max(a[0], b[0]), right = min(a[2], b[2]);
  T top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  T width = max(right - left, (T)0), height = max(bottom - top, (T)0);
  using acc_T = typename acc_type<T>::type;
  acc_T interS = (acc_T)width * height;
  acc_T Sa = ((acc_T)a[2] - a[0]) * (a[3] - a[1]);
  acc_T Sb = ((acc_T)b[2] - b[0]) * (b[3] - b[1]);
  return (interS / (Sa + Sb - interS)) > threshold;
}

template <typename T>
__global__ void nms_kernel_impl(
    int n_boxes,
    double iou_threshold,
    const T* dev_boxes,
    unsigned long long* dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  if (row_start > col_start)
    return;

  const int row_size =
      min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ T block_boxes[threadsPerBlock * 4];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 4 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const T* cur_box = dev_boxes + cur_box_idx * 4;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU<T>(cur_box, block_boxes + i * 4, iou_threshold)) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = ceil_div(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

template <typename T>
__global__ void nms_kernel_iou_impl(
    int n_boxes,
    double iou_threshold,
    const T* dev_iou, // [N, N] row-major IoU matrix
    unsigned long long* dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  if (row_start > col_start)
    return;

  const int row_size =
      min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  if (threadIdx.x < row_size) {
    const int cur_row_idx = threadsPerBlock * row_start + threadIdx.x;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    const int col_base = threadsPerBlock * col_start;
    for (i = start; i < col_size; i++) {
      const int col_idx = col_base + i;
      T iou = dev_iou[cur_row_idx * n_boxes + col_idx];
      if (static_cast<double>(iou) > iou_threshold) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = ceil_div(n_boxes, threadsPerBlock);
    dev_mask[cur_row_idx * col_blocks + col_start] = t;
  }
}

__global__ static void gather_keep_from_mask(
    bool* keep,
    const unsigned long long* dev_mask,
    const int n_boxes) {
  // Taken and adapted from mmcv
  // https://github.com/open-mmlab/mmcv/blob/03ce9208d18c0a63d7ffa087ea1c2f5661f2441a/mmcv/ops/csrc/common/cuda/nms_cuda_kernel.cuh#L76
  const int col_blocks = ceil_div(n_boxes, threadsPerBlock);
  const int thread_id = threadIdx.x;

  // Mark the bboxes which have been removed.
  extern __shared__ unsigned long long removed[];

  // Initialize removed.
  for (int i = thread_id; i < col_blocks; i += blockDim.x) {
    removed[i] = 0;
  }
  __syncthreads();

  for (int nblock = 0; nblock < col_blocks; nblock++) {
    auto removed_val = removed[nblock];
    __syncthreads();
    const int i_offset = nblock * threadsPerBlock;
#pragma unroll
    for (int inblock = 0; inblock < threadsPerBlock; inblock++) {
      const int i = i_offset + inblock;
      if (i >= n_boxes)
        break;
      // Select a candidate, check if it should kept.
      if (!(removed_val & (1ULL << inblock))) {
        if (thread_id == 0) {
          keep[i] = true;
        }
        auto p = dev_mask + i * col_blocks;
        // Remove all bboxes which overlap the candidate.
        for (int j = thread_id; j < col_blocks; j += blockDim.x) {
          if (j >= nblock)
            removed[j] |= p[j];
        }
        __syncthreads();
        removed_val = removed[nblock];
      }
    }
  }
}

} // namespace

Tensor generic_nms(
    const Tensor& dets,
    const Tensor& scores,
    double iou_threshold,
    bool use_iou_matrix) {
  STD_TORCH_CHECK(dets.is_cuda(), "dets must be a CUDA tensor");
  STD_TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
  STD_TORCH_CHECK(
      dets.dim() == 2,
      "first argument should be a 2d tensor, got ",
      dets.dim(),
      "D");
  STD_TORCH_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  STD_TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "first argument and scores should have same number of elements in dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0));

  const torch::stable::accelerator::DeviceGuard device_guard(
      dets.get_device_index());

  if (dets.numel() == 0) {
    return torch::stable::new_empty(dets, {0}, ScalarType::Long);
  }

  auto order_t = sort_indices(
      scores, /*stable=*/true, /*dim=*/0, /*descending=*/true);
  int dets_num = dets.size(0);
  const int col_blocks = ceil_div(dets_num, threadsPerBlock);

  Tensor mask = torch::stable::new_empty(
      dets, {static_cast<int64_t>(dets_num) * col_blocks}, ScalarType::Long);
  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = get_current_cuda_stream(dets);

  if (use_iou_matrix) {
    STD_TORCH_CHECK(
        dets.size(0) == dets.size(1),
        "when use_iou_matrix=True, first argument must be [N,N]");
    auto sorted_iou = torch::stable::contiguous(torch::stable::index_select(
        torch::stable::index_select(dets, 0, order_t), 1, order_t));
    DISPATCH_FLOATING_TYPES_AND_HALF(
        sorted_iou.scalar_type(), "nms_kernel_iou_ex", [&] {
          nms_kernel_iou_impl<scalar_t><<<blocks, threads, 0, stream>>>(
              dets_num,
              iou_threshold,
              sorted_iou.const_data_ptr<scalar_t>(),
              (unsigned long long*)mask.mutable_data_ptr<int64_t>());
        });
  } else {
    STD_TORCH_CHECK(
        dets.size(1) == 4, "when use_iou_matrix=False, boxes must be [N,4]");
    auto dets_sorted =
        torch::stable::contiguous(torch::stable::index_select(dets, 0, order_t));
    DISPATCH_FLOATING_TYPES_AND_HALF(
        dets_sorted.scalar_type(), "nms_kernel_ex", [&] {
          nms_kernel_impl<scalar_t><<<blocks, threads, 0, stream>>>(
              dets_num,
              iou_threshold,
              dets_sorted.const_data_ptr<scalar_t>(),
              (unsigned long long*)mask.mutable_data_ptr<int64_t>());
        });
  }

  Tensor keep = torch::stable::new_zeros(dets, {dets_num}, ScalarType::Bool);
  gather_keep_from_mask<<<
      1,
      min(col_blocks, threadsPerBlock),
      col_blocks * sizeof(unsigned long long),
      stream>>>(
      keep.mutable_data_ptr<bool>(),
      (unsigned long long*)mask.const_data_ptr<int64_t>(),
      dets_num);

  STD_CUDA_CHECK(cudaGetLastError());
  return masked_select(order_t, keep);
}
