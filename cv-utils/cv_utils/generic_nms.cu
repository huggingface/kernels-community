#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include <cstdint>
#include <limits>

namespace {

using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;

template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename T>
struct NmsAcc {
  using type = float;
};

template <>
struct NmsAcc<double> {
  using type = double;
};

template <typename T>
__device__ inline typename NmsAcc<T>::type to_acc(T value) {
  return static_cast<typename NmsAcc<T>::type>(value);
}

template <>
__device__ inline float to_acc<__half>(__half value) {
  return __half2float(value);
}

template <typename T>
__device__ inline bool devIoU(
    T const* const a,
    T const* const b,
    const double threshold) {
  using acc_T = typename NmsAcc<T>::type;
  const acc_T ax0 = to_acc(a[0]);
  const acc_T ay0 = to_acc(a[1]);
  const acc_T ax2 = to_acc(a[2]);
  const acc_T ay2 = to_acc(a[3]);
  const acc_T bx0 = to_acc(b[0]);
  const acc_T by0 = to_acc(b[1]);
  const acc_T bx2 = to_acc(b[2]);
  const acc_T by2 = to_acc(b[3]);
  const acc_T left = max(ax0, bx0);
  const acc_T right = min(ax2, bx2);
  const acc_T top = max(ay0, by0);
  const acc_T bottom = min(ay2, by2);
  const acc_T width = max(right - left, static_cast<acc_T>(0));
  const acc_T height = max(bottom - top, static_cast<acc_T>(0));
  const acc_T interS = width * height;
  const acc_T Sa = (ax2 - ax0) * (ay2 - ay0);
  const acc_T Sb = (bx2 - bx0) * (by2 - by0);
  return (interS / (Sa + Sb - interS)) > static_cast<acc_T>(threshold);
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
      if (to_acc(dev_iou[cur_row_idx * n_boxes + col_idx]) >
          static_cast<typename NmsAcc<T>::type>(iou_threshold)) {
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

__global__ void initialize_indices(int64_t* indices, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    indices[index] = index;
  }
}

template <typename T>
__global__ void gather_rows(
    const T* input,
    const int64_t* order,
    T* output,
    int rows,
    int cols) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < rows * cols) {
    const int row = index / cols;
    const int col = index % cols;
    output[index] = input[order[row] * cols + col];
  }
}

template <typename T>
__global__ void gather_matrix(
    const T* input,
    const int64_t* order,
    T* output,
    int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size * size) {
    const int row = index / size;
    const int col = index % size;
    output[index] = input[order[row] * size + order[col]];
  }
}

inline void check_cuda(cudaError_t error, const char* operation) {
  STD_TORCH_CHECK(
      error == cudaSuccess, operation, ": ", cudaGetErrorString(error));
}

template <typename ScoreT>
void sort_scores_desc(
    const Tensor& scores,
    Tensor& initial_indices,
    Tensor& sorted_scores,
    Tensor& sort_temp,
    Tensor& order,
    cudaStream_t stream) {
  const int n = static_cast<int>(scores.size(0));
  const int blocks = ceil_div(n, 256);
  initial_indices = torch::stable::new_empty(
      scores, {n}, ScalarType::Long);
  initialize_indices<<<blocks, 256, 0, stream>>>(
      static_cast<int64_t*>(initial_indices.data_ptr()), n);
  check_cuda(cudaGetLastError(), "initialize NMS indices");

  sorted_scores = torch::stable::new_empty(
      scores, {n}, scores.scalar_type());
  size_t temp_bytes = 0;
  check_cuda(
      cub::DeviceRadixSort::SortPairsDescending(
          nullptr,
          temp_bytes,
          static_cast<const ScoreT*>(scores.data_ptr()),
          static_cast<ScoreT*>(sorted_scores.data_ptr()),
          static_cast<const int64_t*>(initial_indices.data_ptr()),
          static_cast<int64_t*>(order.data_ptr()),
          n,
          0,
          sizeof(ScoreT) * 8,
          stream),
      "query NMS sort workspace");
  sort_temp = torch::stable::new_empty(
      scores, {static_cast<int64_t>(temp_bytes)}, ScalarType::Byte);
  check_cuda(
      cub::DeviceRadixSort::SortPairsDescending(
          sort_temp.data_ptr(),
          temp_bytes,
          static_cast<const ScoreT*>(scores.data_ptr()),
          static_cast<ScoreT*>(sorted_scores.data_ptr()),
          static_cast<const int64_t*>(initial_indices.data_ptr()),
          static_cast<int64_t*>(order.data_ptr()),
          n,
          0,
          sizeof(ScoreT) * 8,
          stream),
      "sort NMS scores");
}

#define CV_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, ...)                         \
  do {                                                                         \
    if ((TYPE) == ScalarType::Float) {                                         \
      using scalar_t = float;                                                  \
      __VA_ARGS__;                                                             \
    } else if ((TYPE) == ScalarType::Double) {                                 \
      using scalar_t = double;                                                 \
      __VA_ARGS__;                                                             \
    } else if ((TYPE) == ScalarType::Half) {                                   \
      using scalar_t = __half;                                                 \
      __VA_ARGS__;                                                             \
    } else {                                                                   \
      STD_TORCH_CHECK(false, "NMS supports float, double, and half tensors");  \
    }                                                                          \
  } while (false)

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

  torch::stable::accelerator::DeviceGuard device_guard(
      dets.get_device_index());
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(
      dets.get_device_index(), &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  if (dets.numel() == 0) {
    return torch::stable::new_empty(dets, {0}, ScalarType::Long);
  }

  const int64_t dets_num_64 = dets.size(0);
  STD_TORCH_CHECK(
      dets_num_64 <= std::numeric_limits<int>::max(),
      "NMS input is too large");
  const int dets_num = static_cast<int>(dets_num_64);
  const int col_blocks = ceil_div(dets_num, threadsPerBlock);

  Tensor order = torch::stable::new_empty(scores, {dets_num}, ScalarType::Long);
  Tensor initial_indices;
  Tensor sorted_scores;
  Tensor sort_temp;
  CV_DISPATCH_FLOATING_TYPES_AND_HALF(
      scores.scalar_type(),
      sort_scores_desc<scalar_t>(
          scores, initial_indices, sorted_scores, sort_temp, order, stream));

  Tensor mask = torch::stable::new_empty(
      dets,
      {static_cast<int64_t>(dets_num) * col_blocks},
      ScalarType::Long);
  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);

  if (use_iou_matrix) {
    STD_TORCH_CHECK(
        dets.size(0) == dets.size(1),
        "when use_iou_matrix=True, first argument must be [N,N]");
    Tensor sorted_iou = torch::stable::new_empty(
        dets, {dets_num, dets_num}, dets.scalar_type());
    CV_DISPATCH_FLOATING_TYPES_AND_HALF(
        sorted_iou.scalar_type(),
        gather_matrix<scalar_t><<<
            ceil_div(dets_num * dets_num, 256),
            256,
            0,
            stream>>>(
            static_cast<const scalar_t*>(dets.data_ptr()),
            static_cast<const int64_t*>(order.data_ptr()),
            static_cast<scalar_t*>(sorted_iou.data_ptr()),
            dets_num);
        check_cuda(cudaGetLastError(), "gather NMS IoU matrix");
        nms_kernel_iou_impl<scalar_t><<<blocks, threads, 0, stream>>>(
            dets_num,
            iou_threshold,
            static_cast<const scalar_t*>(sorted_iou.data_ptr()),
            static_cast<unsigned long long*>(mask.data_ptr())));
  } else {
    STD_TORCH_CHECK(
        dets.size(1) == 4,
        "when use_iou_matrix=False, boxes must be [N,4]");
    Tensor dets_sorted = torch::stable::new_empty(
        dets, {dets_num, 4}, dets.scalar_type());
    CV_DISPATCH_FLOATING_TYPES_AND_HALF(
        dets_sorted.scalar_type(),
        gather_rows<scalar_t><<<
            ceil_div(dets_num * 4, 256),
            256,
            0,
            stream>>>(
            static_cast<const scalar_t*>(dets.data_ptr()),
            static_cast<const int64_t*>(order.data_ptr()),
            static_cast<scalar_t*>(dets_sorted.data_ptr()),
            dets_num,
            4);
        check_cuda(cudaGetLastError(), "gather NMS boxes");
        nms_kernel_impl<scalar_t><<<blocks, threads, 0, stream>>>(
            dets_num,
            iou_threshold,
            static_cast<const scalar_t*>(dets_sorted.data_ptr()),
            static_cast<unsigned long long*>(mask.data_ptr())));
  }
  check_cuda(cudaGetLastError(), "run NMS kernel");

  Tensor keep = torch::stable::new_zeros(dets, {dets_num}, ScalarType::Bool);
  gather_keep_from_mask<<<
      1,
      min(col_blocks, threadsPerBlock),
      col_blocks * sizeof(unsigned long long),
      stream>>>(
      static_cast<bool*>(keep.data_ptr()),
      static_cast<const unsigned long long*>(mask.data_ptr()),
      dets_num);
  check_cuda(cudaGetLastError(), "gather NMS keep flags");

  // DeviceSelect preserves the score order while compacting the selected
  // indices. The count is copied only after all work on the caller's stream
  // has completed, because the Stable ABI does not expose masked_select.
  Tensor selected = torch::stable::new_empty(dets, {dets_num}, ScalarType::Long);
  Tensor selected_count = torch::stable::new_zeros(dets, {1}, ScalarType::Int);
  size_t select_temp_bytes = 0;
  check_cuda(
      cub::DeviceSelect::Flagged(
          nullptr,
          select_temp_bytes,
          static_cast<const int64_t*>(order.data_ptr()),
          static_cast<const bool*>(keep.data_ptr()),
          static_cast<int64_t*>(selected.data_ptr()),
          static_cast<int*>(selected_count.data_ptr()),
          dets_num,
          stream),
      "query NMS selection workspace");
  Tensor select_temp = torch::stable::new_empty(
      dets,
      {static_cast<int64_t>(select_temp_bytes)},
      ScalarType::Byte);
  check_cuda(
      cub::DeviceSelect::Flagged(
          select_temp.data_ptr(),
          select_temp_bytes,
          static_cast<const int64_t*>(order.data_ptr()),
          static_cast<const bool*>(keep.data_ptr()),
          static_cast<int64_t*>(selected.data_ptr()),
          static_cast<int*>(selected_count.data_ptr()),
          dets_num,
          stream),
      "select NMS indices");

  int selected_count_host = 0;
  check_cuda(
      cudaMemcpyAsync(
          &selected_count_host,
          selected_count.data_ptr(),
          sizeof(selected_count_host),
          cudaMemcpyDeviceToHost,
          stream),
      "copy NMS selection count");
  check_cuda(cudaStreamSynchronize(stream), "synchronize NMS selection count");
  return torch::stable::narrow(selected, 0, 0, selected_count_host);
}
