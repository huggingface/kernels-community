#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cuda_runtime.h>

#include <cstdint>

extern "C" AOTITorchError aoti_torch_get_current_cuda_stream(
    int32_t device_index,
    void** ret_stream);

// 2d
#define BLOCK_ROWS 16
#define BLOCK_COLS 16

namespace cc2d {

template <typename T>
__device__ __forceinline__ unsigned char hasBit(T bitmap, unsigned char pos) {
  return (bitmap >> pos) & 1;
}

__device__ int32_t find(const int32_t* s_buf, int32_t n) {
  while (s_buf[n] != n)
    n = s_buf[n];
  return n;
}

__device__ int32_t find_n_compress(int32_t* s_buf, int32_t n) {
  const int32_t id = n;
  while (s_buf[n] != n) {
    n = s_buf[n];
    s_buf[id] = n;
  }
  return n;
}

__device__ void union_(int32_t* s_buf, int32_t a, int32_t b) {
  bool done;
  do {
    a = find(s_buf, a);
    b = find(s_buf, b);

    if (a < b) {
      int32_t old = atomicMin(s_buf + b, a);
      done = (old == b);
      b = old;
    } else if (b < a) {
      int32_t old = atomicMin(s_buf + a, b);
      done = (old == a);
      a = old;
    } else
      done = true;

  } while (!done);
}

__global__ void
init_labeling(int32_t* label, const uint32_t W, const uint32_t H) {
  const uint32_t n = blockIdx.z; // batch index
  const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  const uint32_t idx = row * W + col;
  const uint32_t offset = n * H * W;

  if (row < H && col < W)
    label[offset + idx] = idx; // each image uses local indexing, later +1
}

__global__ void
merge(const uint8_t* img, int32_t* label, const uint32_t W, const uint32_t H) {
  const uint32_t n = blockIdx.z; // batch index
  const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  const uint32_t idx = row * W + col;
  const uint32_t offset = n * H * W;

  if (row >= H || col >= W)
    return;

  uint32_t P = 0;

  // NOTE : Original Codes, but occurs silent error
  // NOTE : Programs keep runnig, but now showing printf logs, and the result
  // is weird uint8_t buffer[4] = {0}; if (col + 1 < W) {
  //     *(reinterpret_cast<uint16_t*>(buffer)) =
  //     *(reinterpret_cast<uint16_t*>(img + idx)); if (row + 1 < H) {
  //         *(reinterpret_cast<uint16_t*>(buffer + 2)) =
  //         *(reinterpret_cast<uint16_t*>(img + idx + W));
  //     }
  // }
  // else {
  //     buffer[0] = img[idx];
  //     if (row + 1 < H)
  //         buffer[2] = img[idx + W];
  // }
  // if (buffer[0])              P |= 0x777;
  // if (buffer[1])              P |= (0x777 << 1);
  // if (buffer[2])              P |= (0x777 << 4);

  if (img[offset + idx])
    P |= 0x777;
  if (row + 1 < H && img[offset + idx + W])
    P |= 0x777 << 4;
  if (col + 1 < W && img[offset + idx + 1])
    P |= 0x777 << 1;

  if (col == 0)
    P &= 0xEEEE;
  if (col + 1 >= W)
    P &= 0x3333;
  else if (col + 2 >= W)
    P &= 0x7777;

  if (row == 0)
    P &= 0xFFF0;
  if (row + 1 >= H)
    P &= 0xFF;

  if (P > 0) {
    // If need check about top-left pixel(if flag the first bit) and hit the
    // top-left pixel
    if (hasBit(P, 0) && img[offset + idx - W - 1]) {
      union_(label + offset, idx, idx - 2 * W - 2); // top left block
    }

    if ((hasBit(P, 1) && img[offset + idx - W]) ||
        (hasBit(P, 2) && img[offset + idx - W + 1]))
      union_(label + offset, idx, idx - 2 * W); // top bottom block

    if (hasBit(P, 3) && img[offset + idx + 2 - W])
      union_(label + offset, idx, idx - 2 * W + 2); // top right block

    if ((hasBit(P, 4) && img[offset + idx - 1]) ||
        (hasBit(P, 8) && img[offset + idx + W - 1]))
      union_(label + offset, idx, idx - 2); // just left block
  }
}

__global__ void compression(int32_t* label, const int32_t W, const int32_t H) {
  const uint32_t n = blockIdx.z; // batch index
  const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  const uint32_t idx = row * W + col;
  const uint32_t offset = n * H * W;

  if (row < H && col < W)
    find_n_compress(label + offset, idx);
}

__global__ void final_labeling(
    const uint8_t* img,
    int32_t* label,
    const int32_t W,
    const int32_t H) {
  const uint32_t n = blockIdx.z; // batch index
  const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  const uint32_t idx = row * W + col;
  const uint32_t offset = n * H * W;

  if (row >= H || col >= W)
    return;

  int32_t y = label[offset + idx] + 1;

  if (img[offset + idx])
    label[offset + idx] = y;
  else
    label[offset + idx] = 0;

  if (col + 1 < W) {
    if (img[offset + idx + 1])
      label[offset + idx + 1] = y;
    else
      label[offset + idx + 1] = 0;

    if (row + 1 < H) {
      if (img[offset + idx + W + 1])
        label[offset + idx + W + 1] = y;
      else
        label[offset + idx + W + 1] = 0;
    }
  }

  if (row + 1 < H) {
    if (img[offset + idx + W])
      label[offset + idx + W] = y;
    else
      label[offset + idx + W] = 0;
  }
}

__global__ void init_counting(
    const int32_t* label,
    int32_t* count_init,
    const int32_t W,
    const int32_t H) {
  const uint32_t n = blockIdx.z; // batch index
  const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y);
  const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x);
  const uint32_t idx = row * W + col;
  const uint32_t offset = n * H * W;

  if (row >= H || col >= W)
    return;

  int32_t y = label[offset + idx];
  if (y > 0) {
    int32_t count_idx = y - 1;
    atomicAdd(count_init + offset + count_idx, 1);
  }
}

__global__ void final_counting(
    const int32_t* label,
    const int32_t* count_init,
    int32_t* count_final,
    const int32_t W,
    const int32_t H) {
  const uint32_t n = blockIdx.z; // batch index
  const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y);
  const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x);
  const uint32_t idx = row * W + col;
  const uint32_t offset = n * H * W;

  if (row >= H || col >= W)
    return;

  int32_t y = label[offset + idx];
  if (y > 0) {
    int32_t count_idx = y - 1;
    count_final[offset + idx] = count_init[offset + count_idx];
  } else {
    count_final[offset + idx] = 0;
  }
}

} // namespace cc2d

std::tuple<torch::stable::Tensor, torch::stable::Tensor>
connected_components_labeling_2d(
    const torch::stable::Tensor& inputs,
    bool get_counts) {
  using torch::headeronly::ScalarType;
  using torch::stable::Tensor;

  STD_TORCH_CHECK(inputs.is_cuda(), "inputs must be a CUDA tensor");
  STD_TORCH_CHECK(inputs.dim() == 4, "inputs must be [N, 1, H, W] shape");
  STD_TORCH_CHECK(
      inputs.scalar_type() == ScalarType::Byte, "inputs must be a uint8 type");

  const uint32_t N = static_cast<uint32_t>(inputs.size(0));
  const uint32_t C = static_cast<uint32_t>(inputs.size(1));
  const uint32_t H = static_cast<uint32_t>(inputs.size(2));
  const uint32_t W = static_cast<uint32_t>(inputs.size(3));

  STD_TORCH_CHECK(C == 1, "inputs must be [N, 1, H, W] shape");
  STD_TORCH_CHECK((H % 2) == 0, "height must be a even number");
  STD_TORCH_CHECK((W % 2) == 0, "width must be a even number");

  Tensor labels = torch::stable::new_zeros(
      inputs, {N, C, H, W}, ScalarType::Int);
  Tensor counts_init = torch::stable::new_zeros(
      inputs, {N, C, H, W}, ScalarType::Int);
  Tensor counts_final = torch::stable::new_zeros(
      inputs, {N, C, H, W}, ScalarType::Int);

  if (N == 0 || H == 0 || W == 0) {
    // empty input masks, return an empty label and count tensor
    // returned values are [labels, counts]
    return {labels, counts_final};
  }

  dim3 grid = dim3(
      ((W + 1) / 2 + BLOCK_COLS - 1) / BLOCK_COLS,
      ((H + 1) / 2 + BLOCK_ROWS - 1) / BLOCK_ROWS,
      N);
  dim3 block = dim3(BLOCK_COLS, BLOCK_ROWS);
  dim3 grid_count =
      dim3((W + BLOCK_COLS) / BLOCK_COLS, (H + BLOCK_ROWS) / BLOCK_ROWS, N);
  dim3 block_count = dim3(BLOCK_COLS, BLOCK_ROWS);
  torch::stable::accelerator::DeviceGuard device_guard(
      inputs.get_device_index());
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(
      inputs.get_device_index(), &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  cc2d::init_labeling<<<grid, block, 0, stream>>>(
      static_cast<int32_t*>(labels.data_ptr()), W, H);
  cc2d::merge<<<grid, block, 0, stream>>>(
      static_cast<const uint8_t*>(inputs.data_ptr()),
      static_cast<int32_t*>(labels.data_ptr()),
      W,
      H);
  cc2d::compression<<<grid, block, 0, stream>>>(
      static_cast<int32_t*>(labels.data_ptr()), W, H);
  cc2d::final_labeling<<<grid, block, 0, stream>>>(
      static_cast<const uint8_t*>(inputs.data_ptr()),
      static_cast<int32_t*>(labels.data_ptr()),
      W,
      H);

  if (get_counts) {
    cc2d::init_counting<<<grid_count, block_count, 0, stream>>>(
        static_cast<int32_t*>(labels.data_ptr()),
        static_cast<int32_t*>(counts_init.data_ptr()),
        W,
        H);
    cc2d::final_counting<<<grid_count, block_count, 0, stream>>>(
        static_cast<int32_t*>(labels.data_ptr()),
        static_cast<int32_t*>(counts_init.data_ptr()),
        static_cast<int32_t*>(counts_final.data_ptr()),
        W,
        H);
  }

  return {labels, counts_final};
}
