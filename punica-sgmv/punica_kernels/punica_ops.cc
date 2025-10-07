#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/all.h>

#include <cstdint>

#include "bgmv/bgmv_config.h"
#include "sgmv/sgmv.h"
#include "sgmv_flashinfer/sgmv_config.h"

//namespace
//{

  //====== utils ======

  inline constexpr uint64_t pack_u32(uint32_t a, uint32_t b)
  {
    return (uint64_t(a) << 32) | uint64_t(b);
  }

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM(d, x) \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_EQ(a, b) \
  TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define CHECK_GE(a, b) \
  TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)

  //====== dispatch pytorch dtype ======

#define _DISPATCH_SWITCH(cond, ...) \
  [&]() -> bool {                   \
    switch (cond)                   \
    {                               \
      __VA_ARGS__                   \
    default:                        \
      return false;                 \
    }                               \
  }()

#define _DISPATCH_DTYPE_CASE(enum_type, c_type_, ...) \
  case enum_type:                                     \
  {                                                   \
    using c_type = c_type_;                           \
    return __VA_ARGS__();                             \
  }

#define _DISPATCH_DTYPE_CASES(...)                                 \
  _DISPATCH_DTYPE_CASE(at::ScalarType::Half, nv_half, __VA_ARGS__) \
  _DISPATCH_DTYPE_CASE(at::ScalarType::BFloat16, nv_bfloat16, __VA_ARGS__)

#define DISPATCH_TORCH_DTYPE(scalar_type, ...) \
  _DISPATCH_SWITCH(scalar_type, _DISPATCH_DTYPE_CASES(__VA_ARGS__))

  //====== bgmv ======

  template <typename T>
  inline bool launch_bgmv_kernel(T *Y, const T *X, T **W,
                                 const int64_t *lora_indices,
                                 uint16_t in_features, uint16_t out_features,
                                 int64_t y_offset, int64_t full_y_size,
                                 int64_t batch_size,
                                 int64_t layer_idx, float scale)
  {
    switch (pack_u32(in_features, out_features))
    {
#define CASE_ONESIDE(_T, feat_in, feat_out)                         \
  case pack_u32(feat_in, feat_out):                                 \
    bgmv_kernel<feat_in, feat_out>(Y, X, W, lora_indices, y_offset, \
                                   full_y_size, batch_size,         \
                                   layer_idx, scale);               \
    break;
#define CASE(_T, narrow, wide)  \
  CASE_ONESIDE(T, narrow, wide) \
  CASE_ONESIDE(T, wide, narrow)

      FOR_BGMV_WIDE_NARROW(CASE, _)
#undef CASE
#undef CASE_ONESIDE
    default:
      return false;
    }

    return true;
  }

  void dispatch_bgmv(torch::Tensor y, torch::Tensor x, torch::Tensor w_ptr,
                     torch::Tensor indicies, int64_t layer_idx, double scale)
  {
    CHECK_INPUT(y);
    CHECK_INPUT(x);
    CHECK_INPUT(w_ptr);
    CHECK_INPUT(indicies);

    CHECK_DIM(2, y);
    CHECK_DIM(2, x);
    CHECK_DIM(1, w_ptr);
    CHECK_DIM(1, indicies);

    int64_t B = x.size(0);
    int64_t h_in = x.size(1);
    int64_t h_out = y.size(1);
    CHECK_EQ(indicies.size(0), x.size(0));
    CHECK_EQ(y.size(0), x.size(0));
    bool ok = false;
    if (h_in < 65536 && h_out < 65536)
    {
      switch (x.scalar_type())
      {
      case at::ScalarType::Half:
        ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                static_cast<nv_half *>(x.data_ptr()),
                                static_cast<nv_half **>(w_ptr.data_ptr()),
                                indicies.data_ptr<int64_t>(), h_in, h_out, 0, h_out, B,
                                layer_idx, scale);
        break;
      case at::ScalarType::BFloat16:
        ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                static_cast<nv_bfloat16 *>(x.data_ptr()),
                                static_cast<nv_bfloat16 **>(w_ptr.data_ptr()),
                                indicies.data_ptr<int64_t>(), h_in, h_out, 0, h_out, B,
                                layer_idx, scale);
        break;
      default:
        break;
      }
    }
    TORCH_CHECK(ok, "No suitable kernel.", " h_in=", h_in, " h_out=", h_out,
                " dtype=", x.scalar_type());
  }

  //====== sgmv ======

  void dispatch_sgmv_cutlass(torch::Tensor y, torch::Tensor x, torch::Tensor w_ptr,
                             torch::Tensor s_start, torch::Tensor s_end,
                             torch::Tensor tmp, int64_t layer_idx)
  {
    CHECK_INPUT(y);
    CHECK_INPUT(x);
    CHECK_INPUT(w_ptr);
    CHECK_INPUT(s_start);
    CHECK_INPUT(s_end);
    CHECK_INPUT(tmp);

    CHECK_DIM(2, y);
    CHECK_DIM(2, x);
    CHECK_DIM(1, w_ptr);
    CHECK_DIM(1, s_start);
    CHECK_DIM(1, s_end);
    CHECK_DIM(1, tmp);

    int num_problems = s_start.size(0);
    int d_in = x.size(1);
    int d_out = y.size(1);
    CHECK_EQ(tmp.size(0), static_cast<int64_t>(sgmv_tmp_size(num_problems)));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    bool ok = DISPATCH_TORCH_DTYPE(x.scalar_type(), [&]
                                   { return sgmv<c_type>((c_type *)y.data_ptr(), (c_type *)x.data_ptr(), (c_type **)w_ptr.data_ptr(),
                                                         s_start.data_ptr<int32_t>(), s_end.data_ptr<int32_t>(),
                                                         tmp.data_ptr<uint8_t>(), num_problems, d_in, d_out,
                                                         layer_idx, stream); });
    TORCH_CHECK(ok, "No suitable kernel.", " dtype=", x.scalar_type());
  }

  void dispatch_sgmv_shrink(torch::Tensor y, torch::Tensor x, torch::Tensor w_ptr,
                            torch::Tensor s_start, torch::Tensor s_end, torch::Tensor tmp, int64_t layer_idx)
  {
    CHECK_INPUT(y);
    CHECK_INPUT(x);
    CHECK_INPUT(w_ptr);
    CHECK_INPUT(s_start);
    CHECK_INPUT(s_end);
    CHECK_INPUT(tmp);

    CHECK_DIM(2, y);
    CHECK_DIM(2, x);
    CHECK_DIM(1, w_ptr);
    CHECK_DIM(1, s_start);
    CHECK_DIM(1, s_end);
    CHECK_DIM(1, tmp);

    uint32_t num_problems = s_start.size(0);
    uint32_t d_in = x.size(1);
    uint32_t d_out = y.size(1);
    CHECK_EQ(tmp.scalar_type(), at::ScalarType::Byte);
    CHECK_EQ(tmp.size(0), 8 * 1024 * 1024);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

#define CASE(_T, D_OUT)                                                                      \
  case D_OUT:                                                                                \
    return sgmv_shrink<c_type, D_OUT>(                                                       \
        (c_type *)y.data_ptr(), (c_type *)x.data_ptr(),                                      \
        (c_type **)w_ptr.data_ptr(), s_start.data_ptr<int32_t>(), s_end.data_ptr<int32_t>(), \
        tmp.data_ptr<uint8_t>(), num_problems, d_in, layer_idx, stream);

    bool ok = DISPATCH_TORCH_DTYPE(x.scalar_type(), [&]
                                   {
    switch (d_out) {
      FOR_SGMV_NARROW(CASE, c_type);
      default:
        return false;
    } });

#undef CASE
    TORCH_CHECK(ok, "No suitable kernel.", " dtype=", x.scalar_type(),
                " d_out=", d_out);
  }
//} // namespace

