// Fused per-token FP8 cast (local to the Kernel Hub build, no upstream counterpart).
//
// Bit for bit `deep_gemm.utils.per_token_cast_to_fp8`, the PyTorch reference upstream uses in its tests: per row,
// groups of `gran_k` values share a scale `amax / 448` (`amax` clamped to 1e-4), rounded up to a power of two with
// `use_ue8m0`; the values are multiplied by the scale's reciprocal and rounded to E4M3. The scales come out as fp32, or
// with `use_packed_ue8m0` as their biased exponents, four per int32 in K order. One launch reads each value once and
// writes into caller-given outputs, where the reference pads the input, keeps fp32 temporaries the size of the input and
// synchronizes with the host to check the packing.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace {

constexpr int kValuesPerThread = 8;
constexpr int kThreadsPerBlock = 256;

__device__ __forceinline__ float ceil_to_ue8m0(float value) {
    // the power of two at or above `value`, exponent clamped to [1, 254] like the reference
    const uint32_t bits = __float_as_uint(value);
    int exponent = static_cast<int>((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0);
    exponent = min(max(exponent, 1), 254);
    return __uint_as_float(static_cast<uint32_t>(exponent) << 23);
}

template <typename T>
__device__ __forceinline__ void load_values(const T* row, int column, int n, float (&values)[kValuesPerThread]) {
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i)
        values[i] = column + i < n ? static_cast<float>(row[column + i]) : 0.0f;
}

template <>
__device__ __forceinline__ void load_values(const __nv_bfloat16* row, int column, int n,
                                            float (&values)[kValuesPerThread]) {
    if (column + kValuesPerThread <= n) {  // n % 8 == 0 and 16-byte rows: one vector load
        const uint4 packed = *reinterpret_cast<const uint4*>(row + column);
        const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
        for (int i = 0; i < kValuesPerThread / 2; ++i) {
            const float2 pair = __bfloat1622float2(pairs[i]);
            values[2 * i] = pair.x;
            values[2 * i + 1] = pair.y;
        }
    } else {
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i)
            values[i] = column + i < n ? __bfloat162float(row[column + i]) : 0.0f;
    }
}

// One thread per 8 consecutive values of a row; the `kGranK / 8` threads of a scale group are adjacent lanes of a warp.
template <typename T, int kGranK>
__global__ void __launch_bounds__(kThreadsPerBlock) per_token_cast_to_fp8_kernel(
    const T* __restrict__ x, int64_t x_row_stride, __nv_fp8_e4m3* __restrict__ out, int64_t out_row_stride,
    void* __restrict__ out_sf, int64_t sf_row_stride, int n, int num_groups, bool use_ue8m0, bool packed) {
    constexpr int kThreadsPerGroup = kGranK / kValuesPerThread;
    static_assert(kThreadsPerGroup >= 1 && 32 % kThreadsPerGroup == 0, "a scale group must sit within one warp");

    const int64_t row = blockIdx.y;
    const int column = (blockIdx.x * kThreadsPerBlock + threadIdx.x) * kValuesPerThread;
    const int group = column / kGranK;
    // every lane of a warp takes part in the shuffles, so a lane past the row still runs with zeros
    const bool in_row = group < num_groups;

    float values[kValuesPerThread];
    if (in_row)
        load_values(x + row * x_row_stride, column, n, values);
    else
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i)
            values[i] = 0.0f;

    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i)
        amax = fmaxf(amax, fabsf(values[i]));
#pragma unroll
    for (int offset = kThreadsPerGroup / 2; offset > 0; offset /= 2)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, offset));

    // as the reference: clamp, scale = amax / 448 (rounded up to a power of two for UE8M0), multiply by 1 / scale.
    // PyTorch divides a tensor by a scalar as a multiply by the scalar's float reciprocal, so the scale is too.
    float scale = fmaxf(amax, 1e-4f) * (1.0f / 448.0f);
    if (use_ue8m0)
        scale = ceil_to_ue8m0(scale);
    const float inverse = 1.0f / scale;
    if (!in_row)
        return;

    __nv_fp8_e4m3 converted[kValuesPerThread];
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i)
        converted[i] = __nv_fp8_e4m3(values[i] * inverse);
    __nv_fp8_e4m3* out_row = out + row * out_row_stride;
    if (column + kValuesPerThread <= n) {
        *reinterpret_cast<uint2*>(out_row + column) = *reinterpret_cast<const uint2*>(converted);
    } else {
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i)
            if (column + i < n)
                out_row[column + i] = converted[i];
    }

    if (threadIdx.x % kThreadsPerGroup == 0) {
        if (packed)  // the biased exponent, byte `group` of the row's packed int32s (little-endian, K order)
            static_cast<uint8_t*>(out_sf)[row * sf_row_stride * 4 + group] =
                static_cast<uint8_t>(__float_as_uint(scale) >> 23);
        else
            static_cast<float*>(out_sf)[row * sf_row_stride + group] = scale;
    }
}

template <typename T, int kGranK>
void launch(const at::Tensor& x, at::Tensor& out, at::Tensor& out_sf, int num_groups, bool use_ue8m0, bool packed) {
    const int m = static_cast<int>(x.size(0));
    const int n = static_cast<int>(x.size(1));
    const int threads_per_row = (num_groups * kGranK) / kValuesPerThread;
    const dim3 grid((threads_per_row + kThreadsPerBlock - 1) / kThreadsPerBlock, m);
    per_token_cast_to_fp8_kernel<T, kGranK><<<grid, kThreadsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const T*>(x.data_ptr()), x.stride(0),
        reinterpret_cast<__nv_fp8_e4m3*>(out.data_ptr()), out.stride(0), out_sf.data_ptr(), out_sf.stride(0), n,
        num_groups, use_ue8m0, packed);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void deep_gemm_per_token_cast_to_fp8_into(const at::Tensor& x, at::Tensor& out, at::Tensor& out_sf, int64_t gran_k,
                                          bool use_ue8m0, bool use_packed_ue8m0) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.stride(1) == 1, "x must be a 2D CUDA tensor with contiguous rows");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat,
                "x must be bfloat16 or float32, got ", x.scalar_type());
    TORCH_CHECK(gran_k == 32 || gran_k == 128, "gran_k must be 32 or 128, got ", gran_k);
    TORCH_CHECK(!use_packed_ue8m0 || use_ue8m0, "packed scales are UE8M0");
    const int64_t m = x.size(0), n = x.size(1);
    const int64_t num_groups = (n + gran_k - 1) / gran_k;
    TORCH_CHECK(out.scalar_type() == at::kFloat8_e4m3fn && out.dim() == 2 && out.size(0) == m && out.size(1) == n &&
                    out.stride(1) == 1,
                "out must be a float8_e4m3fn (m, n) tensor with contiguous rows");
    if (use_packed_ue8m0) {
        TORCH_CHECK(num_groups % 4 == 0, "packed UE8M0 scales need a multiple of 4 groups per row, got ", num_groups);
        TORCH_CHECK(out_sf.scalar_type() == at::kInt && out_sf.dim() == 2 && out_sf.size(0) == m &&
                        out_sf.size(1) == num_groups / 4 && out_sf.stride(1) == 1,
                    "out_sf must be an int32 (m, ceil(n / gran_k) / 4) tensor with contiguous rows");
    } else {
        TORCH_CHECK(out_sf.scalar_type() == at::kFloat && out_sf.dim() == 2 && out_sf.size(0) == m &&
                        out_sf.size(1) == num_groups && out_sf.stride(1) == 1,
                    "out_sf must be a float32 (m, ceil(n / gran_k)) tensor with contiguous rows");
    }
    TORCH_CHECK(out.device() == x.device() && out_sf.device() == x.device(), "x, out and out_sf must share a device");
    // the vector loads and stores need 16-byte (bf16) and 8-byte (fp8) aligned row starts
    TORCH_CHECK(x.scalar_type() != at::kBFloat16 ||
                    (n % kValuesPerThread == 0 && x.stride(0) % kValuesPerThread == 0 &&
                     reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0),
                "bfloat16 x needs a multiple of 8 columns, a row stride multiple of 8 and a 16-byte aligned start");
    TORCH_CHECK(n % kValuesPerThread == 0 && out.stride(0) % kValuesPerThread == 0 &&
                    reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0,
                "out needs a multiple of 8 columns, a row stride multiple of 8 and an 8-byte aligned start");
    if (m == 0)
        return;
    const c10::cuda::CUDAGuard guard(x.device());
    const int groups = static_cast<int>(num_groups);
    if (x.scalar_type() == at::kBFloat16) {
        if (gran_k == 32)
            launch<__nv_bfloat16, 32>(x, out, out_sf, groups, use_ue8m0, use_packed_ue8m0);
        else
            launch<__nv_bfloat16, 128>(x, out, out_sf, groups, use_ue8m0, use_packed_ue8m0);
    } else {
        if (gran_k == 32)
            launch<float, 32>(x, out, out_sf, groups, use_ue8m0, use_packed_ue8m0);
        else
            launch<float, 128>(x, out, out_sf, groups, use_ue8m0, use_packed_ue8m0);
    }
}
