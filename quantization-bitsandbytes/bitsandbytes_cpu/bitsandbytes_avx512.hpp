// AVX512 implementation - compile with -mavx512f -mavx512bf16
#include <algorithm>
#include <cmath>
#include <common.h>
#include <cstdint>
#include <cstring>
#include <thread>
#include <type_traits>
#include <omp.h>
#include <immintrin.h>

namespace bitsandbytes_cpu
{
    namespace avx512
    {
// amx-bf16
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32
// work around compiler internal error
#define BLOCK_K 128 // 4 * TILE_K

        // block size for AMX gemm
        constexpr int block_size_m() { return 2 * TILE_M; }

        constexpr int block_size_n() { return 2 * TILE_N; }

        template <typename T>
        inline int get_cache_blocks(int chunk_size)
        {
            // L2 2MB and ratio of 50%
            const int L2_size = 2048 * 1024 >> 1;
            return std::max(1, int(L2_size / (chunk_size * sizeof(T))));
        }

// forced unroll for perf critical path
#if __has_attribute(always_inline)
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define ALWAYS_INLINE inline
#endif

        template <int n>
        struct Unroll
        {
            template <typename Func, typename... Args>
            ALWAYS_INLINE void operator()(const Func &f, Args... args) const
            {
                Unroll<n - 1>{}(f, args...);
                f(std::integral_constant<int, n - 1>{}, args...);
            }
        };

        template <>
        struct Unroll<1>
        {
            template <typename Func, typename... Args>
            ALWAYS_INLINE void operator()(const Func &f, Args... args) const
            {
                f(std::integral_constant<int, 0>{}, args...);
            }
        };

        template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
        inline T div_up(T x, T y)
        {
            return (x + y - 1) / y;
        }

        inline int adjust_num_threads(int m)
        {
            int actual_nth = omp_get_max_threads();
            if (m == 1)
                return actual_nth;
            return std::max(1, (actual_nth >> 1) * 2);
        }

        template <typename func_t>
        inline void parallel_2d(int m, int n, const func_t &f)
        {
            int nth = adjust_num_threads(m);
            float r = float(m) / n;
            int nth_m = std::ceil(std::sqrt(r * nth));
            int nth_n = 1;
            for (; nth_m > 0; --nth_m)
            {
                nth_n = nth / nth_m;
                if (nth_m * nth_n == nth)
                {
                    break;
                }
            }
#pragma omp parallel num_threads(nth)
            {
                int ith = omp_get_thread_num();
                int ith_m = ith / nth_n;
                int ith_n = ith % nth_n;

                int thread_block_m = div_up(m, nth_m);
                int thread_block_n = div_up(n, nth_n);

                int begin_m = ith_m * thread_block_m;
                int end_m = std::min(m, begin_m + thread_block_m);
                int begin_n = ith_n * thread_block_n;
                int end_n = std::min(n, begin_n + thread_block_n);

                f(begin_m, end_m, begin_n, end_n);
            }
        }

        typedef enum DataType_t
        {
            NF4 = 0,
            FP4 = 1,
        } DataType_t;

        struct fp16_t
        {
            uint16_t v;
        };

        struct bf16_t
        {
            uint16_t v;
        };

        static inline bf16_t float_to_bf16(float x)
        {
            uint32_t bits;
            std::memcpy(&bits, &x, 4);
            uint32_t r = bits + 0x7FFF + ((bits >> 16) & 1);
            return bf16_t{static_cast<uint16_t>(r >> 16)};
        }

        static float bf16_to_float(uint16_t bf16)
        {
            uint32_t bits = (uint32_t)bf16 << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            return f;
        }

        static inline fp16_t float_to_fp16(float x)
        {
            uint32_t bits;
            std::memcpy(&bits, &x, 4);
            uint32_t sign = (bits >> 31) & 0x1;
            uint32_t exp = (bits >> 23) & 0xFF;
            uint32_t mant = bits & 0x7FFFFF;

            uint16_t h;
            if (exp == 0xFF)
            {                                       // Inf / NaN
                uint16_t mant16 = mant ? 0x200 : 0; // quiet NaN: set MSB of mantissa
                h = (sign << 15) | (0x1F << 10) | mant16;
            }
            else if (exp > 0x70 + 0x1E)
            {                                    // overflow: exp_f -127 +15 > 30  (exp_f > 142)
                h = (sign << 15) | (0x1F << 10); // Inf
            }
            else if (exp < 0x71)
            { // subnormal or zero (exp_f < 113)
                if (exp < 0x67)
                { // too small -> zero (exp_f < 103)
                    h = (sign << 15);
                }
                else
                {
                    // subnormal: implicit leading 1
                    uint32_t shift = 0x71 - exp;
                    uint32_t mant_with_hidden = mant | 0x800000;
                    // add rounding bias before shifting (23-10 =13 bits to drop + shift)
                    uint32_t rounded = (mant_with_hidden + (1u << (shift + 12))) >> (shift + 13);
                    h = (sign << 15) | (uint16_t)rounded;
                }
            }
            else
            {
                // normalized
                uint32_t exp_h = exp - 127 + 15;
                // round mantissa: add 2^(23-10-1) = 0x1000
                uint32_t mant_rounded = mant + 0x00001000;
                if (mant_rounded & 0x00800000)
                { // mantissa overflow after rounding
                    mant_rounded = 0;
                    ++exp_h;
                    if (exp_h >= 0x1F)
                    { // overflow to Inf
                        h = (sign << 15) | (0x1F << 10);
                        return fp16_t{h};
                    }
                }
                h = (sign << 15) | ((uint16_t)exp_h << 10) | ((uint16_t)(mant_rounded >> 13));
            }
            return fp16_t{h};
        }

#ifdef _MSC_VER
#include <intrin.h>

        static inline bool has_avx512f()
        {
            static bool v = []
            {
                int info[4];
                __cpuidex(info, 7, 0);
                return (info[1] & (1 << 16)) != 0; // EBX bit16 AVX512F
            }();
            return v;
        }
        static inline bool has_avx512bf16()
        {
            static bool v = []
            {
                int info[4];
                __cpuidex(info, 7, 1);
                return (info[0] & (1 << 5)) != 0; // EAX bit5 AVX512_BF16
            }();
            return v;
        }
#else
        static inline bool has_avx512f()
        {
            static const bool supported_avx512f = __builtin_cpu_supports("avx512f");
            return supported_avx512f;
        }
        static inline bool has_avx512bf16()
        {
            static const bool supported_avx512bf16 = __builtin_cpu_supports("avx512bf16");
            return supported_avx512bf16;
        }
#endif

        template <typename T, int DATA_TYPE>
        void gemm_4bit_inference(
            int64_t M, int64_t N, int64_t K, const T *__restrict__ x, const unsigned char *__restrict__ w,
            const T *__restrict__ absmax, T *__restrict__ out, int64_t blocksize, int64_t x_stride, int64_t out_stride);
    } // namespace avx512
} // namespace bitsandbytes_cpu