// AVX512 implementation - compile with -mavx512f -mavx512bf16
#define CPU_CAPABILITY_AVX512
#include <ATen/ATen.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <c10/core/ScalarType.h>
#include <bitsandbytes_avx512.hpp>
#include <thread>
#include <omp.h>
#include <immintrin.h>

namespace bitsandbytes_cpu
{
    namespace avx512
    {

        static inline __m512 set_nf4_lut()
        {
            return _mm512_set_ps(
                1.0f, 0.7229568362236023, 0.5626170039176941, 0.44070982933044434, 0.33791524171829224, 0.24611230194568634,
                0.16093020141124725, 0.07958029955625534, 0.0f, -0.09105003625154495, -0.18477343022823334,
                -0.28444138169288635, -0.39491748809814453, -0.5250730514526367, -0.6961928009986877, -1.0f);
        }

        static inline __m512 set_fp4_lut()
        {
            return _mm512_set_ps(
                -0.2500f, -0.16666667f, -0.5000f, -0.33333333f, -1.0000f, -0.66666667f, -5.208333333e-03f, 0.0000f, 0.2500f,
                0.16666667f, 0.5000f, 0.33333333f, 1.0000f, 0.66666667f, 5.208333333e-03f, 0.0000f);
        }

#define CVT_BF16_TO_FP32(a) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(a), 16))

        template <int DATA_TYPE>
        inline void unpack_B(
            at::BFloat16 *__restrict__ Btmp, const unsigned char *__restrict__ packed_B,
            const at::BFloat16 *__restrict__ Bs, // scales [K/gs, N] in bf16
            int64_t N, int64_t K, int blocksize, int64_t ldb, int64_t ldb_tmp, int64_t strideBs)
        {
            // Dequant: (w - z) * s -> bf16
            const int64_t K2 = K >> 1; // 2 weights packed per byte
            const int64_t gs2 = blocksize >> 1;
            const int64_t ldb2 = ldb;                          // packed leading dimension (bytes)
            const int64_t ldb_tmp2 = ldb_tmp;                  // output leading dimension in elements
            float *btmp_ptr = reinterpret_cast<float *>(Btmp); // direct bf16 storage

            __m256i mask = _mm256_set1_epi8(0xF);   // low nibble
            __m256i fifteen = _mm256_set1_epi8(15); // shift [-15,15] -> [0,30] for LUT
            __m512i lut = DATA_TYPE == 1
                              ? _mm512_set_epi16(
                                    0x0000, -0x4180, -0x41D5, -0x4100, -0x4155, -0x4080, -0x40D5, -0x4455, 0x0000, 0x3E80,
                                    0x3E2B, 0x3F00, 0x3EAB, 0x3F80, 0x3F2B, 0x3BAB, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                                    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000)
                              : _mm512_set_epi16(
                                    0x0000, 0x3F80, 0x3F39, 0x3F10, 0x3EE2, 0x3EAD, 0x3E7C, 0x3E25, 0x3DA3, 0x0000, -0x4246,
                                    -0x41C3, -0x416E, -0x4136, -0x40FA, -0x40CE, -0x4080, 0x0000, 0x0000, 0x0000, 0x0000,
                                    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000);
            __m512i s_idx1 = _mm512_set_epi32(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8);
            __m512i s_idx0 = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);

            __m512 scale_lo_fp32, scale_hi_fp32;
            __m512 scales[4];

            for (int64_t n = 0; n < N; n += 32)
            {
                for (int64_t k = 0; k < K2; ++k)
                {
                    if (k % gs2 == 0)
                    {
                        const int64_t kgs = k / gs2;
                        // Load 32 scales (bf16) -> two fp32 vectors (first16, second16)
                        __m512i scales_bf16 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(Bs + kgs * strideBs + n));
                        scale_lo_fp32 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(scales_bf16, 0));
                        scale_hi_fp32 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(scales_bf16, 1));
                        scales[0] = _mm512_permutexvar_ps(s_idx0, scale_lo_fp32);
                        scales[1] = _mm512_permutexvar_ps(s_idx1, scale_lo_fp32);
                        scales[2] = _mm512_permutexvar_ps(s_idx0, scale_hi_fp32);
                        scales[3] = _mm512_permutexvar_ps(s_idx1, scale_hi_fp32);
                    }

                    // Load packed 32 bytes => 64 int4
                    __m256i w_u4 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(packed_B + k * ldb2 + n));

                    // Split nibbles
                    __m256i w_lo = w_u4 & mask;
                    __m256i w_hi = _mm256_srli_epi16(w_u4, 4) & mask;

                    // Shift to [0..30] before LUT
                    w_lo = _mm256_add_epi8(w_lo, fifteen);
                    w_hi = _mm256_add_epi8(w_hi, fifteen);

                    // Lookup (w - z) -> bf16 using LUT (process 16-byte halves)
                    __m512i w_lo_bf16 = _mm512_permutexvar_epi16(_mm512_cvtepi8_epi16(w_lo), lut);
                    __m512i w_hi_bf16 = _mm512_permutexvar_epi16(_mm512_cvtepi8_epi16(w_hi), lut);

                    __m512 w_lo_fp32_0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(w_lo_bf16, 0)) * scales[0];
                    __m512 w_hi_fp32_0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(w_lo_bf16, 1)) * scales[1];
                    __m512 w_lo_fp32_1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(w_hi_bf16, 0)) * scales[2];
                    __m512 w_hi_fp32_1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(w_hi_bf16, 1)) * scales[3];

                    // Pack scaled (first 16 cols) then (second 16 cols) to bf16
                    __m512bh packed0 = _mm512_cvtne2ps_pbh(w_hi_fp32_0, w_lo_fp32_0);
                    __m512bh packed1 = _mm512_cvtne2ps_pbh(w_hi_fp32_1, w_lo_fp32_1);

                    // Store: two blocks of 16 bf16 (32 elements) per k iteration
                    _mm512_storeu_si512(btmp_ptr + (k * ldb_tmp2 + n + 0), (__m512i)packed0);
                    _mm512_storeu_si512(btmp_ptr + (k * ldb_tmp2 + n + 16), (__m512i)packed1);
                }
            }
        }

        template <typename scalar_t, int BLOCK_M, int BLOCK_N, int DATA_TYPE>
        struct tinygemm_kernel_nn
        {
            static inline void apply(
                const scalar_t *, const unsigned char *, scalar_t *, const scalar_t *, int64_t, int, int64_t, int64_t, int64_t,
                int64_t, int64_t)
            {
                static_assert(sizeof(scalar_t) == 0, "tinygemm_kernel_nn primary template should never be instantiated");
            }
        };

        // The brgemm will not be used without HAS_TORCH
        template <typename scalar_t, int DATA_TYPE>
        struct brgemm
        {
            static inline void apply(
                const scalar_t *__restrict__ A, const unsigned char *__restrict__ B, scalar_t *__restrict__ C,
                const scalar_t *__restrict__ Bs, scalar_t *__restrict__ Btmp, float *__restrict__ Ctmp, int64_t M, int64_t N,
                int64_t K, int blocksize, int64_t lda, int64_t ldb, int64_t ldc, int64_t strideBs, bool use_brgemm_dequant_out)
            {
                return;
            }
        };

        template <int BLOCK_M, int BLOCK_N, int DATA_TYPE>
        struct tinygemm_kernel_nn<at::BFloat16, BLOCK_M, BLOCK_N, DATA_TYPE>
        {
            static inline void apply(
                const at::BFloat16 *__restrict__ A, const unsigned char *__restrict__ B, at::BFloat16 *__restrict__ C,
                const at::BFloat16 *__restrict__ Bs, int64_t K, int blocksize, int64_t lda, int64_t ldb, int64_t ldc, int64_t strideBs)
            {
                static_assert(BLOCK_N % 32 == 0);
                constexpr int ROWS = BLOCK_M;      // 32
                constexpr int COLS = BLOCK_N / 16; // 2

                // prefetch distance
                constexpr int PREFETCH_SIZE_K = 16 * 4;

                __m512bh va;
                __m512bh vb[COLS];
                __m512 vc[ROWS * COLS];
                __m512 vc_master[ROWS * COLS];

                __m256i mask = _mm256_set1_epi8(0xF); // lower 4 bit
                __m256i fifteen = _mm256_set1_epi8(15);
                __m512i lut = DATA_TYPE == 1
                                  ? _mm512_set_epi16(
                                        0x0000, -0x4180, -0x41D5, -0x4100, -0x4155, -0x4080, -0x40D5, -0x4455, 0x0000, 0x3E80,
                                        0x3E2B, 0x3F00, 0x3EAB, 0x3F80, 0x3F2B, 0x3BAB, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                                        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000)
                                  : _mm512_set_epi16(
                                        0x0000, 0x3F80, 0x3F39, 0x3F10, 0x3EE2, 0x3EAD, 0x3E7C, 0x3E25, 0x3DA3, 0x0000, -0x4246,
                                        -0x41C3, -0x416E, -0x4136, -0x40FA, -0x40CE, -0x4080, 0x0000, 0x0000, 0x0000, 0x0000,
                                        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000);
                __m512 scales[COLS];
                const int64_t K2 = K >> 1;
                const int64_t lda2 = lda >> 1;
                const int64_t ldb2 = ldb;           // ldb * 2 >> 1;
                const int64_t gs2 = blocksize >> 1; // 64 / 2 = 32
                const float *a_ptr = reinterpret_cast<const float *>(A);

                auto loadc = [&](auto i)
                {
                    constexpr int col = i % COLS;
                    vc_master[i] = _mm512_set1_ps(0.f);
                };
                Unroll<ROWS * COLS>{}(loadc);

                auto pre_compute = [&](auto i, int64_t kgs)
                {
                    constexpr int row = i / COLS;
                    constexpr int col = i % COLS;
                    vc[i] = _mm512_set1_ps(0.f); // reset accumulator

                    // load scales
                    if constexpr (row == 0 && col % 2 == 0)
                    {
                        // Bs layout: [K/gs, BLOCK_N] : [strideBs, 1], dtype=bf16
                        __m512i tmp = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(Bs + kgs * strideBs + col * 16));
                        scales[col] = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(tmp, 0));
                        scales[col + 1] = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(tmp, 1));
                    }
                };
                auto compute = [&](auto i, int64_t k)
                {
                    constexpr int row = i / COLS;
                    constexpr int col = i % COLS;

                    if constexpr (col == 0)
                    {
                        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
                    }
                    if constexpr (row == 0 && col % 2 == 0)
                    {
                        __m256i vb_u4 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(B + k * ldb + col * 16));

                        // deinterleave and lookup to BF16
                        __m256i vb_i8_lo = vb_u4 & mask;
                        __m256i vb_i8_hi = _mm256_srli_epi16(vb_u4, 4) & mask;
                        vb_i8_lo = _mm256_add_epi8(vb_i8_lo, fifteen);
                        vb_i8_hi = _mm256_add_epi8(vb_i8_hi, fifteen);
                        vb[col] = (__m512bh)_mm512_permutexvar_epi16(_mm512_cvtepi8_epi16(vb_i8_lo), lut);
                        vb[col + 1] = (__m512bh)_mm512_permutexvar_epi16(_mm512_cvtepi8_epi16(vb_i8_hi), lut);

                        if constexpr (PREFETCH_SIZE_K > 0)
                        {
                            _mm_prefetch(B + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
                        }
                    }
                    vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
                };
                auto post_compute = [&](auto i, int64_t kgs)
                {
                    vc_master[i] = _mm512_fmadd_ps(vc[i], scales[i % COLS], vc_master[i]);
                };
                for (int64_t k = 0; k < K2; k += gs2)
                {
                    Unroll<ROWS * COLS>{}(pre_compute, k / gs2);
                    for (int64_t k_offset = 0; k_offset < gs2; ++k_offset)
                    {
                        Unroll<ROWS * COLS>{}(compute, k + k_offset);
                    }
                    Unroll<ROWS * COLS>{}(post_compute, k / gs2);
                }

                auto storec = [&](auto i)
                {
                    constexpr int row = i / COLS;
                    constexpr int col = i % COLS;
                    if constexpr (col % 2 == 0)
                    {
                        _mm512_storeu_si512(
                            reinterpret_cast<__m512i *>(C + row * ldc + col * 16),
                            (__m512i)(_mm512_cvtne2ps_pbh(vc_master[i + 1], vc_master[i])));
                    }
                };
                Unroll<ROWS * COLS>{}(storec);
            }
        };

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE, DATA_TYPE)                                                       \
    tinygemm_kernel_nn<scalar_t, MB_SIZE, NB_SIZE, DATA_TYPE>::apply(                                                \
        A + mb_start * lda, B + nb_start, C + mb_start * ldc + nb_start, Bs + nb_start, K, blocksize, lda, ldb, ldc, \
        strideBs);

        inline uint16_t float_to_bf16_round(float x)
        {
            uint32_t u;
            std::memcpy(&u, &x, sizeof(u));
            uint32_t lsb = (u >> 16) & 1;
            uint32_t rounding_bias = 0x7fff + lsb;
            u += rounding_bias;
            uint16_t hi = static_cast<uint16_t>(u >> 16);
            // Quiet NaN handling
            if ((u & 0x7f800000) == 0x7f800000 && (u & 0x007fffff))
            {
                hi = 0xffff;
            }
            return hi;
        }

        template <typename scalar_t, typename std::enable_if_t<at::vec::is_reduced_floating_point_v<scalar_t>, int> = 0>
        inline at::vec::Vectorized<scalar_t> convert_from_float_ext(const at::vec::Vectorized<float>& a, const at::vec::Vectorized<float>& b) {
            return at::vec::convert_from_float<scalar_t>(a, b);
        }

        template <>
        inline at::vec::Vectorized<at::BFloat16>
        convert_from_float_ext<at::BFloat16>(const at::vec::Vectorized<float>& a, const at::vec::Vectorized<float>& b) {
            return (__m512i)(_mm512_cvtne2ps_pbh(__m512(b), __m512(a)));
        }

        template <typename scalar_t>
        inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int64_t size) {
            using bVec = at::vec::Vectorized<scalar_t>;
            using fVec = at::vec::Vectorized<float>;
            constexpr int kVecSize = bVec::size();

            int64_t d;
            #pragma GCC unroll 4
            for (d = 0; d <= size - kVecSize; d += kVecSize) {
                fVec data0 = fVec::loadu(input + d);
                fVec data1 = fVec::loadu(input + d + fVec::size());
                bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
                out_vec.store(out + d);
            }
            for (; d < size; ++d) {
                out[d] = static_cast<scalar_t>(input[d]);
            }
        }

        template <int DATA_TYPE>
        struct brgemm<at::BFloat16, DATA_TYPE>
        {
            static inline void apply(
                const at::BFloat16 *__restrict__ A, const unsigned char *__restrict__ B, at::BFloat16 *__restrict__ C,
                const at::BFloat16 *__restrict__ Bs, at::BFloat16 *__restrict__ Btmp, float *__restrict__ Ctmp, int64_t M, int64_t N,
                int64_t K, int blocksize, int64_t lda, int64_t ldb, int64_t ldc, int64_t strideBs, bool use_brgemm_dequant_out)
            {
                constexpr int BLOCK_N = block_size_n();
                const int ldb_tmp = BLOCK_N;
                if (use_brgemm_dequant_out)
                {
                    at::native::cpublas::brgemm(
                        M, N, K, lda, ldb_tmp, BLOCK_N, false, A, Btmp, Ctmp);
                }
                else
                {
                    for (int64_t k = 0; k < K; k += BLOCK_K)
                    {
                        int64_t kb_size = std::min(static_cast<int64_t>(BLOCK_K), K - k);
                        const int64_t kgs = k / blocksize;

                        unpack_B<DATA_TYPE>(
                            Btmp, B + (k >> 1) * ldb, Bs + kgs * strideBs, N, kb_size, blocksize, ldb, ldb_tmp, strideBs);

                        const bool add_C = k != 0;
                        at::native::cpublas::brgemm(
                            M, N, kb_size, lda, ldb_tmp, BLOCK_N, add_C, A + k, Btmp, Ctmp);
                    }
                }

                // copy from Ctmp to C
                for (int64_t m = 0; m < M; ++m)
                {
                    copy_stub(C + m * ldc, Ctmp + m * BLOCK_N, N);
                }
            }
        };

        template <typename scalar_t, int DATA_TYPE>
        void tinygemm_kernel(
            const scalar_t *__restrict__ A, const unsigned char *__restrict__ B, scalar_t *__restrict__ C,
            const scalar_t *__restrict__ Bs, scalar_t *__restrict__ Btmp, float *__restrict__ Ctmp, int64_t M, int64_t N,
            int64_t K, int blocksize, int64_t lda, int64_t ldb, int64_t ldc, int64_t strideBs, bool brg,
            bool use_brgemm_dequant_out = false)
        {
            if (brg)
            {
                brgemm<scalar_t, DATA_TYPE>::apply(
                    A, B, C, Bs, Btmp, Ctmp, M, N, K, blocksize, lda, ldb, ldc, strideBs, use_brgemm_dequant_out);
                return;
            }
            constexpr int64_t BLOCK_M = 4;
            constexpr int64_t BLOCK_N = 64;
            const int64_t MB = div_up(M, BLOCK_M);
            const int64_t NB = div_up(N, BLOCK_N);
            for (int mb = 0; mb < MB; ++mb)
            {
                int64_t mb_start = mb * BLOCK_M;
                int64_t mb_size = std::min(BLOCK_M, M - mb_start);
                for (int64_t nb = 0; nb < NB; ++nb)
                {
                    int64_t nb_start = nb * BLOCK_N;
                    int64_t nb_size = std::min(BLOCK_N, N - nb_start);

                    switch (mb_size << 4 | nb_size >> 4)
                    {
                    // mb_size = 1
                    case 0x12:
                        LAUNCH_TINYGEMM_KERNEL_NN(1, 32, DATA_TYPE);
                        break;
                    case 0x14:
                        LAUNCH_TINYGEMM_KERNEL_NN(1, 64, DATA_TYPE);
                        break;
                    // mb_size = 2
                    case 0x22:
                        LAUNCH_TINYGEMM_KERNEL_NN(2, 32, DATA_TYPE);
                        break;
                    case 0x24:
                        LAUNCH_TINYGEMM_KERNEL_NN(2, 64, DATA_TYPE);
                        break;
                    // mb_size = 3
                    case 0x32:
                        LAUNCH_TINYGEMM_KERNEL_NN(3, 32, DATA_TYPE);
                        break;
                    case 0x34:
                        LAUNCH_TINYGEMM_KERNEL_NN(3, 64, DATA_TYPE);
                        break;
                    // mb_size = 4
                    case 0x42:
                        LAUNCH_TINYGEMM_KERNEL_NN(4, 32, DATA_TYPE);
                        break;
                    case 0x44:
                        LAUNCH_TINYGEMM_KERNEL_NN(4, 64, DATA_TYPE);
                        break;
                    default:
                    {
                        std::fprintf(
                            stderr, "[bitsandbytes] Unexpected block size %lldx%lld\n", (long long)mb_size, (long long)nb_size);
                        std::abort(); // or return; if you prefer silent exit
                    }
                    }
                }
            }
        }

        template <typename T, int DATA_TYPE>
        void gemm_4bit_inference(
            int64_t M, int64_t N, int64_t K, const T *__restrict__ x, const unsigned char *__restrict__ w,
            const T *__restrict__ absmax, T *__restrict__ out, int64_t blocksize, int64_t x_stride, int64_t out_stride)
        {
            constexpr int64_t BLOCK_M = block_size_m(); // 32
            constexpr int64_t BLOCK_N = block_size_n(); // 32
            const int64_t MB = div_up(M, BLOCK_M);      // （x + y -1）/ y, res = 1 when M <= 32
            const int64_t NB = div_up(N, BLOCK_N);
            // TODO: Find better threshold.
            T *Btmp_start = nullptr;
            const bool use_brgemm = M > 4;
            const bool use_brgemm_dequant_out = M > 100;
            if (use_brgemm_dequant_out)
            {
                // Layout: contiguous [N*K] elements, 64-byte aligned for AVX512 loads
                at::Tensor Btmp_t = at::zeros({N, K}, c10::CppTypeToScalarType<T>::value);
                Btmp_start = Btmp_t.data_ptr<T>();
#pragma omp parallel for
                for (int64_t nb = 0; nb < NB; ++nb)
                {
                    int64_t nb_start = nb * BLOCK_N;
                    int64_t nb_size = std::min<int64_t>(N - nb_start, BLOCK_N);
                    T *Btmp = Btmp_start + nb_start * K;
                    for (int64_t k = 0; k < K; k += BLOCK_K)
                    {
                        int64_t kb_size = std::min<int64_t>(BLOCK_K, K - k);
                        int64_t kgs = k / blocksize;
                        int64_t strideBs = N;
                        int64_t ldb = nb_size;
                        const T *Bs = absmax + nb_start;
                        const unsigned char *Bw = reinterpret_cast<const unsigned char *>(w + nb_start * K / 2);
                        unpack_B<DATA_TYPE>(
                            Btmp + k * BLOCK_N, Bw + (k >> 1) * ldb, Bs + kgs * strideBs, nb_size, kb_size, blocksize, ldb,
                            BLOCK_N, strideBs);
                    }
                }
            }
            // l2 cache block for n
            int64_t cache_blocks_nb = get_cache_blocks<T>(BLOCK_N * K);
            parallel_2d(MB, NB, [&](int64_t begin_mb, int64_t end_mb, int64_t begin_nb, int64_t end_nb)
                        {
            // for brgemm, use float32 for accumulate
            alignas(64) float Ctmp[BLOCK_M * BLOCK_N];
            alignas(64) T Btmp_inner[BLOCK_N * BLOCK_K]; // BLOCK_K = 128
            for (int64_t nbb = begin_nb; nbb < end_nb; nbb += cache_blocks_nb) {
                for (int64_t mb = begin_mb; mb < end_mb; ++mb) { // 0-1
                    for (int64_t nb = nbb; nb < std::min(nbb + cache_blocks_nb, end_nb); ++nb) {
                        int64_t mb_start = mb * BLOCK_M; // 0
                        int64_t mb_size = std::min(M - mb_start, BLOCK_M);
                        int64_t nb_start = nb * BLOCK_N;
                        int64_t nb_size = std::min(N - nb_start, BLOCK_N);
                        tinygemm_kernel<T, DATA_TYPE>(
                            /*   A  */ x + mb_start * x_stride,
                            /*   B  */ w + nb_start * K / 2, // divide by 2 since w is u4 packed in u8, K is w.size(1) * 2
                            /*   C  */ out + mb_start * out_stride + nb_start,
                            /*  Bs  */ absmax + nb_start,
                            /* Btmp */ use_brgemm_dequant_out ? Btmp_start + nb_start * K : Btmp_inner,
                            /* Ctmp */ Ctmp,
                            /*   M  */ mb_size,
                            /*   N  */ nb_size,
                            /*   K  */ K,
                            /*  gs  */ blocksize, // blocksize
                            /* lda  */ x_stride,
                            /* ldb  */ nb_size,
                            /* ldc  */ out_stride,
                            /* sBs  */ N,
                            /* brg  */ use_brgemm,
                            /* dequant choice*/ use_brgemm_dequant_out
                        );
                    }
                }
            }
            if (use_brgemm) {
                at::native::cpublas::brgemm_release();
            } });
        }

        //==============================================================
        //                   TEMPLATE DEFINITIONS
        //==============================================================

        template void gemm_4bit_inference<at::BFloat16, FP4>(
            int64_t M, int64_t N, int64_t K, const at::BFloat16 *__restrict__ x, const unsigned char *__restrict__ w,
            const at::BFloat16 *__restrict__ absmax, at::BFloat16 *__restrict__ out, int64_t blocksize, int64_t x_stride, int64_t out_stride);
        template void gemm_4bit_inference<at::BFloat16, NF4>(
            int64_t M, int64_t N, int64_t K, const at::BFloat16 *__restrict__ x, const unsigned char *__restrict__ w,
            const at::BFloat16 *__restrict__ absmax, at::BFloat16 *__restrict__ out, int64_t blocksize, int64_t x_stride, int64_t out_stride);
    } // namespace avx512
} // namespace bitsandbytes_cpu