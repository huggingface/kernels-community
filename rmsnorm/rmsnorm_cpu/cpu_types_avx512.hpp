
#ifndef CPU_TYPES_AVX512_HPP
#define CPU_TYPES_AVX512_HPP

#include <immintrin.h>
#include <torch/all.h>

namespace vec_op_avx512 {

#define DISPATCH_CASE_FLOATING_TYPES(...)            \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)    \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)

#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#ifndef CPU_OP_GUARD
  #define CPU_KERNEL_GUARD_IN(NAME)
  #define CPU_KERNEL_GUARD_OUT(NAME)
#else
  #define CPU_KERNEL_GUARD_IN(NAME) \
    RECORD_FUNCTION(#NAME, c10::ArrayRef<c10::IValue>({}));
  #define CPU_KERNEL_GUARD_OUT(NAME)
#endif

#define FORCE_INLINE __attribute__((always_inline)) inline

namespace {
template <typename T, T... indexes, typename F>
constexpr void unroll_loop_item(std::integer_sequence<T, indexes...>, F&& f) {
  (f(std::integral_constant<T, indexes>{}), ...);
}
};  // namespace

template <typename T, T count, typename F,
          typename = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr void unroll_loop(F&& f) {
  unroll_loop_item(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

template <typename T>
struct Vec {
  constexpr static int get_elem_num() { return T::VEC_ELEM_NUM; }
};

struct FP32Vec8;
struct FP32Vec16;

struct FP16Vec8 : public Vec<FP16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  __m128i reg;

  explicit FP16Vec8(const void* ptr)
      : reg((__m128i)_mm_loadu_si128((__m128i*)ptr)) {}

  explicit FP16Vec8(const FP32Vec8&);

  void save(void* ptr) const { *reinterpret_cast<__m128i*>(ptr) = reg; }
};

struct FP16Vec16 : public Vec<FP16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  __m256i reg;

  // normal load
  explicit FP16Vec16(const void* ptr)
      : reg((__m256i)_mm256_loadu_si256((__m256i*)ptr)) {}

  // non-temporal load
  explicit FP16Vec16(bool, void* ptr)
      : reg(_mm256_stream_load_si256((__m256i*)ptr)) {}

  explicit FP16Vec16(const FP32Vec16&);

  void save(void* ptr) const { _mm256_storeu_si256((__m256i*)ptr, reg); }

  void save(void* ptr, const int elem_num) const {
    constexpr uint32_t M = 0xFFFFFFFF;
    __mmask16 mask = _cvtu32_mask16(M >> (32 - elem_num));
    _mm256_mask_storeu_epi16(ptr, mask, reg);
  }
};

struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  __m128i reg;

  explicit BF16Vec8(const void* ptr)
      : reg((__m128i)_mm_loadu_si128((__m128i*)ptr)) {}

  explicit BF16Vec8(const FP32Vec8&);

  void save(void* ptr) const { *reinterpret_cast<__m128i*>(ptr) = reg; }
};

struct BF16Vec16 : public Vec<BF16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  __m256i reg;

  // normal load
  explicit BF16Vec16(const void* ptr)
      : reg((__m256i)_mm256_loadu_si256((__m256i*)ptr)) {}

  // non-temporal load
  explicit BF16Vec16(bool, void* ptr)
      : reg(_mm256_stream_load_si256((__m256i*)ptr)) {}

  explicit BF16Vec16(const FP32Vec16&);

  void save(void* ptr) const { _mm256_storeu_si256((__m256i*)ptr, reg); }

  void save(void* ptr, const int elem_num) const {
    constexpr uint32_t M = 0xFFFFFFFF;
    __mmask16 mask = _cvtu32_mask16(M >> (32 - elem_num));
    _mm256_mask_storeu_epi16(ptr, mask, reg);
  }
};

struct BF16Vec32 : public Vec<BF16Vec32> {
  constexpr static int VEC_ELEM_NUM = 32;

  __m512i reg;

  explicit BF16Vec32() : reg(_mm512_setzero_si512()) {}

  explicit BF16Vec32(const void* ptr) : reg((__m512i)_mm512_loadu_si512(ptr)) {}

  explicit BF16Vec32(__m512i data) : reg(data) {}

  explicit BF16Vec32(BF16Vec8& vec8_data)
      : reg((__m512i)_mm512_inserti32x4(
            _mm512_inserti32x4(_mm512_inserti32x4(_mm512_castsi128_si512(
                                                      (__m128i)vec8_data.reg),
                                                  (__m128i)vec8_data.reg, 1),
                               (__m128i)vec8_data.reg, 2),
            (__m128i)vec8_data.reg, 3)) {}

  void save(void* ptr) const { *reinterpret_cast<__m512i*>(ptr) = reg; }
};

struct FP32Vec4 : public Vec<FP32Vec4> {
  constexpr static int VEC_ELEM_NUM = 4;
  union AliasReg {
    __m128 reg;
    float values[VEC_ELEM_NUM];
  };

  __m128 reg;

  explicit FP32Vec4(float v) : reg(_mm_set1_ps(v)) {}

  explicit FP32Vec4() : reg(_mm_set1_ps(0.0)) {}

  explicit FP32Vec4(const float* ptr) : reg(_mm_loadu_ps(ptr)) {}

  explicit FP32Vec4(__m128 data) : reg(data) {}

  explicit FP32Vec4(const FP32Vec4& data) : reg(data.reg) {}
};

struct FP32Vec8 : public Vec<FP32Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  union AliasReg {
    __m256 reg;
    float values[VEC_ELEM_NUM];
  };

  __m256 reg;

  explicit FP32Vec8(float v) : reg(_mm256_set1_ps(v)) {}

  explicit FP32Vec8() : reg(_mm256_set1_ps(0.0)) {}

  explicit FP32Vec8(const float* ptr) : reg(_mm256_loadu_ps(ptr)) {}

  explicit FP32Vec8(__m256 data) : reg(data) {}

  explicit FP32Vec8(const FP32Vec8& data) : reg(data.reg) {}

  explicit FP32Vec8(const FP16Vec8& v) : reg(_mm256_cvtph_ps(v.reg)) {}

  explicit FP32Vec8(const BF16Vec8& v)
      : reg(_mm256_castsi256_ps(
            _mm256_bslli_epi128(_mm256_cvtepu16_epi32(v.reg), 2))) {}

  float reduce_sum() const {
    AliasReg ar;
    ar.reg = reg;
    float result = 0;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&result, &ar](int i) { result += ar.values[i]; });

    return result;
  }

  FP32Vec8 exp() const {
    AliasReg ar;
    ar.reg = reg;
    return FP32Vec8(_mm256_set_ps(expf(ar.values[7]), expf(ar.values[6]),
                                  expf(ar.values[5]), expf(ar.values[4]),
                                  expf(ar.values[3]), expf(ar.values[2]),
                                  expf(ar.values[1]), expf(ar.values[0])));
  }

  FP32Vec8 tanh() const {
    AliasReg ar;
    ar.reg = reg;
    return FP32Vec8(_mm256_set_ps(tanhf(ar.values[7]), tanhf(ar.values[6]),
                                  tanhf(ar.values[5]), tanhf(ar.values[4]),
                                  tanhf(ar.values[3]), tanhf(ar.values[2]),
                                  tanhf(ar.values[1]), tanhf(ar.values[0])));
  }

  FP32Vec8 er() const {
    AliasReg ar;
    ar.reg = reg;
    return FP32Vec8(_mm256_set_ps(erf(ar.values[7]), erf(ar.values[6]),
                                  erf(ar.values[5]), erf(ar.values[4]),
                                  erf(ar.values[3]), erf(ar.values[2]),
                                  erf(ar.values[1]), erf(ar.values[0])));
  }

  FP32Vec8 operator*(const FP32Vec8& b) const {
    return FP32Vec8(_mm256_mul_ps(reg, b.reg));
  }

  FP32Vec8 operator+(const FP32Vec8& b) const {
    return FP32Vec8(_mm256_add_ps(reg, b.reg));
  }

  FP32Vec8 operator-(const FP32Vec8& b) const {
    return FP32Vec8(_mm256_sub_ps(reg, b.reg));
  }

  FP32Vec8 operator/(const FP32Vec8& b) const {
    return FP32Vec8(_mm256_div_ps(reg, b.reg));
  }

  void save(float* ptr) const { _mm256_storeu_ps(ptr, reg); }
};


struct FP32Vec16 : public Vec<FP32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  union AliasReg {
    __m512 reg;
    float values[VEC_ELEM_NUM];
  };

  __m512 reg;

  explicit FP32Vec16(float v) : reg(_mm512_set1_ps(v)) {}

  explicit FP32Vec16() : reg(_mm512_set1_ps(0.0)) {}

  // normal load
  explicit FP32Vec16(const float* ptr) : reg(_mm512_loadu_ps(ptr)) {}

  // non-temporal load
  explicit FP32Vec16(bool, void* ptr)
      : reg((__m512)_mm512_stream_load_si512(ptr)) {}

  explicit FP32Vec16(__m512 data) : reg(data) {}

  explicit FP32Vec16(const FP32Vec4& data)
      : reg((__m512)_mm512_inserti32x4(
            _mm512_inserti32x4(
                _mm512_inserti32x4(_mm512_castsi128_si512((__m128i)data.reg),
                                   (__m128i)data.reg, 1),
                (__m128i)data.reg, 2),
            (__m128i)data.reg, 3)) {}

  explicit FP32Vec16(const FP32Vec8& data)
      : reg((__m512)_mm512_inserti32x8(
            _mm512_castsi256_si512((__m256i)data.reg), (__m256i)data.reg, 1)) {}

  explicit FP32Vec16(const BF16Vec16& v)
      : reg(_mm512_castsi512_ps(
            _mm512_bslli_epi128(_mm512_cvtepu16_epi32(v.reg), 2))) {}

  explicit FP32Vec16(const FP16Vec16& v) : reg(_mm512_cvtph_ps(v.reg)) {}

  explicit FP32Vec16(const FP16Vec8& v) : FP32Vec16(FP32Vec8(v)) {}

  explicit FP32Vec16(const BF16Vec8& v) : FP32Vec16(FP32Vec8(v)) {}

  FP32Vec16 operator*(const FP32Vec16& b) const {
    return FP32Vec16(_mm512_mul_ps(reg, b.reg));
  }

  FP32Vec16 operator+(const FP32Vec16& b) const {
    return FP32Vec16(_mm512_add_ps(reg, b.reg));
  }

  FP32Vec16 operator-(const FP32Vec16& b) const {
    return FP32Vec16(_mm512_sub_ps(reg, b.reg));
  }

  FP32Vec16 operator/(const FP32Vec16& b) const {
    return FP32Vec16(_mm512_div_ps(reg, b.reg));
  }

  FP32Vec16 clamp(const FP32Vec16& min, const FP32Vec16& max) const {
    return FP32Vec16(_mm512_min_ps(max.reg, _mm512_max_ps(min.reg, reg)));
  }

  FP32Vec16 max(const FP32Vec16& b) const {
    return FP32Vec16(_mm512_max_ps(reg, b.reg));
  }

  FP32Vec16 max(const FP32Vec16& b, const int elem_num) const {
    constexpr uint32_t M = 0xFFFFFFFF;
    __mmask16 mask = _cvtu32_mask16(M >> (32 - elem_num));
    return FP32Vec16(_mm512_mask_max_ps(reg, mask, reg, b.reg));
  }

  FP32Vec16 min(const FP32Vec16& b) const {
    return FP32Vec16(_mm512_min_ps(reg, b.reg));
  }

  FP32Vec16 min(const FP32Vec16& b, const int elem_num) const {
    constexpr uint32_t M = 0xFFFFFFFF;
    __mmask16 mask = _cvtu32_mask16(M >> (32 - elem_num));
    return FP32Vec16(_mm512_mask_min_ps(reg, mask, reg, b.reg));
  }

  FP32Vec16 abs() const { return FP32Vec16(_mm512_abs_ps(reg)); }

  float reduce_sum() const { return _mm512_reduce_add_ps(reg); }

  float reduce_max() const { return _mm512_reduce_max_ps(reg); }

  float reduce_min() const { return _mm512_reduce_min_ps(reg); }

  template <int group_size>
  float reduce_sub_sum(int idx) {
    static_assert(VEC_ELEM_NUM % group_size == 0);
    constexpr uint32_t base_mask = (0xFFFF >> (16 - group_size));
    __mmask16 mask = _cvtu32_mask16(base_mask << (idx * group_size));
    return _mm512_mask_reduce_add_ps(mask, reg);
  }

  void save(float* ptr) const { _mm512_storeu_ps(ptr, reg); }

  void save(float* ptr, const int elem_num) const {
    constexpr uint32_t M = 0xFFFFFFFF;
    __mmask16 mask = _cvtu32_mask16(M >> (32 - elem_num));
    _mm512_mask_storeu_ps(ptr, mask, reg);
  }
};

template <typename T>
struct VecType {
  using vec_type = void;
};

template <typename T>
using vec_t = typename VecType<T>::vec_type;

template <>
struct VecType<float> {
  using vec_type = FP32Vec16;
};

template <>
struct VecType<c10::Half> {
  using vec_type = FP16Vec16;
};

template <>
struct VecType<c10::BFloat16> {
  using vec_type = BF16Vec16;
};

template <typename T>
void storeFP32(float v, T* ptr) {
  *ptr = v;
}

inline void fma(FP32Vec16& acc, FP32Vec16& a, FP32Vec16& b) {
  acc = acc + a * b;
}

template <>
inline void storeFP32<c10::Half>(float v, c10::Half* ptr) {
  *reinterpret_cast<unsigned short*>(ptr) =
      _cvtss_sh(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

inline FP16Vec8::FP16Vec8(const FP32Vec8& v)
    : reg(_mm256_cvtps_ph(v.reg,
                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)) {}

inline FP16Vec16::FP16Vec16(const FP32Vec16& v)
    : reg(_mm512_cvtps_ph(v.reg,
                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)) {}

template <>
inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16* ptr) {
  *reinterpret_cast<__bfloat16*>(ptr) = _mm_cvtness_sbh(v);
}

inline BF16Vec8::BF16Vec8(const FP32Vec8& v)
    : reg((__m128i)_mm256_cvtneps_pbh(v.reg)) {}

inline BF16Vec16::BF16Vec16(const FP32Vec16& v)
    : reg((__m256i)_mm512_cvtneps_pbh(v.reg)) {}

inline void fma(FP32Vec16& acc, BF16Vec32& a, BF16Vec32& b) {
  acc.reg = _mm512_dpbf16_ps(acc.reg, (__m512bh)a.reg, (__m512bh)b.reg);
}

inline void prefetch(const void* addr) { _mm_prefetch(addr, _MM_HINT_T1); }

inline void mem_barrier() { _mm_mfence(); }
};  // namespace vec_op_avx512

#endif
