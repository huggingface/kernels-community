
#ifndef CPU_TYPES_AVX_HPP
#define CPU_TYPES_AVX_HPP

#include <immintrin.h>
#include <torch/all.h>

namespace vec_op_avx2 {

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

  __m256i reg_low;
  __m256i reg_high;

  explicit BF16Vec32(const void* ptr)
      : reg_low(_mm256_loadu_si256((__m256i const*)ptr)),
        reg_high(_mm256_loadu_si256((__m256i const*)ptr + 1)) {}

  explicit BF16Vec32(__m256i low, __m256i high)
      : reg_low(low), reg_high(high) {}

  explicit BF16Vec32(BF16Vec8& vec8_data)
      : reg_low((__m256i)_mm256_inserti32x4(
            _mm256_castsi128_si256((__m128i)vec8_data.reg),
            (__m128i)vec8_data.reg, 1)),
        reg_high((__m256i)_mm256_inserti32x4(
            _mm256_castsi128_si256((__m128i)vec8_data.reg),
            (__m128i)vec8_data.reg, 1)) {}

  void save(void* ptr) const {
    _mm256_storeu_si256((__m256i*)ptr, reg_low);
    _mm256_storeu_si256((__m256i*)ptr + 1, reg_high);
  }
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
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(v.reg), 16))) {}

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
    __m256 reg;
    float values[8];
  };

  __m256 reg_low;
  __m256 reg_high;

  explicit FP32Vec16(float v)
      : reg_low(_mm256_set1_ps(v)), reg_high(_mm256_set1_ps(v)) {}

  explicit FP32Vec16()
      : reg_low(_mm256_set1_ps(0.0)), reg_high(_mm256_set1_ps(0.0)) {}

  explicit FP32Vec16(const float* ptr)
      : reg_low(_mm256_loadu_ps(ptr)), reg_high(_mm256_loadu_ps(ptr + 8)) {}

  explicit FP32Vec16(__m256 low, __m256 high) : reg_low(low), reg_high(high) {}

  explicit FP32Vec16(const FP32Vec4& data)
      : reg_low((__m256)_mm256_inserti128_si256(
            _mm256_castsi128_si256((__m128i)data.reg), (__m128i)data.reg, 1)),
        reg_high((__m256)_mm256_inserti128_si256(
            _mm256_castsi128_si256((__m128i)data.reg), (__m128i)data.reg, 1)) {}

  explicit FP32Vec16(const FP32Vec8& data)
      : reg_low(data.reg), reg_high(data.reg) {}

  explicit FP32Vec16(const FP16Vec16& v) {
    __m128i low = _mm256_extractf128_si256(v.reg, 0);
    __m128i high = _mm256_extractf128_si256(v.reg, 1);

    reg_low = _mm256_cvtph_ps(low);
    reg_high = _mm256_cvtph_ps(high);
  }

  explicit FP32Vec16(const FP16Vec8& v) : FP32Vec16(FP32Vec8(v)) {}

  explicit FP32Vec16(const BF16Vec16& v) {
    __m128i low = _mm256_extractf128_si256(v.reg, 0);
    __m128i high = _mm256_extractf128_si256(v.reg, 1);

    __m256i v_low_epi32 = _mm256_cvtepu16_epi32(low);
    __m256i v_high_epi32 = _mm256_cvtepu16_epi32(high);

    __m256i v_low_shifted = _mm256_bslli_epi128(v_low_epi32, 2);
    __m256i v_high_shifted = _mm256_bslli_epi128(v_high_epi32, 2);

    reg_low = _mm256_castsi256_ps(v_low_shifted);
    reg_high = _mm256_castsi256_ps(v_high_shifted);
  }

  explicit FP32Vec16(const BF16Vec8& v) : FP32Vec16(FP32Vec8(v)) {}

  FP32Vec16 operator*(const FP32Vec16& b) const {
    return FP32Vec16(_mm256_mul_ps(reg_low, b.reg_low),
                     _mm256_mul_ps(reg_high, b.reg_high));
  }

  FP32Vec16 operator+(const FP32Vec16& b) const {
    return FP32Vec16(_mm256_add_ps(reg_low, b.reg_low),
                     _mm256_add_ps(reg_high, b.reg_high));
  }

  FP32Vec16 operator-(const FP32Vec16& b) const {
    return FP32Vec16(_mm256_sub_ps(reg_low, b.reg_low),
                     _mm256_sub_ps(reg_high, b.reg_high));
  }

  FP32Vec16 operator/(const FP32Vec16& b) const {
    return FP32Vec16(_mm256_div_ps(reg_low, b.reg_low),
                     _mm256_div_ps(reg_high, b.reg_high));
  }

  FP32Vec16 max(const FP32Vec16& b) const {
    return FP32Vec16(_mm256_max_ps(reg_low, b.reg_low),
                     _mm256_max_ps(reg_high, b.reg_high));
  }

  float reduce_max() const {
    __m256 v = _mm256_max_ps(reg_low, reg_high);
    // Permute to compare elements within 128-bit lanes
    __m256 v_shuffled = _mm256_permute_ps(
        v, 0b00001011);  // Swap halves within each 128-bit lane
    __m256 v_max = _mm256_max_ps(v, v_shuffled);

    v_shuffled = _mm256_permute_ps(
        v_max, 0b00000001);  // Shuffle elements within each 128-bit lane
    v_max = _mm256_max_ps(v_max, v_shuffled);

    // Permute to compare elements between 128-bit lanes
    v_shuffled =
        _mm256_permute2f128_ps(v_max, v_max, 0b00000001);  // Swap 128-bit lanes
    v_max = _mm256_max_ps(v_max, v_shuffled);

    // At this point, the maximum value is present in all elements of v_max.
    // Extract the first element for the scalar result.
    return _mm256_cvtss_f32(v_max);  // Extract the lowest 32-bit float
  }

  float reduce_sum() const {
    FP32Vec8 low = FP32Vec8(reg_low);
    FP32Vec8 high = FP32Vec8(reg_high);
    return low.reduce_sum() + high.reduce_sum();
  }

  template <int group_size>
  float reduce_sub_sum(int idx) {
    float sum = 0.0;
    static_assert(VEC_ELEM_NUM % group_size == 0);
    constexpr uint32_t base_mask = (0xFFFF >> (16 - group_size));
    uint32_t mask = base_mask << (idx * group_size);

    AliasReg ar;

    auto func = [&sum, &mask, &ar](int i) {
      int flag = mask & 0x1;
      mask = mask >> 1;
      if (flag != 0) sum += ar.values[i];
    };

    ar.reg = reg_low;
    unroll_loop<int, 8>(func);

    ar.reg = reg_high;
    unroll_loop<int, 8>(func);

    return sum;
  }

  void save(float* ptr) const {
    _mm256_storeu_ps(ptr, reg_low);
    _mm256_storeu_ps(ptr + 8, reg_high);
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
  using vec_type = FP32Vec8;
};

template <>
struct VecType<c10::Half> {
  using vec_type = FP16Vec8;
};

template <>
struct VecType<c10::BFloat16> {
  using vec_type = BF16Vec8;
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
    : reg(_mm256_insertf128_si256(
          _mm256_castsi128_si256(FP16Vec8(FP32Vec8(v.reg_low)).reg),
          FP16Vec8(FP32Vec8(v.reg_high)).reg, 1)) {}

template <>
inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16* ptr) {
  c10::BFloat16 __attribute__((__may_alias__))* v_ptr =
      reinterpret_cast<c10::BFloat16*>(&v);
  *ptr = *(v_ptr + 1);
}

__m128i FP32Vec8_to_BF16Vec8_avx2(__m256 a) {
  __m256i ai = _mm256_castps_si256(a);
  ai = _mm256_srli_epi32(ai, 16);
  ai = _mm256_packus_epi32(ai, ai);
  ai = _mm256_permute4x64_epi64(ai, 0b00111001);
  return _mm256_extracti128_si256(ai, 0);
}

inline BF16Vec8::BF16Vec8(const FP32Vec8& v)
    : reg(FP32Vec8_to_BF16Vec8_avx2(v.reg)) {}

inline BF16Vec16::BF16Vec16(const FP32Vec16& v) {
  BF16Vec8 low = BF16Vec8(FP32Vec8(v.reg_low));
  BF16Vec8 high = BF16Vec8(FP32Vec8(v.reg_high));
  reg = _mm256_insertf128_si256(_mm256_castsi128_si256(low.reg), high.reg, 1);
}


inline void prefetch(const void* addr) { _mm_prefetch(addr, _MM_HINT_T1); }

inline void mem_barrier() { _mm_mfence(); }
};  // namespace vec_op_avx2

#endif
