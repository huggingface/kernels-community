// bitsandbytes MPS Metal kernels - NF4/FP4 codebook definitions and helpers
// Adapted from bitsandbytes CUDA kernels (kernels.cu) for Apple Metal

#pragma once

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Quant type enum (matches bitsandbytes common.h)
// ============================================================================

enum BnBQuantType {
  BNB_FP4 = 1,
  BNB_NF4 = 2,
};

// ============================================================================
// NF4 codebook - 16 values optimized for normal distributions
// Maps 4-bit indices (0-15) to float values in [-1, 1]
// ============================================================================

constant float NF4_CODEBOOK[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f,
};

// ============================================================================
// FP4 codebook - 16 values using sign-magnitude FP4 encoding
// Indices 0-7: non-negative, indices 8-15: negative (bit 3 = sign)
// ============================================================================

constant float FP4_CODEBOOK[16] = {
    0.0f,
    0.005208333333f,
    0.66666667f,
    1.0f,
    0.33333333f,
    0.5f,
    0.16666667f,
    0.25f,
    0.0f,
    -0.005208333333f,
    -0.66666667f,
    -1.0f,
    -0.33333333f,
    -0.5f,
    -0.16666667f,
    -0.25f,
};

// ============================================================================
// Codebook accessor by quant_type template parameter
// ============================================================================

template <int quant_type>
inline constant float* bnb_codebook() {
  if (quant_type == BNB_NF4) {
    return NF4_CODEBOOK;
  } else {
    return FP4_CODEBOOK;
  }
}

// ============================================================================
// NF4 quantization - binary search (matches CUDA dQuantizeNF4)
// Input: normalized value in [-1, 1]
// Output: 4-bit index (0-15)
// ============================================================================

inline uchar quantize_nf4(float x) {
  if (x > 0.03979014977812767f) {
    if (x > 0.3893125355243683f) {
      if (x > 0.6427869200706482f) {
        return (x > 0.8614784181118011f) ? 15 : 14;
      }
      return (x > 0.5016634166240692f) ? 13 : 12;
    }
    if (x > 0.2035212516784668f) {
      return (x > 0.2920137718319893f) ? 11 : 10;
    }
    return (x > 0.1202552504837513f) ? 9 : 8;
  }
  if (x > -0.33967943489551544f) {
    if (x > -0.13791173323988914f) {
      return (x > -0.045525018125772476f) ? 7 : 6;
    }
    return (x > -0.23460740596055984f) ? 5 : 4;
  }
  if (x > -0.6106329262256622f) {
    return (x > -0.4599952697753906f) ? 3 : 2;
  }
  return (x > -0.8480964004993439f) ? 1 : 0;
}

// ============================================================================
// FP4 quantization - binary search (matches CUDA dQuantizeFP4)
// Input: normalized value in [-1, 1]
// Output: 4-bit index (0-15), MSB = sign bit
// ============================================================================

inline uchar quantize_fp4(float x) {
  uchar sign = (x < 0.0f) ? 8 : 0;
  x = metal::abs(x);
  uchar code;
  if (x > 0.29166667f) {
    if (x > 0.75f) {
      code = (x > 0.8333333f) ? 3 : 2;
    } else {
      code = (x > 0.4166667f) ? 5 : 4;
    }
  } else {
    if (x > 0.0859375f) {
      code = (x > 0.20833333f) ? 7 : 6;
    } else {
      code = (x > 0.00260416f) ? 1 : 0;
    }
  }
  return sign | code;
}

// ============================================================================
// Generic quantize dispatch by quant_type
// ============================================================================

template <int quant_type>
inline uchar bnb_quantize_value(float normalized) {
  if (quant_type == BNB_NF4) {
    return quantize_nf4(normalized);
  } else {
    return quantize_fp4(normalized);
  }
}

// ============================================================================
// Dequantize a single 4-bit value using codebook lookup
// ============================================================================

template <int quant_type>
inline float bnb_dequantize_value(uchar nibble) {
  return bnb_codebook<quant_type>()[nibble & 0x0f];
}

// ============================================================================
// BnB 4-bit dequantize for block loader (adapted from MLX affine dequantize)
// Unpacks N values from packed bytes using codebook lookup.
//
// BnB packing: high nibble = first element, low nibble = second element
// Each byte stores 2 4-bit values.
// ============================================================================

template <typename U, int N, int quant_type>
inline void bnb_dequantize(
    const device uint8_t* w,
    U absmax_val,
    threadgroup U* w_local) {
  constant float* codebook = bnb_codebook<quant_type>();

  for (int i = 0; i < N / 2; i++) {
    uint8_t byte_val = w[i];
    uint8_t high = (byte_val >> 4) & 0x0f;
    uint8_t low = byte_val & 0x0f;
    w_local[2 * i] = U(codebook[high]) * absmax_val;
    w_local[2 * i + 1] = U(codebook[low]) * absmax_val;
  }
}
