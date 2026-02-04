#pragma once
#include <cuda.h>

// Provide fallbacks only if CUDA headers don’t define tensor map
#if !defined(CU_TENSOR_MAP_NUM_QWORDS)

// Layout-compatible stand-in
#if defined(__cplusplus) && (__cplusplus >= 201103L)
struct alignas(64) CUtensorMap_st { unsigned long long opaque[16]; };
#else
struct CUtensorMap_st { unsigned long long opaque[16]; };
#endif
typedef CUtensorMap_st CUtensorMap;

// Minimal enums used by create_tensor_map_4D
typedef enum CUtensorMapDataType_enum {
  CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT16 = 6,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 7,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT64 = 8,
  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 10
} CUtensorMapDataType;

typedef enum CUtensorMapInterleave_enum {
  CU_TENSOR_MAP_INTERLEAVE_NONE = 0
} CUtensorMapInterleave;

typedef enum CUtensorMapSwizzle_enum {
  CU_TENSOR_MAP_SWIZZLE_NONE = 0,
  CU_TENSOR_MAP_SWIZZLE_32B,
  CU_TENSOR_MAP_SWIZZLE_64B,
  CU_TENSOR_MAP_SWIZZLE_128B
} CUtensorMapSwizzle;

typedef enum CUtensorMapL2promotion_enum {
  CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
  CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
  CU_TENSOR_MAP_L2_PROMOTION_L2_128B
} CUtensorMapL2promotion;

typedef enum CUtensorMapFloatOOBfill_enum {
  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0
} CUtensorMapFloatOOBfill;

#endif  // !defined(CU_TENSOR_MAP_NUM_QWORDS)
// no declaration of cuTensorMapEncodeTiled here; it’s resolved at runtime