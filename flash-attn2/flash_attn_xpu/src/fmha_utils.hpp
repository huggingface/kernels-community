#pragma once
#include "torch/all.h"
#include <cute/tensor.hpp>

enum class CutlassType {
  half,
  bfloat16,
};

inline CutlassType aten_to_Cutlass_dtype(const at::Tensor& input) {
  CutlassType cuType;
  if (input.scalar_type() == torch::kHalf) {
    cuType = CutlassType::half;
  } else if (input.scalar_type() == torch::kBFloat16) {
    cuType = CutlassType::bfloat16;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Current cutlass kernel only support half/bf16 data type.");
  }
  return cuType;
}

const int PipelineStages_Decode = 1;
const int PipelineStages_Prefill = 2;

using namespace cute;

struct prefill_policy_head32 {
  using ShapeQK = Shape<_64, _64, _32>;
  using ShapePV = Shape<_64, _32, _64>;
  using ShapeOut = Shape<_64, _32>;
  using SubgroupLayoutQK = Layout<Shape<_4, _1, _1>>;
};

struct prefill_policy_head64 {
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _64>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
};

struct prefill_policy_head96 {
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _96>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
};

struct prefill_policy_head128 {
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _128>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
};

struct prefill_policy_head160 {
  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOut = Shape<_256, _160>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
};

struct prefill_policy_head192 {
  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOut = Shape<_256, _192>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
};

struct prefill_policy_head256 {
  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOut = Shape<_256, _256>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
};

struct decode_policy_head32 {
  using ShapeQK = Shape<_1, _512, _64>;
  using ShapePV = Shape<_1, _32, _512>;
  using ShapeOut = Shape<_1, _32>;
  using SubgroupLayoutQK = Layout<Shape<_1, _8, _1>>;
};

struct decode_policy_head64 {
  using ShapeQK = Shape<_1, _512, _64>;
  using ShapePV = Shape<_1, _32, _512>;
  using ShapeOut = Shape<_1, _64>;
  using SubgroupLayoutQK = Layout<Shape<_1, _8, _1>>;
};

struct decode_policy_head96 {
  using ShapeQK = Shape<_1, _512, _64>;
  using ShapePV = Shape<_1, _32, _512>;
  using ShapeOut = Shape<_1, _96>;
  using SubgroupLayoutQK = Layout<Shape<_1, _8, _1>>;
};

struct decode_policy_head128 {
  using ShapeQK = Shape<_1, _512, _64>;
  using ShapePV = Shape<_1, _32, _512>;
  using ShapeOut = Shape<_1, _128>;
  using SubgroupLayoutQK = Layout<Shape<_1, _8, _1>>;
};

struct decode_policy_head160 {
  using ShapeQK = Shape<_1, _512, _64>;
  using ShapePV = Shape<_1, _32, _512>;
  using ShapeOut = Shape<_1, _160>;
  using SubgroupLayoutQK = Layout<Shape<_1, _8, _1>>;
};

struct decode_policy_head192 {
  using ShapeQK = Shape<_1, _512, _64>;
  using ShapePV = Shape<_1, _32, _512>;
  using ShapeOut = Shape<_1, _192>;
  using SubgroupLayoutQK = Layout<Shape<_1, _8, _1>>;
};

struct decode_policy_head256 {
  using ShapeQK = Shape<_1, _512, _64>;
  using ShapePV = Shape<_1, _32, _512>;
  using ShapeOut = Shape<_1, _256>;
  using SubgroupLayoutQK = Layout<Shape<_1, _8, _1>>;
};
