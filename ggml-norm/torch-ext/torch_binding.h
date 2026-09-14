#pragma once

#include <torch/torch.h>

// `rms_norm(x) * weight` over the last axis of `x`, in one dispatch -- ggml's
// `kernel_rms_norm_mul_f32`, which folds the weight multiply into the normalisation and reads the
// row as float4. Eager torch spells the same thing as five dispatches.
//
// `weight` is one row, broadcast over every row of `x`. Models disagree on what it means -- `x * w`
// for most, `x * (1 + w)` for the zero-centered ones -- and this op takes it already resolved.
at::Tensor rms_norm(const at::Tensor &x, const at::Tensor &weight, double eps);
