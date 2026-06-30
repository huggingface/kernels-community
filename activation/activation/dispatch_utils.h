/*
 * Stable-ABI dtype dispatch for the activation kernels.
 *
 * The non-stable build used ATen's AT_DISPATCH_* machinery (from
 * <torch/all.h>), which is not part of Torch's stable ABI. Here we provide a
 * minimal, header-only replacement that switches on the runtime dtype and
 * binds `scalar_t` to the corresponding header-only scalar type before
 * invoking the provided lambda.
 */
#pragma once

#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>
#include <torch/headeronly/util/Exception.h>

#define VLLM_DISPATCH_CASE_FLOATING_TYPE(ENUM, TYPE, ...) \
  case torch::headeronly::ScalarType::ENUM: {             \
    using scalar_t = TYPE;                                \
    return __VA_ARGS__();                                 \
  }

// Dispatch over the floating-point types supported by the activation kernels
// (float32, float16, bfloat16).
#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                         \
  [&] {                                                                       \
    switch (TYPE) {                                                           \
      VLLM_DISPATCH_CASE_FLOATING_TYPE(Float, float, __VA_ARGS__)             \
      VLLM_DISPATCH_CASE_FLOATING_TYPE(Half, torch::headeronly::Half,         \
                                       __VA_ARGS__)                           \
      VLLM_DISPATCH_CASE_FLOATING_TYPE(BFloat16, torch::headeronly::BFloat16, \
                                       __VA_ARGS__)                           \
      default:                                                                \
        STD_TORCH_CHECK(false, NAME " not implemented for the given dtype");  \
    }                                                                         \
  }()
