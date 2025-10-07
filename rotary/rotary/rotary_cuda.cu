/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <torch/all.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

void _apply_rotary(torch::Tensor const &x1, torch::Tensor const &x2,
                       torch::Tensor const &cos, torch::Tensor const &sin,
                       torch::Tensor &out1, torch::Tensor &out2,
                       bool const conj) {
    auto iter = at::TensorIteratorConfig()
        .add_output(out1)
        .add_output(out2)
        .add_input(x1)
        .add_input(x2)
        .add_input(cos)
        .add_input(sin)
        .check_all_same_dtype(false)
        .promote_inputs_to_common_dtype(false)
        .build();

    if (!conj) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, x1.scalar_type(), "rotary_kernel", [&] {
            at::native::gpu_kernel_multiple_outputs(
                iter, [] GPU_LAMBDA (scalar_t x1, scalar_t x2, scalar_t cos,
                                    scalar_t sin) -> thrust::tuple<scalar_t, scalar_t> {
                scalar_t out1 = float(x1) * float(cos) - float(x2) * float(sin);
                scalar_t out2 = float(x1) * float(sin) + float(x2) * float(cos);
                return {out1, out2};
            });
        });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, x1.scalar_type(), "rotary_kernel", [&] {
            at::native::gpu_kernel_multiple_outputs(
                iter, [] GPU_LAMBDA (scalar_t x1, scalar_t x2, scalar_t cos,
                                    scalar_t sin) -> thrust::tuple<scalar_t, scalar_t> {
                scalar_t out1 = float(x1) * float(cos) + float(x2) * float(sin);
                scalar_t out2 = -float(x1) * float(sin) + float(x2) * float(cos);
                return {out1, out2};
            });
        });
    }
}
