#pragma once

#include <ATen/Tensor.h>


// Launch the forward RMSNorm kernel. Selects the appropriate specialization
// based on tensor dtype and feature size, and commits work on the current MPS
// stream.
void launch_forward_kernel(const at::Tensor& input,
                           const at::Tensor& weight,
                           at::Tensor& output,
                           float epsilon);

// Launch the backward (vector-Jacobian product) RMSNorm kernel. When
// `grad_weight` is non-null the kernel also computes dL/dweight; otherwise the
// weight gradient is skipped.
void launch_backward_kernel(const at::Tensor& input,
                            const at::Tensor& weight,
                            const at::Tensor& grad_output,
                            at::Tensor& grad_input,
                            at::Tensor* grad_weight,
                            float epsilon);


