// AVX512 implementation - compile with -mavx512f -mavx512bf16
#include <immintrin.h>
#include <cmath>
#include <omp.h>
#include <vector>
#include "cpu_types_avx512.hpp"

namespace rmsnorm_cpu
{
    namespace avx512
    {

        void rms_norm(torch::Tensor &out, const torch::Tensor &input, const torch::Tensor &weight,
                      const float epsilon);

        void rms_norm_backward(torch::Tensor &grad_input, torch::Tensor &grad_weight,
                               const torch::Tensor &grad_out, const torch::Tensor &input,
                               const torch::Tensor &weight, const float epsilon);

    } // namespace avx512
} // namespace rmsnorm_cpu
