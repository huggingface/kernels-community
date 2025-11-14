// AVX2 implementation - compile with -mavx2
#include <immintrin.h>
#include <cmath>
#include <omp.h>
#include <vector>
#include "cpu_types_avx2.hpp"

namespace rmsnorm_cpu
{
  namespace avx2
  {

    template <typename scalar_t>
    void rms_norm_impl(scalar_t *__restrict__ out,
                       const scalar_t *__restrict__ input,
                       const scalar_t *__restrict__ weight, const float epsilon,
                       const int num_tokens, const int hidden_size)
    {
      using namespace vec_op_avx2;
      using scalar_vec_t = vec_t<scalar_t>;
      constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();
      TORCH_CHECK(hidden_size % VEC_ELEM_NUM == 0);

#pragma omp parallel for
      for (int i = 0; i < num_tokens; ++i)
      {
        FP32Vec8 variance(0.0);
        auto input_p = input + i * hidden_size;
        auto output_p = out + i * hidden_size;
        for (int j = 0; j < hidden_size; j += VEC_ELEM_NUM)
        {
          scalar_vec_t x(input_p + j);
          FP32Vec8 fp32_x(x);
          variance = variance + fp32_x * fp32_x;
        }

        float s_variance =
            1.0f / sqrtf(variance.reduce_sum() / (float)hidden_size + epsilon);
        FP32Vec8 fp32_s_variance(s_variance);

        for (int j = 0; j < hidden_size; j += VEC_ELEM_NUM)
        {
          scalar_vec_t x(input_p + j);
          scalar_vec_t w(weight + j);

          FP32Vec8 fp32_x(x);
          FP32Vec8 fp32_w(w);

          FP32Vec8 fp32_out = fp32_x * fp32_s_variance * fp32_w;

          scalar_vec_t out(fp32_out);
          out.save(output_p + j);
        }
      }
    }

    void rms_norm(torch::Tensor &out, const torch::Tensor &input, const torch::Tensor &weight,
                  const float epsilon)
    {
      int hidden_size = input.size(-1);
      int num_tokens = input.numel() / hidden_size;

      DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_impl", [&]
                              {
    CPU_KERNEL_GUARD_IN(rms_norm_impl)
    rms_norm_impl(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  weight.data_ptr<scalar_t>(), epsilon, num_tokens,
                  hidden_size);
    CPU_KERNEL_GUARD_OUT(rms_norm_impl) });
    }

    template <typename scalar_t>
    void rms_norm_backward_impl(scalar_t *__restrict__ grad_input,
                                scalar_t *__restrict__ grad_weight,
                                const scalar_t *__restrict__ grad_out,
                                const scalar_t *__restrict__ input,
                                const scalar_t *__restrict__ weight,
                                const float epsilon, const int num_tokens,
                                const int hidden_size)
    {
      using namespace vec_op_avx2;
      using scalar_vec_t = vec_t<scalar_t>;
      constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();
      TORCH_CHECK(hidden_size % VEC_ELEM_NUM == 0);

      int HS = hidden_size;
      int NT = num_tokens;

      // initialize grad_weight to zero
      for (int j = 0; j < HS; ++j)
      {
        grad_weight[j] = (scalar_t)0;
      }

      // Allocate per-thread accumulators and re-run accumulation serially per-thread.
      int max_threads = omp_get_max_threads();
      std::vector<std::vector<scalar_t>> all_acc(max_threads, std::vector<scalar_t>(HS, (scalar_t)0));

      // Parallel over tokens: compute grad_input and accumulate into thread-local
      // buffers for grad_weight to avoid atomics.
#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int start = (NT * tid) / nthreads;
        int end = (NT * (tid + 1)) / nthreads;

        auto &local_acc = all_acc[tid];
        for (int i = start; i < end; ++i)
        {
          const scalar_t *input_p = input + i * HS;
          const scalar_t *grad_out_p = grad_out + i * HS;
          scalar_t *grad_input_p = grad_input + i * HS;

          // compute variance and inv_rms for this token
          FP32Vec8 variance(0.0f);
          for (int j = 0; j < HS; j += VEC_ELEM_NUM)
          {
            scalar_vec_t x(input_p + j);
            FP32Vec8 fp32_x(x);
            variance = variance + fp32_x * fp32_x;
          }

          float inv_rms = 1.0f / sqrtf(variance.reduce_sum() / (float)HS + epsilon);
          FP32Vec8 fp32_inv_rms(inv_rms);

          // compute S = sum_k (g * w * x) for this token
          FP32Vec8 Svec(0.0f);
          for (int j = 0; j < HS; j += VEC_ELEM_NUM)
          {
            scalar_vec_t x(input_p + j);
            scalar_vec_t g_out(grad_out_p + j);
            scalar_vec_t w(weight + j);

            FP32Vec8 fp32_x(x);
            FP32Vec8 fp32_g_out(g_out);
            FP32Vec8 fp32_w(w);

            Svec = Svec + fp32_g_out * fp32_w * fp32_x;
          }
          float S = Svec.reduce_sum();
          float S_over_H = S / (float)HS;
          float inv_rms3 = inv_rms * inv_rms * inv_rms;
          FP32Vec8 fp32_inv_rms3(inv_rms3);
          FP32Vec8 fp32_S_over_H(S_over_H);

          for (int j = 0; j < HS; j += VEC_ELEM_NUM)
          {
            scalar_vec_t x(input_p + j);
            scalar_vec_t g_out(grad_out_p + j);
            scalar_vec_t w(weight + j);

            FP32Vec8 fp32_x(x);
            FP32Vec8 fp32_g_out(g_out);
            FP32Vec8 fp32_w(w);

            // grad_input = g * w * inv_rms - x * inv_rms^3 * (S/H)
            FP32Vec8 term1 = fp32_g_out * fp32_w * fp32_inv_rms;
            FP32Vec8 term2 = fp32_x * fp32_inv_rms3 * fp32_S_over_H;
            FP32Vec8 fp32_grad_in = term1 - term2;
            scalar_vec_t sgrad_in(fp32_grad_in);
            sgrad_in.save(grad_input_p + j);

            // accumulate grad_weight += input * grad_out * inv_rms
            FP32Vec8 prod = fp32_x * fp32_g_out * fp32_inv_rms;
            scalar_vec_t sprod(prod);
            scalar_t tmp[VEC_ELEM_NUM];
            sprod.save(tmp);
            for (int e = 0; e < VEC_ELEM_NUM; ++e)
            {
              local_acc[j + e] += tmp[e];
            }
          }
        }
      }

      // reduce all_acc into grad_weight
      for (int t = 0; t < (int)all_acc.size(); ++t)
      {
        for (int j = 0; j < HS; ++j)
        {
          grad_weight[j] += all_acc[t][j];
        }
      }
    }

    void rms_norm_backward(torch::Tensor &grad_input, torch::Tensor &grad_weight,
                           const torch::Tensor &grad_out, const torch::Tensor &input,
                           const torch::Tensor &weight, const float epsilon)
    {
      int hidden_size = input.size(-1);
      int num_tokens = input.numel() / hidden_size;

      DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_backward_impl", [&]
                              {
    CPU_KERNEL_GUARD_IN(rms_norm_backward_impl)
    rms_norm_backward_impl(grad_input.data_ptr<scalar_t>(),
                           grad_weight.data_ptr<scalar_t>(),
                           grad_out.data_ptr<scalar_t>(),
                           input.data_ptr<scalar_t>(),
                           weight.data_ptr<scalar_t>(), epsilon, num_tokens,
                           hidden_size);
    CPU_KERNEL_GUARD_OUT(rms_norm_backward_impl) });
    }

  } // namespace avx2
} // namespace rmsnorm_cpu
