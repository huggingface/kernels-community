// SPDX-License-Identifier: Apache-2.0
// MegaBlocks CPU Fused MoE Implementation
//
// Optimized with brgemm (batch-reduced GEMM) for Intel AMX

#include "moe_ops.h"
#include <ATen/Parallel.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <torch/extension.h>
#include <cmath>
#include <algorithm>

namespace megablocks {
namespace cpu {

namespace {

// Determine if we should use brgemm based on data type
template <typename scalar_t>
inline bool can_use_brgemm(int64_t M) {
  return false;  // Default: disable
}

template <>
inline bool can_use_brgemm<at::BFloat16>(int64_t M) {
  return M > 4;  // brgemm efficient for larger M
}

template <>
inline bool can_use_brgemm<at::Half>(int64_t M) {
  return true;
}

// Vectorized copy
template <typename scalar_t>
inline void copy_vec(scalar_t* __restrict__ dst, const scalar_t* __restrict__ src, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  int64_t d = 0;
  for (; d <= size - Vec::size(); d += Vec::size()) {
    Vec::loadu(src + d).store(dst + d);
  }
  for (; d < size; ++d) {
    dst[d] = src[d];
  }
}

// Add bias (vectorized)
template <typename scalar_t>
inline void add_bias_vec(scalar_t* __restrict__ data, const scalar_t* __restrict__ bias, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  int64_t d = 0;
  for (; d <= size - Vec::size(); d += Vec::size()) {
    (Vec::loadu(data + d) + Vec::loadu(bias + d)).store(data + d);
  }
  for (; d < size; ++d) {
    data[d] += bias[d];
  }
}

// Swiglu activation (vectorized)
template <typename scalar_t>
inline void swigluoai_act(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    int64_t size,
    float alpha,
    float limit) {
  using Vec = at::vec::Vectorized<scalar_t>;
  const Vec one = Vec(scalar_t(1));
  const Vec limit_v = Vec(scalar_t(limit));
  const Vec nlimit_v = Vec(scalar_t(-limit));
  const Vec alpha_v = Vec(scalar_t(alpha));

  int64_t d = 0;
  for (; d <= size - Vec::size(); d += Vec::size()) {
    Vec g = Vec::loadu(gate + d);
    Vec u = Vec::loadu(up + d);
    
    g = at::vec::minimum(g, limit_v);
    u = at::vec::minimum(limit_v, at::vec::maximum(nlimit_v, u));
    
    Vec glu = g / (one + (g * alpha_v).neg().exp());
    ((u + one) * glu).store(output + d);
  }
  
  for (; d < size; ++d) {
    scalar_t g = std::min(gate[d], scalar_t(limit));
    scalar_t u = std::clamp(up[d], scalar_t(-limit), scalar_t(limit));
    scalar_t glu = g / (scalar_t(1) + std::exp(-g * alpha));
    output[d] = (u + scalar_t(1)) * glu;
  }
}

// SiLU activation (vectorized)
template <typename scalar_t>
inline void silu_act(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  const Vec one = Vec(scalar_t(1));

  int64_t d = 0;
  for (; d <= size - Vec::size(); d += Vec::size()) {
    Vec g = Vec::loadu(gate + d);
    Vec u = Vec::loadu(up + d);
    Vec silu = g / (one + g.neg().exp());
    (silu * u).store(output + d);
  }
  
  for (; d < size; ++d) {
    scalar_t silu = gate[d] / (scalar_t(1) + std::exp(-gate[d]));
    output[d] = silu * up[d];
  }
}

} // namespace

// Main fused MoE kernel with brgemm
torch::Tensor fused_moe_cpu(
    torch::Tensor hidden_states,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    const c10::optional<torch::Tensor>& w1_bias,
    const c10::optional<torch::Tensor>& w2_bias,
    const std::string& activation,
    float alpha,
    float limit,
    bool is_interleaved) {
  
  TORCH_CHECK(hidden_states.is_cpu(), "hidden_states must be CPU");
  TORCH_CHECK(w1.is_cpu() && w2.is_cpu(), "weights must be CPU");
  
  const int64_t num_tokens = hidden_states.size(0);
  const int64_t hidden_size = hidden_states.size(1);
  const int64_t num_experts = w1.size(0);
  const int64_t inter_size = w2.size(1);
  const int64_t topk = topk_weights.size(1);
  
  auto output = torch::zeros_like(hidden_states);
  
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, hidden_states.scalar_type(), "fused_moe_cpu", [&] {
    
    const scalar_t* h_ptr = hidden_states.data_ptr<scalar_t>();
    const scalar_t* w1_ptr = w1.data_ptr<scalar_t>();
    const scalar_t* w2_ptr = w2.data_ptr<scalar_t>();
    const float* wt_ptr = topk_weights.data_ptr<float>();
    const int32_t* id_ptr = topk_ids.data_ptr<int32_t>();
    scalar_t* out_ptr = output.data_ptr<scalar_t>();
    
    const scalar_t* w1_bias_ptr = w1_bias.has_value() ? w1_bias.value().data_ptr<scalar_t>() : nullptr;
    const scalar_t* w2_bias_ptr = w2_bias.has_value() ? w2_bias.value().data_ptr<scalar_t>() : nullptr;
    
    const bool use_swiglu = (activation == "swigluoai");
    const bool use_brgemm = can_use_brgemm<scalar_t>(num_tokens);
    
    // Parallel over tokens (OpenMP via at::parallel_for)
    at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
      // Thread-local buffers
      std::vector<scalar_t> gate_buf(inter_size);
      std::vector<scalar_t> up_buf(inter_size);
      std::vector<scalar_t> activated_buf(inter_size);
      std::vector<scalar_t> expert_out_buf(hidden_size);
      
      for (int64_t tok = begin; tok < end; ++tok) {
        const scalar_t* h_tok = h_ptr + tok * hidden_size;
        scalar_t* out_tok = out_ptr + tok * hidden_size;
        
        // Process top-k experts
        for (int64_t k = 0; k < topk; ++k) {
          int32_t expert_id = id_ptr[tok * topk + k];
          float weight = wt_ptr[tok * topk + k];
          
          if (expert_id < 0 || expert_id >= num_experts) continue;
          
          const scalar_t* w1_exp = w1_ptr + expert_id * hidden_size * 2 * inter_size;
          const scalar_t* w2_exp = w2_ptr + expert_id * inter_size * hidden_size;
          
          // ========== GEMM 1: hidden @ w1 -> [gate; up] or [g0,u0,g1,u1,...] ==========
          if (use_brgemm) {
            // Use brgemm for better performance with AMX
            // gate_up layout: [2*inter_size], split to gate[inter_size] and up[inter_size]
            
            // Compute gate = h @ w1[:, :inter_size*2:2] (even columns)
            // Compute up = h @ w1[:, 1:inter_size*2:2] (odd columns)
            
            if (is_interleaved) {
              // w1 layout: [h0_g, h0_u, h1_g, h1_u, ...] need to deinterleave
              // For brgemm, we need contiguous layout, so use regular matmul here
              // TODO: optimize by pre-transposing weights to [2, inter_size, hidden_size]
              
              std::vector<scalar_t> gate_up_buf(2 * inter_size);
              
              // gate_up = h @ w1
              for (int64_t i = 0; i < 2 * inter_size; ++i) {
                scalar_t sum = 0;
                for (int64_t j = 0; j < hidden_size; ++j) {
                  sum += h_tok[j] * w1_exp[j * 2 * inter_size + i];
                }
                gate_up_buf[i] = sum;
              }
              
              // Deinterleave
              for (int64_t i = 0; i < inter_size; ++i) {
                gate_buf[i] = gate_up_buf[2*i];
                up_buf[i] = gate_up_buf[2*i + 1];
              }
            } else {
              // Stacked layout: [gate; up]
              at::native::cpublas::brgemm(
                  /* M */ 1,
                  /* N */ inter_size,
                  /* K */ hidden_size,
                  /* lda */ hidden_size,
                  /* ldb */ inter_size,
                  /* ldc */ inter_size,
                  /* add_C */ false,
                  /* A */ h_tok,
                  /* B */ w1_exp,
                  /* C */ gate_buf.data());
              
              at::native::cpublas::brgemm(
                  /* M */ 1,
                  /* N */ inter_size,
                  /* K */ hidden_size,
                  /* lda */ hidden_size,
                  /* ldb */ inter_size,
                  /* ldc */ inter_size,
                  /* add_C */ false,
                  /* A */ h_tok,
                  /* B */ w1_exp + hidden_size * inter_size,
                  /* C */ up_buf.data());
            }
          } else {
            // Fallback: manual GEMM
            std::vector<scalar_t> gate_up_buf(2 * inter_size, 0);
            
            for (int64_t i = 0; i < 2 * inter_size; ++i) {
              scalar_t sum = 0;
              for (int64_t j = 0; j < hidden_size; ++j) {
                sum += h_tok[j] * w1_exp[j * 2 * inter_size + i];
              }
              gate_up_buf[i] = sum;
            }
            
            if (is_interleaved) {
              for (int64_t i = 0; i < inter_size; ++i) {
                gate_buf[i] = gate_up_buf[2*i];
                up_buf[i] = gate_up_buf[2*i + 1];
              }
            } else {
              copy_vec(gate_buf.data(), gate_up_buf.data(), inter_size);
              copy_vec(up_buf.data(), gate_up_buf.data() + inter_size, inter_size);
            }
          }
          
          // Add w1 bias
          if (w1_bias_ptr) {
            const scalar_t* w1_bias_exp = w1_bias_ptr + expert_id * 2 * inter_size;
            if (is_interleaved) {
              for (int64_t i = 0; i < inter_size; ++i) {
                gate_buf[i] += w1_bias_exp[2*i];
                up_buf[i] += w1_bias_exp[2*i + 1];
              }
            } else {
              add_bias_vec(gate_buf.data(), w1_bias_exp, inter_size);
              add_bias_vec(up_buf.data(), w1_bias_exp + inter_size, inter_size);
            }
          }
          
          // ========== Activation ==========
          if (use_swiglu) {
            swigluoai_act(activated_buf.data(), gate_buf.data(), up_buf.data(),
                          inter_size, alpha, limit);
          } else {
            silu_act(activated_buf.data(), gate_buf.data(), up_buf.data(), inter_size);
          }
          
          // ========== GEMM 2: activated @ w2 -> expert_out ==========
          if (use_brgemm) {
            at::native::cpublas::brgemm(
                /* M */ 1,
                /* N */ hidden_size,
                /* K */ inter_size,
                /* lda */ inter_size,
                /* ldb */ hidden_size,
                /* ldc */ hidden_size,
                /* add_C */ false,
                /* A */ activated_buf.data(),
                /* B */ w2_exp,
                /* C */ expert_out_buf.data());
          } else {
            std::fill(expert_out_buf.begin(), expert_out_buf.end(), scalar_t(0));
            for (int64_t i = 0; i < hidden_size; ++i) {
              scalar_t sum = 0;
              for (int64_t j = 0; j < inter_size; ++j) {
                sum += activated_buf[j] * w2_exp[j * hidden_size + i];
              }
              expert_out_buf[i] = sum;
            }
          }
          
          // Add w2 bias
          if (w2_bias_ptr) {
            const scalar_t* w2_bias_exp = w2_bias_ptr + expert_id * hidden_size;
            add_bias_vec(expert_out_buf.data(), w2_bias_exp, hidden_size);
          }
          
          // Accumulate weighted output
          scalar_t weight_s = scalar_t(weight);
          for (int64_t i = 0; i < hidden_size; ++i) {
            out_tok[i] += expert_out_buf[i] * weight_s;
          }
        }
      }
    });
    
    // Release brgemm resources if used
    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
  
  return output;
}

} // namespace cpu
} // namespace megablocks
