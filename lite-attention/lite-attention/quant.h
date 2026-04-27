/******************************************************************************
 * TMA-based Q/K quantization with mean centering for attention (FP16/BF16)
 * Header file for quant.cu
 ******************************************************************************/

#pragma once

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

// Runtime dispatcher for Q/K quantization
// Launches TMA-based CUDA kernels for efficient quantization
// Tile sizes are determined internally using tile_size_fwd_sm90()
template <typename Element>
void launch_quantize_qk_runtime(
    const Element* Q, const Element* K,
    int8_t* Q_q, int8_t* K_q,
    float* q_scales, float* k_scales, const float* k_mean,
    int batch, int seqlen_q, int seqlen_k, int num_heads,
    int head_dim, bool v_colmajor, bool is_skipable,
    double q_scale,
    cudaStream_t stream);

// Explicit instantiation declarations
extern template void launch_quantize_qk_runtime<cutlass::half_t>(
    const cutlass::half_t*, const cutlass::half_t*,
    int8_t*, int8_t*,
    float*, float*, const float*,
    int, int, int, int,
    int, bool, bool,
    double,
    cudaStream_t);

extern template void launch_quantize_qk_runtime<cutlass::bfloat16_t>(
    const cutlass::bfloat16_t*, const cutlass::bfloat16_t*,
    int8_t*, int8_t*,
    float*, float*, const float*,
    int, int, int, int,
    int, bool, bool,
    double,
    cudaStream_t);