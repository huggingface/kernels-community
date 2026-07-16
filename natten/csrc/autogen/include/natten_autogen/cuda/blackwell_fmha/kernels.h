#pragma once


#include <iostream> 
#include <stdexcept> 
#include <string> 
#include <type_traits> 
#ifdef NATTEN_WITH_CUTLASS
#ifdef NATTEN_WITH_BLACKWELL_FNA
#include <natten/natten.h> 
#include <ATen/ATen.h> 
#include <ATen/cuda/CUDAContext.h> 
#include <c10/cuda/CUDAGuard.h> 
#include <c10/cuda/CUDAStream.h> 
#include <natten/natten.h> 
#include <natten/helpers.h> 
#include <natten/cuda/fmha_blackwell/fmha_forward.cuh> 
namespace natten { 
namespace cuda { 
namespace fmha_blackwell { 

void blackwell_fmha_float16_256x128x32(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_float16_256x128x32_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_float16_256x128x64(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_float16_256x128x64_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_float16_256x128x128(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_float16_256x128x128_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_bfloat16_256x128x32(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_bfloat16_256x128x32_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_bfloat16_256x128x64(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_bfloat16_256x128x64_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_bfloat16_256x128x128(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_bfloat16_256x128x128_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_e4m3_256x128x32(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_e4m3_256x128x32_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_e4m3_256x128x64(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_e4m3_256x128x64_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_e4m3_256x128x128(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_e4m3_256x128x128_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_e5m2_256x128x32(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_e5m2_256x128x32_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_e5m2_256x128x64(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_e5m2_256x128x64_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_e5m2_256x128x128(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_e5m2_256x128x128_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      bool is_causal,
      float attn_scale,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options);


} // namespace natten 
} // namespace cuda 
} // namespace fmha_blackwell 
#endif 
#endif 

