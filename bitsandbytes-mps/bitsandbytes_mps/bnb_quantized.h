// bitsandbytes MPS Metal kernels - 4-bit quantized operations
// Adapted from MLX quantized.h for bitsandbytes NF4/FP4 format.
//
// Key differences from MLX affine quantization:
//   MLX:  dequant(q) = scale * q_int + bias     (linear mapping)
//   BnB:  dequant(q) = codebook[q_int] * absmax  (lookup-based)
//
// Packing format:
//   BnB: high nibble = first element, low nibble = second element
//   Two 4-bit values per byte, pack_factor = 2

#include <metal_simdgroup>
#include <metal_stdlib>

#include "bnb_types.h"

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;

// ============================================================================
// BnBQuantizedBlockLoader
//
// Loads blocks of BnB 4-bit packed weights into threadgroup memory,
// performing codebook dequantization on the fly.
// Adapted from MLX QuantizedBlockLoader.
//
// Template parameters:
//   T            - output scalar type (float16_t, bfloat16_t, float)
//   BROWS        - number of rows in the tile
//   BCOLS        - number of columns in the tile (unpacked)
//   dst_ld       - leading dimension of destination (threadgroup memory)
//   reduction_dim - 0 for K along rows, 1 for K along columns
//   tgp_size     - threads per threadgroup
//   blocksize    - BnB blocksize (elements per absmax value)
//   quant_type   - BNB_FP4 (1) or BNB_NF4 (2)
// ============================================================================

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short blocksize,
    int quant_type>
struct BnBQuantizedBlockLoader {
  static_assert(
      BCOLS <= blocksize,
      "The blocksize should be larger than the tile columns");
  static_assert(
      blocksize % BCOLS == 0,
      "The blocksize should be divisible by the tile columns");

  MLX_MTL_CONST short pack_factor = 2;
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1
                                        : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = blocksize / BCOLS;

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  const device float* absmax_ptr;

  BnBQuantizedBlockLoader(
      const device uint8_t* src_,
      const device float* absmax_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED : BROWS * src_ld / pack_factor),
        group_step_cnt(0),
        group_stride(BROWS * src_ld / blocksize),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld / pack_factor + bj),
        absmax_ptr(absmax_ + bi * src_ld / blocksize) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    float am = *absmax_ptr;
    for (int i = 0; i < n_reads; i++) {
      bnb_dequantize<T, pack_factor, quant_type>(src + i, T(am), dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    float am = *absmax_ptr;
    for (int i = 0; i < n_reads; i++) {
      bnb_dequantize<T, pack_factor, quant_type>(src + i, T(am), dst + i * pack_factor);
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          absmax_ptr++;
        }
      } else {
        absmax_ptr++;
      }
    } else {
      absmax_ptr += group_stride;
    }
  }
};

// ============================================================================
// BnB GEMV (matrix-vector multiply with 4-bit quantized weights)
//
// Computes y = dequant(W) @ x
// W: [N, K/2] packed bytes, absmax: [N, ceil(K/blocksize)], x: [K], y: [N]
//
// Each simdgroup handles results_per_simdgroup output rows.
// Each thread processes values_per_thread elements of K per iteration.
// ============================================================================

template <typename T, int blocksize, int quant_type>
METAL_FUNC void bnb_qmv_impl(
    const device uint8_t* w,
    const device float* absmax,
    const device T* x,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int bytes_per_thread = 4;
  constexpr int values_per_thread = bytes_per_thread * 2;
  constexpr int block_size_k = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = blocksize / values_per_thread;

  constant float* codebook = bnb_codebook<quant_type>();

  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int K_packed = in_vec_size / 2;
  const int K_groups = (in_vec_size + blocksize - 1) / blocksize;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size) {
    return;
  }

  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  const device uint8_t* ws =
      w + used_out_row * K_packed + simd_lid * bytes_per_thread;
  const device float* am =
      absmax + used_out_row * K_groups + simd_lid / scale_step_per_thread;
  const device T* xi = x + tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + used_out_row;

  int k = 0;
  for (; k < in_vec_size - block_size_k; k += block_size_k) {
    // Load x values
    for (int i = 0; i < values_per_thread; i++) {
      x_thread[i] = U(xi[i]);
    }

    // Compute dot product for each output row
    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * K_packed;
      U scale = U(am[row * K_groups]);

      U accum = 0;
      for (int i = 0; i < bytes_per_thread; i++) {
        uint8_t byte_val = wl[i];
        U w0 = U(codebook[(byte_val >> 4) & 0x0f]);
        U w1 = U(codebook[byte_val & 0x0f]);
        accum += x_thread[2 * i] * w0 + x_thread[2 * i + 1] * w1;
      }
      result[row] += accum * scale;
    }

    ws += block_size_k / 2;
    am += block_size_k / blocksize;
    xi += block_size_k;
  }

  // Handle remaining K elements
  const int remaining = clamp(
      static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    for (int i = 0; i < remaining; i++) {
      x_thread[i] = U(xi[i]);
    }
    for (int i = remaining; i < values_per_thread; i++) {
      x_thread[i] = 0;
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      const device uint8_t* wl = ws + row * K_packed;
      U scale = U(am[row * K_groups]);

      U accum = 0;
      int bytes_to_read = (remaining + 1) / 2;
      for (int i = 0; i < bytes_to_read; i++) {
        uint8_t byte_val = wl[i];
        U w0 = U(codebook[(byte_val >> 4) & 0x0f]);
        U w1 = U(codebook[byte_val & 0x0f]);
        accum += x_thread[2 * i] * w0 + x_thread[2 * i + 1] * w1;
      }
      result[row] += accum * scale;
    }
  }

  // Reduce across SIMD lanes
  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

// ============================================================================
// BnB GEMM with transposed weight (y = x @ dequant(w).T)
//
// x: [M, K], w: [N, K/2] packed, absmax: [N, ceil(K/blocksize)], y: [M, N]
//
// Uses tiled matrix multiply with BnBQuantizedBlockLoader for on-the-fly
// dequantization of weights during the GEMM computation.
// ============================================================================

template <
    typename T,
    const int blocksize,
    const int quant_type,
    const int BM = 32,
    const int BK = 32,
    const int BN = 32>
METAL_FUNC void bnb_qmm_t_impl(
    const device uint8_t* w,
    const device float* absmax,
    const device T* x,
    device T* y,
    threadgroup T* Xs,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = 2;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = BnBQuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      blocksize,
      quant_type>;

  const int K_packed = K / pack_factor;
  const int K_groups = (K + blocksize - 1) / blocksize;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  x += y_row * static_cast<int64_t>(K);
  w += y_col * K_packed;
  absmax += y_col * K_groups;
  y += y_row * static_cast<int64_t>(N) + y_col;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(
      (const device uint8_t*)w, absmax, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if (num_outs < BN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (num_outs < BN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  // Store results
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

// ============================================================================
// Kernel entry points
// ============================================================================

// ---- Standalone blockwise quantize ----
// Each thread handles one block of elements.

template <typename T, int blocksize, int quant_type>
[[kernel]] void bnb_quantize_blockwise(
    const device T* input [[buffer(0)]],
    device float* absmax [[buffer(1)]],
    device uint8_t* packed [[buffer(2)]],
    const constant int& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const int num_blocks = (n + blocksize - 1) / blocksize;
  if (static_cast<int>(gid) >= num_blocks) {
    return;
  }

  int block_start = gid * blocksize;
  int block_end = min(block_start + blocksize, n);

  // Find absmax for this block
  float max_val = 0.0f;
  for (int i = block_start; i < block_end; i++) {
    float current = metal::abs(float(input[i]));
    max_val = metal::max(max_val, current);
  }
  absmax[gid] = max_val;

  float inv = (max_val > 0.0f) ? 1.0f / max_val : 0.0f;

  // Quantize and pack pairs of values
  int out_byte = block_start / 2;
  for (int i = block_start; i < block_end; i += 2) {
    float norm0 = (max_val > 0.0f) ? clamp(float(input[i]) * inv, -1.0f, 1.0f)
                                    : 0.0f;
    uchar q0 = bnb_quantize_value<quant_type>(norm0);

    uchar q1 = 0;
    if (i + 1 < block_end) {
      float norm1 = (max_val > 0.0f)
          ? clamp(float(input[i + 1]) * inv, -1.0f, 1.0f)
          : 0.0f;
      q1 = bnb_quantize_value<quant_type>(norm1);
    }

    packed[out_byte++] = (q0 << 4) | (q1 & 0x0f);
  }
}

// ---- Standalone blockwise dequantize ----
// Each threadgroup handles one block. Threads within share the absmax.

template <typename T, int blocksize, int quant_type>
[[kernel]] void bnb_dequantize_blockwise(
    const device uint8_t* packed [[buffer(0)]],
    const device float* absmax [[buffer(1)]],
    device T* output [[buffer(2)]],
    const constant int& n [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  const int num_blocks = (n + blocksize - 1) / blocksize;
  if (static_cast<int>(tgid) >= num_blocks) {
    return;
  }

  constant float* codebook = bnb_codebook<quant_type>();

  int block_start = tgid * blocksize;
  int block_end = min(block_start + blocksize, n);

  threadgroup float shared_scale = 0.0f;
  if (tid == 0) {
    shared_scale = absmax[tgid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float scale = shared_scale;

  int pairs_in_block = (block_end - block_start + 1) / 2;

  for (int pair = static_cast<int>(tid); pair < pairs_in_block;
       pair += static_cast<int>(tg_size)) {
    int elem_idx = block_start + pair * 2;
    int byte_idx = elem_idx / 2;
    uint8_t byte_val = packed[byte_idx];

    uint8_t high = (byte_val >> 4) & 0x0f;
    uint8_t low = byte_val & 0x0f;

    output[elem_idx] = T(codebook[high] * scale);
    if (elem_idx + 1 < block_end) {
      output[elem_idx + 1] = T(codebook[low] * scale);
    }
  }
}

// ---- GEMV kernel entry point ----
// y = dequant(W) @ x
// W: [N, K/2], absmax: [N, K_groups], x: [K], y: [N]

template <typename T, int blocksize, int quant_type>
[[kernel]] void bnb_qmv(
    const device uint8_t* w [[buffer(0)]],
    const device float* absmax [[buffer(1)]],
    const device T* x [[buffer(2)]],
    device T* y [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  bnb_qmv_impl<T, blocksize, quant_type>(
      w, absmax, x, y, in_vec_size, out_vec_size, tid, simd_gid, simd_lid);
}

// ---- GEMM (transposed weight) kernel entry point ----
// Y = X @ dequant(W).T
// X: [M, K], W: [N, K/2], absmax: [N, K_groups], Y: [M, N]

template <typename T, int blocksize, int quant_type>
[[kernel]] void bnb_qmm_t(
    const device uint8_t* w [[buffer(0)]],
    const device float* absmax [[buffer(1)]],
    const device T* x [[buffer(2)]],
    device T* y [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BM = 32;
  constexpr int BK = 32;
  constexpr int BN = 32;
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  bnb_qmm_t_impl<T, blocksize, quant_type, BM, BK, BN>(
      w, absmax, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}
