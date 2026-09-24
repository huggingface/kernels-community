// Copyright © 2024-25 Apple Inc.
//
// Variable-length (cu_seqlens) flash attention. This is MLX's steel attention
// kernel (steel/attn/kernels/steel_attention.h) adapted to packed
// [total_tokens, heads, head_dim] inputs. The building blocks in steel/ are
// vendored from MLX unmodified. Differences from upstream:
//
// - Sequence bounds come from cu_seqlens_q/cu_seqlens_k; partial Q/K blocks are
//   detected per sequence instead of through align_Q/align_K.
// - Inputs are addressed by (token, head) strides; head_dim must be contiguous.
// - Optional tanh softcapping, applied before any masking.
// - Causal masking is aligned to the bottom-right when q_len != k_len, like
//   flash-attn. Query rows that see no keys (and sequences with k_len == 0)
//   produce zeros.
// - Attention sinks are float32, independent of the input dtype.
// - No array masks.

// clang-format off
#include "utils.h"
#include "steel/attn/attn.h"
#include "varlen_params.h"

using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];
constant bool has_softcap [[function_constant(303)]];

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_varlen(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant VarlenAttnParams* params [[buffer(4)]],
    const device int* cu_seqlens_q [[buffer(5)]],
    const device int* cu_seqlens_k [[buffer(6)]],
    const device float* sinks [[buffer(7), function_constant(has_sinks)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;

  const int q_block = tid.x;
  const int head_idx = tid.y;
  const int seq_idx = tid.z;

  const int q_seq_start = cu_seqlens_q[seq_idx];
  const int qL = cu_seqlens_q[seq_idx + 1] - q_seq_start;
  const int k_seq_start = cu_seqlens_k[seq_idx];
  const int kL = cu_seqlens_k[seq_idx + 1] - k_seq_start;

  // The grid is sized for the longest sequence. The whole threadgroup exits
  // here, before any barrier.
  if (q_block * BQ >= qL) {
    return;
  }

  const int q_block_size = min(BQ, qL - q_block * BQ);
  const int NK = (kL + BK - 1) / BK;
  const int kL_rem = kL - (NK - 1) * BK;
  // Offset of the query rows in the key sequence (bottom-right alignment).
  const int qL_off = kL - qL;

  // Move to correct block
  const int kv_head_idx = head_idx / (params->H / params->H_kv);
  const int64_t q_row = int64_t(q_seq_start) + q_block * BQ;

  Q += q_row * params->q_strides[0] + head_idx * params->q_strides[1];
  K += int64_t(k_seq_start) * params->k_strides[0] +
      kv_head_idx * params->k_strides[1];
  V += int64_t(k_seq_start) * params->v_strides[0] +
      kv_head_idx * params->v_strides[1];
  O += q_row * params->o_strides[0] + head_idx * params->o_strides[1];

  // Prepare threadgroup memory
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  constexpr short tgp_mem_0 = (BK + padK) * (BD);
  constexpr short tgp_mem_1 = BK * (BD + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T Q_smem[BQ * (BD + padQ)];
  threadgroup T KV_smem[tgp_mem_s];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  // Prepare block loaders
  using QBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BQ,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDQ_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 1,
      /* short tgp_size = */ WM * WN * 32>;

  // K is loaded in transposed
  using KBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ 1,
      /* short kDstStrCol = */ LDK_tgp,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  using VBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDV_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  QBlockLoader loader_q(
      Q, int(params->q_strides[0]), Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(
      K, int(params->k_strides[0]), Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(
      V, int(params->v_strides[0]), Vs, simd_group_id, simd_lane_id);

  // Scores are computed in log2 space, so exp2 can be used in the softmax.
  // With softcapping: s = log2(e) * cap * tanh(scale * qk / cap).
  const AccumType scale = has_softcap ? params->scale / params->softcap
                                      : params->scale * M_LOG2E_F;
  const AccumType softcap_scale = has_softcap ? params->softcap * M_LOG2E_F : 1;

  // Prepare MMA tiles
  constexpr short kFragSize = 8; // MMAFrag size
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  // KV sequence frags (all warps load the same frags)
  constexpr int TK = BK / kFragSize;
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kFragSize;

  static_assert(TQ == 1, "Check TQ");

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load Q blocks
  if (q_block_size < BQ) {
    loader_q.load_safe(short2(BD, q_block_size));
  } else {
    loader_q.load_unsafe();
  }

  // Init row reduction variables
  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  if (has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[head_idx]);
      sum_score[i] = 1;
    }
  }

  int kb_lim = NK;
  int kb_min_causal = NK;

  if (do_causal) {
    int q_max = q_block * BQ + q_block_size + qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = max(0, min(NK, kb_lim));

    int q_min = q_block * BQ + qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
    const bool k_partial = kb == NK - 1 && kL_rem < BK;

    // Load K block and apply scale
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k_partial) {
      loader_k.load_safe(short2(BD, kL_rem));
    } else {
      loader_k.load_unsafe();
    }

    // Do S = Q @ K.T
    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
          &Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * Ks_tile_stride]);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    // Apply scale (and softcap) in float32, before masking
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      if (has_softcap) {
        Stile.elems()[ii] =
            softcap_scale * metal::precise::tanh(Stile.elems()[ii] * scale);
      } else {
        Stile.elems()[ii] *= scale;
      }
    }

    // Mask out length sequence
    if (k_partial) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= kL_rem) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos =
            q_block * BQ + qL_off + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load V blocks
    if (k_partial) {
      loader_v.load_safe(short2(BD, kL_rem));
    } else {
      loader_v.load_unsafe();
    }

    // Do softmax

    // Temp variables
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
    }

    // Save max for next iteration
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }

    // Row Sum
    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);

    // Update norm
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    // Update O
    Otile.template row_bin_op<MulOp>(factor);

    // Load V into registers
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if constexpr (BD >= 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;

          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);

          if constexpr (BD >= 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }

    // Prepare for next iteration
    loader_k.next();
    loader_v.next();
  }

  // Normalize output. Rows without any visible key produce zeros.
  AccumType inv_sum[kRowsPT];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    const int row_pos = q_block * BQ + qL_off + tm + sm +
        (i * decltype(Stile)::kFragRows);
    const bool has_keys = kL > 0 && (!do_causal || row_pos >= 0);
    inv_sum[i] = has_keys ? AccumType(1) / sum_score[i] : AccumType(0);
  }
  Otile.template row_bin_op<MulOp>(inv_sum);
  threadgroup_barrier(mem_flags::mem_none);

  // Store results
  const int ldo = int(params->o_strides[0]);
  O += (tm + sm) * ldo + sn;

  if (q_block_size < BQ) {
    auto dst_tile_dims = short2(BD - sn, q_block_size - (tm + sm));

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Otile.template store_safe<T, 1, 1>(O, ldo, dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, ldo);
  }
}

// clang-format off
#define instantiate_attn(tname, dtype, bq, bk, bd, wm, wn) \
  instantiate_kernel(                                     \
      "attention_varlen_" #tname "_bq" #bq "_bk" #bk      \
      "_bd" #bd "_wm" #wm "_wn" #wn,                      \
  attention_varlen, dtype, bq, bk, bd, wm, wn, float)

// Same tiles as upstream MLX; head_dim 32 is a local addition.
#define instantiate_attn_shapes_common(iname, itype) \
    instantiate_attn(iname, itype, 32, 16, 128, 4, 1) \
    instantiate_attn(iname, itype, 32, 32,  96, 4, 1) \
    instantiate_attn(iname, itype, 32, 32,  80, 4, 1) \
    instantiate_attn(iname, itype, 32, 32,  72, 4, 1) \
    instantiate_attn(iname, itype, 32, 32,  64, 4, 1) \
    instantiate_attn(iname, itype, 32, 32,  32, 4, 1)

#define instantiate_attn_shapes_half(iname, itype) \
    instantiate_attn(iname, itype, 32, 16, 256, 4, 1) \
    instantiate_attn(iname, itype, 32, 16, 192, 4, 1) \
    instantiate_attn_shapes_common(iname, itype)

// The upstream head_dim 192/256 tiles exceed 32KB of threadgroup memory for
// float32, so float32 uses smaller tiles there.
#define instantiate_attn_shapes_float(iname, itype) \
    instantiate_attn(iname, itype, 16, 8, 256, 2, 1) \
    instantiate_attn(iname, itype, 16, 8, 192, 2, 1) \
    instantiate_attn_shapes_common(iname, itype)

instantiate_attn_shapes_half(float16, half);
instantiate_attn_shapes_half(bfloat16, bfloat16_t);
instantiate_attn_shapes_float(float32, float);
// clang-format on
