#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

// Paged decode tuning for XE3P (calibrated on CRI): head_dim 128/256 with
// 64-token KV tiles, for 16-bit (bf16/fp16) Q/KV, fp8 KV with a 16-bit query
// and full fp8. Shared by the host split heuristic (flash_api.cpp) and the
// kernel launcher (xe_3/paged_decode.hpp) so both agree on the concurrency.
//
// Decode streams K/V from DRAM, and CRI loses bandwidth when too many
// work-groups per Xe core stream different KV heads at once (16-bit head_dim
// 256 reaches ~92% of peak with two streams per Xe core, ~60% with eight).
// The launcher reserves SLM to cap work-groups per Xe core, and the host picks
// the split count that fills and balances the capped slots.
namespace flash_xpu_splitkv::paged_decode_tuning {

enum class Precision : int {
  k16Bit,    // bf16/fp16 Q and KV
  kFullFp8,  // fp8 Q and KV (fp8 MMAs)
  kFp8KV,    // bf16/fp16 Q, fp8 KV (converted to 16-bit in registers)
};

// The calibration is in 64-token KV tiles, so it applies only to kernels whose
// KV tile (TileShapeQK's KV extent) matches. Retune before adding a tile size.
constexpr int kCalibratedKvTile = 64;

// Whether a decode kernel with this head-size bucket and KV tile is tuned.
constexpr bool is_tuned(int head_dim, int kv_tile) {
  return (head_dim == 128 || head_dim == 256) && kv_tile == kCalibratedKvTile;
}

// Sequences shorter than this many KV tiles are not split (the kernel uses it
// as XeFMHAFwdSplitKVKernel::kMinBlocksForSplit).
constexpr int kMinTilesForSplit = 4;

// Fitted to exhaustive occupancy and split sweeps on CRI. Streams are KV heads
// streamed concurrently per Xe core; fp8 tiles carry half the bytes of 16-bit
// ones, so they need more streams to saturate DRAM.
struct Calibration {
  int cap_streams;    // occupancy cap; 0 = no cap
  int split_streams;  // streams the split model fills
  int sat_quarters;   // grids with sat_quarters / 4 streams are not split
  int reduce_tiles;   // split-K reduction cost per 8-row q-tile, in KV tiles
};

constexpr Calibration calibration(int head_dim, Precision prec) {
  const bool hd128 = head_dim <= 128;
  if (prec == Precision::kFullFp8) {
    return hd128 ? Calibration{6, 4, 16, 2} : Calibration{4, 3, 8, 1};
  }
  if (prec == Precision::kFp8KV) {
    // head_dim 128 was slower with any cap.
    return hd128 ? Calibration{0, 6, 16, 2} : Calibration{4, 4, 16, 2};
  }
  return hd128 ? Calibration{4, 3, 6, 4} : Calibration{2, 2, 8, 2};
}

// Work-groups per Xe core for the launcher's SLM-based occupancy cap, or 0 for
// no cap. wgs_per_kv_head is the launch grid's q-tile count: the q-tiles of a
// KV head re-read the same K/V (mostly from L2), so the cap scales with it.
// The cap pays off when work-groups stream many KV tiles, or when a grid of
// <= 4-subgroup work-groups is several capped waves deep (for fp8 KV only from
// four tiles per work-group). Small grids of short work-groups are latency
// bound and keep every slot.
inline int occupancy_cap(
    int head_dim,
    Precision prec,
    int64_t tiles_per_wg,
    int64_t num_wgs,
    int num_xe_cores,
    int wgs_per_kv_head,
    int subgroups_per_wg) {
  const int streams = calibration(head_dim, prec).cap_streams;
  if (streams <= 0) return 0;
  // Each subgroup takes one of the 32 hardware threads of an Xe core (256-GRF
  // mode).
  const int max_wgs = std::max(1, 32 / std::max(1, subgroups_per_wg));
  const int wgs = std::clamp(streams * wgs_per_kv_head, 1, max_wgs);
  const bool deep_grid = subgroups_per_wg <= 4 &&
                         (prec != Precision::kFp8KV || tiles_per_wg >= 4) &&
                         num_wgs >= 4 * int64_t{num_xe_cores} * wgs;
  return (tiles_per_wg >= 8 || deep_grid) ? wgs : 0;
}

// Split count for uniform-length decode: minimise, in KV-tile units,
//   waves(s) * (ceil(kv_tiles / s) + 2) + (s > 1 ? reduce_cost(s) : 0)
// where waves(s) = ceil(batch * num_heads_kv * s / slots), slots is
// split_streams per Xe core and 2 is the fixed per-work-group cost (Q load,
// epilogue, partial write). This fills an under-occupied GPU with splits only
// while the extra parallelism pays for the reduction; ties keep fewer splits.
// kv_token_stride_bytes is the distance between consecutive tokens of one KV
// head in the cache (num_heads_kv * head_dim * element size when token-major).
inline int num_splits(
    int head_dim,
    Precision prec,
    int batch,
    int num_heads_q,
    int num_heads_kv,
    int kv_tiles,
    int num_xe_cores,
    int64_t kv_token_stride_bytes) {
  if (kv_tiles < kMinTilesForSplit) return 1;
  const Calibration cal = calibration(head_dim, prec);
  // GQA group size in 8-row q-tiles.
  const int q_tiles = std::max(1, (num_heads_q / num_heads_kv + 7) / 8);

  // A grid that already fills the GPU is balanced by the hardware dispatcher;
  // splitting it only adds the reduction. Thresholds are in quarter streams
  // per Xe core. Head_dim 256 saturates earlier than calibrated: short
  // sequences (<= 16 tiles) at one stream per Xe core for 16-bit and at two
  // for fp8 KV on the 16-row tile; long sequences with fp8 KV on the 8-row
  // tile (e.g. Qwen3.8, 24 Q / 4 KV heads) at two. 16-bit head_dim 128
  // reaches the DRAM peak from 1.5 streams per Xe core, or from 1.75 when a
  // KV head's tokens are more than 1 KB apart (8 KV heads in a token-major
  // cache: each stream reads 256-byte pieces of the token rows). Splitting
  // it further only fragments the reads and adds the reduction (VLLMZ-3208:
  // Qwen3-30B-A3B at TP4, 8 Q / 1 KV head, 58 sequences of 3.5K-5K tokens
  // ran 14-17% slower with 3 splits).
  // TODO: packed spec decode (#456) folds q_len positions into the rows, so
  // q_tiles can grow; re-check this upper bound once it lands.
  int sat_quarters = cal.sat_quarters;
  if (head_dim > 128) {
    if (kv_tiles <= 16) {
      if (prec == Precision::k16Bit) sat_quarters = 4;
      if (prec == Precision::kFp8KV && q_tiles > 1) sat_quarters = 8;
    } else if (prec == Precision::kFp8KV && q_tiles == 1) {
      sat_quarters = 8;
    }
  } else if (prec == Precision::k16Bit && kv_token_stride_bytes > 1024) {
    sat_quarters = 7;
  }
  const int64_t streams = int64_t{batch} * num_heads_kv;
  if (4 * streams >= int64_t{num_xe_cores} * sat_quarters) return 1;

  // Costs are in half-tile units: full fp8 head_dim 256 on the 8-row tile also
  // pays half a tile per split.
  const int64_t slots =
      std::max<int64_t>(1, int64_t{num_xe_cores} * cal.split_streams);
  const int64_t reduce_cost = 2 * int64_t{cal.reduce_tiles} * q_tiles;
  const int64_t split_cost =
      (prec == Precision::kFullFp8 && head_dim > 128 && q_tiles == 1) ? 1 : 0;
  // At most 32 splits, within every tuned kernel's max_num_kv_splits.
  int max_splits = std::min(32, kv_tiles);
  // 16-bit head_dim 128 saturates DRAM within one wave of slots (from 1.5 or
  // 1.75 of its 3 streams per Xe core), so a deeper grid streams no faster:
  // its extra splits only add work-groups and reduction work.
  if (prec == Precision::k16Bit && head_dim <= 128) {
    max_splits = static_cast<int>(std::clamp<int64_t>(
        slots / std::max<int64_t>(1, streams), 1, max_splits));
  }
  int best_splits = 1;
  int64_t best_cost = std::numeric_limits<int64_t>::max();
  for (int s = 1; s <= max_splits; ++s) {
    const int64_t waves = (streams * s + slots - 1) / slots;
    const int64_t cost = 2 * waves * ((kv_tiles + s - 1) / s + 2) +
                         (s > 1 ? reduce_cost + split_cost * s : 0);
    if (cost < best_cost) {
      best_cost = cost;
      best_splits = s;
    }
  }
  // fp8 KV on the head_dim-256 16-row tile (8-subgroup work-groups) streams
  // best with 3/4 of the modelled splits once each split covers two KV tiles.
  if (prec == Precision::kFp8KV && head_dim > 128 && q_tiles == 2 &&
      best_splits >= 4 && kv_tiles >= 2 * best_splits) {
    best_splits = best_splits * 3 / 4;
  }
  return best_splits;
}

}  // namespace flash_xpu_splitkv::paged_decode_tuning
