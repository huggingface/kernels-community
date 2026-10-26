//! Measured tile/split heuristics and CUDA specialization selection.

use tvm_ffi::Tensor;
use tvm_ffi::error::{RUNTIME_ERROR, Result};
use tvm_ffi_sys::dlpack::DLDevice;

use crate::host::cuda::CudaFunction;
use crate::host::layout::Strides;
use crate::host::err;
use crate::host::runtime::Runtime;
use crate::host::tiles::{Tile, alternate_tile, default_tile, dropout_tile, flash_tile, split_tile};

/// Validated forward inputs and buffers, kept until all launches finish.
pub(super) struct PreparedForward {
    pub(super) q: Tensor,
    pub(super) k: Tensor,
    pub(super) v: Tensor,
    pub(super) _alibi: Option<Tensor>,
    pub(super) q_str: Strides,
    pub(super) k_str: Strides,
    pub(super) v_str: Strides,
    pub(super) is_bf16: bool,
    pub(super) b: i64,
    pub(super) h: i64,
    pub(super) d: i64,
    pub(super) seqlen_q: i64,
    pub(super) seqlen_k: i64,
    pub(super) hk: i64,
    pub(super) bh: i64,
    pub(super) device: DLDevice,
    pub(super) has_alibi: u32,
    pub(super) alibi_data: u64,
    pub(super) alibi_numel: u64,
    pub(super) alibi_bs: u32,
    pub(super) scale: f64,
    pub(super) scale_eff: f64,
    pub(super) tanh_scale: f64,
    pub(super) softcap: f64,
    pub(super) wl: i64,
    pub(super) wr: i64,
    pub(super) alibi_causal: u32,
    pub(super) supplied_out: Option<Tensor>,
    pub(super) copy_out: bool,
    pub(super) out: Tensor,
    pub(super) lse: Tensor,
    pub(super) probabilities: Option<Tensor>,
    pub(super) rng_state: Option<Tensor>,
    pub(super) dropout: bool,
    pub(super) dropout_seed: u64,
    pub(super) dropout_offset: u64,
    pub(super) dropout_keep: f32,
    pub(super) native_reservation: bool,
    pub(super) return_softmax: bool,
    pub(super) descriptor_mode: bool,
    pub(super) descriptor: [u64; 3],
    pub(super) descriptor_captured: u32,
    pub(super) bucket: i64,
    pub(super) lse_only: bool,
}

pub(super) struct ForwardPlan {
    pub(super) tile: Tile,
    pub(super) short_d32_alt: bool,
    pub(super) cross_split: bool,
    pub(super) groups: i64,
    pub(super) wide_queries: bool,
    pub(super) splits: i64,
    pub(super) wide_keys: bool,
}

pub(super) fn plan_forward(input: &PreparedForward, rt: &Runtime) -> ForwardPlan {
    let PreparedForward {
        bucket,
        wl,
        wr,
        bh,
        seqlen_q,
        seqlen_k,
        has_alibi,
        softcap,
        dropout,
        d,
        h,
        hk,
        b,
        q_str,
        ..
    } = *input;
    let ctas_128 = ((seqlen_q + 127) / 128) * bh;
    let mut tile = flash_tile(bucket, wl >= 0 || wr >= 0, ctas_128, rt.sm_count);
    let short_d32_alt = bucket == 32
        && seqlen_q >= 256
        && seqlen_q < 1024
        && seqlen_k >= 128
        && ctas_128 >= (3 * rt.sm_count + 1) / 2
        && ctas_128 < 2 * rt.sm_count
        && wl < 0
        && wr < 0
        && has_alibi == 0
        && softcap == 0.0;
    if short_d32_alt {
        // Enough work for the larger query tile.
        tile = alternate_tile(bucket);
    }

    if bucket == 96 && wl < 0 && wr < 0 && has_alibi == 0 && softcap == 0.0 && seqlen_k >= 1024 {
        // Small final wave: 64 query rows in this measured range.
        let wave = 2 * rt.sm_count;
        let tail = ctas_128 % wave;
        if ctas_128 > wave && ctas_128 <= 7 * rt.sm_count && tail > 0 && tail <= rt.sm_count {
            tile = default_tile(bucket);
        } else if ctas_128 >= (3 * rt.sm_count + 1) / 2 && ctas_128 < wave {
            // A well filled first wave benefits from 128-row K/V reuse.
            tile = alternate_tile(bucket);
        }
    }

    // d192 fixes the grid layout to save coordinate registers.
    let preliminary_blocks = (seqlen_q + tile.rows - 1) / tile.rows;
    if bucket == 192
        && !(wr == 0 && wl >= seqlen_k && preliminary_blocks > 1 && preliminary_blocks <= 65535)
    {
        tile = default_tile(bucket);
    }
    // Split long keys when the query grid cannot fill Ada.
    let cross_split = !dropout
        && seqlen_q > 1
        && (seqlen_q > 16 || (wl >= seqlen_k && wr == 0))
        && seqlen_q <= 256
        && seqlen_k >= 1024
        && has_alibi == 0
        && ((wl < 0 && wr < 0) || (wl >= seqlen_k && wr == 0))
        && ((seqlen_q + 63) / 64) * bh <= rt.sm_count;
    // Enough key segments to fill Ada when queries give few CTAs.
    let split_eligible = !dropout
        && seqlen_q <= 16
        && seqlen_k >= 1024
        && has_alibi == 0
        && ((wl < 0 && wr < 0) || (seqlen_q == 1 && wl >= seqlen_k && wr == 0))
        && (bh < rt.sm_count || (seqlen_q == 1 && h > hk && b * hk < rt.sm_count));
    // Single-query GQA heads become rows sharing one K/V head.
    let groups =
        if split_eligible && seqlen_q == 1 && q_str.head.checked_mul((h / hk) as u32).is_some() {
            h / hk
        } else {
            1
        };
    let wide_queries = if cross_split && d == 32 && seqlen_q >= 64 && seqlen_q % 64 == 0 {
        let ctas = ((seqlen_q + 63) / 64) * bh;
        let n = ((rt.sm_count * 4 + ctas - 1) / ctas)
            .min(32)
            .min(seqlen_k / 256);
        let chunk = (seqlen_k + n - 1) / n;
        n > 1 && chunk <= 1024 && chunk % 64 == 0
    } else {
        false
    };
    let cross_rows = if wide_queries {
        64
    } else if bucket >= 192 && seqlen_q > 32 && seqlen_q <= 64 {
        64
    } else {
        32
    };
    let cross_ctas = ((seqlen_q + cross_rows - 1) / cross_rows) * bh;
    let mut splits = if cross_split {
        ((rt.sm_count * 4 + cross_ctas - 1) / cross_ctas)
            .min(32)
            .min(seqlen_k / 256)
    } else if groups > 1 {
        ((rt.sm_count * 4 + b * hk - 1) / (b * hk))
            .min(32)
            .min(seqlen_k / 128)
    } else if split_eligible {
        ((rt.sm_count * 4 + bh - 1) / bh)
            .min(32)
            .min(seqlen_k / 256)
    } else {
        1
    };
    // Full unmasked grids trade one split for less merge traffic.
    if wide_queries && wr < 0 && cross_ctas == rt.sm_count && splits == 4 {
        let chunk = (seqlen_k + 2) / 3;
        let padded_keys = ((chunk + 63) / 64) * 64 * 3;
        if padded_keys * 16 <= seqlen_k * 17 {
            splits = 3;
        }
    }
    // Less scratch traffic for wide-head MHA cross-attention, 48-64 queries.
    if cross_split
        && h == hk
        && bucket == 256
        && seqlen_q >= 48
        && seqlen_q <= 64
        && rt.l2_bytes > 0
    {
        let working_set = b * hk * seqlen_k * d * 4 + bh * seqlen_q * d * 4 * (splits + 1);
        let fewer_splits = (rt.sm_count + cross_ctas - 1) / cross_ctas;
        if working_set > rt.l2_bytes * 17 / 16 && fewer_splits >= 2 {
            splits = splits.min(fewer_splits);
        }
    }
    // Larger key tiles for short aligned segments.
    let wide_keys = cross_split
        && splits > 1
        && bucket == 64
        && seqlen_q <= 128
        && (seqlen_k + splits - 1) / splits <= 512
        && ((seqlen_k + splits - 1) / splits) % 64 == 0
        && (seqlen_q <= 32 || seqlen_q % 32 == 0)
        && (seqlen_q <= 32 || (d >= 48 && (wr < 0 || seqlen_q <= 64 || d == 64)));
    // Larger key tiles for exact d32 and aligned segments.
    let wide_keys = wide_keys
        || (cross_split
            && splits > 1
            && d == 32
            && ((seqlen_q <= 32 && splits <= 8) || seqlen_q % 32 == 0)
            && (seqlen_k + splits - 1) / splits <= 1024
            && ((seqlen_k + splits - 1) / splits) % 64 == 0);
    if splits > 1 {
        // Four-warp blocks when very few K/V heads supply the grid.
        let rows =
            if b * hk <= 2 && groups >= 16 && (bucket == 256 || (bucket == 192 && groups <= 32)) {
                64
            } else if cross_split {
                cross_rows
            } else if groups > 1 {
                32
            } else {
                16
            };
        tile = split_tile(bucket, rows);
        if wide_keys || wide_queries {
            tile.keys = 64;
        }
    }
    if dropout {
        tile = dropout_tile(bucket);
    }
    ForwardPlan {
        tile,
        short_d32_alt,
        cross_split,
        groups,
        wide_queries,
        splits,
        wide_keys,
    }
}

pub(super) fn select_kernel<'a>(
    input: &PreparedForward,
    plan: &mut ForwardPlan,
    rt: &'a Runtime,
) -> Result<(&'a CudaFunction, bool)> {
    let PreparedForward {
        bucket,
        wl,
        wr,
        bh,
        seqlen_q,
        seqlen_k,
        has_alibi,
        softcap,
        dropout,
        d,
        h,
        hk,
        q_str,
        k_str,
        v_str,
        scale,
        is_bf16,
        descriptor_mode,
        return_softmax,
        ..
    } = *input;
    let ForwardPlan {
        mut tile,
        short_d32_alt,
        cross_split,
        groups: _,
        wide_queries,
        splits,
        wide_keys,
    } = *plan;
    // Unrolled copies help short key segments, not multi-tile causal.
    let fast_split = splits > 1
        && bucket == 128
        && (seqlen_k + splits - 1) / splits <= 256
        && !(cross_split && wr == 0 && seqlen_q > 32);
    // Shared-Q tile for mostly filled waves on SM89.
    let dense192_ctas = ((seqlen_q + 127) / 128) * bh;
    let dense192_selected = !dropout
        && dense192_ctas >= (5 * rt.sm_count + 7) / 8
        && dense192_ctas <= rt.sm_count
        && seqlen_k <= 512
        && d == 192
        && h == hk
        && splits == 1
        && seqlen_q % 128 == 0
        && seqlen_k % 64 == 0
        && wl < 0
        && wr < 0
        && has_alibi == 0
        && softcap == 0.0
        && (scale as f32) > 0.0
        && ((scale as f32) * std::f32::consts::LOG2_E).is_finite()
        && q_str.head == 96
        && k_str.head == 96
        && v_str.head == 96
        && q_str.seq as i64 == h * 96
        && k_str.seq as i64 == h * 96
        && v_str.seq as i64 == h * 96
        && q_str.batch as i64 == seqlen_q * h * 96
        && k_str.batch as i64 == seqlen_k * h * 96
        && v_str.batch as i64 == seqlen_k * h * 96;
    let flash_fn = if dropout {
        rt.dropout_fns
            .get(&(
                bucket,
                is_bf16,
                if descriptor_mode {
                    3
                } else if return_softmax {
                    2
                } else {
                    input.rng_state.is_some() as u8
                },
            ))
            .unwrap()
    } else if dense192_selected {
        tile = Tile {
            rows: 128,
            keys: 64,
            threads: 256,
            alternate: false,
        };
        rt.dense_d192_fns.get(&is_bf16).unwrap()
    } else if wide_queries {
        // Three splits favor rolled copies; other Q64 splits unroll.
        rt.wide_query_split_fns
            .get(&(is_bf16, wr == 0, splits != 3))
            .unwrap()
    } else if wide_keys {
        rt.wide_key_split_fns
            .get(&(bucket, is_bf16, wr == 0))
            .unwrap()
    } else if fast_split {
        rt.fast_split_fns
            .get(&(tile.rows, is_bf16, cross_split && wr == 0))
            .unwrap()
    } else if cross_split && splits > 1 && wr == 0 {
        rt.causal_split_fns
            .get(&(bucket, is_bf16, tile.rows, bucket == 256 && seqlen_q > 32))
            .unwrap()
    } else if splits > 1 {
        rt.split_fns.get(&(bucket, is_bf16, tile.rows)).unwrap()
    } else if bucket > 32 && ((wl >= 0 && wl < seqlen_k) || (wr > 0 && wr < seqlen_k)) {
        // Full causal keeps vector stores; windows use Q64/K32 triple pipeline.
        if bucket == 192 {
            tile.keys = 32;
        }
        rt.scalar_fns
            .get(&(bucket, tile.alternate, is_bf16))
            .unwrap()
    } else if bucket == 192
        && seqlen_k > 48
        && seqlen_k <= 256
        && ((seqlen_q + tile.rows - 1) / tile.rows) * bh <= rt.sm_count
    {
        // Coordinate increments for short sequences, <= 1 CTA per SM.
        rt.short_d192_fns.get(&(tile.alternate, is_bf16)).unwrap()
    } else if bucket == 32
        && tile.alternate
        && seqlen_q >= 1024
        && (if d == 8 {
            seqlen_q >= 2048 && seqlen_k >= 2048 && (seqlen_q >= 4096 || seqlen_k >= 4096)
        } else {
            seqlen_k >= if d == 32 { 1024 } else { 2048 }
        })
        && wl < 0
        && wr < 0
        && has_alibi == 0
        && softcap == 0.0
        && (scale as f32) > 0.0
        && ((scale as f32) * std::f32::consts::LOG2_E).is_finite()
    {
        // Wider key tiles for these dense workloads; grid recomputed below.
        tile = Tile {
            rows: 64,
            keys: 128,
            ..tile
        };
        rt.wide_d32_fns.get(&is_bf16).unwrap()
    } else if bucket == 32
        && tile.alternate
        && (short_d32_alt || (seqlen_q >= 1024 && seqlen_k >= if d == 8 { 2048 } else { 1024 }))
        && wl < 0
        && wr < 0
        && has_alibi == 0
        && softcap == 0.0
        && (scale as f32) > 0.0
        && ((scale as f32) * std::f32::consts::LOG2_E).is_finite()
    {
        rt.raw_d32_fns.get(&is_bf16).unwrap()
    } else if bucket == 128
        && seqlen_q >= 512
        && seqlen_k >= 4096
        && ((seqlen_q + 63) / 64) * bh >= 2 * rt.sm_count
        && wl < 0
        && wr < 0
        && has_alibi == 0
        && softcap == 0.0
        && (scale as f32) > 0.0
        && ((scale as f32) * std::f32::consts::LOG2_E).is_finite()
    {
        // exp2 scale fusion needs a positive, representable scale.
        rt.raw_d128_fns.get(&is_bf16).unwrap()
    } else {
        rt.fns
            .get(&(bucket, tile.alternate, is_bf16))
            .ok_or_else(|| {
                err(
                    RUNTIME_ERROR,
                    &format!(
                        "no kernel for bucket {bucket} alternate={} bf16={is_bf16}",
                        tile.alternate
                    ),
                )
            })?
    };
    plan.tile = tile;
    Ok((flash_fn, dense192_selected))
}
