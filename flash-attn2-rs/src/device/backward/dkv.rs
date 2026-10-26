use super::common::{
    copy_rows, copy_rows_swizzled, fused_dq_tile, fused_dq_tile_d64, fused_store_ds,
    probability_gradient, query_fragments, query_fragments_swizzled, transposed_keep,
    transposed_paired_words, transposed_word_mask,
};
use crate::device::config::{FwdConfig, LaunchTile, Strides};
use crate::device::dropout::Dropout;
use crate::device::forward::{SmemLayout, accumulate_values, compute_qk};
use crate::device::inlined::cvt;
use cuda_device::shared::DynamicSharedArray;
use cuda_device::{DisjointSlice, kernel, launch_bounds, ptx_asm, thread};

// Looping over query heads gives a deterministic GQA reduction.
#[inline(always)]
unsafe fn dkv_core<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const PARTIAL: bool,
>(
    q: *const u32,
    k: *const u32,
    v: *const u32,
    dout: *const u32,
    lse: *const f32,
    delta: *const f32,
    dk: *mut u32,
    dv: *mut u32,
    alibi: *const f32,
    cfg: FwdConfig,
    do_strides: Strides,
    gradient_scale: f32,
    dropout: Dropout,
) {
    unsafe {
        let tid = thread::threadIdx_x();
        let lane = tid % 32;
        let is_dk = tid < 64;
        let b = thread::blockIdx_z();
        let output_head = thread::blockIdx_y();
        let kh = if PARTIAL {
            output_head / cfg.gqa
        } else {
            output_head
        };
        let key_start = thread::blockIdx_x() * 32;
        let mut tile = LaunchTile {
            tid,
            lane_id: lane,
            bh: 0,
            group: lane >> 2,
            pair: lane & 3,
            q_row_base: key_start,
            warp_row: (tid / 32 % 2) * 16,
            b,
            head: 0,
            kv_head: kh,
        };
        let key = key_start + tile.warp_row + tile.group;
        let layout = SmemLayout::new::<HD, 32, 32>();
        let k_shared = DynamicSharedArray::<u32, 16>::get() as usize as *mut u32;
        let v_shared = k_shared.add(32 * layout.stride);
        let q_shared = k_shared.add(64 * layout.stride);
        let do_shared = k_shared.add(96 * layout.stride);
        copy_rows::<HD, 32>(
            k,
            k_shared,
            b as usize * cfg.k.batch as usize + kh as usize * cfg.k.head as usize,
            key_start,
            cfg.k.seq,
            cfg.seqlen_k,
            cfg.hs_u32,
            tid,
        );
        copy_rows::<HD, 32>(
            v,
            v_shared,
            b as usize * cfg.v.batch as usize + kh as usize * cfg.v.head as usize,
            key_start,
            cfg.v.seq,
            cfg.seqlen_k,
            cfg.hs_u32,
            tid,
        );
        ptx_asm!(
            "cp.async.commit_group; cp.async.wait_group 0;",
            clobber("memory")
        );
        thread::sync_threads();
        let rounded_q = cfg.seqlen_q.div_ceil(128) * 128;
        // Restrict the query loop to tiles that can attend to this key tile.
        let off = cfg.seqlen_k as i64 - cfg.seqlen_q as i64;
        let first = if cfg.window_right >= 0 {
            (key_start as i64 - off - cfg.window_right as i64).clamp(0, cfg.seqlen_q as i64) as u32
                / 32
        } else {
            0
        };
        let end = if cfg.window_left >= 0 {
            ((key_start + 32).min(cfg.seqlen_k) as i64 - off + cfg.window_left as i64)
                .clamp(0, cfg.seqlen_q as i64) as u32
        } else {
            cfg.seqlen_q
        }
        .div_ceil(32);
        let mut acc = [[0.0f32; 4]; NFO];
        let mut group_head = if PARTIAL { output_head % cfg.gqa } else { 0 };
        while group_head
            < if PARTIAL {
                output_head % cfg.gqa + 1
            } else {
                cfg.gqa
            }
        {
            tile.head = kh * cfg.gqa + group_head;
            tile.bh = b * cfg.h + tile.head;
            let q_base =
                b as usize * cfg.q.batch as usize + tile.head as usize * cfg.q.head as usize;
            let do_base = b as usize * do_strides.batch as usize
                + tile.head as usize * do_strides.head as usize;
            let slope = if cfg.has_alibi != 0 {
                *alibi.add(b as usize * cfg.alibi_bs as usize + tile.head as usize)
            } else {
                0.0
            };
            let mut block = first;
            while block < end {
                let query_start = block * 32;
                copy_rows::<HD, 32>(
                    q,
                    q_shared,
                    q_base,
                    query_start,
                    cfg.q.seq,
                    cfg.seqlen_q,
                    cfg.hs_u32,
                    tid,
                );
                copy_rows::<HD, 32>(
                    dout,
                    do_shared,
                    do_base,
                    query_start,
                    do_strides.seq,
                    cfg.seqlen_q,
                    cfg.hs_u32,
                    tid,
                );
                ptx_asm!(
                    "cp.async.commit_group; cp.async.wait_group 0;",
                    clobber("memory")
                );
                thread::sync_threads();
                let mut scores = [[0.0f32; 4]; 4];
                {
                    let fragments = query_fragments::<NFQ>(k_shared, &tile, layout);
                    compute_qk::<BF16, 1, NFQ, 4>(&mut scores, &fragments, q_shared, &tile, layout);
                }
                let mut nt = 0;
                while nt < 4 {
                    thread::__unroll_config::<0>();
                    let query = query_start + nt as u32 * 8 + tile.pair * 2;
                    let l0 = if query < cfg.seqlen_q {
                        *lse.add(tile.bh as usize * cfg.seqlen_q as usize + query as usize)
                    } else {
                        f32::INFINITY
                    };
                    let l1 = if query + 1 < cfg.seqlen_q {
                        *lse.add(tile.bh as usize * cfg.seqlen_q as usize + query as usize + 1)
                    } else {
                        f32::INFINITY
                    };
                    let f = scores[nt];
                    scores[nt] = [
                        probability_gradient(f[0], query, key, l0, &cfg, slope, is_dk),
                        probability_gradient(f[1], query + 1, key, l1, &cfg, slope, is_dk),
                        probability_gradient(f[2], query, key + 8, l0, &cfg, slope, is_dk),
                        probability_gradient(f[3], query + 1, key + 8, l1, &cfg, slope, is_dk),
                    ];
                    nt += 1;
                }
                let mut dp = [[0.0f32; 4]; 4];
                if is_dk {
                    let fragments = query_fragments::<NFQ>(v_shared, &tile, layout);
                    compute_qk::<BF16, 1, NFQ, 4>(&mut dp, &fragments, do_shared, &tile, layout);
                }
                let mut packed = [[0u32; 2]; 4];
                nt = 0;
                while nt < 4 {
                    thread::__unroll_config::<0>();
                    let query = query_start + nt as u32 * 8 + tile.pair * 2;
                    let mask = transposed_keep(dropout, tile.bh, query, key);
                    let s = scores[nt];
                    let f = if is_dk {
                        let d0 = *delta.add(tile.bh as usize * rounded_q as usize + query as usize);
                        let d1 =
                            *delta.add(tile.bh as usize * rounded_q as usize + query as usize + 1);
                        let x = dp[nt];
                        [
                            s[0] * (x[0] * mask[0] - d0),
                            s[1] * (x[1] * mask[1] - d1),
                            s[2] * (x[2] * mask[2] - d0),
                            s[3] * (x[3] * mask[3] - d1),
                        ]
                    } else {
                        [
                            s[0] * mask[0],
                            s[1] * mask[1],
                            s[2] * mask[2],
                            s[3] * mask[3],
                        ]
                    };
                    packed[nt] = [cvt::<BF16>(f[0], f[1]), cvt::<BF16>(f[2], f[3])];
                    nt += 1;
                }
                let source = if is_dk { q_shared } else { do_shared };
                accumulate_values::<BF16, 1, 4, NFO, 2>(&mut acc, &packed, source, &tile, layout);
                thread::sync_threads();
                block += 1;
            }
            group_head += 1;
        }
        let output = if is_dk { dk } else { dv };
        let output_scale = if is_dk {
            gradient_scale
        } else {
            dropout.inv_keep
        };
        let output_heads = if PARTIAL { cfg.h } else { cfg.h / cfg.gqa };
        let output_stride = output_heads as usize * cfg.hs_u32 as usize;
        let output_base = b as usize * cfg.seqlen_k as usize * output_stride
            + output_head as usize * cfg.hs_u32 as usize;
        let mut dim = 0;
        while dim < NFO {
            thread::__unroll_config::<0>();
            let col = dim as u32 * 4 + tile.pair;
            if col < cfg.hs_u32 {
                if key < cfg.seqlen_k {
                    let index = output_base + (key as usize) * output_stride + col as usize;
                    if PARTIAL {
                        let partial = output.cast::<f32>().add(index * 2);
                        *partial = acc[dim][0] * output_scale;
                        *partial.add(1) = acc[dim][1] * output_scale;
                    } else {
                        *output.add(output_base + key as usize * output_stride + col as usize) =
                            cvt::<BF16>(acc[dim][0] * output_scale, acc[dim][1] * output_scale);
                    }
                }
                if key + 8 < cfg.seqlen_k {
                    let index = output_base + (key as usize + 8) * output_stride + col as usize;
                    if PARTIAL {
                        let partial = output.cast::<f32>().add(index * 2);
                        *partial = acc[dim][2] * output_scale;
                        *partial.add(1) = acc[dim][3] * output_scale;
                    } else {
                        *output
                            .add(output_base + (key as usize + 8) * output_stride + col as usize) =
                            cvt::<BF16>(acc[dim][2] * output_scale, acc[dim][3] * output_scale);
                    }
                }
            }
            dim += 1;
        }
    }
}
#[inline(always)]
pub(super) unsafe fn dkv_both_core<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const PARTIAL: bool,
    const DROPOUT: bool,
    const PLAIN: bool,
    const FUSED: bool,
    const ORDER: u32,
    const KSTRIDE: usize,
>(
    q: *const u32,
    k: *const u32,
    v: *const u32,
    dout: *const u32,
    lse: *const f32,
    delta: *const f32,
    dk: *mut u32,
    dv: *mut u32,
    dq: *mut f32,
    alibi: *const f32,
    cfg: FwdConfig,
    do_strides: Strides,
    gradient_scale: f32,
    dropout: Dropout,
) {
    unsafe {
        let tid = thread::threadIdx_x();
        let lane = tid % 32;
        let b = thread::blockIdx_z();
        let output_head = thread::blockIdx_y();
        let kh = if PARTIAL {
            output_head / cfg.gqa
        } else {
            output_head
        };
        let key_block = if ORDER == 3 && b % 2 == 1 {
            cfg.seqlen_k.div_ceil(64) - 1 - thread::blockIdx_x()
        } else {
            thread::blockIdx_x()
        };
        let key_start = key_block * 64;
        let mut tile = LaunchTile {
            tid,
            lane_id: lane,
            bh: 0,
            group: lane >> 2,
            pair: lane & 3,
            q_row_base: key_start,
            warp_row: (tid / 32) * 16,
            b,
            head: 0,
            kv_head: kh,
        };
        let key = key_start + tile.warp_row + tile.group;
        let layout = SmemLayout::new::<HD, 64, 32>();
        let scratch = DynamicSharedArray::<u32, 16>::get() as usize as *mut u32;
        let mut k_frag = [[0u32; 4]; NFQ];
        let mut v_frag = [[0u32; 4]; NFQ];
        let (q_shared, do_shared) = if FUSED || DROPOUT || !PLAIN {
            // K/V stay in shared memory to free registers before Philox.
            copy_rows_swizzled::<64, KSTRIDE>(
                k,
                scratch,
                b as usize * cfg.k.batch as usize + kh as usize * cfg.k.head as usize,
                key_start,
                cfg.k.seq,
                cfg.seqlen_k,
                cfg.hs_u32,
                tid,
            );
            copy_rows_swizzled::<64, KSTRIDE>(
                v,
                scratch.add(64 * KSTRIDE),
                b as usize * cfg.v.batch as usize + kh as usize * cfg.v.head as usize,
                key_start,
                cfg.v.seq,
                cfg.seqlen_k,
                cfg.hs_u32,
                tid,
            );
            ptx_asm!(
                "cp.async.commit_group; cp.async.wait_group 0;",
                clobber("memory")
            );
            thread::sync_threads();
            let q_shared = scratch.add(128 * KSTRIDE);
            (q_shared, q_shared.add(32 * layout.stride))
        } else {
            // Barriers protect each change of shared-memory ownership.
            copy_rows::<HD, 64>(
                k,
                scratch,
                b as usize * cfg.k.batch as usize + kh as usize * cfg.k.head as usize,
                key_start,
                cfg.k.seq,
                cfg.seqlen_k,
                cfg.hs_u32,
                tid,
            );
            ptx_asm!(
                "cp.async.commit_group; cp.async.wait_group 0;",
                clobber("memory")
            );
            thread::sync_threads();
            k_frag = query_fragments::<NFQ>(scratch, &tile, layout);
            thread::sync_threads();
            copy_rows::<HD, 64>(
                v,
                scratch,
                b as usize * cfg.v.batch as usize + kh as usize * cfg.v.head as usize,
                key_start,
                cfg.v.seq,
                cfg.seqlen_k,
                cfg.hs_u32,
                tid,
            );
            ptx_asm!(
                "cp.async.commit_group; cp.async.wait_group 0;",
                clobber("memory")
            );
            thread::sync_threads();
            v_frag = query_fragments::<NFQ>(scratch, &tile, layout);
            thread::sync_threads();
            (scratch, scratch.add(32 * layout.stride))
        };
        let rounded_q = cfg.seqlen_q.div_ceil(128) * 128;
        // Restrict the query loop to tiles that can attend to this key tile.
        let off = cfg.seqlen_k as i64 - cfg.seqlen_q as i64;
        let first = if cfg.window_right >= 0 {
            (key_start as i64 - off - cfg.window_right as i64).clamp(0, cfg.seqlen_q as i64) as u32
                / 32
        } else {
            0
        };
        let end = if cfg.window_left >= 0 {
            ((key_start + 64).min(cfg.seqlen_k) as i64 - off + cfg.window_left as i64)
                .clamp(0, cfg.seqlen_q as i64) as u32
        } else {
            cfg.seqlen_q
        }
        .div_ceil(32);
        let mut acc_k = [[0.0f32; 4]; NFO];
        let mut acc_v = [[0.0f32; 4]; NFO];
        let mut group_head = if PARTIAL { output_head % cfg.gqa } else { 0 };
        while group_head
            < if PARTIAL {
                output_head % cfg.gqa + 1
            } else {
                cfg.gqa
            }
        {
            tile.head = kh * cfg.gqa + group_head;
            tile.bh = b * cfg.h + tile.head;
            let q_base =
                b as usize * cfg.q.batch as usize + tile.head as usize * cfg.q.head as usize;
            let do_base = b as usize * do_strides.batch as usize
                + tile.head as usize * do_strides.head as usize;
            let slope = if cfg.has_alibi != 0 {
                *alibi.add(b as usize * cfg.alibi_bs as usize + tile.head as usize)
            } else {
                0.0
            };
            let mut block = first;
            while block < end {
                let query_start = block * 32;
                copy_rows::<HD, 32>(
                    q,
                    q_shared,
                    q_base,
                    query_start,
                    cfg.q.seq,
                    cfg.seqlen_q,
                    cfg.hs_u32,
                    tid,
                );
                copy_rows::<HD, 32>(
                    dout,
                    do_shared,
                    do_base,
                    query_start,
                    do_strides.seq,
                    cfg.seqlen_q,
                    cfg.hs_u32,
                    tid,
                );
                ptx_asm!(
                    "cp.async.commit_group; cp.async.wait_group 0;",
                    clobber("memory")
                );
                thread::sync_threads();
                let mut scores = [[0.0f32; 4]; 4];
                if FUSED || DROPOUT || !PLAIN {
                    let fragments = query_fragments_swizzled::<NFQ, KSTRIDE>(scratch, &tile);
                    compute_qk::<BF16, 1, NFQ, 4>(&mut scores, &fragments, q_shared, &tile, layout);
                } else {
                    compute_qk::<BF16, 1, NFQ, 4>(&mut scores, &k_frag, q_shared, &tile, layout);
                }
                let mut nt = 0;
                while nt < 4 {
                    thread::__unroll_config::<0>();
                    let query = query_start + nt as u32 * 8 + tile.pair * 2;
                    let l0 = if query < cfg.seqlen_q {
                        *lse.add(tile.bh as usize * cfg.seqlen_q as usize + query as usize)
                    } else {
                        f32::INFINITY
                    };
                    let l1 = if query + 1 < cfg.seqlen_q {
                        *lse.add(tile.bh as usize * cfg.seqlen_q as usize + query as usize + 1)
                    } else {
                        f32::INFINITY
                    };
                    let f = scores[nt];
                    scores[nt] = [
                        probability_gradient(f[0], query, key, l0, &cfg, slope, false),
                        probability_gradient(f[1], query + 1, key, l1, &cfg, slope, false),
                        probability_gradient(f[2], query, key + 8, l0, &cfg, slope, false),
                        probability_gradient(f[3], query + 1, key + 8, l1, &cfg, slope, false),
                    ];
                    nt += 1;
                }
                if DROPOUT {
                    let mut dp = [[0.0f32; 4]; 4];
                    {
                        let fragments = query_fragments_swizzled::<NFQ, KSTRIDE>(
                            scratch.add(64 * KSTRIDE),
                            &tile,
                        );
                        compute_qk::<BF16, 1, NFQ, 4>(
                            &mut dp, &fragments, do_shared, &tile, layout,
                        );
                    }
                    let mut packed_k = [[0u32; 2]; 4];
                    let mut packed_v = [[0u32; 2]; 4];
                    // 8-query fragments pick different bytes of the same words.
                    let mut mask_words = [0u32; 4];
                    nt = 0;
                    while nt < 4 {
                        thread::__unroll_config::<0>();
                        let query = query_start + nt as u32 * 8 + tile.pair * 2;
                        let mask = if FUSED {
                            if nt % 2 == 0 {
                                mask_words = transposed_paired_words(dropout, tile.bh, query, key);
                            }
                            transposed_word_mask(mask_words, dropout.threshold, query, key)
                        } else {
                            transposed_keep(dropout, tile.bh, query, key)
                        };
                        let d0 = *delta.add(tile.bh as usize * rounded_q as usize + query as usize);
                        let d1 =
                            *delta.add(tile.bh as usize * rounded_q as usize + query as usize + 1);
                        let p = scores[nt];
                        let x = dp[nt];
                        let dk = [
                            p[0] * (x[0] * mask[0] - d0),
                            p[1] * (x[1] * mask[1] - d1),
                            p[2] * (x[2] * mask[2] - d0),
                            p[3] * (x[3] * mask[3] - d1),
                        ];
                        let dv = [
                            p[0] * mask[0],
                            p[1] * mask[1],
                            p[2] * mask[2],
                            p[3] * mask[3],
                        ];
                        packed_k[nt] = [cvt::<BF16>(dk[0], dk[1]), cvt::<BF16>(dk[2], dk[3])];
                        packed_v[nt] = [cvt::<BF16>(dv[0], dv[1]), cvt::<BF16>(dv[2], dv[3])];
                        nt += 1;
                    }
                    accumulate_values::<BF16, 1, 4, NFO, 2>(
                        &mut acc_v, &packed_v, do_shared, &tile, layout,
                    );
                    accumulate_values::<BF16, 1, 4, NFO, 2>(
                        &mut acc_k, &packed_k, q_shared, &tile, layout,
                    );
                    if FUSED {
                        // Q/dOut are dead for every warp before dS replaces Q.
                        thread::sync_threads();
                        fused_store_ds(&packed_k, q_shared, tid);
                    }
                } else {
                    let mut packed = [[0u32; 2]; 4];
                    nt = 0;
                    while nt < 4 {
                        thread::__unroll_config::<0>();
                        let f = scores[nt];
                        packed[nt] = [cvt::<BF16>(f[0], f[1]), cvt::<BF16>(f[2], f[3])];
                        nt += 1;
                    }
                    accumulate_values::<BF16, 1, 4, NFO, 2>(
                        &mut acc_v, &packed, do_shared, &tile, layout,
                    );
                    let mut dp = [[0.0f32; 4]; 4];
                    if FUSED || !PLAIN {
                        let fragments = query_fragments_swizzled::<NFQ, KSTRIDE>(
                            scratch.add(64 * KSTRIDE),
                            &tile,
                        );
                        compute_qk::<BF16, 1, NFQ, 4>(
                            &mut dp, &fragments, do_shared, &tile, layout,
                        );
                    } else {
                        compute_qk::<BF16, 1, NFQ, 4>(&mut dp, &v_frag, do_shared, &tile, layout);
                    }
                    nt = 0;
                    while nt < 4 {
                        thread::__unroll_config::<0>();
                        let query = query_start + nt as u32 * 8 + tile.pair * 2;
                        let d0 = *delta.add(tile.bh as usize * rounded_q as usize + query as usize);
                        let d1 =
                            *delta.add(tile.bh as usize * rounded_q as usize + query as usize + 1);
                        let s = scores[nt];
                        let x = dp[nt];
                        let f = [
                            s[0] * (x[0] - d0),
                            s[1] * (x[1] - d1),
                            s[2] * (x[2] - d0),
                            s[3] * (x[3] - d1),
                        ];
                        packed[nt] = [cvt::<BF16>(f[0], f[1]), cvt::<BF16>(f[2], f[3])];
                        nt += 1;
                    }
                    accumulate_values::<BF16, 1, 4, NFO, 2>(
                        &mut acc_k, &packed, q_shared, &tile, layout,
                    );
                    if FUSED {
                        // Q/dOut are dead for every warp before dS replaces Q.
                        thread::sync_threads();
                        fused_store_ds(&packed, q_shared, tid);
                    }
                }
                if FUSED {
                    thread::sync_threads();
                    if HD == 128 {
                        fused_dq_tile::<BF16>(scratch, q_shared, dq, &cfg, &tile, query_start);
                    } else {
                        fused_dq_tile_d64::<BF16, HD, NFQ, KSTRIDE>(
                            scratch,
                            q_shared,
                            dq,
                            &cfg,
                            &tile,
                            query_start,
                        );
                    }
                }
                thread::sync_threads();
                block += 1;
            }
            group_head += 1;
        }
        let output_heads = if PARTIAL { cfg.h } else { cfg.h / cfg.gqa };
        let output_stride = output_heads as usize * cfg.hs_u32 as usize;
        let output_base = b as usize * cfg.seqlen_k as usize * output_stride
            + output_head as usize * cfg.hs_u32 as usize;
        store_dkv_tile::<BF16, NFO, PARTIAL>(
            dk,
            &acc_k,
            gradient_scale,
            &cfg,
            &tile,
            key,
            output_base,
            output_stride,
        );
        store_dkv_tile::<BF16, NFO, PARTIAL>(
            dv,
            &acc_v,
            if DROPOUT { dropout.inv_keep } else { 1.0 },
            &cfg,
            &tile,
            key,
            output_base,
            output_stride,
        );
    }
}
#[inline(always)]
unsafe fn store_dkv_tile<const BF16: bool, const NFO: usize, const PARTIAL: bool>(
    output: *mut u32,
    acc: &[[f32; 4]; NFO],
    output_scale: f32,
    cfg: &FwdConfig,
    tile: &LaunchTile,
    key: u32,
    output_base: usize,
    output_stride: usize,
) {
    unsafe {
        let mut dim = 0;
        while dim < NFO {
            thread::__unroll_config::<0>();
            let col = dim as u32 * 4 + tile.pair;
            if col < cfg.hs_u32 {
                if key < cfg.seqlen_k {
                    let index = output_base + (key as usize) * output_stride + col as usize;
                    if PARTIAL {
                        let partial = output.cast::<f32>().add(index * 2);
                        *partial = acc[dim][0] * output_scale;
                        *partial.add(1) = acc[dim][1] * output_scale;
                    } else {
                        *output.add(output_base + key as usize * output_stride + col as usize) =
                            cvt::<BF16>(acc[dim][0] * output_scale, acc[dim][1] * output_scale);
                    }
                }
                if key + 8 < cfg.seqlen_k {
                    let index = output_base + (key as usize + 8) * output_stride + col as usize;
                    if PARTIAL {
                        let partial = output.cast::<f32>().add(index * 2);
                        *partial = acc[dim][2] * output_scale;
                        *partial.add(1) = acc[dim][3] * output_scale;
                    } else {
                        *output
                            .add(output_base + (key as usize + 8) * output_stride + col as usize) =
                            cvt::<BF16>(acc[dim][2] * output_scale, acc[dim][3] * output_scale);
                    }
                }
            }
            dim += 1;
        }
    }
}
pub trait DkvSpec {
    const BOTH: bool;
    const DROPOUT: bool;
    const PLAIN: bool;
    unsafe fn run(
        q: *const u32,
        k: *const u32,
        v: *const u32,
        dout: *const u32,
        lse: *const f32,
        delta: *const f32,
        dk: *mut u32,
        dv: *mut u32,
        alibi: *const f32,
        cfg: FwdConfig,
        do_strides: Strides,
        gradient_scale: f32,
        dropout: Dropout,
    );
}
pub struct Dkv<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const PARTIAL: bool = false,
    const DROPOUT: bool = true,
    const PLAIN: bool = false,
    const BOTH: bool = false,
>;
impl<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const PARTIAL: bool,
    const DROPOUT: bool,
    const PLAIN: bool,
    const BOTH: bool,
> DkvSpec for Dkv<BF16, HD, NFQ, NFO, PARTIAL, DROPOUT, PLAIN, BOTH>
{
    const BOTH: bool = BOTH;
    const DROPOUT: bool = DROPOUT;
    const PLAIN: bool = PLAIN;
    #[inline(always)]
    unsafe fn run(
        q: *const u32,
        k: *const u32,
        v: *const u32,
        dout: *const u32,
        lse: *const f32,
        delta: *const f32,
        dk: *mut u32,
        dv: *mut u32,
        alibi: *const f32,
        cfg: FwdConfig,
        do_strides: Strides,
        gradient_scale: f32,
        dropout: Dropout,
    ) {
        unsafe {
            if BOTH {
                dkv_both_core::<BF16, HD, NFQ, NFO, PARTIAL, DROPOUT, PLAIN, false, 0, 64>(
                    q,
                    k,
                    v,
                    dout,
                    lse,
                    delta,
                    dk,
                    dv,
                    core::ptr::null_mut(),
                    alibi,
                    cfg,
                    do_strides,
                    gradient_scale,
                    dropout,
                );
                return;
            }
            dkv_core::<BF16, HD, NFQ, NFO, PARTIAL>(
                q,
                k,
                v,
                dout,
                lse,
                delta,
                dk,
                dv,
                alibi,
                cfg,
                do_strides,
                gradient_scale,
                dropout,
            );
        }
    }
}
#[allow(non_camel_case_types)]
pub type dkv_d32 = Dkv<false, 32, 2, 4>;
#[allow(non_camel_case_types)]
pub type dkv_d32_bf16 = Dkv<true, 32, 2, 4>;
#[allow(non_camel_case_types)]
pub type dkv_d64 = Dkv<false, 64, 4, 8>;
#[allow(non_camel_case_types)]
pub type dkv_d64_bf16 = Dkv<true, 64, 4, 8>;
#[allow(non_camel_case_types)]
pub type dkv_d96 = Dkv<false, 96, 6, 12>;
#[allow(non_camel_case_types)]
pub type dkv_d96_bf16 = Dkv<true, 96, 6, 12>;
#[allow(non_camel_case_types)]
pub type dkv_d128 = Dkv<false, 128, 8, 16>;
#[allow(non_camel_case_types)]
pub type dkv_d128_bf16 = Dkv<true, 128, 8, 16>;
#[allow(non_camel_case_types)]
pub type dkv_d192 = Dkv<false, 192, 12, 24>;
#[allow(non_camel_case_types)]
pub type dkv_d192_bf16 = Dkv<true, 192, 12, 24>;
#[allow(non_camel_case_types)]
pub type dkv_d256 = Dkv<false, 256, 16, 32>;
#[allow(non_camel_case_types)]
pub type dkv_d256_bf16 = Dkv<true, 256, 16, 32>;
#[allow(non_camel_case_types)]
pub type dkv_d128_partial = Dkv<false, 128, 8, 16, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_partial_bf16 = Dkv<true, 128, 8, 16, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_partial_no_dropout = Dkv<false, 128, 8, 16, true, false>;
#[allow(non_camel_case_types)]
pub type dkv_d128_partial_no_dropout_bf16 = Dkv<true, 128, 8, 16, true, false>;
#[allow(non_camel_case_types)]
pub type dkv_d128_partial_plain = Dkv<false, 128, 8, 16, true, false, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_partial_plain_bf16 = Dkv<true, 128, 8, 16, true, false, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_plain = Dkv<false, 128, 8, 16, false, false, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_plain_bf16 = Dkv<true, 128, 8, 16, false, false, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_no_dropout = Dkv<false, 128, 8, 16, false, false, false>;
#[allow(non_camel_case_types)]
pub type dkv_d128_no_dropout_bf16 = Dkv<true, 128, 8, 16, false, false, false>;
#[allow(non_camel_case_types)]
pub type dkv_d128_both = Dkv<false, 128, 8, 16, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_both_bf16 = Dkv<true, 128, 8, 16, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_partial_both = Dkv<false, 128, 8, 16, true, false, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_partial_both_bf16 = Dkv<true, 128, 8, 16, true, false, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_both_drop = Dkv<false, 128, 8, 16, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_both_drop_bf16 = Dkv<true, 128, 8, 16, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_partial_both_drop = Dkv<false, 128, 8, 16, true, true, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_partial_both_drop_bf16 = Dkv<true, 128, 8, 16, true, true, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_both_masked = Dkv<false, 128, 8, 16, false, false, false, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_both_masked_bf16 = Dkv<true, 128, 8, 16, false, false, false, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_both_masked_drop = Dkv<false, 128, 8, 16, false, true, false, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_both_masked_drop_bf16 = Dkv<true, 128, 8, 16, false, true, false, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_partial_both_masked = Dkv<false, 128, 8, 16, true, false, false, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_partial_both_masked_bf16 = Dkv<true, 128, 8, 16, true, false, false, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_partial_both_masked_drop = Dkv<false, 128, 8, 16, true, true, false, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_partial_both_masked_drop_bf16 = Dkv<true, 128, 8, 16, true, true, false, true>;
#[kernel(
    dkv_d128_both_masked,
    dkv_d128_both_masked_bf16,
    dkv_d128_both_masked_drop,
    dkv_d128_both_masked_drop_bf16,
    dkv_d128_partial_both_masked,
    dkv_d128_partial_both_masked_bf16,
    dkv_d128_partial_both_masked_drop,
    dkv_d128_partial_both_masked_drop_bf16,
    dkv_d128_both_drop,
    dkv_d128_both_drop_bf16,
    dkv_d128_partial_both_drop,
    dkv_d128_partial_both_drop_bf16,
    dkv_d128_both,
    dkv_d128_both_bf16,
    dkv_d128_partial_both,
    dkv_d128_partial_both_bf16,
    dkv_d128_plain,
    dkv_d128_plain_bf16,
    dkv_d128_no_dropout,
    dkv_d128_no_dropout_bf16,
    dkv_d128_partial_plain,
    dkv_d128_partial_plain_bf16,
    dkv_d128_partial_no_dropout,
    dkv_d128_partial_no_dropout_bf16,
    dkv_d128_partial,
    dkv_d128_partial_bf16,
    dkv_d32,
    dkv_d32_bf16,
    dkv_d64,
    dkv_d64_bf16,
    dkv_d96,
    dkv_d96_bf16,
    dkv_d128,
    dkv_d128_bf16,
    dkv_d192,
    dkv_d192_bf16,
    dkv_d256,
    dkv_d256_bf16
)]
#[launch_bounds(128)]
pub fn backward_kv<S: DkvSpec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    dout: &[u32],
    lse: &[f32],
    delta: &[f32],
    dk: DisjointSlice<u32>,
    dv: DisjointSlice<u32>,
    alibi: &[f32],
    rng: &[u64],
    scale: f32,
    tanh_scale: f32,
    window_left: i32,
    window_right: i32,
    seqlen_q: u32,
    seqlen_k: u32,
    has_alibi: u32,
    alibi_causal: u32,
    h: u32,
    gqa: u32,
    hs_u32: u32,
    q_bs: u32,
    q_ss: u32,
    q_hs: u32,
    k_bs: u32,
    k_ss: u32,
    k_hs: u32,
    v_bs: u32,
    v_ss: u32,
    v_hs: u32,
    alibi_bs: u32,
    do_bs: u32,
    do_ss: u32,
    do_hs: u32,
    gradient_scale: f32,
    keep: f32,
) {
    let mut dk = dk;
    let mut dv = dv;
    let cfg = FwdConfig::new(
        scale,
        if S::PLAIN || S::BOTH { 0.0 } else { tanh_scale },
        if S::PLAIN { -1 } else { window_left },
        if S::PLAIN { -1 } else { window_right },
        seqlen_q,
        seqlen_k,
        if S::PLAIN { 0 } else { has_alibi },
        if S::PLAIN { 0 } else { alibi_causal },
        h,
        gqa,
        hs_u32,
        q_bs,
        q_ss,
        q_hs,
        k_bs,
        k_ss,
        k_hs,
        v_bs,
        v_ss,
        v_hs,
        alibi_bs,
    );
    // Init outside the branch: the backend drops const fields across a phi.
    let seed = if S::DROPOUT && keep < 1.0 {
        unsafe { *rng.as_ptr() }
    } else {
        0
    };
    let offset = if S::DROPOUT && keep < 1.0 {
        unsafe { *rng.as_ptr().add(1) }
    } else {
        0
    };
    let dropout = Dropout {
        seed,
        offset,
        threshold: if S::DROPOUT {
            (keep * 255.0) as u32
        } else {
            255
        },
        inv_keep: if S::DROPOUT { 1.0 / keep } else { 1.0 },
        ..Dropout::NONE
    };
    unsafe {
        S::run(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            dout.as_ptr(),
            lse.as_ptr(),
            delta.as_ptr(),
            dk.as_mut_ptr(),
            dv.as_mut_ptr(),
            alibi.as_ptr(),
            cfg,
            Strides {
                batch: do_bs,
                seq: do_ss,
                head: do_hs,
            },
            gradient_scale,
            dropout,
        );
    }
}
