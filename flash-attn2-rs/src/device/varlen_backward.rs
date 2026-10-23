use crate::device::backward::{
    ReduceSpec, bf16_8, bf16_16, bf16_32, copy_rows, f16_8, f16_16, f16_32, probability_gradient,
    query_fragments, transposed_paired_words, transposed_word_mask, unpack,
};
use crate::device::config::{FwdConfig, GRID_PLAIN, KvRange, LaunchTile, Strides, TensorBases};
use crate::device::dropout::Dropout;
use crate::device::forward::{SmemLayout, accumulate_values, compute_qk};
use crate::device::inlined::cvt;
use cuda_device::shared::DynamicSharedArray;
use cuda_device::warp::shuffle_xor_f32;
use cuda_device::{DisjointSlice, kernel, launch_bounds, ptx_asm, thread};

#[derive(Clone, Copy)]
pub struct Packed {
    cuq: *const i32,
    cuk: *const i32,
    total_q: u32,
    total_k: u32,
    delta_stride: u32,
}
#[derive(Clone, Copy)]
struct Sequence {
    qs: u32,
    ks: u32,
    sq: u32,
    sk: u32,
}
impl Packed {
    #[inline(always)]
    unsafe fn sequence(self, b: u32) -> Sequence {
        unsafe {
            let qs = *self.cuq.add(b as usize) as u32;
            let ks = *self.cuk.add(b as usize) as u32;
            Sequence {
                qs,
                ks,
                sq: (*self.cuq.add(b as usize + 1) as u32).wrapping_sub(qs),
                sk: (*self.cuk.add(b as usize + 1) as u32).wrapping_sub(ks),
            }
        }
    }
    #[inline(always)]
    fn valid(self, seq: Sequence, cfg: &FwdConfig) -> bool {
        seq.qs <= self.total_q
            && seq.sq <= self.total_q - seq.qs
            && seq.sq <= cfg.seqlen_q
            && seq.ks <= self.total_k
            && seq.sk <= self.total_k - seq.ks
            && seq.sk <= cfg.seqlen_k
    }
}

#[kernel(f16_8, f16_16, f16_32, bf16_8, bf16_16, bf16_32)]
#[launch_bounds(256)]
pub fn varlen_preprocess<S: ReduceSpec>(
    dout: &[u32],
    out: &[u32],
    delta: DisjointSlice<f32>,
    cuq: &[i32],
    total_q: u32,
    delta_stride: u32,
    max_q: u32,
    dim_pairs: u32,
    do_seq: u32,
    do_head: u32,
    out_seq: u32,
    out_head: u32,
    keep: f32,
) {
    let mut delta = delta;
    let b = thread::blockIdx_z();
    let h = thread::blockIdx_y();
    let tid = thread::threadIdx_x();
    let lane = tid % S::LANES;
    let row = thread::blockIdx_x() * (256 / S::LANES) + tid / S::LANES;
    let start = unsafe { *cuq.as_ptr().add(b as usize) } as u32;
    let end = unsafe { *cuq.as_ptr().add(b as usize + 1) } as u32;
    if start > total_q || end < start || end > total_q || end - start > max_q {
        return;
    }
    let seq = end - start;
    if row >= seq.div_ceil(128) * 128 {
        return;
    }
    let mut sum = 0.0f32;
    if row < seq {
        let a_base = (start + row) as usize * do_seq as usize + h as usize * do_head as usize;
        let b_base = (start + row) as usize * out_seq as usize + h as usize * out_head as usize;
        let mut d = lane;
        while d < dim_pairs {
            let (a, b) = unsafe {
                (
                    *dout.as_ptr().add(a_base + d as usize),
                    *out.as_ptr().add(b_base + d as usize),
                )
            };
            let (a0, a1, b0, b1) = if S::BF16 {
                (
                    unpack::<true>(a),
                    unpack::<true>(a >> 16),
                    unpack::<true>(b),
                    unpack::<true>(b >> 16),
                )
            } else {
                (
                    unpack::<false>(a),
                    unpack::<false>(a >> 16),
                    unpack::<false>(b),
                    unpack::<false>(b >> 16),
                )
            };
            sum += a0 * b0;
            sum += a1 * b1;
            d += S::LANES;
        }
    }
    let mut distance = S::LANES / 2;
    while distance > 0 {
        sum += shuffle_xor_f32(sum, distance);
        distance /= 2;
    }
    if lane == 0 {
        unsafe {
            *delta.as_mut_ptr().add(
                h as usize * delta_stride as usize
                    + start as usize
                    + b as usize * 128
                    + row as usize,
            ) = sum * keep;
        }
    }
}
#[inline(always)]
unsafe fn dq_core<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const SPLITS: u32,
>(
    q: *const u32,
    k: *const u32,
    v: *const u32,
    dout: *const u32,
    lse: *const f32,
    delta: *const f32,
    dq: *mut u32,
    alibi: *const f32,
    mut cfg: FwdConfig,
    do_strides: Strides,
    gradient_scale: f32,
    dropout: Dropout,
    packed_meta: Packed,
) {
    unsafe {
        let mut tile = LaunchTile::new::<64, 1, GRID_PLAIN>(cfg.h, cfg.gqa, cfg.seqlen_q, false);
        let split = thread::blockIdx_x() % SPLITS;
        if SPLITS > 1 {
            tile.q_row_base = ((cfg.seqlen_q - 1) / 64 - thread::blockIdx_x() / SPLITS) * 64;
        }
        let seq = packed_meta.sequence(tile.b);
        if !packed_meta.valid(seq, &cfg) || tile.q_row_base >= seq.sq {
            return;
        }
        cfg.seqlen_q = seq.sq;
        cfg.seqlen_k = seq.sk;
        let out_stride = cfg.h as usize * cfg.hs_u32 as usize;
        let bases = TensorBases {
            q: seq.qs as usize * cfg.q.seq as usize + tile.head as usize * cfg.q.head as usize,
            k: seq.ks as usize * cfg.k.seq as usize + tile.kv_head as usize * cfg.k.head as usize,
            v: seq.ks as usize * cfg.v.seq as usize + tile.kv_head as usize * cfg.v.head as usize,
            o: seq.qs as usize * out_stride + tile.head as usize * cfg.hs_u32 as usize,
            o_ss: out_stride,
        };
        let layout = SmemLayout::new::<HD, 32, 64>();
        let q_shared = DynamicSharedArray::<u32, 16>::get() as usize as *mut u32;
        let do_shared = q_shared.add(64 * layout.stride);
        let k_shared = q_shared.add(128 * layout.stride);
        let v_shared = q_shared.add(160 * layout.stride);
        let do_base = seq.qs as usize * do_strides.seq as usize
            + tile.head as usize * do_strides.head as usize;
        copy_rows::<HD, 64>(
            q,
            q_shared,
            bases.q,
            tile.q_row_base,
            cfg.q.seq,
            cfg.seqlen_q,
            cfg.hs_u32,
            tile.tid,
        );
        copy_rows::<HD, 64>(
            dout,
            do_shared,
            do_base,
            tile.q_row_base,
            do_strides.seq,
            cfg.seqlen_q,
            cfg.hs_u32,
            tile.tid,
        );
        ptx_asm!(
            "cp.async.commit_group; cp.async.wait_group 0;",
            clobber("memory")
        );
        thread::sync_threads();
        let row = tile.q_row_base + tile.warp_row + tile.group;
        let lse_base = tile.head as usize * packed_meta.total_q as usize + seq.qs as usize;
        let delta_base = tile.head as usize * packed_meta.delta_stride as usize
            + seq.qs as usize
            + tile.b as usize * 128;
        let l0 = if row < cfg.seqlen_q {
            *lse.add(lse_base + row as usize)
        } else {
            f32::INFINITY
        };
        let l1 = if row + 8 < cfg.seqlen_q {
            *lse.add(lse_base + row as usize + 8)
        } else {
            f32::INFINITY
        };
        let d0 = *delta.add(delta_base + row as usize);
        let d1 = *delta.add(delta_base + row as usize + 8);
        let slope = if cfg.has_alibi != 0 {
            *alibi.add(tile.b as usize * cfg.alibi_bs as usize + tile.head as usize)
        } else {
            0.0
        };
        let range = KvRange::new::<64, 32>(
            &cfg,
            tile.q_row_base,
            cfg.seqlen_k as i32 - cfg.seqlen_q as i32,
        );
        let mut acc = [[0.0f32; 4]; NFO];
        // Empty partitions still write zero partials.
        let (first, end) = if SPLITS > 1 {
            let chunk = range
                .num_tiles
                .saturating_sub(range.start_tile)
                .div_ceil(SPLITS);
            let first = range.start_tile + split * chunk;
            (first, (first + chunk).min(range.num_tiles))
        } else {
            (range.start_tile, range.num_tiles)
        };
        let mut block = first;
        while block < end {
            let key_start = block * 32;
            copy_rows::<HD, 32>(
                k,
                k_shared,
                bases.k,
                key_start,
                cfg.k.seq,
                cfg.seqlen_k,
                cfg.hs_u32,
                tile.tid,
            );
            copy_rows::<HD, 32>(
                v,
                v_shared,
                bases.v,
                key_start,
                cfg.v.seq,
                cfg.seqlen_k,
                cfg.hs_u32,
                tile.tid,
            );
            ptx_asm!(
                "cp.async.commit_group; cp.async.wait_group 0;",
                clobber("memory")
            );
            thread::sync_threads();
            let mut scores = [[0.0f32; 4]; 4];
            {
                let q_frag = query_fragments::<NFQ>(q_shared, &tile, layout);
                compute_qk::<BF16, 1, NFQ, 4>(&mut scores, &q_frag, k_shared, &tile, layout);
            }
            let mut nt = 0;
            while nt < 4 {
                thread::__unroll_config::<0>();
                let col = key_start + nt as u32 * 8 + tile.pair * 2;
                let f = scores[nt];
                scores[nt] = [
                    probability_gradient(f[0], row, col, l0, &cfg, slope, true),
                    probability_gradient(f[1], row, col + 1, l0, &cfg, slope, true),
                    probability_gradient(f[2], row + 8, col, l1, &cfg, slope, true),
                    probability_gradient(f[3], row + 8, col + 1, l1, &cfg, slope, true),
                ];
                nt += 1;
            }
            let mut dp = [[0.0f32; 4]; 4];
            {
                let do_frag = query_fragments::<NFQ>(do_shared, &tile, layout);
                compute_qk::<BF16, 1, NFQ, 4>(&mut dp, &do_frag, v_shared, &tile, layout);
            }
            let random = if dropout.threshold < 255 {
                dropout.random(
                    (tile.q_row_base + tile.warp_row) / 16,
                    block,
                    dropout
                        .offset
                        .wrapping_add(tile.bh as u64 * 32 + tile.lane_id as u64),
                )
            } else {
                [0u32; 4]
            };
            let mut packed = [[0u32; 2]; 4];
            nt = 0;
            while nt < 4 {
                thread::__unroll_config::<0>();
                let r = random[nt];
                let s = scores[nt];
                let x = dp[nt];
                let a = s[0]
                    * ((if (r & 255) <= dropout.threshold {
                        x[0]
                    } else {
                        0.0
                    }) - d0);
                let b = s[1]
                    * ((if ((r >> 8) & 255) <= dropout.threshold {
                        x[1]
                    } else {
                        0.0
                    }) - d0);
                let c = s[2]
                    * ((if ((r >> 16) & 255) <= dropout.threshold {
                        x[2]
                    } else {
                        0.0
                    }) - d1);
                let d = s[3]
                    * ((if (r >> 24) <= dropout.threshold {
                        x[3]
                    } else {
                        0.0
                    }) - d1);
                packed[nt] = [cvt::<BF16>(a, b), cvt::<BF16>(c, d)];
                nt += 1;
            }
            accumulate_values::<BF16, 1, 4, NFO, 2>(&mut acc, &packed, k_shared, &tile, layout);
            thread::sync_threads();
            block += 1;
        }
        let mut dim = 0;
        while dim < NFO {
            thread::__unroll_config::<0>();
            let col = dim as u32 * 4 + tile.pair;
            if col < cfg.hs_u32 {
                if row < cfg.seqlen_q {
                    if SPLITS > 1 {
                        let plane = packed_meta.total_q as usize * bases.o_ss;
                        let index = (split as usize * plane
                            + bases.o
                            + row as usize * bases.o_ss
                            + col as usize)
                            * 2;
                        *(dq as *mut f32).add(index) = acc[dim][0];
                        *(dq as *mut f32).add(index + 1) = acc[dim][1];
                    } else {
                        *dq.add(bases.o + row as usize * bases.o_ss + col as usize) =
                            cvt::<BF16>(acc[dim][0] * gradient_scale, acc[dim][1] * gradient_scale);
                    }
                }
                if row + 8 < cfg.seqlen_q {
                    if SPLITS > 1 {
                        let plane = packed_meta.total_q as usize * bases.o_ss;
                        let index = (split as usize * plane
                            + bases.o
                            + (row as usize + 8) * bases.o_ss
                            + col as usize)
                            * 2;
                        *(dq as *mut f32).add(index) = acc[dim][2];
                        *(dq as *mut f32).add(index + 1) = acc[dim][3];
                    } else {
                        *dq.add(bases.o + (row as usize + 8) * bases.o_ss + col as usize) =
                            cvt::<BF16>(acc[dim][2] * gradient_scale, acc[dim][3] * gradient_scale);
                    }
                }
            }
            dim += 1;
        }
    }
}
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
    mut cfg: FwdConfig,
    do_strides: Strides,
    gradient_scale: f32,
    dropout: Dropout,
    packed_meta: Packed,
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
        let seq = packed_meta.sequence(b);
        if !packed_meta.valid(seq, &cfg) || key_start >= seq.sk {
            return;
        }
        cfg.seqlen_q = seq.sq;
        cfg.seqlen_k = seq.sk;
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
            seq.ks as usize * cfg.k.seq as usize + kh as usize * cfg.k.head as usize,
            key_start,
            cfg.k.seq,
            cfg.seqlen_k,
            cfg.hs_u32,
            tid,
        );
        copy_rows::<HD, 32>(
            v,
            v_shared,
            seq.ks as usize * cfg.v.seq as usize + kh as usize * cfg.v.head as usize,
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
        let first_head = if PARTIAL { output_head % cfg.gqa } else { 0 };
        let last_head = if PARTIAL { first_head + 1 } else { cfg.gqa };
        let mut group_head = first_head;
        while group_head < last_head {
            tile.head = kh * cfg.gqa + group_head;
            tile.bh = b * cfg.h + tile.head;
            let lse_base = tile.head as usize * packed_meta.total_q as usize + seq.qs as usize;
            let delta_base = tile.head as usize * packed_meta.delta_stride as usize
                + seq.qs as usize
                + b as usize * 128;
            let q_base =
                seq.qs as usize * cfg.q.seq as usize + tile.head as usize * cfg.q.head as usize;
            let do_base = seq.qs as usize * do_strides.seq as usize
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
                        *lse.add(lse_base + query as usize)
                    } else {
                        f32::INFINITY
                    };
                    let l1 = if query + 1 < cfg.seqlen_q {
                        *lse.add(lse_base + query as usize + 1)
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
                let mut mask_words = [0u32; 4];
                nt = 0;
                while nt < 4 {
                    thread::__unroll_config::<0>();
                    let query = query_start + nt as u32 * 8 + tile.pair * 2;
                    // Key lanes exchange shared Philox words.
                    if nt % 2 == 0 {
                        mask_words = transposed_paired_words(dropout, tile.bh, query, key);
                    }
                    let mask = transposed_word_mask(mask_words, dropout.threshold, query, key);
                    let s = scores[nt];
                    let f = if is_dk {
                        let d0 = *delta.add(delta_base + query as usize);
                        let d1 = *delta.add(delta_base + query as usize + 1);
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
        let output_base =
            seq.ks as usize * output_stride + output_head as usize * cfg.hs_u32 as usize;
        let mut dim = 0;
        while dim < NFO {
            thread::__unroll_config::<0>();
            let col = dim as u32 * 4 + tile.pair;
            if col < cfg.hs_u32 {
                if key < cfg.seqlen_k {
                    let index = output_base + (key) as usize * output_stride + col as usize;
                    if PARTIAL {
                        let partial = output.cast::<f32>().add(index * 2);
                        *partial = acc[dim][0] * output_scale;
                        *partial.add(1) = acc[dim][1] * output_scale;
                    } else {
                        *output.add(index) =
                            cvt::<BF16>(acc[dim][0] * output_scale, acc[dim][1] * output_scale);
                    }
                }
                if key + 8 < cfg.seqlen_k {
                    let index = output_base + (key + 8) as usize * output_stride + col as usize;
                    if PARTIAL {
                        let partial = output.cast::<f32>().add(index * 2);
                        *partial = acc[dim][2] * output_scale;
                        *partial.add(1) = acc[dim][3] * output_scale;
                    } else {
                        *output.add(index) =
                            cvt::<BF16>(acc[dim][2] * output_scale, acc[dim][3] * output_scale);
                    }
                }
            }
            dim += 1;
        }
    }
}
pub trait Spec {
    unsafe fn run(
        q: *const u32,
        k: *const u32,
        v: *const u32,
        dout: *const u32,
        lse: *const f32,
        delta: *const f32,
        dq: *mut u32,
        dk: *mut u32,
        dv: *mut u32,
        alibi: *const f32,
        cfg: FwdConfig,
        do_strides: Strides,
        gradient_scale: f32,
        dropout: Dropout,
        meta: Packed,
    );
}
pub struct Backward<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const KV: bool,
    const PARTIAL: bool = false,
    const SPLITS: u32 = 1,
    const CAUSAL: bool = false,
>;
impl<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const KV: bool,
    const PARTIAL: bool,
    const SPLITS: u32,
    const CAUSAL: bool,
> Spec for Backward<BF16, HD, NFQ, NFO, KV, PARTIAL, SPLITS, CAUSAL>
{
    #[inline(always)]
    unsafe fn run(
        q: *const u32,
        k: *const u32,
        v: *const u32,
        dout: *const u32,
        lse: *const f32,
        delta: *const f32,
        dq: *mut u32,
        dk: *mut u32,
        dv: *mut u32,
        alibi: *const f32,
        mut cfg: FwdConfig,
        do_strides: Strides,
        gradient_scale: f32,
        dropout: Dropout,
        meta: Packed,
    ) {
        // Host restricts this to unbounded causal without ALiBi or softcap.
        if CAUSAL {
            cfg.tanh_scale = 0.0;
            cfg.has_alibi = 0;
            cfg.window_left = -1;
            cfg.window_right = 0;
        }
        unsafe {
            if KV {
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
                    meta,
                );
            } else {
                dq_core::<BF16, HD, NFQ, NFO, SPLITS>(
                    q,
                    k,
                    v,
                    dout,
                    lse,
                    delta,
                    dq,
                    alibi,
                    cfg,
                    do_strides,
                    gradient_scale,
                    dropout,
                    meta,
                );
            }
        }
    }
}
#[allow(non_camel_case_types)]
pub type dq_d32 = Backward<false, 32, 2, 4, false>;
#[allow(non_camel_case_types)]
pub type dkv_d32 = Backward<false, 32, 2, 4, true>;
#[allow(non_camel_case_types)]
pub type dq_d32_bf16 = Backward<true, 32, 2, 4, false>;
#[allow(non_camel_case_types)]
pub type dkv_d32_bf16 = Backward<true, 32, 2, 4, true>;
#[allow(non_camel_case_types)]
pub type dq_d64 = Backward<false, 64, 4, 8, false>;
#[allow(non_camel_case_types)]
pub type dkv_d64 = Backward<false, 64, 4, 8, true>;
#[allow(non_camel_case_types)]
pub type dq_d64_bf16 = Backward<true, 64, 4, 8, false>;
#[allow(non_camel_case_types)]
pub type dkv_d64_bf16 = Backward<true, 64, 4, 8, true>;
#[allow(non_camel_case_types)]
pub type dq_d96 = Backward<false, 96, 6, 12, false>;
#[allow(non_camel_case_types)]
pub type dkv_d96 = Backward<false, 96, 6, 12, true>;
#[allow(non_camel_case_types)]
pub type dq_d96_bf16 = Backward<true, 96, 6, 12, false>;
#[allow(non_camel_case_types)]
pub type dkv_d96_bf16 = Backward<true, 96, 6, 12, true>;
#[allow(non_camel_case_types)]
pub type dq_d128 = Backward<false, 128, 8, 16, false>;
#[allow(non_camel_case_types)]
pub type dkv_d128 = Backward<false, 128, 8, 16, true>;
#[allow(non_camel_case_types)]
pub type dq_d128_bf16 = Backward<true, 128, 8, 16, false>;
#[allow(non_camel_case_types)]
pub type dkv_d128_bf16 = Backward<true, 128, 8, 16, true>;
#[allow(non_camel_case_types)]
pub type dq_d192 = Backward<false, 192, 12, 24, false>;
#[allow(non_camel_case_types)]
pub type dkv_d192 = Backward<false, 192, 12, 24, true>;
#[allow(non_camel_case_types)]
pub type dq_d192_bf16 = Backward<true, 192, 12, 24, false>;
#[allow(non_camel_case_types)]
pub type dkv_d192_bf16 = Backward<true, 192, 12, 24, true>;
#[allow(non_camel_case_types)]
pub type dq_d256 = Backward<false, 256, 16, 32, false>;
#[allow(non_camel_case_types)]
pub type dkv_d256 = Backward<false, 256, 16, 32, true>;
#[allow(non_camel_case_types)]
pub type dq_d256_bf16 = Backward<true, 256, 16, 32, false>;
#[allow(non_camel_case_types)]
pub type dkv_d256_bf16 = Backward<true, 256, 16, 32, true>;
#[allow(non_camel_case_types)]
pub type dkv_d32_partial = Backward<false, 32, 2, 4, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d32_partial_bf16 = Backward<true, 32, 2, 4, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d64_partial = Backward<false, 64, 4, 8, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d64_partial_bf16 = Backward<true, 64, 4, 8, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d96_partial = Backward<false, 96, 6, 12, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d96_partial_bf16 = Backward<true, 96, 6, 12, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_partial = Backward<false, 128, 8, 16, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d128_partial_bf16 = Backward<true, 128, 8, 16, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d192_partial = Backward<false, 192, 12, 24, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d192_partial_bf16 = Backward<true, 192, 12, 24, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d256_partial = Backward<false, 256, 16, 32, true, true>;
#[allow(non_camel_case_types)]
pub type dkv_d256_partial_bf16 = Backward<true, 256, 16, 32, true, true>;
#[allow(non_camel_case_types)]
pub type dq_d64_split2 = Backward<false, 64, 4, 8, false, false, 2>;
#[allow(non_camel_case_types)]
pub type dq_d64_split2_bf16 = Backward<true, 64, 4, 8, false, false, 2>;
#[allow(non_camel_case_types)]
pub type dq_d128_split2 = Backward<false, 128, 8, 16, false, false, 2>;
#[allow(non_camel_case_types)]
pub type dq_d128_split2_bf16 = Backward<true, 128, 8, 16, false, false, 2>;
#[allow(non_camel_case_types)]
pub type dkv_d64_partial_causal = Backward<false, 64, 4, 8, true, true, 1, true>;
#[allow(non_camel_case_types)]
pub type dkv_d64_partial_causal_bf16 = Backward<true, 64, 4, 8, true, true, 1, true>;
#[allow(non_camel_case_types)]
pub type dq_d64_split2_causal = Backward<false, 64, 4, 8, false, false, 2, true>;
#[allow(non_camel_case_types)]
pub type dq_d64_split2_causal_bf16 = Backward<true, 64, 4, 8, false, false, 2, true>;
#[kernel(
    dq_d64_split2_causal,
    dq_d64_split2_causal_bf16,
    dkv_d64_partial_causal,
    dkv_d64_partial_causal_bf16,
    dq_d128_split2,
    dq_d128_split2_bf16,
    dq_d64_split2,
    dq_d64_split2_bf16,
    dkv_d32_partial,
    dkv_d32_partial_bf16,
    dkv_d64_partial,
    dkv_d64_partial_bf16,
    dkv_d96_partial,
    dkv_d96_partial_bf16,
    dkv_d128_partial,
    dkv_d128_partial_bf16,
    dkv_d192_partial,
    dkv_d192_partial_bf16,
    dkv_d256_partial,
    dkv_d256_partial_bf16,
    dq_d32,
    dkv_d32,
    dq_d32_bf16,
    dkv_d32_bf16,
    dq_d64,
    dkv_d64,
    dq_d64_bf16,
    dkv_d64_bf16,
    dq_d96,
    dkv_d96,
    dq_d96_bf16,
    dkv_d96_bf16,
    dq_d128,
    dkv_d128,
    dq_d128_bf16,
    dkv_d128_bf16,
    dq_d192,
    dkv_d192,
    dq_d192_bf16,
    dkv_d192_bf16,
    dq_d256,
    dkv_d256,
    dq_d256_bf16,
    dkv_d256_bf16
)]
#[launch_bounds(128)]
pub fn varlen_backward<S: Spec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    dout: &[u32],
    lse: &[f32],
    delta: &[f32],
    dq: DisjointSlice<u32>,
    dk: DisjointSlice<u32>,
    dv: DisjointSlice<u32>,
    alibi: &[f32],
    rng: &[u64],
    cuq: &[i32],
    cuk: &[i32],
    scale: f32,
    tanh_scale: f32,
    window_left: i32,
    window_right: i32,
    max_q: u32,
    max_k: u32,
    has_alibi: u32,
    alibi_causal: u32,
    h: u32,
    gqa: u32,
    hs_u32: u32,
    q_ss: u32,
    q_hs: u32,
    k_ss: u32,
    k_hs: u32,
    v_ss: u32,
    v_hs: u32,
    alibi_bs: u32,
    do_ss: u32,
    do_hs: u32,
    total_q: u32,
    total_k: u32,
    delta_stride: u32,
    gradient_scale: f32,
    keep: f32,
) {
    let mut dq = dq;
    let mut dk = dk;
    let mut dv = dv;
    let cfg = FwdConfig::new(
        scale,
        tanh_scale,
        window_left,
        window_right,
        max_q,
        max_k,
        has_alibi,
        alibi_causal,
        h,
        gqa,
        hs_u32,
        0,
        q_ss,
        q_hs,
        0,
        k_ss,
        k_hs,
        0,
        v_ss,
        v_hs,
        alibi_bs,
    );
    let seed = if keep < 1.0 {
        unsafe { *rng.as_ptr() }
    } else {
        0
    };
    let offset = if keep < 1.0 {
        unsafe { *rng.as_ptr().add(1) }
    } else {
        0
    };
    let dropout = Dropout {
        seed,
        offset,
        threshold: (keep * 255.0) as u32,
        inv_keep: 1.0 / keep,
        ..Dropout::NONE
    };
    let meta = Packed {
        cuq: cuq.as_ptr(),
        cuk: cuk.as_ptr(),
        total_q,
        total_k,
        delta_stride,
    };
    unsafe {
        S::run(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            dout.as_ptr(),
            lse.as_ptr(),
            delta.as_ptr(),
            dq.as_mut_ptr(),
            dk.as_mut_ptr(),
            dv.as_mut_ptr(),
            alibi.as_ptr(),
            cfg,
            Strides {
                batch: 0,
                seq: do_ss,
                head: do_hs,
            },
            gradient_scale,
            dropout,
            meta,
        );
    }
}
