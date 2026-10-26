use super::common::{
    copy_rows, copy_rows_swizzled, probability_gradient, query_fragments, query_fragments_swizzled,
};
use crate::device::config::{FwdConfig, GRID_PLAIN, KvRange, LaunchTile, Strides, TensorBases};
use crate::device::dropout::Dropout;
use crate::device::forward::{SmemLayout, accumulate_values, compute_qk};
use crate::device::inlined::cvt;
use cuda_device::shared::DynamicSharedArray;
use cuda_device::{DisjointSlice, kernel, launch_bounds, ptx_asm, thread};

#[inline(always)]
unsafe fn dq_core<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const SWIZZLED_Q: bool,
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
    cfg: FwdConfig,
    do_strides: Strides,
    gradient_scale: f32,
    dropout: Dropout,
) {
    unsafe {
        let mut tile = LaunchTile::new::<64, 1, GRID_PLAIN>(cfg.h, cfg.gqa, cfg.seqlen_q, false);
        let split = if SPLITS > 1 {
            thread::blockIdx_x() % SPLITS
        } else {
            0
        };
        if SPLITS > 1 {
            tile.q_row_base = ((cfg.seqlen_q - 1) / 64 - thread::blockIdx_x() / SPLITS) * 64;
        }
        let bases = TensorBases::new(&cfg, &tile);
        let layout = SmemLayout::new::<HD, 32, 64>();
        let q_shared = DynamicSharedArray::<u32, 16>::get() as usize as *mut u32;
        let do_shared = q_shared.add(if SWIZZLED_Q {
            64 * 64
        } else {
            64 * layout.stride
        });
        let k_shared = q_shared.add(if SWIZZLED_Q {
            128 * 64
        } else {
            128 * layout.stride
        });
        let v_shared = q_shared.add(if SWIZZLED_Q {
            128 * 64 + 32 * layout.stride
        } else {
            160 * layout.stride
        });
        let do_base = tile.b as usize * do_strides.batch as usize
            + tile.head as usize * do_strides.head as usize;
        if SWIZZLED_Q {
            copy_rows_swizzled::<64, 64>(
                q,
                q_shared,
                bases.q,
                tile.q_row_base,
                cfg.q.seq,
                cfg.seqlen_q,
                cfg.hs_u32,
                tile.tid,
            );
        } else {
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
        }
        if SWIZZLED_Q {
            copy_rows_swizzled::<64, 64>(
                dout,
                do_shared,
                do_base,
                tile.q_row_base,
                do_strides.seq,
                cfg.seqlen_q,
                cfg.hs_u32,
                tile.tid,
            );
        } else {
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
        }
        ptx_asm!(
            "cp.async.commit_group; cp.async.wait_group 0;",
            clobber("memory")
        );
        thread::sync_threads();
        let row = tile.q_row_base + tile.warp_row + tile.group;
        let rounded_q = cfg.seqlen_q.div_ceil(128) * 128;
        let l0 = if row < cfg.seqlen_q {
            *lse.add(tile.bh as usize * cfg.seqlen_q as usize + row as usize)
        } else {
            f32::INFINITY
        };
        let l1 = if row + 8 < cfg.seqlen_q {
            *lse.add(tile.bh as usize * cfg.seqlen_q as usize + row as usize + 8)
        } else {
            f32::INFINITY
        };
        let d0 = *delta.add(tile.bh as usize * rounded_q as usize + row as usize);
        let d1 = *delta.add(tile.bh as usize * rounded_q as usize + row as usize + 8);
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
        let chunk = range
            .num_tiles
            .saturating_sub(range.start_tile)
            .div_ceil(SPLITS);
        let first = if SPLITS > 1 {
            range.start_tile + split * chunk
        } else {
            range.start_tile
        };
        let end = if SPLITS > 1 {
            (first + chunk).min(range.num_tiles)
        } else {
            range.num_tiles
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
                let q_frag = if SWIZZLED_Q {
                    query_fragments_swizzled::<NFQ, 64>(q_shared, &tile)
                } else {
                    query_fragments::<NFQ>(q_shared, &tile, layout)
                };
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
                let do_frag = if SWIZZLED_Q {
                    query_fragments_swizzled::<NFQ, 64>(do_shared, &tile)
                } else {
                    query_fragments::<NFQ>(do_shared, &tile, layout)
                };
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
                        let plane =
                            thread::gridDim_z() as usize * cfg.seqlen_q as usize * bases.o_ss;
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
                        let plane =
                            thread::gridDim_z() as usize * cfg.seqlen_q as usize * bases.o_ss;
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

pub trait DqSpec {
    const NOCAP: bool;
    const DROPOUT: bool;
    const PLAIN: bool;
    unsafe fn run(
        q: *const u32,
        k: *const u32,
        v: *const u32,
        dout: *const u32,
        lse: *const f32,
        delta: *const f32,
        dq: *mut u32,
        alibi: *const f32,
        cfg: FwdConfig,
        do_strides: Strides,
        gradient_scale: f32,
        dropout: Dropout,
    );
}
pub struct Dq<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const SWIZZLED_Q: bool = false,
    const SPLITS: u32 = 1,
    const DROPOUT: bool = true,
    const PLAIN: bool = false,
    const NOCAP: bool = false,
>;
impl<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const SWIZZLED_Q: bool,
    const SPLITS: u32,
    const DROPOUT: bool,
    const PLAIN: bool,
    const NOCAP: bool,
> DqSpec for Dq<BF16, HD, NFQ, NFO, SWIZZLED_Q, SPLITS, DROPOUT, PLAIN, NOCAP>
{
    const DROPOUT: bool = DROPOUT;
    const PLAIN: bool = PLAIN;
    const NOCAP: bool = NOCAP;
    #[inline(always)]
    unsafe fn run(
        q: *const u32,
        k: *const u32,
        v: *const u32,
        dout: *const u32,
        lse: *const f32,
        delta: *const f32,
        dq: *mut u32,
        alibi: *const f32,
        cfg: FwdConfig,
        do_strides: Strides,
        gradient_scale: f32,
        dropout: Dropout,
    ) {
        unsafe {
            dq_core::<BF16, HD, NFQ, NFO, SWIZZLED_Q, SPLITS>(
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
            );
        }
    }
}
#[allow(non_camel_case_types)]
pub type dq_d32 = Dq<false, 32, 2, 4>;
#[allow(non_camel_case_types)]
pub type dq_d32_bf16 = Dq<true, 32, 2, 4>;
#[allow(non_camel_case_types)]
pub type dq_d64 = Dq<false, 64, 4, 8>;
#[allow(non_camel_case_types)]
pub type dq_d64_bf16 = Dq<true, 64, 4, 8>;
#[allow(non_camel_case_types)]
pub type dq_d96 = Dq<false, 96, 6, 12>;
#[allow(non_camel_case_types)]
pub type dq_d96_bf16 = Dq<true, 96, 6, 12>;
#[allow(non_camel_case_types)]
pub type dq_d128 = Dq<false, 128, 8, 16>;
#[allow(non_camel_case_types)]
pub type dq_d128_bf16 = Dq<true, 128, 8, 16>;
#[allow(non_camel_case_types)]
pub type dq_d192 = Dq<false, 192, 12, 24>;
#[allow(non_camel_case_types)]
pub type dq_d192_bf16 = Dq<true, 192, 12, 24>;
#[allow(non_camel_case_types)]
pub type dq_d256 = Dq<false, 256, 16, 32>;
#[allow(non_camel_case_types)]
pub type dq_d256_bf16 = Dq<true, 256, 16, 32>;
#[allow(non_camel_case_types)]
pub type dq_d128_swizzled = Dq<false, 128, 8, 16, true>;
#[allow(non_camel_case_types)]
pub type dq_d128_swizzled_bf16 = Dq<true, 128, 8, 16, true>;
#[allow(non_camel_case_types)]
pub type dq_d128_swizzled_nodrop = Dq<false, 128, 8, 16, true, 1, false, false>;
#[allow(non_camel_case_types)]
pub type dq_d128_swizzled_plain = Dq<false, 128, 8, 16, true, 1, false, true>;
#[allow(non_camel_case_types)]
pub type dq_d128_swizzled_nodrop_bf16 = Dq<true, 128, 8, 16, true, 1, false, false>;
#[allow(non_camel_case_types)]
pub type dq_d128_swizzled_plain_bf16 = Dq<true, 128, 8, 16, true, 1, false, true>;
#[allow(non_camel_case_types)]
pub type dq_d128_swizzled_nocap = Dq<false, 128, 8, 16, true, 1, true, false, true>;
#[allow(non_camel_case_types)]
pub type dq_d128_swizzled_nodrop_nocap = Dq<false, 128, 8, 16, true, 1, false, false, true>;
#[allow(non_camel_case_types)]
pub type dq_d128_swizzled_nocap_bf16 = Dq<true, 128, 8, 16, true, 1, true, false, true>;
#[allow(non_camel_case_types)]
pub type dq_d128_swizzled_nodrop_nocap_bf16 = Dq<true, 128, 8, 16, true, 1, false, false, true>;
#[allow(non_camel_case_types)]
pub type dq_d128_split2 = Dq<false, 128, 8, 16, false, 2>;
#[allow(non_camel_case_types)]
pub type dq_d128_split2_bf16 = Dq<true, 128, 8, 16, false, 2>;
#[allow(non_camel_case_types)]
pub type dq_d128_split4 = Dq<false, 128, 8, 16, false, 4>;
#[allow(non_camel_case_types)]
pub type dq_d128_split4_bf16 = Dq<true, 128, 8, 16, false, 4>;
#[allow(non_camel_case_types)]
pub type dq_d128_split8 = Dq<false, 128, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type dq_d128_split8_bf16 = Dq<true, 128, 8, 16, false, 8>;
#[kernel(
    dq_d128_swizzled_nocap,
    dq_d128_swizzled_nocap_bf16,
    dq_d128_swizzled_nodrop_nocap,
    dq_d128_swizzled_nodrop_nocap_bf16,
    dq_d128_split2,
    dq_d128_split2_bf16,
    dq_d128_split4,
    dq_d128_split4_bf16,
    dq_d128_split8,
    dq_d128_split8_bf16,
    dq_d128_swizzled,
    dq_d128_swizzled_bf16,
    dq_d128_swizzled_plain,
    dq_d128_swizzled_plain_bf16,
    dq_d128_swizzled_nodrop,
    dq_d128_swizzled_nodrop_bf16,
    dq_d32,
    dq_d32_bf16,
    dq_d64,
    dq_d64_bf16,
    dq_d96,
    dq_d96_bf16,
    dq_d128,
    dq_d128_bf16,
    dq_d192,
    dq_d192_bf16,
    dq_d256,
    dq_d256_bf16
)]
#[launch_bounds(128)]
pub fn backward<S: DqSpec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    dout: &[u32],
    lse: &[f32],
    delta: &[f32],
    dq: DisjointSlice<u32>,
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
    let mut dq = dq;
    let cfg = FwdConfig::new(
        scale,
        if S::PLAIN || S::NOCAP {
            0.0
        } else {
            tanh_scale
        },
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
            dq.as_mut_ptr(),
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
