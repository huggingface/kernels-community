//! Shared forward core: smem layout, QK^T, online softmax and PV accumulation.
// Helpers stay inlined: #[device] calls force fragments to local memory.

mod dense;
mod dense192;

use crate::device::config::{FwdConfig, KvRange, LaunchTile, TensorBases};
use crate::device::dropout::Dropout;
use crate::device::inlined::{cvt, fabs, fast_exp2, fmax, lg2_approx, mma, tanh_approx};
use crate::device::inlined::{
    ldmatrix_x2_shared, ldmatrix_x2_trans_shared, ldmatrix_x4_shared, ldmatrix_x4_trans_shared,
    shared_address,
};
use cuda_device::async_copy::cp_async_ca_zfill_16;
use cuda_device::ptx_asm;
use cuda_device::shared::DynamicSharedArray;
use cuda_device::thread;
use cuda_device::warp::shuffle_xor_f32;

pub const LOG2_E: f32 = core::f32::consts::LOG2_E;

const LN_2: f32 = core::f32::consts::LN_2;

const SMEM_PAD_U32: usize = 4;

const SMEM_BUDGET: usize = 101_376;

#[derive(Clone, Copy)]
pub struct SmemLayout {
    pub d_u32: usize,
    pub stride: usize,
    tile_bytes: usize,
    pub double: bool,
    pub triple: bool,
}

impl SmemLayout {
    pub fn new<const HD: usize, const KEYS: u32, const BLOCK_ROWS: u32>() -> Self {
        let d_u32 = HD / 2;
        let stride = d_u32 + SMEM_PAD_U32;
        let tile_bytes = KEYS as usize * stride * 4;
        Self {
            d_u32,
            stride,
            tile_bytes,
            triple: (HD == 192 && BLOCK_ROWS == 64 && KEYS == 32)
                || (HD == 96 && BLOCK_ROWS == 128 && KEYS == 32),
            double: !(HD == 192 && KEYS == 48)
                && !(HD == 96 && BLOCK_ROWS == 128 && KEYS == 32)
                && BLOCK_ROWS > 32
                && 4 * tile_bytes <= SMEM_BUDGET
                && !((HD >= 192 && BLOCK_ROWS == 64 && KEYS == 32)
                    || (HD == 96 && BLOCK_ROWS == 64 && KEYS == 64)),
        }
    }
}

#[derive(Clone, Copy)]
pub struct SmemBuffers {
    pub k0: *mut u32,
    pub v0: *mut u32,
    pub k1: *mut u32,
    pub v1: *mut u32,
}

impl SmemBuffers {
    pub unsafe fn new(layout: SmemLayout) -> Self {
        let v0_off = if layout.double || layout.triple {
            2 * layout.tile_bytes
        } else {
            layout.tile_bytes
        };
        Self {
            k0: DynamicSharedArray::<u32, 16>::get() as usize as *mut u32,
            v0: DynamicSharedArray::<u32, 16>::offset(v0_off),
            k1: DynamicSharedArray::<u32, 16>::offset(layout.tile_bytes),
            v1: DynamicSharedArray::<u32, 16>::offset(3 * layout.tile_bytes),
        }
    }
}

#[inline(always)]
pub unsafe fn load_q_frag<
    const FRAGS: usize,
    const RFRAGS: usize,
    const NFQ: usize,
    const BLOCK_ROWS: u32,
    const THREADS: u32,
>(
    q: *const u32,
    scratch: *mut u32,
    cfg: &FwdConfig,
    tile: &LaunchTile,
    bases: &TensorBases,
    layout: SmemLayout,
) -> [[u32; 4]; FRAGS] {
    unsafe {
        let mut i = tile.tid as usize;
        while i < BLOCK_ROWS as usize * layout.d_u32 {
            let row = i / layout.d_u32;
            let col = i % layout.d_u32;
            let qrow = tile.q_row_base as usize + row;
            let val = if qrow < cfg.seqlen_q as usize && col < cfg.hs_u32 as usize {
                *q.add(bases.q + qrow * cfg.q.seq as usize + col)
            } else {
                0
            };
            *scratch.add(row * layout.stride + col) = val;
            i += THREADS as usize;
        }
        thread::sync_threads();

        let a_row = tile.warp_row + (tile.lane_id & 7) + ((tile.lane_id >> 3) & 1) * 8;
        let a_col_half = ((tile.lane_id >> 4) as usize) * 4;
        let base = shared_address(scratch);
        let mut f = [[0u32; 4]; FRAGS];
        let mut rf = 0usize;
        while rf < RFRAGS {
            cuda_device::thread::__unroll_config::<0>();
            let row = a_row as usize + rf * 16;
            let mut kc = 0usize;
            while kc < NFQ {
                cuda_device::thread::__unroll_config::<0>();
                f[rf * NFQ + kc] = ldmatrix_x4_shared(
                    base + ((row * layout.stride + kc * 8 + a_col_half) * 4) as u32,
                );
                kc += 1;
            }
            rf += 1;
        }
        thread::sync_threads();
        f
    }
}

#[inline(always)]
pub unsafe fn stage_tile<
    const HD: usize,
    const KEYS: u32,
    const THREADS: u32,
    const UNROLL_COPY: bool,
    const RECURRENT_COPY: bool,
>(
    smem: *mut u32,
    src: *const u32,
    base: u32,
    gbase: usize,
    seq_stride: u32,
    cfg: &FwdConfig,
    tile: &LaunchTile,
    layout: SmemLayout,
) {
    unsafe {
        if UNROLL_COPY {
            stage_tile_unrolled::<HD, KEYS, THREADS>(
                smem, src, base, gbase, seq_stride, cfg, tile, layout,
            );
            return;
        }
        if RECURRENT_COPY && HD == 192 && KEYS == 48 && THREADS == 128 {
            // Advance coordinates directly instead of dividing an index.
            let mut row = tile.tid / 24;
            let mut col = (tile.tid % 24) * 4;
            while row < KEYS {
                let krow = base + row;
                let row_ok = krow < cfg.seqlen_k;
                let col_ok = col < cfg.hs_u32;
                let src_row = if row_ok { krow } else { cfg.seqlen_k - 1 };
                let src_col = if col_ok { col } else { 0 };
                let g = gbase + src_row as usize * seq_stride as usize + src_col as usize;
                cp_async_ca_zfill_16(
                    smem.add(row as usize * layout.stride + col as usize),
                    src.add(g) as *const u8,
                    if row_ok && col_ok { 16 } else { 0 },
                );
                col += 32;
                row += 5;
                if col >= 96 {
                    col -= 96;
                    row += 1;
                }
            }
            return;
        }
        let mut i = tile.tid as usize * 4;
        while i < KEYS as usize * layout.d_u32 {
            let row = i / layout.d_u32;
            let col = i % layout.d_u32;
            let krow = base as usize + row;
            let row_ok = krow < cfg.seqlen_k as usize;
            let col_ok = col < cfg.hs_u32 as usize;
            let src_row = if row_ok {
                krow
            } else {
                cfg.seqlen_k as usize - 1
            };
            let src_col = if col_ok { col } else { 0 };
            let g = gbase + src_row * seq_stride as usize + src_col;
            let sz: u32 = if row_ok && col_ok { 16 } else { 0 };
            cp_async_ca_zfill_16(
                smem.add(row * layout.stride + col),
                src.add(g) as *const u8,
                sz,
            );
            i += THREADS as usize * 4;
        }
    }
}

#[inline(always)]
unsafe fn stage_kv_pair<
    const HD: usize,
    const KEYS: u32,
    const THREADS: u32,
    const UNROLL_COPY: bool,
    const RECURRENT_COPY: bool,
>(
    k_smem: *mut u32,
    v_smem: *mut u32,
    k: *const u32,
    v: *const u32,
    base: u32,
    cfg: &FwdConfig,
    tile: &LaunchTile,
    bases: &TensorBases,
    layout: SmemLayout,
) {
    unsafe {
        stage_tile::<HD, KEYS, THREADS, UNROLL_COPY, RECURRENT_COPY>(
            k_smem, k, base, bases.k, cfg.k.seq, cfg, tile, layout,
        );
        stage_tile::<HD, KEYS, THREADS, UNROLL_COPY, RECURRENT_COPY>(
            v_smem, v, base, bases.v, cfg.v.seq, cfg, tile, layout,
        );
    }
}

#[inline(always)]
pub unsafe fn prefetch_first_kv<
    const HD: usize,
    const KEYS: u32,
    const THREADS: u32,
    const UNROLL_COPY: bool,
    const RECURRENT_COPY: bool,
>(
    k: *const u32,
    v: *const u32,
    cfg: &FwdConfig,
    tile: &LaunchTile,
    bases: &TensorBases,
    range: KvRange,
    smem: SmemBuffers,
    layout: SmemLayout,
    reverse: bool,
) {
    unsafe {
        if !layout.double && range.start_tile < range.num_tiles {
            stage_tile::<HD, KEYS, THREADS, UNROLL_COPY, RECURRENT_COPY>(
                smem.k0,
                k,
                (if reverse {
                    range.num_tiles - 1
                } else {
                    range.start_tile
                }) * KEYS,
                bases.k,
                cfg.k.seq,
                cfg,
                tile,
                layout,
            );
            ptx_asm!("cp.async.commit_group;", clobber("memory"));
        }
        if layout.double && range.start_tile < range.num_tiles {
            stage_kv_pair::<HD, KEYS, THREADS, UNROLL_COPY, RECURRENT_COPY>(
                smem.k0,
                smem.v0,
                k,
                v,
                (if reverse {
                    range.num_tiles - 1
                } else {
                    range.start_tile
                }) * KEYS,
                cfg,
                tile,
                bases,
                layout,
            );
            ptx_asm!("cp.async.commit_group;", clobber("memory"));
        }
    }
}

#[inline(always)]
pub unsafe fn stage_kv_tile<
    const HD: usize,
    const KEYS: u32,
    const THREADS: u32,
    const UNROLL_COPY: bool,
    const RECURRENT_COPY: bool,
>(
    k: *const u32,
    v: *const u32,
    cfg: &FwdConfig,
    tile: &LaunchTile,
    bases: &TensorBases,
    range: KvRange,
    smem: SmemBuffers,
    layout: SmemLayout,
    j: u32,
    reverse: bool,
) -> (*const u32, *const u32) {
    unsafe {
        let kv_base = j * KEYS;
        if layout.triple {
            ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
            // Finish reading the previous V tile and the reusable K buffer.
            thread::sync_threads();
            let odd = ((if reverse {
                range.num_tiles - 1 - j
            } else {
                j - range.start_tile
            }) & 1)
                == 1;
            let current_k = if odd { smem.k1 } else { smem.k0 };
            stage_tile::<HD, KEYS, THREADS, UNROLL_COPY, RECURRENT_COPY>(
                smem.v0, v, kv_base, bases.v, cfg.v.seq, cfg, tile, layout,
            );
            ptx_asm!("cp.async.commit_group;", clobber("memory"));
            if if reverse {
                j > range.start_tile
            } else {
                j + 1 < range.num_tiles
            } {
                let next_k = if odd { smem.k0 } else { smem.k1 };
                let next_base = (if reverse { j - 1 } else { j + 1 }) * KEYS;
                stage_tile::<HD, KEYS, THREADS, UNROLL_COPY, RECURRENT_COPY>(
                    next_k, k, next_base, bases.k, cfg.k.seq, cfg, tile, layout,
                );
                ptx_asm!("cp.async.commit_group;", clobber("memory"));
            }
            return (current_k as *const u32, smem.v0 as *const u32);
        }
        if layout.double {
            let odd = ((if reverse {
                range.num_tiles - 1 - j
            } else {
                j - range.start_tile
            }) & 1)
                == 1;
            if if reverse {
                j > range.start_tile
            } else {
                j + 1 < range.num_tiles
            } {
                let next_base = (if reverse { j - 1 } else { j + 1 }) * KEYS;
                let nxt_k = if odd { smem.k0 } else { smem.k1 };
                let nxt_v = if odd { smem.v0 } else { smem.v1 };
                stage_kv_pair::<HD, KEYS, THREADS, UNROLL_COPY, RECURRENT_COPY>(
                    nxt_k, nxt_v, k, v, next_base, cfg, tile, bases, layout,
                );
                ptx_asm!("cp.async.commit_group;", clobber("memory"));
                ptx_asm!("cp.async.wait_group 1;", clobber("memory"));
            } else {
                ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
            }
            thread::sync_threads();
            let kb = if odd { smem.k1 } else { smem.k0 };
            let vb = if odd { smem.v1 } else { smem.v0 };
            (kb as *const u32, vb as *const u32)
        } else {
            ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
            thread::sync_threads();
            stage_tile::<HD, KEYS, THREADS, UNROLL_COPY, RECURRENT_COPY>(
                smem.v0, v, kv_base, bases.v, cfg.v.seq, cfg, tile, layout,
            );
            ptx_asm!("cp.async.commit_group;", clobber("memory"));
            (smem.k0 as *const u32, smem.v0 as *const u32)
        }
    }
}

#[inline(always)]
pub unsafe fn compute_qk<
    const BF16: bool,
    const RFRAGS: usize,
    const NFQ: usize,
    const NKT: usize,
>(
    s: &mut [[f32; 4]],
    q_frag: &[[u32; 4]],
    k_smem_base: *const u32,
    tile: &LaunchTile,
    layout: SmemLayout,
) {
    unsafe {
        // Literal steps: dynamic fragment indices stay in local memory.
        if NFQ != 2 && NFQ != 12 && !(NFQ == 16 && NKT == 2) {
            compute_qk_single::<BF16, RFRAGS, NFQ, NKT>(s, q_frag, k_smem_base, tile, layout);
            return;
        }
        let k_addr = shared_address(k_smem_base);
        let mut kc = 0usize;
        while kc < NFQ {
            cuda_device::thread::__unroll_config::<0>();
            let mut nt = 0usize;
            while nt < NKT {
                cuda_device::thread::__unroll_config::<0>();
                let l16 = (tile.lane_id & 15) as usize;
                let b_row = nt * 8 + (l16 & 7) + ((tile.lane_id >> 4) * 8) as usize;
                let b_col_u32 = kc * 8 + (l16 >> 3) * 4;
                let b =
                    ldmatrix_x4_shared(k_addr + ((b_row * layout.stride + b_col_u32) * 4) as u32);
                let mut rf = 0usize;
                while rf < RFRAGS {
                    cuda_device::thread::__unroll_config::<0>();
                    s[rf * NKT + nt] =
                        mma::<BF16>(s[rf * NKT + nt], q_frag[rf * NFQ + kc], [b[0], b[1]]);
                    s[rf * NKT + nt + 1] =
                        mma::<BF16>(s[rf * NKT + nt + 1], q_frag[rf * NFQ + kc], [b[2], b[3]]);
                    rf += 1;
                }
                nt += 2;
            }
            kc += 1;
        }
    }
}

#[derive(Clone, Copy)]
struct ScoreArgs {
    off: i32,
    key_len: i32,
    scale: f32,
    tanh_scale: f32,
    slope: f32,
    has_alibi: u32,
    alibi_causal: u32,
    window_left: i32,
    window_right: i32,
}

#[inline(always)]
fn score_base(args: ScoreArgs, raw: f32, row: i32, col: i32) -> f32 {
    let x = if args.tanh_scale > 0.0 {
        tanh_approx(raw * args.tanh_scale)
    } else {
        raw
    };
    let mut s = x * args.scale;
    if args.has_alibi != 0 {
        if args.alibi_causal != 0 {
            s += args.slope * (col as f32);
        } else {
            s -= args.slope * fabs((row + args.off - col) as f32);
        }
    }
    s
}

#[inline(always)]
fn score_masked(args: ScoreArgs, raw: f32, row: i32, col: i32) -> f32 {
    let mut s = score_base(args, raw, row, col);
    let mask_pad = col >= args.key_len;
    let mask_right = args.window_right >= 0 && col >= row + 1 + args.off + args.window_right;
    let mask_left = args.window_left >= 0 && col < row + args.off - args.window_left;
    if mask_pad || mask_right || mask_left {
        s = f32::NEG_INFINITY;
    }
    s
}

#[inline(always)]
fn score_elem<const MASK: bool>(args: ScoreArgs, raw: f32, row: i32, col: i32) -> f32 {
    if MASK {
        score_masked(args, raw, row, col)
    } else {
        score_base(args, raw, row, col)
    }
}

#[inline(always)]
fn score_frag<const MASK: bool>(f: &mut [f32; 4], row0: i32, row1: i32, col: i32, args: ScoreArgs) {
    f[0] = score_elem::<MASK>(args, f[0], row0, col);
    f[1] = score_elem::<MASK>(args, f[1], row0, col + 1);
    f[2] = score_elem::<MASK>(args, f[2], row1, col);
    f[3] = score_elem::<MASK>(args, f[3], row1, col + 1);
}

#[inline(always)]
pub fn apply_scores<
    const RFRAGS: usize,
    const NKT: usize,
    const BLOCK_ROWS: u32,
    const KEYS: u32,
>(
    s: &mut [[f32; 4]],
    cfg: &FwdConfig,
    tile: &LaunchTile,
    kv_base: u32,
    off: i32,
    scale: f32,
    tanh_scale: f32,
    slope: f32,
) {
    let warp_row_g = (tile.q_row_base + tile.warp_row + tile.group) as i32;
    let args = ScoreArgs {
        off,
        key_len: cfg.seqlen_k as i32,
        scale,
        tanh_scale,
        slope,
        has_alibi: cfg.has_alibi,
        alibi_causal: cfg.alibi_causal,
        window_left: cfg.window_left,
        window_right: cfg.window_right,
    };
    let needs_mask = kv_base + KEYS > cfg.seqlen_k
        || (cfg.window_right >= 0
            && (kv_base + KEYS - 1) as i32 >= tile.q_row_base as i32 + 1 + off + cfg.window_right)
        || (cfg.window_left >= 0
            && (kv_base as i32)
                < (tile.q_row_base + BLOCK_ROWS - 1) as i32 + off - cfg.window_left);
    // Optional scoring branches stay outside the unrolled loop.
    if !needs_mask && cfg.has_alibi == 0 && tanh_scale == 0.0 {
        // Separate constant bounds; a flat loop prevents scalarization.
        let mut rf = 0usize;
        while rf < RFRAGS {
            cuda_device::thread::__unroll_config::<0>();
            let mut nt = 0usize;
            while nt < NKT {
                cuda_device::thread::__unroll_config::<0>();
                let i = rf * NKT + nt;
                let f = s[i];
                s[i] = [f[0] * scale, f[1] * scale, f[2] * scale, f[3] * scale];
                nt += 1;
            }
            rf += 1;
        }
        return;
    }
    if needs_mask {
        if cfg.has_alibi == 0 && tanh_scale == 0.0 {
            let dense_args = ScoreArgs {
                has_alibi: 0,
                tanh_scale: 0.0,
                ..args
            };
            apply_score_tile::<true, RFRAGS, NKT>(s, tile, kv_base, warp_row_g, dense_args);
        } else {
            apply_score_tile::<true, RFRAGS, NKT>(s, tile, kv_base, warp_row_g, args);
        }
    } else {
        apply_score_tile::<false, RFRAGS, NKT>(s, tile, kv_base, warp_row_g, args);
    }
}

#[inline(always)]
fn apply_score_tile<const MASK: bool, const RFRAGS: usize, const NKT: usize>(
    s: &mut [[f32; 4]],
    tile: &LaunchTile,
    kv_base: u32,
    warp_row_g: i32,
    args: ScoreArgs,
) {
    let mut rf = 0usize;
    while rf < RFRAGS {
        cuda_device::thread::__unroll_config::<0>();
        let row_g0 = warp_row_g + (rf as i32) * 16;
        let row_g1 = row_g0 + 8;
        let mut nt = 0usize;
        while nt < NKT {
            cuda_device::thread::__unroll_config::<0>();
            let col = (kv_base as usize + nt * 8 + tile.pair as usize * 2) as i32;
            score_frag::<MASK>(&mut s[rf * NKT + nt], row_g0, row_g1, col, args);
            nt += 1;
        }
        rf += 1;
    }
}

#[inline(always)]
pub fn online_softmax_pack<
    const BF16: bool,
    const RFRAGS: usize,
    const NKT: usize,
    const NFO: usize,
>(
    s: &[[f32; 4]],
    p_pack: &mut [[u32; 2]],
    o_acc: &mut [[f32; 4]],
    m_row: &mut [f32],
    l_row: &mut [f32],
) {
    let mut rf = 0usize;
    while rf < RFRAGS {
        cuda_device::thread::__unroll_config::<0>();
        let m_new0;
        let m_new1;
        // Balanced reduction only helps wide-head dense kernels.
        if NFO == 32 && NKT == 4 {
            let mut maxima = [[f32::NEG_INFINITY; 2]; NKT];
            let mut nt = 0usize;
            while nt < NKT {
                cuda_device::thread::__unroll_config::<0>();
                let f = &s[rf * NKT + nt];
                maxima[nt] = [fmax(f[0], f[1]), fmax(f[2], f[3])];
                nt += 1;
            }
            let mut span = NKT / 2;
            while span > 0 {
                cuda_device::thread::__unroll_config::<0>();
                let mut nt = 0usize;
                while nt < span {
                    cuda_device::thread::__unroll_config::<0>();
                    maxima[nt] = [
                        fmax(maxima[nt][0], maxima[nt + span][0]),
                        fmax(maxima[nt][1], maxima[nt + span][1]),
                    ];
                    nt += 1;
                }
                span /= 2;
            }
            let mut cm0 = fmax(m_row[rf * 2], maxima[0][0]);
            let mut cm1 = fmax(m_row[rf * 2 + 1], maxima[0][1]);
            cm0 = fmax(cm0, shuffle_xor_f32(cm0, 1));
            cm0 = fmax(cm0, shuffle_xor_f32(cm0, 2));
            cm1 = fmax(cm1, shuffle_xor_f32(cm1, 1));
            cm1 = fmax(cm1, shuffle_xor_f32(cm1, 2));
            m_new0 = cm0;
            m_new1 = cm1;
        } else {
            let mut cm0 = f32::NEG_INFINITY;
            let mut cm1 = f32::NEG_INFINITY;
            let mut nt = 0usize;
            while nt < NKT {
                cuda_device::thread::__unroll_config::<0>();
                let f = &s[rf * NKT + nt];
                cm0 = fmax(fmax(cm0, f[0]), f[1]);
                cm1 = fmax(fmax(cm1, f[2]), f[3]);
                nt += 1;
            }
            cm0 = fmax(cm0, shuffle_xor_f32(cm0, 1));
            cm0 = fmax(cm0, shuffle_xor_f32(cm0, 2));
            cm1 = fmax(cm1, shuffle_xor_f32(cm1, 1));
            cm1 = fmax(cm1, shuffle_xor_f32(cm1, 2));

            m_new0 = fmax(m_row[rf * 2], cm0);
            m_new1 = fmax(m_row[rf * 2 + 1], cm1);
        }
        // Fully masked rows must stay zero: avoid -inf - -inf NaN.
        let exp_max0 = if m_new0 == f32::NEG_INFINITY {
            0.0
        } else {
            m_new0
        };
        let exp_max1 = if m_new1 == f32::NEG_INFINITY {
            0.0
        } else {
            m_new1
        };
        let alpha0 = fast_exp2(m_row[rf * 2] - exp_max0);
        let alpha1 = fast_exp2(m_row[rf * 2 + 1] - exp_max1);
        m_row[rf * 2] = m_new0;
        m_row[rf * 2 + 1] = m_new1;

        if alpha0 != 1.0 || alpha1 != 1.0 {
            let mut d = 0usize;
            while d < NFO {
                cuda_device::thread::__unroll_config::<0>();
                let f = &mut o_acc[rf * NFO + d];
                f[0] *= alpha0;
                f[1] *= alpha0;
                f[2] *= alpha1;
                f[3] *= alpha1;
                d += 1;
            }
        }

        let mut cs0 = 0.0f32;
        let mut cs1 = 0.0f32;
        let mut nt = 0usize;
        while nt < NKT {
            cuda_device::thread::__unroll_config::<0>();
            let f = &s[rf * NKT + nt];
            let e0 = fast_exp2(f[0] - exp_max0);
            let e1 = fast_exp2(f[1] - exp_max0);
            let e2 = fast_exp2(f[2] - exp_max1);
            let e3 = fast_exp2(f[3] - exp_max1);
            cs0 += e0 + e1;
            cs1 += e2 + e3;
            p_pack[rf * NKT + nt] = [cvt::<BF16>(e0, e1), cvt::<BF16>(e2, e3)];
            nt += 1;
        }
        // Partial denominators; reduced once in write_output.
        l_row[rf * 2] = l_row[rf * 2] * alpha0 + cs0;
        l_row[rf * 2 + 1] = l_row[rf * 2 + 1] * alpha1 + cs1;
        rf += 1;
    }
}

#[inline(always)]
pub unsafe fn accumulate_values<
    const BF16: bool,
    const RFRAGS: usize,
    const NKT: usize,
    const NFO: usize,
    const PV_CHUNKS: usize,
>(
    o_acc: &mut [[f32; 4]],
    p_pack: &[[u32; 2]],
    v_smem_base: *const u32,
    tile: &LaunchTile,
    layout: SmemLayout,
) {
    unsafe {
        if NFO != 4 && NFO != 24 && !(NFO == 32 && NKT == 2) {
            accumulate_values_single::<BF16, RFRAGS, NKT, NFO, PV_CHUNKS>(
                o_acc,
                p_pack,
                v_smem_base,
                tile,
                layout,
            );
            return;
        }
        let v_addr = shared_address(v_smem_base);
        let mut kc = 0usize;
        while kc < PV_CHUNKS {
            cuda_device::thread::__unroll_config::<0>();
            let mut d = 0usize;
            while d < NFO {
                cuda_device::thread::__unroll_config::<0>();
                let b_row = kc * 16 + (tile.lane_id & 15) as usize;
                let b_col_u32 = d * 4 + ((tile.lane_id >> 4) * 4) as usize;
                let b = ldmatrix_x4_trans_shared(
                    v_addr + ((b_row * layout.stride + b_col_u32) * 4) as u32,
                );
                let mut rf = 0usize;
                while rf < RFRAGS {
                    cuda_device::thread::__unroll_config::<0>();
                    let p_lo = &p_pack[rf * NKT + kc * 2];
                    let p_hi = &p_pack[rf * NKT + kc * 2 + 1];
                    let a = [p_lo[0], p_lo[1], p_hi[0], p_hi[1]];
                    o_acc[rf * NFO + d] = mma::<BF16>(o_acc[rf * NFO + d], a, [b[0], b[1]]);
                    o_acc[rf * NFO + d + 1] = mma::<BF16>(o_acc[rf * NFO + d + 1], a, [b[2], b[3]]);
                    rf += 1;
                }
                d += 2;
            }
            kc += 1;
        }
    }
}

#[inline(always)]
pub unsafe fn write_output<
    const BF16: bool,
    const RFRAGS: usize,
    const NFO: usize,
    const PARTIAL: bool,
    const BLOCK_ROWS: u32,
    const THREADS: u32,
    const VECTOR_STORE: bool,
>(
    o: *mut u32,
    lse: *mut f32,
    o_acc: &[[f32; 4]],
    m_row: &[f32],
    l_row: &[f32],
    cfg: &FwdConfig,
    tile: &LaunchTile,
    bases: &TensorBases,
    scratch: *mut u32,
    layout: SmemLayout,
    output_scale: f32,
) {
    unsafe {
        // Small heads and local windows store directly.
        let vector_store = !PARTIAL && NFO > 4 && VECTOR_STORE;
        // Last PV read may still be in flight in single-buffer kernels.
        if vector_store && !layout.double {
            thread::sync_threads();
        }
        let mut rf = 0usize;
        while rf < RFRAGS {
            cuda_device::thread::__unroll_config::<0>();
            let mut l0 = l_row[rf * 2];
            let mut l1 = l_row[rf * 2 + 1];
            l0 += shuffle_xor_f32(l0, 1);
            l0 += shuffle_xor_f32(l0, 2);
            l1 += shuffle_xor_f32(l1, 1);
            l1 += shuffle_xor_f32(l1, 2);
            let inv_l0 = if l0 > 0.0 {
                (1.0f32 / l0) * output_scale
            } else {
                0.0
            };
            let inv_l1 = if l1 > 0.0 {
                (1.0f32 / l1) * output_scale
            } else {
                0.0
            };

            let orow0 = (tile.q_row_base + tile.warp_row + tile.group) as usize + rf * 16;
            let orow1 = orow0 + 8;
            let valid0 = orow0 < cfg.seqlen_q as usize;
            let valid1 = orow1 < cfg.seqlen_q as usize;
            let out_row0 = bases.o + orow0 * bases.o_ss;
            let out_row1 = bases.o + orow1 * bases.o_ss;
            let mut d = 0usize;
            while d < NFO {
                cuda_device::thread::__unroll_config::<0>();
                let f = &o_acc[rf * NFO + d];
                let col = d * 4 + tile.pair as usize;
                if col < cfg.hs_u32 as usize {
                    if valid0 {
                        if PARTIAL {
                            *o.add(2 * (out_row0 + col)) = (f[0] * inv_l0).to_bits();
                            *o.add(2 * (out_row0 + col) + 1) = (f[1] * inv_l0).to_bits();
                        } else if vector_store {
                            *scratch
                                .add((orow0 - tile.q_row_base as usize) * layout.stride + col) =
                                cvt::<BF16>(f[0] * inv_l0, f[1] * inv_l0);
                        } else {
                            *o.add(out_row0 + col) = cvt::<BF16>(f[0] * inv_l0, f[1] * inv_l0);
                        }
                    }
                    if valid1 {
                        if PARTIAL {
                            *o.add(2 * (out_row1 + col)) = (f[2] * inv_l1).to_bits();
                            *o.add(2 * (out_row1 + col) + 1) = (f[3] * inv_l1).to_bits();
                        } else if vector_store {
                            *scratch
                                .add((orow1 - tile.q_row_base as usize) * layout.stride + col) =
                                cvt::<BF16>(f[2] * inv_l1, f[3] * inv_l1);
                        } else {
                            *o.add(out_row1 + col) = cvt::<BF16>(f[2] * inv_l1, f[3] * inv_l1);
                        }
                    }
                }
                d += 1;
            }

            if tile.pair == 0 {
                let lse_base = tile.bh as usize * cfg.seqlen_q as usize;
                let lse0 = if l0 > 0.0 {
                    (m_row[rf * 2] + lg2_approx(l0)) * LN_2
                } else {
                    if PARTIAL {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    }
                };
                let lse1 = if l1 > 0.0 {
                    (m_row[rf * 2 + 1] + lg2_approx(l1)) * LN_2
                } else {
                    if PARTIAL {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    }
                };
                if valid0 {
                    *lse.add(lse_base + orow0) = lse0;
                }
                if valid1 {
                    *lse.add(lse_base + orow1) = lse1;
                }
            }
            rf += 1;
        }
        // Fragment stores must be visible before gathering; reuses Q staging.
        if vector_store {
            thread::sync_threads();
            let mut i = tile.tid as usize * 4;
            while i < BLOCK_ROWS as usize * layout.d_u32 {
                let row = i / layout.d_u32;
                let col = i % layout.d_u32;
                let qrow = tile.q_row_base as usize + row;
                if qrow < cfg.seqlen_q as usize && col < cfg.hs_u32 as usize {
                    let src = scratch.add(row * layout.stride + col);
                    let dst = o.add(bases.o + qrow * bases.o_ss + col);
                    let x0 = *src;
                    let x1 = *src.add(1);
                    let x2 = *src.add(2);
                    let x3 = *src.add(3);
                    ptx_asm!("st.global.v4.u32 [%0], {%1, %2, %3, %4};", in("l") dst, in("r") x0, in("r") x1, in("r") x2, in("r") x3, clobber("memory"));
                }
                i += THREADS as usize * 4;
            }
        }
    }
}

#[inline(always)]
unsafe fn compute_qk_single<
    const BF16: bool,
    const RFRAGS: usize,
    const NFQ: usize,
    const NKT: usize,
>(
    s: &mut [[f32; 4]],
    q_frag: &[[u32; 4]],
    k_smem_base: *const u32,
    tile: &LaunchTile,
    layout: SmemLayout,
) {
    unsafe {
        let k_addr = shared_address(k_smem_base);
        let mut kc = 0usize;
        while kc < NFQ {
            cuda_device::thread::__unroll_config::<0>();
            let mut nt = 0usize;
            while nt < NKT {
                cuda_device::thread::__unroll_config::<0>();
                let l16 = (tile.lane_id & 15) as usize;
                let b_row = nt * 8 + (l16 & 7);
                let b_col_u32 = kc * 8 + (l16 >> 3) * 4;
                let b =
                    ldmatrix_x2_shared(k_addr + ((b_row * layout.stride + b_col_u32) * 4) as u32);
                let mut rf = 0usize;
                while rf < RFRAGS {
                    cuda_device::thread::__unroll_config::<0>();
                    s[rf * NKT + nt] = mma::<BF16>(s[rf * NKT + nt], q_frag[rf * NFQ + kc], b);
                    rf += 1;
                }
                nt += 1;
            }
            kc += 1;
        }
    }
}

#[inline(always)]
unsafe fn accumulate_values_single<
    const BF16: bool,
    const RFRAGS: usize,
    const NKT: usize,
    const NFO: usize,
    const PV_CHUNKS: usize,
>(
    o_acc: &mut [[f32; 4]],
    p_pack: &[[u32; 2]],
    v_smem_base: *const u32,
    tile: &LaunchTile,
    layout: SmemLayout,
) {
    unsafe {
        let v_addr = shared_address(v_smem_base);
        let mut kc = 0usize;
        while kc < PV_CHUNKS {
            cuda_device::thread::__unroll_config::<0>();
            let mut a = [[0u32; 4]; RFRAGS];
            let mut rf = 0usize;
            while rf < RFRAGS {
                cuda_device::thread::__unroll_config::<0>();
                let p_lo = &p_pack[rf * NKT + kc * 2];
                let p_hi = &p_pack[rf * NKT + kc * 2 + 1];
                a[rf] = [p_lo[0], p_lo[1], p_hi[0], p_hi[1]];
                rf += 1;
            }
            let mut d = 0usize;
            while d < NFO {
                cuda_device::thread::__unroll_config::<0>();
                let l16 = (tile.lane_id & 15) as usize;
                let b_row = kc * 16 + l16;
                let b_col_u32 = d * 4;
                let b = ldmatrix_x2_trans_shared(
                    v_addr + ((b_row * layout.stride + b_col_u32) * 4) as u32,
                );
                let mut rf = 0usize;
                while rf < RFRAGS {
                    cuda_device::thread::__unroll_config::<0>();
                    o_acc[rf * NFO + d] = mma::<BF16>(o_acc[rf * NFO + d], a[rf], b);
                    rf += 1;
                }
                d += 1;
            }
            kc += 1;
        }
    }
}

// d128 split tiles only: constant indices for short key segments.
#[inline(always)]
unsafe fn stage_tile_unrolled<const HD: usize, const KEYS: u32, const THREADS: u32>(
    smem: *mut u32,
    src: *const u32,
    base: u32,
    gbase: usize,
    seq_stride: u32,
    cfg: &FwdConfig,
    tile: &LaunchTile,
    layout: SmemLayout,
) {
    unsafe {
        let mut copy = 0usize;
        while copy < (KEYS as usize * (HD / 2)) / (THREADS as usize * 4) {
            cuda_device::thread::__unroll_config::<0>();
            let i = tile.tid as usize * 4 + copy * THREADS as usize * 4;
            let row = i / (HD / 2);
            let col = i % (HD / 2);
            let krow = base as usize + row;
            let row_ok = krow < cfg.seqlen_k as usize;
            let col_ok = col < cfg.hs_u32 as usize;
            let src_row = if row_ok {
                krow
            } else {
                cfg.seqlen_k as usize - 1
            };
            let src_col = if col_ok { col } else { 0 };
            let g = gbase + src_row * seq_stride as usize + src_col;
            let sz: u32 = if row_ok && col_ok { 16 } else { 0 };
            cp_async_ca_zfill_16(
                smem.add(row * layout.stride + col),
                src.add(g) as *const u8,
                sz,
            );
            copy += 1;
        }
    }
}

// Positive-scale specialization; kept separate to preserve codegen.
#[inline(always)]
pub fn online_softmax_pack_raw<
    const BF16: bool,
    const RFRAGS: usize,
    const NKT: usize,
    const NFO: usize,
>(
    s: &[[f32; 4]],
    p_pack: &mut [[u32; 2]],
    o_acc: &mut [[f32; 4]],
    m_row: &mut [f32],
    l_row: &mut [f32],
    scale: f32,
) {
    let mut rf = 0usize;
    while rf < RFRAGS {
        cuda_device::thread::__unroll_config::<0>();
        let m_new0;
        let m_new1;
        // Balanced reduction only helps wide-head dense kernels.
        if NFO == 32 && NKT == 4 {
            let mut maxima = [[f32::NEG_INFINITY; 2]; NKT];
            let mut nt = 0usize;
            while nt < NKT {
                cuda_device::thread::__unroll_config::<0>();
                let f = &s[rf * NKT + nt];
                maxima[nt] = [fmax(f[0], f[1]), fmax(f[2], f[3])];
                nt += 1;
            }
            let mut span = NKT / 2;
            while span > 0 {
                cuda_device::thread::__unroll_config::<0>();
                let mut nt = 0usize;
                while nt < span {
                    cuda_device::thread::__unroll_config::<0>();
                    maxima[nt] = [
                        fmax(maxima[nt][0], maxima[nt + span][0]),
                        fmax(maxima[nt][1], maxima[nt + span][1]),
                    ];
                    nt += 1;
                }
                span /= 2;
            }
            let mut cm0 = fmax(m_row[rf * 2], maxima[0][0]);
            let mut cm1 = fmax(m_row[rf * 2 + 1], maxima[0][1]);
            cm0 = fmax(cm0, shuffle_xor_f32(cm0, 1));
            cm0 = fmax(cm0, shuffle_xor_f32(cm0, 2));
            cm1 = fmax(cm1, shuffle_xor_f32(cm1, 1));
            cm1 = fmax(cm1, shuffle_xor_f32(cm1, 2));
            m_new0 = cm0;
            m_new1 = cm1;
        } else {
            let mut cm0 = f32::NEG_INFINITY;
            let mut cm1 = f32::NEG_INFINITY;
            let mut nt = 0usize;
            while nt < NKT {
                cuda_device::thread::__unroll_config::<0>();
                let f = &s[rf * NKT + nt];
                cm0 = fmax(fmax(cm0, f[0]), f[1]);
                cm1 = fmax(fmax(cm1, f[2]), f[3]);
                nt += 1;
            }
            cm0 = fmax(cm0, shuffle_xor_f32(cm0, 1));
            cm0 = fmax(cm0, shuffle_xor_f32(cm0, 2));
            cm1 = fmax(cm1, shuffle_xor_f32(cm1, 1));
            cm1 = fmax(cm1, shuffle_xor_f32(cm1, 2));

            m_new0 = fmax(m_row[rf * 2], cm0);
            m_new1 = fmax(m_row[rf * 2 + 1], cm1);
        }
        // Fully masked rows must stay zero: avoid -inf - -inf NaN.
        let exp_max0 = if m_new0 == f32::NEG_INFINITY {
            0.0
        } else {
            m_new0
        };
        let exp_max1 = if m_new1 == f32::NEG_INFINITY {
            0.0
        } else {
            m_new1
        };
        let alpha0 = fast_exp2((m_row[rf * 2] - exp_max0) * scale);
        let alpha1 = fast_exp2((m_row[rf * 2 + 1] - exp_max1) * scale);
        m_row[rf * 2] = m_new0;
        m_row[rf * 2 + 1] = m_new1;

        if alpha0 != 1.0 || alpha1 != 1.0 {
            let mut d = 0usize;
            while d < NFO {
                cuda_device::thread::__unroll_config::<0>();
                let f = &mut o_acc[rf * NFO + d];
                f[0] *= alpha0;
                f[1] *= alpha0;
                f[2] *= alpha1;
                f[3] *= alpha1;
                d += 1;
            }
        }

        let exp_bias0 = -exp_max0 * scale;
        let exp_bias1 = -exp_max1 * scale;
        let mut cs0 = 0.0f32;
        let mut cs1 = 0.0f32;
        let mut nt = 0usize;
        while nt < NKT {
            cuda_device::thread::__unroll_config::<0>();
            let f = &s[rf * NKT + nt];
            let e0 = fast_exp2(fma_scaled(f[0], scale, exp_bias0));
            let e1 = fast_exp2(fma_scaled(f[1], scale, exp_bias0));
            let e2 = fast_exp2(fma_scaled(f[2], scale, exp_bias1));
            let e3 = fast_exp2(fma_scaled(f[3], scale, exp_bias1));
            cs0 += e0 + e1;
            cs1 += e2 + e3;
            p_pack[rf * NKT + nt] = [cvt::<BF16>(e0, e1), cvt::<BF16>(e2, e3)];
            nt += 1;
        }
        // Partial denominators; reduced once in write_output.
        l_row[rf * 2] = l_row[rf * 2] * alpha0 + cs0;
        l_row[rf * 2 + 1] = l_row[rf * 2 + 1] * alpha1 + cs1;
        rf += 1;
    }
}

#[inline(always)]
fn fma_scaled(score: f32, scale: f32, bias: f32) -> f32 {
    let value: f32;
    unsafe {
        ptx_asm!("fma.rn.f32 %0, %1, %2, %3;", out("=f") value,
            in("f") score, in("f") scale, in("f") bias, options(register_only));
    }
    value
}

#[inline(always)]
pub unsafe fn flash_attn_core<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const BLOCK_ROWS: u32,
    const KEYS: u32,
    const RFRAGS: usize,
    const THREADS: u32,
    const NKT: usize,
    const PV_CHUNKS: usize,
    const Q_FRAGS: usize,
    const S_FRAGS: usize,
    const O_FRAGS: usize,
    const ROWS: usize,
    const GRID: u32,
    const PARTIAL: bool,
    const UNROLL_COPY: bool,
    const REVERSE_KEYS: bool,
    const VECTOR_STORE: bool,
    const RECURRENT_COPY: bool,
    const RAW_SCORES: bool,
    const DROPOUT: bool,
>(
    q: *const u32,
    k: *const u32,
    v: *const u32,
    o: *mut u32,
    lse: *mut f32,
    alibi: *const f32,
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
    dropout: Dropout,
) {
    unsafe {
        let mut layout = SmemLayout::new::<HD, KEYS, BLOCK_ROWS>();
        if PARTIAL {
            layout.double = false;
            // Narrow split tiles overlap next K without doubling the V storage.
            layout.triple = HD == 32 && BLOCK_ROWS >= 32;
        }
        let smem = SmemBuffers::new(layout);

        let cfg = FwdConfig::new(
            scale,
            tanh_scale,
            window_left,
            window_right,
            seqlen_q,
            seqlen_k,
            has_alibi,
            alibi_causal,
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
        let tile = LaunchTile::new::<BLOCK_ROWS, RFRAGS, GRID>(
            cfg.h,
            cfg.gqa,
            cfg.seqlen_q,
            cfg.window_right == 0 && cfg.window_left >= cfg.seqlen_k as i32,
        );
        let bases = TensorBases::new(&cfg, &tile);
        let off = cfg.seqlen_k as i32 - cfg.seqlen_q as i32;
        let slope = if cfg.has_alibi != 0 {
            *alibi.add(tile.b as usize * cfg.alibi_bs as usize + tile.head as usize)
        } else {
            0.0
        };

        let scale = cfg.scale * LOG2_E;
        let tanh_scale = cfg.tanh_scale;
        let slope = slope * LOG2_E;

        let q_frag = load_q_frag::<Q_FRAGS, RFRAGS, NFQ, BLOCK_ROWS, THREADS>(
            q, smem.k0, &cfg, &tile, &bases, layout,
        );

        let mut o_acc = [[0.0f32; 4]; O_FRAGS];
        let mut m_row = [f32::NEG_INFINITY; ROWS];
        let mut l_row = [0.0f32; ROWS];

        let kv_range = KvRange::new::<BLOCK_ROWS, KEYS>(&cfg, tile.q_row_base, off);
        let reverse = REVERSE_KEYS;
        prefetch_first_kv::<HD, KEYS, THREADS, UNROLL_COPY, RECURRENT_COPY>(
            k, v, &cfg, &tile, &bases, kv_range, smem, layout, reverse,
        );

        let mut j = if reverse {
            kv_range.num_tiles
        } else {
            kv_range.start_tile
        };
        while if reverse {
            j > kv_range.start_tile
        } else {
            j < kv_range.num_tiles
        } {
            if reverse {
                j -= 1;
            }
            let kv_base = j * KEYS;
            let (k_smem_base, v_smem_base) =
                stage_kv_tile::<HD, KEYS, THREADS, UNROLL_COPY, RECURRENT_COPY>(
                    k, v, &cfg, &tile, &bases, kv_range, smem, layout, j, reverse,
                );
            let mut s = [[0.0f32; 4]; S_FRAGS];
            compute_qk::<BF16, RFRAGS, NFQ, NKT>(&mut s, &q_frag, k_smem_base, &tile, layout);
            if !layout.double && !layout.triple {
                // d64: one barrier finishes V staging and releases K.
                if HD == 64 {
                    ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
                }
                thread::sync_threads();
                if if reverse {
                    j > kv_range.start_tile
                } else {
                    j + 1 < kv_range.num_tiles
                } {
                    stage_tile::<HD, KEYS, THREADS, UNROLL_COPY, RECURRENT_COPY>(
                        smem.k0,
                        k,
                        (if reverse { j - 1 } else { j + 1 }) * KEYS,
                        bases.k,
                        cfg.k.seq,
                        &cfg,
                        &tile,
                        layout,
                    );
                    ptx_asm!("cp.async.commit_group;", clobber("memory"));
                }
            }
            apply_scores::<RFRAGS, NKT, BLOCK_ROWS, KEYS>(
                &mut s,
                &cfg,
                &tile,
                kv_base,
                off,
                if RAW_SCORES { 1.0 } else { scale },
                tanh_scale,
                slope,
            );
            let mut p_pack = [[0u32; 2]; S_FRAGS];
            if RAW_SCORES {
                online_softmax_pack_raw::<BF16, RFRAGS, NKT, NFO>(
                    &s,
                    &mut p_pack,
                    &mut o_acc,
                    &mut m_row,
                    &mut l_row,
                    scale,
                );
            } else {
                online_softmax_pack::<BF16, RFRAGS, NKT, NFO>(
                    &s,
                    &mut p_pack,
                    &mut o_acc,
                    &mut m_row,
                    &mut l_row,
                );
            }
            if DROPOUT {
                dropout.apply::<RFRAGS, NKT>(&mut p_pack, &tile, kv_base);
            }
            if !layout.double && HD != 64 {
                // V is the older committed group; next K may remain in flight.
                if if reverse {
                    j > kv_range.start_tile
                } else {
                    j + 1 < kv_range.num_tiles
                } {
                    ptx_asm!("cp.async.wait_group 1;", clobber("memory"));
                } else {
                    ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
                }
                thread::sync_threads();
            }
            accumulate_values::<BF16, RFRAGS, NKT, NFO, PV_CHUNKS>(
                &mut o_acc,
                &p_pack,
                v_smem_base,
                &tile,
                layout,
            );

            // Double buffering may enqueue the V overwrite early.
            if layout.double {
                thread::sync_threads();
            }
            if !reverse {
                j += 1;
            }
        }

        if RAW_SCORES {
            let mut row = 0usize;
            while row < ROWS {
                cuda_device::thread::__unroll_config::<0>();
                m_row[row] *= scale;
                row += 1;
            }
        }
        write_output::<BF16, RFRAGS, NFO, PARTIAL, BLOCK_ROWS, THREADS, VECTOR_STORE>(
            o,
            lse,
            &o_acc,
            &m_row,
            &l_row,
            &cfg,
            &tile,
            &bases,
            smem.k0,
            layout,
            if DROPOUT { dropout.inv_keep } else { 1.0 },
        );
    }
}
