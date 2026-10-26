mod backward;
mod merge;
mod paged;

use crate::device::config::{FwdConfig, GRID_PLAIN, KvRange, LaunchTile, TensorBases};
use crate::device::dropout::Dropout;
use crate::device::forward::*;
use crate::device::inlined::{cvt, lg2_approx};
use crate::device::varlen::paged::Pages;
use cuda_device::warp::shuffle_xor_f32;
use cuda_device::{DisjointSlice, kernel, launch_bounds, ptx_asm, thread};

#[derive(Clone, Copy)]
pub struct Cache {
    pub indices: *const i32,
    pub batches: u32,
    pub new_keys: u32,
    pub splits: u32,
}
impl Cache {
    pub const NONE: Self = Self {
        indices: core::ptr::null(),
        batches: 0,
        new_keys: 0,
        splits: 1,
    };
}

#[inline(always)]
unsafe fn core<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const KEYS: u32,
    const NKT: usize,
    const PV_CHUNKS: usize,
    const DROPOUT: bool,
    const PAGED: bool,
    const CACHED: bool,
    const SPLIT: bool,
    const GROUPED: bool,
>(
    q: *const u32,
    k: *const u32,
    v: *const u32,
    o: *mut u32,
    lse: *mut f32,
    alibi: *const f32,
    cuq: *const i32,
    cuk: *const i32,
    used: *const i32,
    leftpad: *const i32,
    total_q: u32,
    total_k: u32,
    mut cfg: FwdConfig,
    mut dropout: Dropout,
    pages: Pages,
    cache: Cache,
) {
    unsafe {
        let groups = if GROUPED { cfg.gqa } else { 1 };
        let heads = if GROUPED { cfg.h / groups } else { cfg.h };
        let query_rows = if GROUPED { groups } else { cfg.seqlen_q };
        let mut tile = LaunchTile::new::<64, 1, GRID_PLAIN>(
            heads,
            if GROUPED { 1 } else { cfg.gqa },
            query_rows,
            false,
        );
        let split = if SPLIT {
            // Fold splits into grid.x so batch remains within CUDA's grid.z limit.
            tile.q_row_base =
                (query_rows.div_ceil(64) - 1 - thread::blockIdx_x() / cache.splits) * 64;
            thread::blockIdx_x() % cache.splits
        } else {
            0
        };
        let qs = if CACHED {
            (tile.b * cfg.seqlen_q) as i32
        } else {
            *cuq.add(tile.b as usize)
        };
        let qe = if CACHED {
            qs + cfg.seqlen_q as i32
        } else {
            *cuq.add(tile.b as usize + 1)
        };
        let ks = if CACHED { 0 } else { *cuk.add(tile.b as usize) };
        let ke = if CACHED {
            cfg.seqlen_k as i32
        } else {
            *cuk.add(tile.b as usize + 1)
        };
        let pad = if leftpad.is_null() {
            0
        } else {
            *leftpad.add(tile.b as usize)
        };
        let length = if CACHED {
            (if cuk.is_null() {
                cfg.seqlen_k as i32
            } else {
                *cuk.add(tile.b as usize)
            })
            .wrapping_add(cache.new_keys as i32)
        } else if used.is_null() {
            ke - ks
        } else {
            *used.add(tile.b as usize)
        };
        // As in v2, metadata is trusted; only memory bounds are guarded.
        if qs < 0
            || qe < qs
            || qe as u32 > total_q
            || (qe - qs) as u32 > cfg.seqlen_q
            || ks < 0
            || ke < ks
            || (!PAGED && ke as u32 > total_k)
            || (PAGED && length as u32 > pages.columns.saturating_mul(pages.size))
            || pad < 0
            || length < pad
            || length > ke - ks
            || (length - pad) as u32 > cfg.seqlen_k
        {
            return;
        }
        cfg.seqlen_q = (qe - qs) as u32 * groups;
        cfg.seqlen_k = (length - pad) as u32;
        if GROUPED {
            cfg.h = heads;
            cfg.gqa = 1;
        }
        if tile.q_row_base >= cfg.seqlen_q {
            return;
        }
        dropout.seqlen_q = cfg.seqlen_q;
        dropout.seqlen_k = cfg.seqlen_k;
        let cache_batch = if CACHED && !PAGED && !cache.indices.is_null() {
            *cache.indices.add(tile.b as usize)
        } else {
            tile.b as i32
        };
        if CACHED && !PAGED && (cache_batch < 0 || cache_batch as u32 >= cache.batches) {
            return;
        }
        let o_ss = cfg.h as usize * cfg.hs_u32 as usize;
        let bases = TensorBases {
            q: (if CACHED {
                tile.b as usize * cfg.q.batch as usize
            } else {
                qs as usize * cfg.q.seq as usize
            }) + tile.head as usize * groups as usize * cfg.q.head as usize,
            k: (if CACHED && !PAGED {
                cache_batch as usize * cfg.k.batch as usize
            } else {
                0
            }) + (if PAGED { 0 } else { ks + pad }) as usize * cfg.k.seq as usize
                + tile.kv_head as usize * cfg.k.head as usize,
            v: (if CACHED && !PAGED {
                cache_batch as usize * cfg.v.batch as usize
            } else {
                0
            }) + (if PAGED { 0 } else { ks + pad }) as usize * cfg.v.seq as usize
                + tile.kv_head as usize * cfg.v.head as usize,
            o: qs as usize * o_ss + tile.head as usize * cfg.hs_u32 as usize,
            o_ss,
        };
        if GROUPED {
            // Query rows are adjacent GQA heads at the same position.
            cfg.q.seq = cfg.q.head;
        }
        let layout = SmemLayout::new::<HD, KEYS, 64>();
        let smem = SmemBuffers::new(layout);
        let off = cfg.seqlen_k as i32 - cfg.seqlen_q as i32;
        let slope = if cfg.has_alibi != 0 {
            *alibi.add(tile.b as usize * cfg.alibi_bs as usize + tile.head as usize)
        } else {
            0.0
        };

        let scale = cfg.scale * LOG2_E;
        let tanh_scale = cfg.tanh_scale;
        let slope = slope * LOG2_E;

        let q_frag = load_q_frag::<NFQ, 1, NFQ, 64, 128>(q, smem.k0, &cfg, &tile, &bases, layout);

        let mut o_acc = [[0.0f32; 4]; NFO];
        let mut m_row = [f32::NEG_INFINITY; 2];
        let mut l_row = [0.0f32; 2];

        let mut kv_range = KvRange::new::<64, KEYS>(&cfg, tile.q_row_base, off);
        if SPLIT {
            // Preserve logical positions for masks/ALiBi and page-table lookup.
            let tiles_per_split = cfg.seqlen_k.div_ceil(KEYS).div_ceil(cache.splits);
            kv_range.start_tile = kv_range.start_tile.max(split * tiles_per_split);
            kv_range.num_tiles = kv_range.num_tiles.min((split + 1) * tiles_per_split);
        }
        let reverse = false;
        if PAGED {
            crate::device::varlen::paged::prefetch_first_kv::<HD, KEYS, 128, false, false>(
                k, v, &cfg, &tile, &bases, kv_range, smem, layout, reverse, pages,
            );
        } else {
            prefetch_first_kv::<HD, KEYS, 128, false, false>(
                k, v, &cfg, &tile, &bases, kv_range, smem, layout, reverse,
            );
        }

        let mut j = kv_range.start_tile;
        while j < kv_range.num_tiles {
            let kv_base = j * KEYS;
            let (k_smem_base, v_smem_base) = if PAGED {
                crate::device::varlen::paged::stage_kv_tile::<HD, KEYS, 128, false, false>(
                    k, v, &cfg, &tile, &bases, kv_range, smem, layout, j, reverse, pages,
                )
            } else {
                stage_kv_tile::<HD, KEYS, 128, false, false>(
                    k, v, &cfg, &tile, &bases, kv_range, smem, layout, j, reverse,
                )
            };
            let mut s = [[0.0f32; 4]; NKT];
            compute_qk::<BF16, 1, NFQ, NKT>(&mut s, &q_frag, k_smem_base, &tile, layout);
            if !layout.double && !layout.triple {
                // d64: one barrier finishes V staging and releases K.
                if HD == 64 {
                    ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
                }
                thread::sync_threads();
                if j + 1 < kv_range.num_tiles {
                    if PAGED {
                        crate::device::varlen::paged::stage_tile::<HD, KEYS, 128>(
                            smem.k0,
                            k,
                            (j + 1) * KEYS,
                            bases.k,
                            cfg.k.seq,
                            cfg.k.batch,
                            &cfg,
                            &tile,
                            layout,
                            pages,
                        );
                    } else {
                        stage_tile::<HD, KEYS, 128, false, false>(
                            smem.k0,
                            k,
                            (j + 1) * KEYS,
                            bases.k,
                            cfg.k.seq,
                            &cfg,
                            &tile,
                            layout,
                        );
                    }
                    ptx_asm!("cp.async.commit_group;", clobber("memory"));
                }
            }
            apply_scores::<1, NKT, 64, KEYS>(
                &mut s, &cfg, &tile, kv_base, off, scale, tanh_scale, slope,
            );
            let mut p_pack = [[0u32; 2]; NKT];
            online_softmax_pack::<BF16, 1, NKT, NFO>(
                &s,
                &mut p_pack,
                &mut o_acc,
                &mut m_row,
                &mut l_row,
            );
            if DROPOUT {
                dropout.apply::<1, NKT>(&mut p_pack, &tile, kv_base);
            }
            if !layout.double && HD != 64 {
                // V is the older committed group; next K may remain in flight.
                if j + 1 < kv_range.num_tiles {
                    ptx_asm!("cp.async.wait_group 1;", clobber("memory"));
                } else {
                    ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
                }
                thread::sync_threads();
            }
            accumulate_values::<BF16, 1, NKT, NFO, PV_CHUNKS>(
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
            j += 1;
        }

        write_packed_output::<BF16, NFO, SPLIT>(
            o,
            lse,
            &o_acc,
            &m_row,
            &l_row,
            &cfg,
            &tile,
            &bases,
            if SPLIT {
                ((tile.b as usize * cache.splits as usize + split as usize) * cfg.h as usize
                    + tile.head as usize)
                    * query_rows as usize
            } else if CACHED {
                (tile.b as usize * cfg.h as usize + tile.head as usize) * cfg.seqlen_q as usize
            } else {
                tile.head as usize * total_q as usize + qs as usize
            },
            if DROPOUT { dropout.inv_keep } else { 1.0 },
        );
    }
}

pub trait Spec {
    unsafe fn run(
        q: *const u32,
        k: *const u32,
        v: *const u32,
        o: *mut u32,
        lse: *mut f32,
        alibi: *const f32,
        cuq: *const i32,
        cuk: *const i32,
        used: *const i32,
        leftpad: *const i32,
        total_q: u32,
        total_k: u32,
        cfg: FwdConfig,
        dropout: Dropout,
        pages: Pages,
        cache: Cache,
    );
}
pub struct Varlen<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const KEYS: u32,
    const NKT: usize,
    const PV_CHUNKS: usize,
    const DROPOUT: bool,
    const PAGED: bool,
    const CACHED: bool,
    const SPLIT: bool = false,
    const GROUPED: bool = false,
>;
impl<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const KEYS: u32,
    const NKT: usize,
    const PV_CHUNKS: usize,
    const DROPOUT: bool,
    const PAGED: bool,
    const CACHED: bool,
    const SPLIT: bool,
    const GROUPED: bool,
> Spec
    for Varlen<BF16, HD, NFQ, NFO, KEYS, NKT, PV_CHUNKS, DROPOUT, PAGED, CACHED, SPLIT, GROUPED>
{
    #[inline(always)]
    unsafe fn run(
        q: *const u32,
        k: *const u32,
        v: *const u32,
        o: *mut u32,
        lse: *mut f32,
        alibi: *const f32,
        cuq: *const i32,
        cuk: *const i32,
        used: *const i32,
        leftpad: *const i32,
        total_q: u32,
        total_k: u32,
        cfg: FwdConfig,
        dropout: Dropout,
        pages: Pages,
        cache: Cache,
    ) {
        unsafe {
            core::<BF16, HD, NFQ, NFO, KEYS, NKT, PV_CHUNKS, DROPOUT, PAGED, CACHED, SPLIT, GROUPED>(
                q, k, v, o, lse, alibi, cuq, cuk, used, leftpad, total_q, total_k, cfg, dropout,
                pages, cache,
            );
        }
    }
}
#[allow(non_camel_case_types)]
pub type d32 = Varlen<false, 32, 2, 4, 64, 8, 4, false, false, false>;
#[allow(non_camel_case_types)]
pub type d32_dropout = Varlen<false, 32, 2, 4, 64, 8, 4, true, false, false>;
#[allow(non_camel_case_types)]
pub type d32_bf16 = Varlen<true, 32, 2, 4, 64, 8, 4, false, false, false>;
#[allow(non_camel_case_types)]
pub type d32_bf16_dropout = Varlen<true, 32, 2, 4, 64, 8, 4, true, false, false>;
#[allow(non_camel_case_types)]
pub type d64 = Varlen<false, 64, 4, 8, 64, 8, 4, false, false, false>;
#[allow(non_camel_case_types)]
pub type d64_dropout = Varlen<false, 64, 4, 8, 64, 8, 4, true, false, false>;
#[allow(non_camel_case_types)]
pub type d64_bf16 = Varlen<true, 64, 4, 8, 64, 8, 4, false, false, false>;
#[allow(non_camel_case_types)]
pub type d64_bf16_dropout = Varlen<true, 64, 4, 8, 64, 8, 4, true, false, false>;
#[allow(non_camel_case_types)]
pub type d96 = Varlen<false, 96, 6, 12, 32, 4, 2, false, false, false>;
#[allow(non_camel_case_types)]
pub type d96_dropout = Varlen<false, 96, 6, 12, 32, 4, 2, true, false, false>;
#[allow(non_camel_case_types)]
pub type d96_bf16 = Varlen<true, 96, 6, 12, 32, 4, 2, false, false, false>;
#[allow(non_camel_case_types)]
pub type d96_bf16_dropout = Varlen<true, 96, 6, 12, 32, 4, 2, true, false, false>;
#[allow(non_camel_case_types)]
pub type d128 = Varlen<false, 128, 8, 16, 32, 4, 2, false, false, false>;
#[allow(non_camel_case_types)]
pub type d128_dropout = Varlen<false, 128, 8, 16, 32, 4, 2, true, false, false>;
#[allow(non_camel_case_types)]
pub type d128_bf16 = Varlen<true, 128, 8, 16, 32, 4, 2, false, false, false>;
#[allow(non_camel_case_types)]
pub type d128_bf16_dropout = Varlen<true, 128, 8, 16, 32, 4, 2, true, false, false>;
#[allow(non_camel_case_types)]
pub type d192 = Varlen<false, 192, 12, 24, 32, 4, 2, false, false, false>;
#[allow(non_camel_case_types)]
pub type d192_dropout = Varlen<false, 192, 12, 24, 32, 4, 2, true, false, false>;
#[allow(non_camel_case_types)]
pub type d192_bf16 = Varlen<true, 192, 12, 24, 32, 4, 2, false, false, false>;
#[allow(non_camel_case_types)]
pub type d192_bf16_dropout = Varlen<true, 192, 12, 24, 32, 4, 2, true, false, false>;
#[allow(non_camel_case_types)]
pub type d256 = Varlen<false, 256, 16, 32, 32, 4, 2, false, false, false>;
#[allow(non_camel_case_types)]
pub type d256_dropout = Varlen<false, 256, 16, 32, 32, 4, 2, true, false, false>;
#[allow(non_camel_case_types)]
pub type d256_bf16 = Varlen<true, 256, 16, 32, 32, 4, 2, false, false, false>;
#[allow(non_camel_case_types)]
pub type d256_bf16_dropout = Varlen<true, 256, 16, 32, 32, 4, 2, true, false, false>;
#[kernel(
    d32,
    d32_dropout,
    d32_bf16,
    d32_bf16_dropout,
    d64,
    d64_dropout,
    d64_bf16,
    d64_bf16_dropout,
    d96,
    d96_dropout,
    d96_bf16,
    d96_bf16_dropout,
    d128,
    d128_dropout,
    d128_bf16,
    d128_bf16_dropout,
    d192,
    d192_dropout,
    d192_bf16,
    d192_bf16_dropout,
    d256,
    d256_dropout,
    d256_bf16,
    d256_bf16_dropout
)]
#[launch_bounds(128)]
pub fn varlen_fwd<S: Spec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    out: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    alibi: &[f32],
    cuq: &[i32],
    cuk: &[i32],
    used: &[i32],
    leftpad: &[i32],
    rng: &[u64],
    probabilities: DisjointSlice<u32>,
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
    total_q: u32,
    total_k: u32,
    keep: f32,
) {
    let mut out = out;
    let mut lse = lse;
    let mut probabilities = probabilities;
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
        probabilities: if probabilities.len() == 0 {
            core::ptr::null_mut()
        } else {
            probabilities.as_mut_ptr()
        },
        rounded_q: max_q.div_ceil(128) as u64 * 128,
        rounded_k: max_k.div_ceil(128) as u64 * 128,
        seqlen_q: max_q,
        seqlen_k: max_k,
    };
    unsafe {
        S::run(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            out.as_mut_ptr(),
            lse.as_mut_ptr(),
            alibi.as_ptr(),
            cuq.as_ptr(),
            crate::device::inlined::optional_ptr(cuk),
            crate::device::inlined::optional_ptr(used),
            crate::device::inlined::optional_ptr(leftpad),
            total_q,
            total_k,
            cfg,
            dropout,
            Pages::NONE,
            Cache::NONE,
        );
    }
}

// Packed dropout resolves replay-time Philox state without a separate launch.
#[kernel(d64_dropout, d64_bf16_dropout)]
#[launch_bounds(128)]
pub fn varlen_rng_descriptor<S: Spec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    out: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    alibi: &[f32],
    cuq: &[i32],
    cuk: &[i32],
    used: &[i32],
    leftpad: &[i32],
    rng: DisjointSlice<u64>,
    probabilities: DisjointSlice<u32>,
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
    total_q: u32,
    total_k: u32,
    keep: f32,
    descriptor_seed: u64,
    descriptor_offset: u64,
    intragraph: u64,
    captured: u32,
) {
    let mut out = out;
    let mut lse = lse;
    let mut probabilities = probabilities;
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
    let mut rng = rng;
    let (seed, offset) = unsafe {
        if captured != 0 {
            (
                *(descriptor_seed as *const u64),
                (*(descriptor_offset as *const u64)).wrapping_add(intragraph),
            )
        } else {
            (descriptor_seed, descriptor_offset)
        }
    };
    // Save before early returns; no block depends on this write.
    if thread::blockIdx_x() == 0
        && thread::blockIdx_y() == 0
        && thread::blockIdx_z() == 0
        && thread::threadIdx_x() == 0
    {
        unsafe {
            *rng.as_mut_ptr() = seed;
            *rng.as_mut_ptr().add(1) = offset;
        }
    }
    let dropout = Dropout {
        seed,
        offset,
        threshold: (keep * 255.0) as u32,
        inv_keep: 1.0 / keep,
        probabilities: if probabilities.len() == 0 {
            core::ptr::null_mut()
        } else {
            probabilities.as_mut_ptr()
        },
        rounded_q: max_q.div_ceil(128) as u64 * 128,
        rounded_k: max_k.div_ceil(128) as u64 * 128,
        seqlen_q: max_q,
        seqlen_k: max_k,
    };
    unsafe {
        S::run(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            out.as_mut_ptr(),
            lse.as_mut_ptr(),
            alibi.as_ptr(),
            cuq.as_ptr(),
            crate::device::inlined::optional_ptr(cuk),
            crate::device::inlined::optional_ptr(used),
            crate::device::inlined::optional_ptr(leftpad),
            total_q,
            total_k,
            cfg,
            dropout,
            Pages::NONE,
            Cache::NONE,
        );
    }
}

// Separate path so dense output register allocation stays unchanged.
#[inline(always)]
unsafe fn write_packed_output<const BF16: bool, const NFO: usize, const PARTIAL: bool>(
    out: *mut u32,
    lse: *mut f32,
    acc: &[[f32; 4]],
    maxima: &[f32],
    sums: &[f32],
    cfg: &FwdConfig,
    tile: &LaunchTile,
    bases: &TensorBases,
    lse_base: usize,
    output_scale: f32,
) {
    unsafe {
        let mut l0 = sums[0];
        let mut l1 = sums[1];
        l0 += shuffle_xor_f32(l0, 1);
        l0 += shuffle_xor_f32(l0, 2);
        l1 += shuffle_xor_f32(l1, 1);
        l1 += shuffle_xor_f32(l1, 2);
        let inv0 = if l0 > 0.0 {
            1.0 / l0 * output_scale
        } else {
            0.0
        };
        let inv1 = if l1 > 0.0 {
            1.0 / l1 * output_scale
        } else {
            0.0
        };
        let row = (tile.q_row_base + tile.warp_row + tile.group) as usize;
        let mut dim = 0;
        while dim < NFO {
            thread::__unroll_config::<0>();
            let col = dim * 4 + tile.pair as usize;
            let f = acc[dim];
            if col < cfg.hs_u32 as usize {
                if row < cfg.seqlen_q as usize {
                    if PARTIAL {
                        let target = out
                            .cast::<f32>()
                            .add((lse_base + row) * cfg.hs_u32 as usize * 2 + col * 2);
                        *target = f[0] * inv0;
                        *target.add(1) = f[1] * inv0;
                    } else {
                        *out.add(bases.o + row * bases.o_ss + col) =
                            cvt::<BF16>(f[0] * inv0, f[1] * inv0);
                    }
                }
                if row + 8 < cfg.seqlen_q as usize {
                    if PARTIAL {
                        let target = out
                            .cast::<f32>()
                            .add((lse_base + row + 8) * cfg.hs_u32 as usize * 2 + col * 2);
                        *target = f[2] * inv1;
                        *target.add(1) = f[3] * inv1;
                    } else {
                        *out.add(bases.o + (row + 8) * bases.o_ss + col) =
                            cvt::<BF16>(f[2] * inv1, f[3] * inv1);
                    }
                }
            }
            dim += 1;
        }
        if tile.pair == 0 {
            if row < cfg.seqlen_q as usize {
                *lse.add(lse_base + row) = if l0 > 0.0 {
                    (maxima[0] + lg2_approx(l0)) * core::f32::consts::LN_2
                } else {
                    if PARTIAL {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    }
                };
            }
            if row + 8 < cfg.seqlen_q as usize {
                *lse.add(lse_base + row + 8) = if l1 > 0.0 {
                    (maxima[1] + lg2_approx(l1)) * core::f32::consts::LN_2
                } else {
                    if PARTIAL {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    }
                };
            }
        }
    }
}

#[allow(non_camel_case_types)]
pub type paged_d32 = Varlen<false, 32, 2, 4, 64, 8, 4, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d32_dropout = Varlen<false, 32, 2, 4, 64, 8, 4, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d32_bf16 = Varlen<true, 32, 2, 4, 64, 8, 4, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d32_bf16_dropout = Varlen<true, 32, 2, 4, 64, 8, 4, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d64 = Varlen<false, 64, 4, 8, 64, 8, 4, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d64_dropout = Varlen<false, 64, 4, 8, 64, 8, 4, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d64_bf16 = Varlen<true, 64, 4, 8, 64, 8, 4, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d64_bf16_dropout = Varlen<true, 64, 4, 8, 64, 8, 4, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d96 = Varlen<false, 96, 6, 12, 32, 4, 2, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d96_dropout = Varlen<false, 96, 6, 12, 32, 4, 2, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d96_bf16 = Varlen<true, 96, 6, 12, 32, 4, 2, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d96_bf16_dropout = Varlen<true, 96, 6, 12, 32, 4, 2, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d128 = Varlen<false, 128, 8, 16, 32, 4, 2, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d128_dropout = Varlen<false, 128, 8, 16, 32, 4, 2, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d128_bf16 = Varlen<true, 128, 8, 16, 32, 4, 2, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d128_bf16_dropout = Varlen<true, 128, 8, 16, 32, 4, 2, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d192 = Varlen<false, 192, 12, 24, 32, 4, 2, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d192_dropout = Varlen<false, 192, 12, 24, 32, 4, 2, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d192_bf16 = Varlen<true, 192, 12, 24, 32, 4, 2, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d192_bf16_dropout = Varlen<true, 192, 12, 24, 32, 4, 2, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d256 = Varlen<false, 256, 16, 32, 32, 4, 2, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d256_dropout = Varlen<false, 256, 16, 32, 32, 4, 2, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d256_bf16 = Varlen<true, 256, 16, 32, 32, 4, 2, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d256_bf16_dropout = Varlen<true, 256, 16, 32, 32, 4, 2, true, true, false>;
#[kernel(
    paged_d32,
    paged_d32_dropout,
    paged_d32_bf16,
    paged_d32_bf16_dropout,
    paged_d64,
    paged_d64_dropout,
    paged_d64_bf16,
    paged_d64_bf16_dropout,
    paged_d96,
    paged_d96_dropout,
    paged_d96_bf16,
    paged_d96_bf16_dropout,
    paged_d128,
    paged_d128_dropout,
    paged_d128_bf16,
    paged_d128_bf16_dropout,
    paged_d192,
    paged_d192_dropout,
    paged_d192_bf16,
    paged_d192_bf16_dropout,
    paged_d256,
    paged_d256_dropout,
    paged_d256_bf16,
    paged_d256_bf16_dropout
)]
#[launch_bounds(128)]
pub fn paged_fwd<S: Spec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    out: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    alibi: &[f32],
    cuq: &[i32],
    cuk: &[i32],
    used: &[i32],
    leftpad: &[i32],
    rng: &[u64],
    probabilities: DisjointSlice<u32>,
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
    total_q: u32,
    total_k: u32,
    keep: f32,
    table: &[i32],
    table_stride: u32,
    table_columns: u32,
    page_size: u32,
    page_count: u32,
    k_bs: u32,
    v_bs: u32,
) {
    let mut out = out;
    let mut lse = lse;
    let mut probabilities = probabilities;
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
        k_bs,
        k_ss,
        k_hs,
        v_bs,
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
        probabilities: if probabilities.len() == 0 {
            core::ptr::null_mut()
        } else {
            probabilities.as_mut_ptr()
        },
        rounded_q: max_q.div_ceil(128) as u64 * 128,
        rounded_k: max_k.div_ceil(128) as u64 * 128,
        seqlen_q: max_q,
        seqlen_k: max_k,
    };
    unsafe {
        S::run(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            out.as_mut_ptr(),
            lse.as_mut_ptr(),
            alibi.as_ptr(),
            cuq.as_ptr(),
            crate::device::inlined::optional_ptr(cuk),
            crate::device::inlined::optional_ptr(used),
            crate::device::inlined::optional_ptr(leftpad),
            total_q,
            total_k,
            cfg,
            dropout,
            Pages {
                table: table.as_ptr(),
                row_stride: table_stride,
                columns: table_columns,
                size: page_size,
                count: page_count,
            },
            Cache::NONE,
        );
    }
}

#[allow(non_camel_case_types)]
pub type cache_d32 = Varlen<false, 32, 2, 4, 64, 8, 4, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d32_paged = Varlen<false, 32, 2, 4, 64, 8, 4, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d32_bf16 = Varlen<true, 32, 2, 4, 64, 8, 4, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d32_bf16_paged = Varlen<true, 32, 2, 4, 64, 8, 4, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d64 = Varlen<false, 64, 4, 8, 64, 8, 4, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d64_paged = Varlen<false, 64, 4, 8, 64, 8, 4, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d64_bf16 = Varlen<true, 64, 4, 8, 64, 8, 4, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d64_bf16_paged = Varlen<true, 64, 4, 8, 64, 8, 4, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d96 = Varlen<false, 96, 6, 12, 32, 4, 2, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d96_paged = Varlen<false, 96, 6, 12, 32, 4, 2, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d96_bf16 = Varlen<true, 96, 6, 12, 32, 4, 2, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d96_bf16_paged = Varlen<true, 96, 6, 12, 32, 4, 2, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d128 = Varlen<false, 128, 8, 16, 32, 4, 2, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d128_paged = Varlen<false, 128, 8, 16, 32, 4, 2, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d128_bf16 = Varlen<true, 128, 8, 16, 32, 4, 2, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d128_bf16_paged = Varlen<true, 128, 8, 16, 32, 4, 2, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d192 = Varlen<false, 192, 12, 24, 32, 4, 2, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d192_paged = Varlen<false, 192, 12, 24, 32, 4, 2, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d192_bf16 = Varlen<true, 192, 12, 24, 32, 4, 2, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d192_bf16_paged = Varlen<true, 192, 12, 24, 32, 4, 2, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d256 = Varlen<false, 256, 16, 32, 32, 4, 2, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d256_paged = Varlen<false, 256, 16, 32, 32, 4, 2, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d256_bf16 = Varlen<true, 256, 16, 32, 32, 4, 2, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d256_bf16_paged = Varlen<true, 256, 16, 32, 32, 4, 2, false, true, true>;
#[kernel(
    cache_d32,
    cache_d32_paged,
    cache_d32_bf16,
    cache_d32_bf16_paged,
    cache_d64,
    cache_d64_paged,
    cache_d64_bf16,
    cache_d64_bf16_paged,
    cache_d96,
    cache_d96_paged,
    cache_d96_bf16,
    cache_d96_bf16_paged,
    cache_d128,
    cache_d128_paged,
    cache_d128_bf16,
    cache_d128_bf16_paged,
    cache_d192,
    cache_d192_paged,
    cache_d192_bf16,
    cache_d192_bf16_paged,
    cache_d256,
    cache_d256_paged,
    cache_d256_bf16,
    cache_d256_bf16_paged
)]
#[launch_bounds(128)]
pub fn cached_fwd<S: Spec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    out: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    alibi: &[f32],
    cuq: &[i32],
    cuk: &[i32],
    used: &[i32],
    leftpad: &[i32],
    rng: &[u64],
    probabilities: DisjointSlice<u32>,
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
    total_q: u32,
    total_k: u32,
    keep: f32,
    table: &[i32],
    table_stride: u32,
    table_columns: u32,
    page_size: u32,
    page_count: u32,
    k_bs: u32,
    v_bs: u32,
    indices: &[i32],
    cache_batches: u32,
    new_keys: u32,
    q_bs: u32,
) {
    let mut out = out;
    let mut lse = lse;
    let mut probabilities = probabilities;
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
        probabilities: if probabilities.len() == 0 {
            core::ptr::null_mut()
        } else {
            probabilities.as_mut_ptr()
        },
        rounded_q: max_q.div_ceil(128) as u64 * 128,
        rounded_k: max_k.div_ceil(128) as u64 * 128,
        seqlen_q: max_q,
        seqlen_k: max_k,
    };
    unsafe {
        S::run(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            out.as_mut_ptr(),
            lse.as_mut_ptr(),
            alibi.as_ptr(),
            cuq.as_ptr(),
            crate::device::inlined::optional_ptr(cuk),
            crate::device::inlined::optional_ptr(used),
            crate::device::inlined::optional_ptr(leftpad),
            total_q,
            total_k,
            cfg,
            dropout,
            Pages {
                table: table.as_ptr(),
                row_stride: table_stride,
                columns: table_columns,
                size: page_size,
                count: page_count,
            },
            Cache {
                indices: crate::device::inlined::optional_ptr(indices),
                batches: cache_batches,
                new_keys,
                splits: 1,
            },
        );
    }
}

// FP32 partial-output specializations for cached split-K attention.
#[allow(non_camel_case_types)]
pub type split_d32 = Varlen<false, 32, 2, 4, 64, 8, 4, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d32_paged = Varlen<false, 32, 2, 4, 64, 8, 4, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d32_bf16 = Varlen<true, 32, 2, 4, 64, 8, 4, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d32_bf16_paged = Varlen<true, 32, 2, 4, 64, 8, 4, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d64 = Varlen<false, 64, 4, 8, 64, 8, 4, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d64_paged = Varlen<false, 64, 4, 8, 64, 8, 4, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d64_bf16 = Varlen<true, 64, 4, 8, 64, 8, 4, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d64_bf16_paged = Varlen<true, 64, 4, 8, 64, 8, 4, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d96 = Varlen<false, 96, 6, 12, 32, 4, 2, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d96_paged = Varlen<false, 96, 6, 12, 32, 4, 2, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d96_bf16 = Varlen<true, 96, 6, 12, 32, 4, 2, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d96_bf16_paged = Varlen<true, 96, 6, 12, 32, 4, 2, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d128 = Varlen<false, 128, 8, 16, 32, 4, 2, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d128_paged = Varlen<false, 128, 8, 16, 32, 4, 2, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d128_bf16 = Varlen<true, 128, 8, 16, 32, 4, 2, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d128_bf16_paged = Varlen<true, 128, 8, 16, 32, 4, 2, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d192 = Varlen<false, 192, 12, 24, 32, 4, 2, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d192_paged = Varlen<false, 192, 12, 24, 32, 4, 2, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d192_bf16 = Varlen<true, 192, 12, 24, 32, 4, 2, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d192_bf16_paged = Varlen<true, 192, 12, 24, 32, 4, 2, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d256 = Varlen<false, 256, 16, 32, 32, 4, 2, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d256_paged = Varlen<false, 256, 16, 32, 32, 4, 2, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d256_bf16 = Varlen<true, 256, 16, 32, 32, 4, 2, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d256_bf16_paged = Varlen<true, 256, 16, 32, 32, 4, 2, false, true, true, true>;
// Packed decode reuses the split core while reading cumulative lengths.
#[allow(non_camel_case_types)]
pub type packed_split_d64 = Varlen<false, 64, 4, 8, 64, 8, 4, false, false, false, true>;
#[allow(non_camel_case_types)]
pub type packed_split_d64_paged = Varlen<false, 64, 4, 8, 64, 8, 4, false, true, false, true>;
#[allow(non_camel_case_types)]
pub type packed_split_d64_bf16 = Varlen<true, 64, 4, 8, 64, 8, 4, false, false, false, true>;
#[allow(non_camel_case_types)]
pub type packed_split_d64_bf16_paged = Varlen<true, 64, 4, 8, 64, 8, 4, false, true, false, true>;
#[allow(non_camel_case_types)]
pub type packed_split_d128 = Varlen<false, 128, 8, 16, 32, 4, 2, false, false, false, true>;
#[allow(non_camel_case_types)]
pub type packed_split_d128_paged = Varlen<false, 128, 8, 16, 32, 4, 2, false, true, false, true>;
#[allow(non_camel_case_types)]
pub type packed_split_d128_bf16 = Varlen<true, 128, 8, 16, 32, 4, 2, false, false, false, true>;
#[allow(non_camel_case_types)]
pub type packed_split_d128_bf16_paged =
    Varlen<true, 128, 8, 16, 32, 4, 2, false, true, false, true>;

#[allow(non_camel_case_types)]
pub type packed_grouped_split_d128 =
    Varlen<false, 128, 8, 16, 32, 4, 2, false, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type packed_grouped_split_d128_paged =
    Varlen<false, 128, 8, 16, 32, 4, 2, false, true, false, true, true>;
#[allow(non_camel_case_types)]
pub type packed_grouped_split_d128_bf16 =
    Varlen<true, 128, 8, 16, 32, 4, 2, false, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type packed_grouped_split_d128_bf16_paged =
    Varlen<true, 128, 8, 16, 32, 4, 2, false, true, false, true, true>;

#[kernel(
    packed_grouped_split_d128,
    packed_grouped_split_d128_paged,
    packed_grouped_split_d128_bf16,
    packed_grouped_split_d128_bf16_paged,
    packed_split_d64,
    packed_split_d64_paged,
    packed_split_d64_bf16,
    packed_split_d64_bf16_paged,
    packed_split_d128,
    packed_split_d128_paged,
    packed_split_d128_bf16,
    packed_split_d128_bf16_paged,
    split_d32,
    split_d32_paged,
    split_d32_bf16,
    split_d32_bf16_paged,
    split_d64,
    split_d64_paged,
    split_d64_bf16,
    split_d64_bf16_paged,
    split_d96,
    split_d96_paged,
    split_d96_bf16,
    split_d96_bf16_paged,
    split_d128,
    split_d128_paged,
    split_d128_bf16,
    split_d128_bf16_paged,
    split_d192,
    split_d192_paged,
    split_d192_bf16,
    split_d192_bf16_paged,
    split_d256,
    split_d256_paged,
    split_d256_bf16,
    split_d256_bf16_paged
)]
#[launch_bounds(128)]
pub fn cached_split_fwd<S: Spec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    out: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    alibi: &[f32],
    cuq: &[i32],
    cuk: &[i32],
    used: &[i32],
    leftpad: &[i32],
    rng: &[u64],
    probabilities: DisjointSlice<u32>,
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
    total_q: u32,
    total_k: u32,
    keep: f32,
    table: &[i32],
    table_stride: u32,
    table_columns: u32,
    page_size: u32,
    page_count: u32,
    k_bs: u32,
    v_bs: u32,
    indices: &[i32],
    cache_batches: u32,
    new_keys: u32,
    q_bs: u32,
    splits: u32,
) {
    let mut out = out;
    let mut lse = lse;
    let mut probabilities = probabilities;
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
        probabilities: if probabilities.len() == 0 {
            core::ptr::null_mut()
        } else {
            probabilities.as_mut_ptr()
        },
        rounded_q: max_q.div_ceil(128) as u64 * 128,
        rounded_k: max_k.div_ceil(128) as u64 * 128,
        seqlen_q: max_q,
        seqlen_k: max_k,
    };
    unsafe {
        S::run(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            out.as_mut_ptr(),
            lse.as_mut_ptr(),
            alibi.as_ptr(),
            cuq.as_ptr(),
            crate::device::inlined::optional_ptr(cuk),
            crate::device::inlined::optional_ptr(used),
            crate::device::inlined::optional_ptr(leftpad),
            total_q,
            total_k,
            cfg,
            dropout,
            Pages {
                table: table.as_ptr(),
                row_stride: table_stride,
                columns: table_columns,
                size: page_size,
                count: page_count,
            },
            Cache {
                indices: crate::device::inlined::optional_ptr(indices),
                batches: cache_batches,
                new_keys,
                splits,
            },
        );
    }
}
