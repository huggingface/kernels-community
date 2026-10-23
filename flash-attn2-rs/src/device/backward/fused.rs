use super::*;

#[inline(always)]
pub(super) unsafe fn fused_store_ds(packed: &[[u32; 2]; 4], shared: *mut u32, tid: u32) {
    unsafe {
        let lane = tid % 32;
        let key = (tid / 32) * 16 + lane / 4;
        let mut nt = 0;
        while nt < 4 {
            thread::__unroll_config::<0>();
            let query = nt * 8 + (lane % 4) * 2;
            *shared.add(key as usize * 16 + query as usize / 2) = packed[nt as usize][0];
            *shared.add((key as usize + 8) * 16 + query as usize / 2) = packed[nt as usize][1];
            nt += 1;
        }
    }
}
#[inline(always)]
unsafe fn fused_atomic_add(ptr: *mut f32, value: f32) {
    unsafe {
        ptx_asm!("red.relaxed.gpu.global.add.f32 [%0], %1;", in("l") ptr as u64, in("f") value, clobber("memory"));
    }
}
#[inline(always)]
pub(super) unsafe fn fused_dq_tile_d64<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const KSTRIDE: usize,
>(
    keys: *const u32,
    ds: *const u32,
    dq: *mut f32,
    cfg: &FwdConfig,
    tile: &LaunchTile,
    query_start: u32,
) {
    unsafe {
        let lane = tile.lane_id;
        let warp = tile.tid / 32;
        let ds_addr = shared_address(ds);
        let k_addr = shared_address(keys);
        let mut acc = [[0.0f32; 4]; NFQ];
        let mut chunk = 0;
        while chunk < 4 {
            thread::__unroll_config::<0>();
            let key_row = chunk * 16 + lane % 8 + (lane / 16) * 8;
            let query = (warp % 2) * 16 + ((lane / 8) % 2) * 8;
            let a = crate::device::inlined::ldmatrix_x4_trans_shared(
                ds_addr + (key_row * 16 + query / 2) * 4,
            );
            let mut d = 0;
            while d < NFQ as u32 {
                thread::__unroll_config::<0>();
                let row = chunk * 16 + lane % 16;
                let col = (warp / 2) * (HD as u32 / 4) + d * 4 + (lane / 16) * 4;
                let b = crate::device::inlined::ldmatrix_x4_trans_shared(
                    k_addr + (row * KSTRIDE as u32 + (col ^ ((row & 7) * 4))) * 4,
                );
                acc[d as usize] =
                    crate::device::inlined::mma::<BF16>(acc[d as usize], a, [b[0], b[1]]);
                acc[d as usize + 1] =
                    crate::device::inlined::mma::<BF16>(acc[d as usize + 1], a, [b[2], b[3]]);
                d += 2;
            }
            chunk += 1;
        }
        // Reuse the full Q/dOut staging region after all dS reads finish.
        thread::sync_threads();
        let staged = ds as *mut f32;
        let row = (warp % 2) * 16 + lane / 4;
        let mut d = 0;
        while d < NFQ as u32 {
            thread::__unroll_config::<0>();
            let col = (warp / 2) * (HD as u32 / 2) + d * 8 + (lane % 4) * 2;
            let values = acc[d as usize];
            let swizzle = ((row & 3) * 8) | ((row >> 2) & 1);
            *staged.add((row * HD as u32 + (col ^ swizzle)) as usize) = values[0];
            *staged.add((row * HD as u32 + ((col + 1) ^ swizzle)) as usize) = values[1];
            *staged.add(((row + 8) * HD as u32 + (col ^ swizzle)) as usize) = values[2];
            *staged.add(((row + 8) * HD as u32 + ((col + 1) ^ swizzle)) as usize) = values[3];
            d += 1;
        }
        thread::sync_threads();
        let col = tile.tid % HD as u32;
        let first_row = tile.tid / HD as u32;
        if col < cfg.hs_u32 * 2 {
            let rows = (cfg.seqlen_q - query_start).min(32);
            let stride = cfg.h as usize * cfg.hs_u32 as usize * 2;
            let mut base = ((tile.b as usize * cfg.seqlen_q as usize
                + (query_start + first_row) as usize)
                * cfg.h as usize
                + tile.head as usize)
                * cfg.hs_u32 as usize
                * 2
                + col as usize;
            let mut row = first_row;
            while row < rows {
                let swizzle = ((row & 3) * 8) | ((row >> 2) & 1);
                let value = *staged.add((row * HD as u32 + (col ^ swizzle)) as usize);
                fused_atomic_add(dq.add(base), value);
                base += stride * (128 / HD);
                row += (128 / HD) as u32;
            }
        }
    }
}

#[inline(always)]
pub(super) unsafe fn fused_dq_tile<const BF16: bool>(
    keys: *const u32,
    ds: *const u32,
    dq: *mut f32,
    cfg: &FwdConfig,
    tile: &LaunchTile,
    query_start: u32,
) {
    unsafe {
        let lane = tile.lane_id;
        let warp = tile.tid / 32;
        let ds_addr = shared_address(ds);
        let k_addr = shared_address(keys);
        let mut acc = [[0.0f32; 4]; 8];
        let mut chunk = 0;
        while chunk < 4 {
            thread::__unroll_config::<0>();
            let key_row = chunk * 16 + lane % 8 + (lane / 16) * 8;
            let query = (warp % 2) * 16 + ((lane / 8) % 2) * 8;
            let a = crate::device::inlined::ldmatrix_x4_trans_shared(
                ds_addr + (key_row * 16 + query / 2) * 4,
            );
            let mut d = 0;
            while d < 8 {
                thread::__unroll_config::<0>();
                let row = chunk * 16 + lane % 16;
                let col = (warp / 2) * 32 + d * 4 + (lane / 16) * 4;
                let b = crate::device::inlined::ldmatrix_x4_trans_shared(
                    k_addr + (row * 64 + (col ^ ((row & 7) * 4))) * 4,
                );
                acc[d as usize] =
                    crate::device::inlined::mma::<BF16>(acc[d as usize], a, [b[0], b[1]]);
                acc[d as usize + 1] =
                    crate::device::inlined::mma::<BF16>(acc[d as usize + 1], a, [b[2], b[3]]);
                d += 2;
            }
            chunk += 1;
        }
        // Reuse the full Q/dOut staging region after all dS reads finish.
        thread::sync_threads();
        let staged = ds as *mut f32;
        let row = (warp % 2) * 16 + lane / 4;
        let mut d = 0;
        while d < 8 {
            thread::__unroll_config::<0>();
            let col = (warp / 2) * 64 + d * 8 + (lane % 4) * 2;
            let values = acc[d as usize];
            let swizzle = ((row & 3) * 8) | ((row >> 2) & 1);
            *staged.add((row * 128 + (col ^ swizzle)) as usize) = values[0];
            *staged.add((row * 128 + ((col + 1) ^ swizzle)) as usize) = values[1];
            *staged.add(((row + 8) * 128 + (col ^ swizzle)) as usize) = values[2];
            *staged.add(((row + 8) * 128 + ((col + 1) ^ swizzle)) as usize) = values[3];
            d += 1;
        }
        thread::sync_threads();
        let col = tile.tid;
        if col < cfg.hs_u32 * 2 {
            let rows = (cfg.seqlen_q - query_start).min(32);
            let stride = cfg.h as usize * cfg.hs_u32 as usize * 2;
            let mut base = ((tile.b as usize * cfg.seqlen_q as usize + query_start as usize)
                * cfg.h as usize
                + tile.head as usize)
                * cfg.hs_u32 as usize
                * 2
                + col as usize;
            let mut row = 0;
            while row < rows {
                let swizzle = ((row & 3) * 8) | ((row >> 2) & 1);
                let value = *staged.add((row * 128 + (col ^ swizzle)) as usize);
                fused_atomic_add(dq.add(base), value);
                base += stride;
                row += 1;
            }
        }
    }
}

pub trait FusedSpec {
    const MODE: u32;
    const DROPOUT: bool;
    unsafe fn run(
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
    );
}
pub struct Fused<
    const BF16: bool,
    const DROP: bool,
    const PARTIAL: bool,
    const MODE: u32 = 2,
    const HD: usize = 128,
    const NFQ: usize = 8,
    const NFO: usize = 16,
    const KSTRIDE: usize = 64,
>;
impl<
    const BF16: bool,
    const DROP: bool,
    const PARTIAL: bool,
    const MODE: u32,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const KSTRIDE: usize,
> FusedSpec for Fused<BF16, DROP, PARTIAL, MODE, HD, NFQ, NFO, KSTRIDE>
{
    const DROPOUT: bool = DROP;
    const MODE: u32 = MODE;
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
        dq: *mut f32,
        alibi: *const f32,
        cfg: FwdConfig,
        do_strides: Strides,
        gradient_scale: f32,
        dropout: Dropout,
    ) {
        unsafe {
            dkv_both_core::<BF16, HD, NFQ, NFO, PARTIAL, DROP, false, true, MODE, KSTRIDE>(
                q,
                k,
                v,
                dout,
                lse,
                delta,
                dk,
                dv,
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
pub type fused_f16 = Fused<false, false, false>;
#[allow(non_camel_case_types)]
pub type fused_f16_partial = Fused<false, false, true>;
#[allow(non_camel_case_types)]
pub type fused_f16_drop = Fused<false, true, false>;
#[allow(non_camel_case_types)]
pub type fused_f16_drop_partial = Fused<false, true, true>;
#[allow(non_camel_case_types)]
pub type fused_bf16 = Fused<true, false, false>;
#[allow(non_camel_case_types)]
pub type fused_bf16_partial = Fused<true, false, true>;
#[allow(non_camel_case_types)]
pub type fused_bf16_drop = Fused<true, true, false>;
#[allow(non_camel_case_types)]
pub type fused_bf16_drop_partial = Fused<true, true, true>;

#[allow(non_camel_case_types)]
pub type fused_f16_plain = Fused<false, false, false, 0>;
#[allow(non_camel_case_types)]
pub type fused_f16_partial_plain = Fused<false, false, true, 0>;
#[allow(non_camel_case_types)]
pub type fused_f16_drop_plain = Fused<false, true, false, 0>;
#[allow(non_camel_case_types)]
pub type fused_f16_drop_partial_plain = Fused<false, true, true, 0>;
#[allow(non_camel_case_types)]
pub type fused_bf16_plain = Fused<true, false, false, 0>;
#[allow(non_camel_case_types)]
pub type fused_bf16_partial_plain = Fused<true, false, true, 0>;
#[allow(non_camel_case_types)]
pub type fused_bf16_drop_plain = Fused<true, true, false, 0>;
#[allow(non_camel_case_types)]
pub type fused_bf16_drop_partial_plain = Fused<true, true, true, 0>;
#[allow(non_camel_case_types)]
pub type fused_f16_causal = Fused<false, false, false, 1>;
#[allow(non_camel_case_types)]
pub type fused_f16_partial_causal = Fused<false, false, true, 1>;
#[allow(non_camel_case_types)]
pub type fused_f16_drop_causal = Fused<false, true, false, 1>;
#[allow(non_camel_case_types)]
pub type fused_f16_drop_partial_causal = Fused<false, true, true, 1>;
#[allow(non_camel_case_types)]
pub type fused_bf16_causal = Fused<true, false, false, 1>;
#[allow(non_camel_case_types)]
pub type fused_bf16_partial_causal = Fused<true, false, true, 1>;
#[allow(non_camel_case_types)]
pub type fused_bf16_drop_causal = Fused<true, true, false, 1>;
#[allow(non_camel_case_types)]
pub type fused_bf16_drop_partial_causal = Fused<true, true, true, 1>;

#[allow(non_camel_case_types)]
pub type fused_f16_causal_balanced = Fused<false, false, false, 3>;
#[allow(non_camel_case_types)]
pub type fused_f16_partial_causal_balanced = Fused<false, false, true, 3>;
#[allow(non_camel_case_types)]
pub type fused_f16_drop_causal_balanced = Fused<false, true, false, 3>;
#[allow(non_camel_case_types)]
pub type fused_f16_drop_partial_causal_balanced = Fused<false, true, true, 3>;
#[allow(non_camel_case_types)]
pub type fused_bf16_causal_balanced = Fused<true, false, false, 3>;
#[allow(non_camel_case_types)]
pub type fused_bf16_partial_causal_balanced = Fused<true, false, true, 3>;
#[allow(non_camel_case_types)]
pub type fused_bf16_drop_causal_balanced = Fused<true, true, false, 3>;
#[allow(non_camel_case_types)]
pub type fused_bf16_drop_partial_causal_balanced = Fused<true, true, true, 3>;

#[allow(non_camel_case_types)]
pub type fused_d64_f16 = Fused<false, false, false, 2, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_f16_partial = Fused<false, false, true, 2, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_f16_drop = Fused<false, true, false, 2, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_f16_drop_partial = Fused<false, true, true, 2, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_bf16 = Fused<true, false, false, 2, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_bf16_partial = Fused<true, false, true, 2, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_bf16_drop = Fused<true, true, false, 2, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_bf16_drop_partial = Fused<true, true, true, 2, 64, 4, 8, 32>;

#[allow(non_camel_case_types)]
pub type fused_d64_f16_plain = Fused<false, false, false, 0, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_f16_partial_plain = Fused<false, false, true, 0, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_f16_drop_plain = Fused<false, true, false, 0, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_f16_drop_partial_plain = Fused<false, true, true, 0, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_bf16_plain = Fused<true, false, false, 0, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_bf16_partial_plain = Fused<true, false, true, 0, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_bf16_drop_plain = Fused<true, true, false, 0, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_bf16_drop_partial_plain = Fused<true, true, true, 0, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_f16_causal = Fused<false, false, false, 1, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_f16_partial_causal = Fused<false, false, true, 1, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_f16_drop_causal = Fused<false, true, false, 1, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_f16_drop_partial_causal = Fused<false, true, true, 1, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_bf16_causal = Fused<true, false, false, 1, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_bf16_partial_causal = Fused<true, false, true, 1, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_bf16_drop_causal = Fused<true, true, false, 1, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_bf16_drop_partial_causal = Fused<true, true, true, 1, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_f16_causal_balanced = Fused<false, false, false, 3, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_f16_partial_causal_balanced = Fused<false, false, true, 3, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_f16_drop_causal_balanced = Fused<false, true, false, 3, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_f16_drop_partial_causal_balanced = Fused<false, true, true, 3, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_bf16_causal_balanced = Fused<true, false, false, 3, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_bf16_partial_causal_balanced = Fused<true, false, true, 3, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_bf16_drop_causal_balanced = Fused<true, true, false, 3, 64, 4, 8, 32>;
#[allow(non_camel_case_types)]
pub type fused_d64_bf16_drop_partial_causal_balanced = Fused<true, true, true, 3, 64, 4, 8, 32>;

#[kernel(
    fused_d64_f16_plain,
    fused_d64_f16_partial_plain,
    fused_d64_f16_drop_plain,
    fused_d64_f16_drop_partial_plain,
    fused_d64_bf16_plain,
    fused_d64_bf16_partial_plain,
    fused_d64_bf16_drop_plain,
    fused_d64_bf16_drop_partial_plain,
    fused_d64_f16_causal,
    fused_d64_f16_partial_causal,
    fused_d64_f16_drop_causal,
    fused_d64_f16_drop_partial_causal,
    fused_d64_bf16_causal,
    fused_d64_bf16_partial_causal,
    fused_d64_bf16_drop_causal,
    fused_d64_bf16_drop_partial_causal,
    fused_d64_f16_causal_balanced,
    fused_d64_f16_partial_causal_balanced,
    fused_d64_f16_drop_causal_balanced,
    fused_d64_f16_drop_partial_causal_balanced,
    fused_d64_bf16_causal_balanced,
    fused_d64_bf16_partial_causal_balanced,
    fused_d64_bf16_drop_causal_balanced,
    fused_d64_bf16_drop_partial_causal_balanced,
    fused_d64_f16,
    fused_d64_f16_partial,
    fused_d64_f16_drop,
    fused_d64_f16_drop_partial,
    fused_d64_bf16,
    fused_d64_bf16_partial,
    fused_d64_bf16_drop,
    fused_d64_bf16_drop_partial,
    fused_f16,
    fused_f16_partial,
    fused_f16_drop,
    fused_f16_drop_partial,
    fused_bf16,
    fused_bf16_partial,
    fused_bf16_drop,
    fused_bf16_drop_partial,
    fused_f16_plain,
    fused_f16_partial_plain,
    fused_f16_drop_plain,
    fused_f16_drop_partial_plain,
    fused_bf16_plain,
    fused_bf16_partial_plain,
    fused_bf16_drop_plain,
    fused_bf16_drop_partial_plain,
    fused_f16_causal,
    fused_f16_partial_causal,
    fused_f16_drop_causal,
    fused_f16_drop_partial_causal,
    fused_bf16_causal,
    fused_bf16_partial_causal,
    fused_bf16_drop_causal,
    fused_bf16_drop_partial_causal,
    fused_f16_causal_balanced,
    fused_f16_partial_causal_balanced,
    fused_f16_drop_causal_balanced,
    fused_f16_drop_partial_causal_balanced,
    fused_bf16_causal_balanced,
    fused_bf16_partial_causal_balanced,
    fused_bf16_drop_causal_balanced,
    fused_bf16_drop_partial_causal_balanced
)]
#[launch_bounds(128)]
pub fn fused_backward<S: FusedSpec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    dout: &[u32],
    lse: &[f32],
    delta: &[f32],
    dk: DisjointSlice<u32>,
    dv: DisjointSlice<u32>,
    dq: *mut f32,
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
    // Softcap is excluded by host dispatch; kept for the shared ABI.
    let _ = tanh_scale;
    let mut dk = dk;
    let mut dv = dv;
    let cfg = FwdConfig::new(
        scale,
        0.0,
        if S::MODE == 2 { window_left } else { -1 },
        if S::MODE == 0 {
            -1
        } else if S::MODE == 1 || S::MODE == 3 {
            0
        } else {
            window_right
        },
        seqlen_q,
        seqlen_k,
        if S::MODE == 2 { has_alibi } else { 0 },
        if S::MODE == 2 { alibi_causal } else { 0 },
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
            dq,
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
