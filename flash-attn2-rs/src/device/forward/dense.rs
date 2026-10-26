use cuda_device::{DisjointSlice, kernel, launch_bounds};

use crate::device::config::{GRID_CAUSAL, GRID_PLAIN};
use crate::device::dropout::Dropout;
use crate::device::forward::flash_attn_core;

pub trait FlashSpec {
    unsafe fn run(
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
    );
}

pub struct Flash<
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
    const GRID: u32 = 0,
    const PARTIAL: bool = false,
    const UNROLL_COPY: bool = false,
    const REVERSE_KEYS: bool = false,
    const VECTOR_STORE: bool = true,
    const RECURRENT_COPY: bool = false,
    const RAW_SCORES: bool = false,
    const DROPOUT: bool = false,
>;

impl<
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
> FlashSpec
    for Flash<
        BF16,
        HD,
        NFQ,
        NFO,
        BLOCK_ROWS,
        KEYS,
        RFRAGS,
        THREADS,
        NKT,
        PV_CHUNKS,
        Q_FRAGS,
        S_FRAGS,
        O_FRAGS,
        ROWS,
        GRID,
        PARTIAL,
        UNROLL_COPY,
        REVERSE_KEYS,
        VECTOR_STORE,
        RECURRENT_COPY,
        RAW_SCORES,
        DROPOUT,
    >
{
    #[inline(always)]
    unsafe fn run(
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
            if HD == 192 && BLOCK_ROWS == 128 && KEYS == 64 {
                super::dense192::run::<BF16>(q, k, v, o, lse, scale, seqlen_q, seqlen_k, h);
                return;
            }
            flash_attn_core::<
                BF16,
                HD,
                NFQ,
                NFO,
                BLOCK_ROWS,
                KEYS,
                RFRAGS,
                THREADS,
                NKT,
                PV_CHUNKS,
                Q_FRAGS,
                S_FRAGS,
                O_FRAGS,
                ROWS,
                GRID,
                PARTIAL,
                UNROLL_COPY,
                REVERSE_KEYS,
                VECTOR_STORE,
                RECURRENT_COPY,
                RAW_SCORES,
                DROPOUT,
            >(
                q,
                k,
                v,
                o,
                lse,
                alibi,
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
                dropout,
            );
        }
    }
}

#[allow(non_camel_case_types)]
pub type d32_q128 = Flash<false, 32, 2, 4, 128, 64, 2, 128, 8, 4, 4, 16, 8, 4>;
#[allow(non_camel_case_types)]
pub type d32_q128_bf16 = Flash<true, 32, 2, 4, 128, 64, 2, 128, 8, 4, 4, 16, 8, 4>;
#[allow(non_camel_case_types)]
pub type d32 = Flash<false, 32, 2, 4, 64, 64, 1, 128, 8, 4, 2, 8, 4, 2>;
#[allow(non_camel_case_types)]
pub type d32_bf16 = Flash<true, 32, 2, 4, 64, 64, 1, 128, 8, 4, 2, 8, 4, 2>;
#[allow(non_camel_case_types)]
pub type d64 = Flash<false, 64, 4, 8, 64, 64, 1, 128, 8, 4, 4, 8, 8, 2>;
#[allow(non_camel_case_types)]
pub type d64_bf16 = Flash<true, 64, 4, 8, 64, 64, 1, 128, 8, 4, 4, 8, 8, 2>;
#[allow(non_camel_case_types)]
pub type d96 = Flash<false, 96, 6, 12, 64, 64, 1, 128, 8, 4, 6, 8, 12, 2>;
#[allow(non_camel_case_types)]
pub type d96_bf16 = Flash<true, 96, 6, 12, 64, 64, 1, 128, 8, 4, 6, 8, 12, 2>;
#[allow(non_camel_case_types)]
pub type d128 = Flash<false, 128, 8, 16, 64, 32, 1, 128, 4, 2, 8, 4, 16, 2>;
#[allow(non_camel_case_types)]
pub type d128_bf16 = Flash<true, 128, 8, 16, 64, 32, 1, 128, 4, 2, 8, 4, 16, 2>;
#[allow(non_camel_case_types)]
pub type d192_short = Flash<
    false,
    192,
    12,
    24,
    64,
    48,
    1,
    128,
    6,
    3,
    12,
    6,
    24,
    2,
    GRID_PLAIN,
    false,
    false,
    false,
    true,
    true,
>;
pub type d192_short_causal = Flash<
    false,
    192,
    12,
    24,
    64,
    48,
    1,
    128,
    6,
    3,
    12,
    6,
    24,
    2,
    GRID_CAUSAL,
    false,
    false,
    false,
    true,
    true,
>;
pub type d192_short_bf16 = Flash<
    true,
    192,
    12,
    24,
    64,
    48,
    1,
    128,
    6,
    3,
    12,
    6,
    24,
    2,
    GRID_PLAIN,
    false,
    false,
    false,
    true,
    true,
>;
pub type d192_short_causal_bf16 = Flash<
    true,
    192,
    12,
    24,
    64,
    48,
    1,
    128,
    6,
    3,
    12,
    6,
    24,
    2,
    GRID_CAUSAL,
    false,
    false,
    false,
    true,
    true,
>;
pub type d192 = Flash<false, 192, 12, 24, 64, 48, 1, 128, 6, 3, 12, 6, 24, 2, GRID_PLAIN>;
#[allow(non_camel_case_types)]
pub type d192_bf16 = Flash<true, 192, 12, 24, 64, 48, 1, 128, 6, 3, 12, 6, 24, 2, GRID_PLAIN>;
#[allow(non_camel_case_types)]
pub type d256 = Flash<false, 256, 16, 32, 128, 32, 1, 256, 4, 2, 16, 4, 32, 2>;
#[allow(non_camel_case_types)]
pub type d256_bf16 = Flash<true, 256, 16, 32, 128, 32, 1, 256, 4, 2, 16, 4, 32, 2>;
#[allow(non_camel_case_types)]
pub type d96_k32 = Flash<false, 96, 6, 12, 128, 32, 2, 128, 4, 2, 12, 8, 24, 4>;
#[allow(non_camel_case_types)]
pub type d96_k32_bf16 = Flash<true, 96, 6, 12, 128, 32, 2, 128, 4, 2, 12, 8, 24, 4>;
#[allow(non_camel_case_types)]
pub type d128_k32 = Flash<false, 128, 8, 16, 128, 32, 2, 128, 4, 2, 16, 8, 32, 4>;
#[allow(non_camel_case_types)]
pub type d128_k32_bf16 = Flash<true, 128, 8, 16, 128, 32, 2, 128, 4, 2, 16, 8, 32, 4>;
#[allow(non_camel_case_types)]
pub type d192_k32 = Flash<false, 192, 12, 24, 64, 48, 1, 128, 6, 3, 12, 6, 24, 2, GRID_CAUSAL>;
#[allow(non_camel_case_types)]
pub type d192_k32_bf16 = Flash<true, 192, 12, 24, 64, 48, 1, 128, 6, 3, 12, 6, 24, 2, GRID_CAUSAL>;
#[allow(non_camel_case_types)]
pub type d256_k32 = Flash<false, 256, 16, 32, 64, 32, 1, 128, 4, 2, 16, 4, 32, 2>;
#[allow(non_camel_case_types)]
pub type d256_k32_bf16 = Flash<true, 256, 16, 32, 64, 32, 1, 128, 4, 2, 16, 4, 32, 2>;

// Direct-store variants preserve the original code for local windows.
#[allow(non_camel_case_types)]
pub type d64_scalar =
    Flash<false, 64, 4, 8, 64, 64, 1, 128, 8, 4, 4, 8, 8, 2, 0, false, false, false, false>;
#[allow(non_camel_case_types)]
pub type d64_bf16_scalar =
    Flash<true, 64, 4, 8, 64, 64, 1, 128, 8, 4, 4, 8, 8, 2, 0, false, false, false, false>;
#[allow(non_camel_case_types)]
pub type d96_scalar =
    Flash<false, 96, 6, 12, 64, 64, 1, 128, 8, 4, 6, 8, 12, 2, 0, false, false, false, false>;
#[allow(non_camel_case_types)]
pub type d96_bf16_scalar =
    Flash<true, 96, 6, 12, 64, 64, 1, 128, 8, 4, 6, 8, 12, 2, 0, false, false, false, false>;
#[allow(non_camel_case_types)]
pub type d128_scalar =
    Flash<false, 128, 8, 16, 64, 32, 1, 128, 4, 2, 8, 4, 16, 2, 0, false, false, false, false>;
#[allow(non_camel_case_types)]
pub type d128_bf16_scalar =
    Flash<true, 128, 8, 16, 64, 32, 1, 128, 4, 2, 8, 4, 16, 2, 0, false, false, false, false>;
#[allow(non_camel_case_types)]
pub type d192_scalar = Flash<
    false,
    192,
    12,
    24,
    64,
    32,
    1,
    128,
    4,
    2,
    12,
    4,
    24,
    2,
    GRID_PLAIN,
    false,
    false,
    false,
    false,
>;
#[allow(non_camel_case_types)]
pub type d192_bf16_scalar = Flash<
    true,
    192,
    12,
    24,
    64,
    32,
    1,
    128,
    4,
    2,
    12,
    4,
    24,
    2,
    GRID_PLAIN,
    false,
    false,
    false,
    false,
>;
#[allow(non_camel_case_types)]
pub type d256_scalar =
    Flash<false, 256, 16, 32, 128, 32, 1, 256, 4, 2, 16, 4, 32, 2, 0, false, false, false, false>;
#[allow(non_camel_case_types)]
pub type d256_bf16_scalar =
    Flash<true, 256, 16, 32, 128, 32, 1, 256, 4, 2, 16, 4, 32, 2, 0, false, false, false, false>;
#[allow(non_camel_case_types)]
pub type d96_k32_scalar =
    Flash<false, 96, 6, 12, 128, 32, 2, 128, 4, 2, 12, 8, 24, 4, 0, false, false, false, false>;
#[allow(non_camel_case_types)]
pub type d96_k32_bf16_scalar =
    Flash<true, 96, 6, 12, 128, 32, 2, 128, 4, 2, 12, 8, 24, 4, 0, false, false, false, false>;
#[allow(non_camel_case_types)]
pub type d128_k32_scalar =
    Flash<false, 128, 8, 16, 128, 32, 2, 128, 4, 2, 16, 8, 32, 4, 0, false, false, false, false>;
#[allow(non_camel_case_types)]
pub type d128_k32_bf16_scalar =
    Flash<true, 128, 8, 16, 128, 32, 2, 128, 4, 2, 16, 8, 32, 4, 0, false, false, false, false>;
#[allow(non_camel_case_types)]
pub type d192_k32_scalar = Flash<
    false,
    192,
    12,
    24,
    64,
    32,
    1,
    128,
    4,
    2,
    12,
    4,
    24,
    2,
    GRID_CAUSAL,
    false,
    false,
    false,
    false,
>;
#[allow(non_camel_case_types)]
pub type d192_k32_bf16_scalar = Flash<
    true,
    192,
    12,
    24,
    64,
    32,
    1,
    128,
    4,
    2,
    12,
    4,
    24,
    2,
    GRID_CAUSAL,
    false,
    false,
    false,
    false,
>;
#[allow(non_camel_case_types)]
pub type d256_k32_scalar =
    Flash<false, 256, 16, 32, 64, 32, 1, 128, 4, 2, 16, 4, 32, 2, 0, false, false, false, false>;
#[allow(non_camel_case_types)]
pub type d256_k32_bf16_scalar =
    Flash<true, 256, 16, 32, 64, 32, 1, 128, 4, 2, 16, 4, 32, 2, 0, false, false, false, false>;

#[kernel(
    d64_scalar,
    d64_bf16_scalar,
    d96_scalar,
    d96_bf16_scalar,
    d128_scalar,
    d128_bf16_scalar,
    d192_scalar,
    d192_bf16_scalar,
    d256_scalar,
    d256_bf16_scalar,
    d96_k32_scalar,
    d96_k32_bf16_scalar,
    d128_k32_scalar,
    d128_k32_bf16_scalar,
    d192_k32_scalar,
    d192_k32_bf16_scalar,
    d256_k32_scalar,
    d256_k32_bf16_scalar,
    d32,
    d32_bf16,
    d32_q128,
    d32_raw,
    d32_wide_raw,
    d32_wide_raw_bf16,
    d32_raw_bf16,
    d32_q128_bf16,
    d64,
    d64_bf16,
    d96,
    d96_bf16,
    d128,
    d128_bf16,
    d128_raw,
    d128_raw_bf16,
    d192_dense,
    d192_dense_bf16,
    d192,
    d192_bf16,
    d192_short,
    d192_short_bf16,
    d192_short_causal,
    d192_short_causal_bf16,
    d256,
    d256_bf16,
    d96_k32,
    d96_k32_bf16,
    d128_k32,
    d128_k32_bf16,
    d192_k32,
    d192_k32_bf16,
    d256_k32,
    d256_k32_bf16
)]
#[launch_bounds(256)]
pub fn flash_attn_fwd<S: FlashSpec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    o: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    alibi: &[f32],
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
) {
    let mut o = o;
    let mut lse = lse;
    unsafe {
        S::run(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            o.as_mut_ptr(),
            lse.as_mut_ptr(),
            alibi.as_ptr(),
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
            Dropout::NONE,
        );
    }
}

#[allow(non_camel_case_types)]
pub type split_d32 = Flash<false, 32, 2, 4, 16, 32, 1, 32, 4, 2, 2, 4, 4, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d32_bf16 = Flash<true, 32, 2, 4, 16, 32, 1, 32, 4, 2, 2, 4, 4, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d64 = Flash<false, 64, 4, 8, 16, 32, 1, 32, 4, 2, 4, 4, 8, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d64_bf16 = Flash<true, 64, 4, 8, 16, 32, 1, 32, 4, 2, 4, 4, 8, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d96 = Flash<false, 96, 6, 12, 16, 32, 1, 32, 4, 2, 6, 4, 12, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d96_bf16 =
    Flash<true, 96, 6, 12, 16, 32, 1, 32, 4, 2, 6, 4, 12, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d128 = Flash<false, 128, 8, 16, 16, 32, 1, 32, 4, 2, 8, 4, 16, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d128_bf16 =
    Flash<true, 128, 8, 16, 16, 32, 1, 32, 4, 2, 8, 4, 16, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d192 =
    Flash<false, 192, 12, 24, 16, 16, 1, 32, 2, 1, 12, 2, 24, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d192_bf16 =
    Flash<true, 192, 12, 24, 16, 16, 1, 32, 2, 1, 12, 2, 24, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d256 =
    Flash<false, 256, 16, 32, 16, 16, 1, 32, 2, 1, 16, 2, 32, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d256_bf16 =
    Flash<true, 256, 16, 32, 16, 16, 1, 32, 2, 1, 16, 2, 32, 2, GRID_PLAIN, true>;

#[allow(non_camel_case_types)]
pub type split_fast_d32_q64 =
    Flash<false, 32, 2, 4, 64, 64, 1, 128, 8, 4, 2, 8, 4, 2, GRID_PLAIN, true, true>;
#[allow(non_camel_case_types)]
pub type split_fast_d32_q64_bf16 =
    Flash<true, 32, 2, 4, 64, 64, 1, 128, 8, 4, 2, 8, 4, 2, GRID_PLAIN, true, true>;
#[allow(non_camel_case_types)]
pub type split_k64_d32_q64 =
    Flash<false, 32, 2, 4, 64, 64, 1, 128, 8, 4, 2, 8, 4, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_k64_d32_q64_bf16 =
    Flash<true, 32, 2, 4, 64, 64, 1, 128, 8, 4, 2, 8, 4, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_k64_d32_q32 =
    Flash<false, 32, 2, 4, 32, 64, 1, 64, 8, 4, 2, 8, 4, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_k64_d32_q32_bf16 =
    Flash<true, 32, 2, 4, 32, 64, 1, 64, 8, 4, 2, 8, 4, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d32_q32 = Flash<false, 32, 2, 4, 32, 32, 1, 64, 4, 2, 2, 4, 4, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d32_q32_bf16 =
    Flash<true, 32, 2, 4, 32, 32, 1, 64, 4, 2, 2, 4, 4, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d64_q32 = Flash<false, 64, 4, 8, 32, 32, 1, 64, 4, 2, 4, 4, 8, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d64_q32_bf16 =
    Flash<true, 64, 4, 8, 32, 32, 1, 64, 4, 2, 4, 4, 8, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d96_q32 =
    Flash<false, 96, 6, 12, 32, 32, 1, 64, 4, 2, 6, 4, 12, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d96_q32_bf16 =
    Flash<true, 96, 6, 12, 32, 32, 1, 64, 4, 2, 6, 4, 12, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d128_q32 =
    Flash<false, 128, 8, 16, 32, 32, 1, 64, 4, 2, 8, 4, 16, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d128_q32_bf16 =
    Flash<true, 128, 8, 16, 32, 32, 1, 64, 4, 2, 8, 4, 16, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d192_q32 =
    Flash<false, 192, 12, 24, 32, 16, 1, 64, 2, 1, 12, 2, 24, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d192_q32_bf16 =
    Flash<true, 192, 12, 24, 32, 16, 1, 64, 2, 1, 12, 2, 24, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d256_q32 =
    Flash<false, 256, 16, 32, 32, 16, 1, 64, 2, 1, 16, 2, 32, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d256_q32_bf16 =
    Flash<true, 256, 16, 32, 32, 16, 1, 64, 2, 1, 16, 2, 32, 2, GRID_PLAIN, true>;

#[allow(non_camel_case_types)]
pub type split_d192_q64 =
    Flash<false, 192, 12, 24, 64, 16, 1, 128, 2, 1, 12, 2, 24, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d192_q64_bf16 =
    Flash<true, 192, 12, 24, 64, 16, 1, 128, 2, 1, 12, 2, 24, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d256_q64 =
    Flash<false, 256, 16, 32, 64, 16, 1, 128, 2, 1, 16, 2, 32, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_d256_q64_bf16 =
    Flash<true, 256, 16, 32, 64, 16, 1, 128, 2, 1, 16, 2, 32, 2, GRID_PLAIN, true>;

#[allow(non_camel_case_types)]
pub type split_fast_d128_q16 =
    Flash<false, 128, 8, 16, 16, 32, 1, 32, 4, 2, 8, 4, 16, 2, GRID_PLAIN, true, true>;
#[allow(non_camel_case_types)]
pub type split_fast_d128_q16_bf16 =
    Flash<true, 128, 8, 16, 16, 32, 1, 32, 4, 2, 8, 4, 16, 2, GRID_PLAIN, true, true>;
#[allow(non_camel_case_types)]
pub type split_fast_d128_q32 =
    Flash<false, 128, 8, 16, 32, 32, 1, 64, 4, 2, 8, 4, 16, 2, GRID_PLAIN, true, true>;
#[allow(non_camel_case_types)]
pub type split_fast_d128_q32_bf16 =
    Flash<true, 128, 8, 16, 32, 32, 1, 64, 4, 2, 8, 4, 16, 2, GRID_PLAIN, true, true>;

#[allow(non_camel_case_types)]
pub type split_k64_d64_q32 =
    Flash<false, 64, 4, 8, 32, 64, 1, 64, 8, 4, 4, 8, 8, 2, GRID_PLAIN, true>;
#[allow(non_camel_case_types)]
pub type split_k64_d64_q32_bf16 =
    Flash<true, 64, 4, 8, 32, 64, 1, 64, 8, 4, 4, 8, 8, 2, GRID_PLAIN, true>;

// Independent key segments for short, unmasked queries.
#[kernel(
    split_fast_d32_q64,
    split_fast_d32_q64_bf16,
    split_k64_d32_q64,
    split_k64_d32_q64_bf16,
    split_k64_d32_q32,
    split_k64_d32_q32_bf16,
    split_k64_d64_q32,
    split_k64_d64_q32_bf16,
    split_fast_d128_q16,
    split_fast_d128_q16_bf16,
    split_fast_d128_q32,
    split_fast_d128_q32_bf16,
    split_d192_q64,
    split_d192_q64_bf16,
    split_d256_q64,
    split_d256_q64_bf16,
    split_d32_q32,
    split_d32_q32_bf16,
    split_d64_q32,
    split_d64_q32_bf16,
    split_d96_q32,
    split_d96_q32_bf16,
    split_d128_q32,
    split_d128_q32_bf16,
    split_d192_q32,
    split_d192_q32_bf16,
    split_d256_q32,
    split_d256_q32_bf16,
    split_d32,
    split_d32_bf16,
    split_d64,
    split_d64_bf16,
    split_d96,
    split_d96_bf16,
    split_d128,
    split_d128_bf16,
    split_d192,
    split_d192_bf16,
    split_d256,
    split_d256_bf16
)]
#[launch_bounds(256)]
pub fn flash_attn_split<S: FlashSpec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    o: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    alibi: &[f32],
    scale: f32,
    tanh_scale: f32,
    _window_left: i32,
    _window_right: i32,
    seqlen_q: u32,
    seqlen_k: u32,
    _has_alibi: u32,
    _alibi_causal: u32,
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
    _alibi_bs: u32,
    splits: u32,
) {
    let mut o = o;
    let mut lse = lse;
    let split_batch = cuda_device::thread::blockIdx_z();
    let batch = split_batch / splits;
    let split = split_batch % splits;
    let chunk = (seqlen_k + splits - 1) / splits;
    let start = split * chunk;
    let count = core::cmp::min(chunk, seqlen_k - start);
    unsafe {
        S::run(
            q.as_ptr().add(batch as usize * q_bs as usize),
            k.as_ptr()
                .add(batch as usize * k_bs as usize + start as usize * k_ss as usize),
            v.as_ptr()
                .add(batch as usize * v_bs as usize + start as usize * v_ss as usize),
            o.as_mut_ptr(),
            lse.as_mut_ptr(),
            alibi.as_ptr(),
            scale,
            tanh_scale,
            -1,
            -1,
            seqlen_q,
            count,
            0,
            0,
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
            0,
            Dropout::NONE,
        );
    }
}

#[allow(non_camel_case_types)]
pub type reverse_split_d256_q32 =
    Flash<false, 256, 16, 32, 32, 16, 1, 64, 2, 1, 16, 2, 32, 2, GRID_PLAIN, true, false, true>;
#[allow(non_camel_case_types)]
pub type reverse_split_d256_q32_bf16 =
    Flash<true, 256, 16, 32, 32, 16, 1, 64, 2, 1, 16, 2, 32, 2, GRID_PLAIN, true, false, true>;
#[allow(non_camel_case_types)]
pub type reverse_split_d256_q64 =
    Flash<false, 256, 16, 32, 64, 16, 1, 128, 2, 1, 16, 2, 32, 2, GRID_PLAIN, true, false, true>;
#[allow(non_camel_case_types)]
pub type reverse_split_d256_q64_bf16 =
    Flash<true, 256, 16, 32, 64, 16, 1, 128, 2, 1, 16, 2, 32, 2, GRID_PLAIN, true, false, true>;

// Translate the global causal boundary into each key segment's coordinates.
#[kernel(
    split_fast_d32_q64,
    split_fast_d32_q64_bf16,
    split_k64_d32_q64,
    split_k64_d32_q64_bf16,
    split_k64_d32_q32,
    split_k64_d32_q32_bf16,
    split_k64_d64_q32,
    split_k64_d64_q32_bf16,
    reverse_split_d256_q32,
    reverse_split_d256_q32_bf16,
    reverse_split_d256_q64,
    reverse_split_d256_q64_bf16,
    split_fast_d128_q32,
    split_fast_d128_q32_bf16,
    split_d32_q32,
    split_d32_q32_bf16,
    split_d64_q32,
    split_d64_q32_bf16,
    split_d96_q32,
    split_d96_q32_bf16,
    split_d128_q32,
    split_d128_q32_bf16,
    split_d192_q32,
    split_d192_q32_bf16,
    split_d256_q32,
    split_d256_q32_bf16,
    split_d192_q64,
    split_d192_q64_bf16,
    split_d256_q64,
    split_d256_q64_bf16
)]
#[launch_bounds(128)]
pub fn flash_attn_split_causal<S: FlashSpec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    o: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    alibi: &[f32],
    scale: f32,
    tanh_scale: f32,
    _window_left: i32,
    _window_right: i32,
    seqlen_q: u32,
    seqlen_k: u32,
    _has_alibi: u32,
    _alibi_causal: u32,
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
    _alibi_bs: u32,
    splits: u32,
) {
    let mut o = o;
    let mut lse = lse;
    let split_batch = cuda_device::thread::blockIdx_z();
    let batch = split_batch / splits;
    let split = split_batch % splits;
    let chunk = (seqlen_k + splits - 1) / splits;
    let start = split * chunk;
    let count = core::cmp::min(chunk, seqlen_k - start);
    unsafe {
        S::run(
            q.as_ptr().add(batch as usize * q_bs as usize),
            k.as_ptr()
                .add(batch as usize * k_bs as usize + start as usize * k_ss as usize),
            v.as_ptr()
                .add(batch as usize * v_bs as usize + start as usize * v_ss as usize),
            o.as_mut_ptr(),
            lse.as_mut_ptr(),
            alibi.as_ptr(),
            scale,
            tanh_scale,
            -1,
            (seqlen_k - start - count) as i32,
            seqlen_q,
            count,
            0,
            0,
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
            0,
            Dropout::NONE,
        );
    }
}

pub struct Merge<const BF16: bool>;
pub trait MergeSpec {
    const BF16: bool;
}
impl<const BF16: bool> MergeSpec for Merge<BF16> {
    const BF16: bool = BF16;
}
#[allow(non_camel_case_types)]
pub type merge_f16 = Merge<false>;
#[allow(non_camel_case_types)]
pub type merge_bf16 = Merge<true>;

#[kernel(merge_f16, merge_bf16)]
#[launch_bounds(128)]
pub fn flash_attn_merge<S: MergeSpec>(
    partial: &[u32],
    stats: &[f32],
    out: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    sq: u32,
    heads: u32,
    words: u32,
    splits: u32,
    groups: u32,
) {
    use crate::device::inlined::{cvt, fast_exp2, fmax, lg2_approx};
    use cuda_device::{shared::DynamicSharedArray, thread, warp::shuffle_xor_f32};
    let mut out = out;
    let mut lse = lse;
    let row = thread::blockIdx_x();
    let batch = row / (heads * sq);
    let head = (row / sq) % heads;
    let query = row % sq;
    let col = thread::threadIdx_x();
    unsafe {
        // Inactive lanes contribute -inf to max and zero to the sum.
        let weights = DynamicSharedArray::<f32, 16>::get() as usize as *mut f32;
        if col < 32 {
            let index = ((batch * splits + col) * heads + head) * sq + query;
            let value = if col < splits {
                *stats.as_ptr().add(index as usize)
            } else {
                f32::NEG_INFINITY
            };
            let mut maximum = value;
            maximum = fmax(maximum, shuffle_xor_f32(maximum, 16));
            maximum = fmax(maximum, shuffle_xor_f32(maximum, 8));
            maximum = fmax(maximum, shuffle_xor_f32(maximum, 4));
            maximum = fmax(maximum, shuffle_xor_f32(maximum, 2));
            maximum = fmax(maximum, shuffle_xor_f32(maximum, 1));
            let weight = fast_exp2((value - maximum) * crate::device::forward::LOG2_E);
            let mut denom = weight;
            denom += shuffle_xor_f32(denom, 16);
            denom += shuffle_xor_f32(denom, 8);
            denom += shuffle_xor_f32(denom, 4);
            denom += shuffle_xor_f32(denom, 2);
            denom += shuffle_xor_f32(denom, 1);
            *weights.add(col as usize) = weight / denom;
            if col == 0 {
                *lse.as_mut_ptr().add(row as usize) =
                    maximum + lg2_approx(denom) / crate::device::forward::LOG2_E;
            }
        }
        thread::sync_threads();
        if col >= words {
            return;
        }
        let mut lo = 0.0f32;
        let mut hi = 0.0f32;
        let mut split = 0;
        while split < splits {
            let base = batch * splits + split;
            let weight = *weights.add(split as usize);
            let oi = if groups > 1 {
                ((base * groups + head % groups) * (heads / groups) + head / groups) * words + col
            } else {
                ((base * sq + query) * heads + head) * words + col
            };
            let a = f32::from_bits(*partial.as_ptr().add((oi * 2) as usize));
            let b = f32::from_bits(*partial.as_ptr().add((oi * 2 + 1) as usize));
            lo += weight * a;
            hi += weight * b;
            split += 1;
        }
        let oi = ((batch * sq + query) * heads + head) * words + col;
        *out.as_mut_ptr().add(oi as usize) = if S::BF16 {
            cvt::<true>(lo, hi)
        } else {
            cvt::<false>(lo, hi)
        };
    }
}

#[kernel(merge_f16, merge_bf16)]
#[launch_bounds(128)]
pub fn flash_attn_merge4<S: MergeSpec>(
    partial: &[u32],
    stats: &[f32],
    out: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    sq: u32,
    heads: u32,
    words: u32,
    splits: u32,
    groups: u32,
) {
    use crate::device::inlined::{cvt, fast_exp2, fmax, lg2_approx};
    use cuda_device::{shared::DynamicSharedArray, thread, warp::shuffle_xor_f32};
    let mut out = out;
    let mut lse = lse;
    // Each warp owns one row, including its shared weights and synchronization.
    let warp = thread::threadIdx_x() / 32;
    let row = thread::blockIdx_x() * 4 + warp;
    if row as usize >= lse.len() {
        return;
    }
    let batch = row / (heads * sq);
    let head = (row / sq) % heads;
    let query = row % sq;
    let col = thread::threadIdx_x() % 32;
    unsafe {
        // Inactive lanes contribute -inf to max and zero to the sum.
        let weights =
            (DynamicSharedArray::<f32, 16>::get() as usize as *mut f32).add(warp as usize * 32);
        if col < 32 {
            let index = ((batch * splits + col) * heads + head) * sq + query;
            let value = if col < splits {
                *stats.as_ptr().add(index as usize)
            } else {
                f32::NEG_INFINITY
            };
            let mut maximum = value;
            maximum = fmax(maximum, shuffle_xor_f32(maximum, 16));
            maximum = fmax(maximum, shuffle_xor_f32(maximum, 8));
            maximum = fmax(maximum, shuffle_xor_f32(maximum, 4));
            maximum = fmax(maximum, shuffle_xor_f32(maximum, 2));
            maximum = fmax(maximum, shuffle_xor_f32(maximum, 1));
            let weight = fast_exp2((value - maximum) * crate::device::forward::LOG2_E);
            let mut denom = weight;
            denom += shuffle_xor_f32(denom, 16);
            denom += shuffle_xor_f32(denom, 8);
            denom += shuffle_xor_f32(denom, 4);
            denom += shuffle_xor_f32(denom, 2);
            denom += shuffle_xor_f32(denom, 1);
            *weights.add(col as usize) = weight / denom;
            if col == 0 {
                *lse.as_mut_ptr().add(row as usize) =
                    maximum + lg2_approx(denom) / crate::device::forward::LOG2_E;
            }
        }
        cuda_device::ptx_asm!("bar.warp.sync 0xffffffff;", clobber("memory"));
        if col >= words {
            return;
        }
        let mut lo = 0.0f32;
        let mut hi = 0.0f32;
        let mut split = 0;
        while split < splits {
            let base = batch * splits + split;
            let weight = *weights.add(split as usize);
            let oi = if groups > 1 {
                ((base * groups + head % groups) * (heads / groups) + head / groups) * words + col
            } else {
                ((base * sq + query) * heads + head) * words + col
            };
            let a = f32::from_bits(*partial.as_ptr().add((oi * 2) as usize));
            let b = f32::from_bits(*partial.as_ptr().add((oi * 2 + 1) as usize));
            lo += weight * a;
            hi += weight * b;
            split += 1;
        }
        let oi = ((batch * sq + query) * heads + head) * words + col;
        *out.as_mut_ptr().add(oi as usize) = if S::BF16 {
            cvt::<true>(lo, hi)
        } else {
            cvt::<false>(lo, hi)
        };
    }
}

pub type d128_raw = Flash<
    false,
    128,
    8,
    16,
    64,
    32,
    1,
    128,
    4,
    2,
    8,
    4,
    16,
    2,
    0,
    false,
    false,
    false,
    true,
    false,
    true,
>;
pub type d128_raw_bf16 = Flash<
    true,
    128,
    8,
    16,
    64,
    32,
    1,
    128,
    4,
    2,
    8,
    4,
    16,
    2,
    0,
    false,
    false,
    false,
    true,
    false,
    true,
>;

#[allow(non_camel_case_types)]
pub type d32_raw = Flash<
    false,
    32,
    2,
    4,
    128,
    64,
    2,
    128,
    8,
    4,
    4,
    16,
    8,
    4,
    0,
    false,
    false,
    false,
    true,
    false,
    true,
>;

#[allow(non_camel_case_types)]
pub type d32_raw_bf16 = Flash<
    true,
    32,
    2,
    4,
    128,
    64,
    2,
    128,
    8,
    4,
    4,
    16,
    8,
    4,
    0,
    false,
    false,
    false,
    true,
    false,
    true,
>;

#[allow(non_camel_case_types)]
pub type d32_wide_raw = Flash<
    false,
    32,
    2,
    4,
    64,
    128,
    1,
    128,
    16,
    8,
    2,
    16,
    4,
    2,
    0,
    false,
    false,
    false,
    true,
    false,
    true,
>;

#[allow(non_camel_case_types)]
pub type d32_wide_raw_bf16 = Flash<
    true,
    32,
    2,
    4,
    64,
    128,
    1,
    128,
    16,
    8,
    2,
    16,
    4,
    2,
    0,
    false,
    false,
    false,
    true,
    false,
    true,
>;

pub type d192_dense = Flash<false, 192, 12, 24, 128, 64, 1, 256, 8, 4, 12, 8, 24, 2>;
pub type d192_dense_bf16 = Flash<true, 192, 12, 24, 128, 64, 1, 256, 8, 4, 12, 8, 24, 2>;

type DropoutFlash<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const KEYS: u32,
    const NKT: usize,
    const PV_CHUNKS: usize,
> = Flash<
    BF16,
    HD,
    NFQ,
    NFO,
    64,
    KEYS,
    1,
    128,
    NKT,
    PV_CHUNKS,
    NFQ,
    NKT,
    NFO,
    2,
    0,
    false,
    false,
    false,
    false,
    false,
    false,
    true,
>;

#[allow(non_camel_case_types)]
pub type dropout_d32 = DropoutFlash<false, 32, 2, 4, 64, 8, 4>;
#[allow(non_camel_case_types)]
pub type dropout_d32_bf16 = DropoutFlash<true, 32, 2, 4, 64, 8, 4>;
#[allow(non_camel_case_types)]
pub type dropout_d64 = DropoutFlash<false, 64, 4, 8, 64, 8, 4>;
#[allow(non_camel_case_types)]
pub type dropout_d64_bf16 = DropoutFlash<true, 64, 4, 8, 64, 8, 4>;
#[allow(non_camel_case_types)]
pub type dropout_d96 = DropoutFlash<false, 96, 6, 12, 32, 4, 2>;
#[allow(non_camel_case_types)]
pub type dropout_d96_bf16 = DropoutFlash<true, 96, 6, 12, 32, 4, 2>;
#[allow(non_camel_case_types)]
pub type dropout_d128 = DropoutFlash<false, 128, 8, 16, 32, 4, 2>;
#[allow(non_camel_case_types)]
pub type dropout_d128_bf16 = DropoutFlash<true, 128, 8, 16, 32, 4, 2>;
#[allow(non_camel_case_types)]
pub type dropout_d192 = DropoutFlash<false, 192, 12, 24, 32, 4, 2>;
#[allow(non_camel_case_types)]
pub type dropout_d192_bf16 = DropoutFlash<true, 192, 12, 24, 32, 4, 2>;
#[allow(non_camel_case_types)]
pub type dropout_d256 = DropoutFlash<false, 256, 16, 32, 32, 4, 2>;
#[allow(non_camel_case_types)]
pub type dropout_d256_bf16 = DropoutFlash<true, 256, 16, 32, 32, 4, 2>;

#[kernel(
    dropout_d32,
    dropout_d32_bf16,
    dropout_d64,
    dropout_d64_bf16,
    dropout_d96,
    dropout_d96_bf16,
    dropout_d128,
    dropout_d128_bf16,
    dropout_d192,
    dropout_d192_bf16,
    dropout_d256,
    dropout_d256_bf16
)]
#[launch_bounds(128)]
pub fn flash_attn<S: FlashSpec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    o: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    alibi: &[f32],
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
    seed: u64,
    offset: u64,
    keep: f32,
) {
    let mut o = o;
    let mut lse = lse;
    unsafe {
        S::run(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            o.as_mut_ptr(),
            lse.as_mut_ptr(),
            alibi.as_ptr(),
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
            Dropout {
                seed,
                offset,
                threshold: (keep * 255.0) as u32,
                inv_keep: 1.0 / keep,
                ..Dropout::NONE
            },
        );
    }
}

// Saved CUDA RNG state is read on the launch stream, including graph replay.
#[kernel(
    dropout_d32,
    dropout_d32_bf16,
    dropout_d64,
    dropout_d64_bf16,
    dropout_d96,
    dropout_d96_bf16,
    dropout_d128,
    dropout_d128_bf16,
    dropout_d192,
    dropout_d192_bf16,
    dropout_d256,
    dropout_d256_bf16
)]
#[launch_bounds(128)]
pub fn flash_attn_rng<S: FlashSpec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    o: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    alibi: &[f32],
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
    rng_state: &[u64],
    keep: f32,
) {
    let mut o = o;
    let mut lse = lse;
    unsafe {
        S::run(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            o.as_mut_ptr(),
            lse.as_mut_ptr(),
            alibi.as_ptr(),
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
            Dropout {
                seed: *rng_state.as_ptr(),
                offset: *rng_state.as_ptr().add(1),
                threshold: (keep * 255.0) as u32,
                inv_keep: 1.0 / keep,
                ..Dropout::NONE
            },
        );
    }
}

// Saved CUDA RNG state is read on the launch stream, including graph replay.
#[kernel(
    dropout_d32,
    dropout_d32_bf16,
    dropout_d64,
    dropout_d64_bf16,
    dropout_d96,
    dropout_d96_bf16,
    dropout_d128,
    dropout_d128_bf16,
    dropout_d192,
    dropout_d192_bf16,
    dropout_d256,
    dropout_d256_bf16
)]
#[launch_bounds(128)]
pub fn flash_attn_probs<S: FlashSpec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    o: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    alibi: &[f32],
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
    rng_state: &[u64],
    keep: f32,
    probabilities: DisjointSlice<u32>,
) {
    let mut probabilities = probabilities;
    let mut o = o;
    let mut lse = lse;
    unsafe {
        S::run(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            o.as_mut_ptr(),
            lse.as_mut_ptr(),
            alibi.as_ptr(),
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
            Dropout {
                seed: *rng_state.as_ptr(),
                offset: *rng_state.as_ptr().add(1),
                threshold: (keep * 255.0) as u32,
                inv_keep: 1.0 / keep,
                probabilities: if probabilities.len() == 0 {
                    core::ptr::null_mut()
                } else {
                    probabilities.as_mut_ptr()
                },
                rounded_q: ((seqlen_q as u64 + 127) / 128) * 128,
                rounded_k: ((seqlen_k as u64 + 127) / 128) * 128,
                seqlen_q,
                seqlen_k,
            },
        );
    }
}

// Resolve each block independently; only one thread saves state for backward.
#[kernel(dropout_d64, dropout_d64_bf16)]
#[launch_bounds(128)]
pub fn flash_attn_rng_descriptor<S: FlashSpec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    o: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    alibi: &[f32],
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
    rng_state: DisjointSlice<u64>,
    keep: f32,
    descriptor_seed: u64,
    descriptor_offset: u64,
    intragraph: u64,
    captured: u32,
) {
    let mut rng_state = rng_state;
    let (resolved_seed, resolved_offset) = unsafe {
        if captured != 0 {
            (
                *(descriptor_seed as *const u64),
                (*(descriptor_offset as *const u64)).wrapping_add(intragraph),
            )
        } else {
            (descriptor_seed, descriptor_offset)
        }
    };
    if cuda_device::thread::blockIdx_x() == 0
        && cuda_device::thread::blockIdx_y() == 0
        && cuda_device::thread::blockIdx_z() == 0
        && cuda_device::thread::threadIdx_x() == 0
    {
        unsafe {
            *rng_state.as_mut_ptr() = resolved_seed;
            *rng_state.as_mut_ptr().add(1) = resolved_offset;
        }
    }
    let mut o = o;
    let mut lse = lse;
    unsafe {
        S::run(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            o.as_mut_ptr(),
            lse.as_mut_ptr(),
            alibi.as_ptr(),
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
            Dropout {
                seed: resolved_seed,
                offset: resolved_offset,
                threshold: (keep * 255.0) as u32,
                inv_keep: 1.0 / keep,
                ..Dropout::NONE
            },
        );
    }
}
