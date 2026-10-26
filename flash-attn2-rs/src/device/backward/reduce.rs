use crate::device::inlined::{cvt, unpack};
use cuda_device::warp::shuffle_xor_f32;
use cuda_device::{DisjointSlice, kernel, launch_bounds, thread};

/// Output precision of the fixed-order partial-sum merges.
pub trait MergeSpec {
    const BF16: bool;
}
pub struct Merge<const BF16: bool>;
impl<const BF16: bool> MergeSpec for Merge<BF16> {
    const BF16: bool = BF16;
}
#[allow(non_camel_case_types)]
pub type f16 = Merge<false>;
#[allow(non_camel_case_types)]
pub type bf16 = Merge<true>;

pub trait ReduceSpec {
    const BF16: bool;
    const LANES: u32;
}
pub struct Reduce<const BF16: bool, const LANES: u32>;
impl<const BF16: bool, const LANES: u32> ReduceSpec for Reduce<BF16, LANES> {
    const BF16: bool = BF16;
    const LANES: u32 = LANES;
}
#[allow(non_camel_case_types)]
pub type f16_8 = Reduce<false, 8>;
#[allow(non_camel_case_types)]
pub type f16_16 = Reduce<false, 16>;
#[allow(non_camel_case_types)]
pub type f16_32 = Reduce<false, 32>;
#[allow(non_camel_case_types)]
pub type bf16_8 = Reduce<true, 8>;
#[allow(non_camel_case_types)]
pub type bf16_16 = Reduce<true, 16>;
#[allow(non_camel_case_types)]
pub type bf16_32 = Reduce<true, 32>;

#[kernel(f16_8, f16_16, f16_32, bf16_8, bf16_16, bf16_32)]
#[launch_bounds(256)]
pub fn bwd_preprocess<S: ReduceSpec>(
    dout: &[u32],
    out: &[u32],
    delta: DisjointSlice<f32>,
    seq: u32,
    rounded_seq: u32,
    heads: u32,
    dim_pairs: u32,
    do_batch: u32,
    do_seq: u32,
    do_head: u32,
    out_batch: u32,
    out_seq: u32,
    out_head: u32,
    keep: f32,
) {
    let mut delta = delta;
    let tid = thread::threadIdx_x();
    let row = thread::blockIdx_x() as u64 * (256 / S::LANES) as u64 + (tid / S::LANES) as u64;
    // rounded_seq is a multiple of 128, so every launched subgroup is valid.
    let s = row % rounded_seq as u64;
    let h = row / rounded_seq as u64 % heads as u64;
    let b = row / (rounded_seq as u64 * heads as u64);
    let lane = tid % S::LANES;
    let mut sum = 0.0f32;
    if s < seq as u64 {
        let a_base = b * do_batch as u64 + s * do_seq as u64 + h * do_head as u64;
        let b_base = b * out_batch as u64 + s * out_seq as u64 + h * out_head as u64;
        let mut d = lane;
        while d < dim_pairs {
            let (a, b) = unsafe {
                (
                    *dout.as_ptr().add((a_base + d as u64) as usize),
                    *out.as_ptr().add((b_base + d as u64) as usize),
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
            *delta.as_mut_ptr().add(row as usize) = sum * keep;
        }
    }
}

// Fixed-order FP32 sum over heads, rounded once for determinism.
#[kernel(f16, bf16)]
#[launch_bounds(256)]
pub fn backward_kv_merge<S: MergeSpec>(
    partial_k: &[f32],
    partial_v: &[f32],
    dk: DisjointSlice<u32>,
    dv: DisjointSlice<u32>,
    pairs: u64,
    heads_k: u32,
    groups: u32,
    dim_pairs: u32,
) {
    let mut dk = dk;
    let mut dv = dv;
    let mut i = thread::blockIdx_x() as u64 * 256 + thread::threadIdx_x() as u64;
    let step = thread::gridDim_x() as u64 * 256;
    while i < pairs {
        let col = i % dim_pairs as u64;
        let head = (i / dim_pairs as u64) % heads_k as u64;
        let row = i / (dim_pairs as u64 * heads_k as u64);
        let base = ((row * heads_k as u64 + head) * groups as u64 * dim_pairs as u64 + col) * 2;
        let mut k0 = 0.0f32;
        let mut k1 = 0.0f32;
        let mut v0 = 0.0f32;
        let mut v1 = 0.0f32;
        let mut g = 0;
        while g < groups {
            let from = base as usize + g as usize * dim_pairs as usize * 2;
            unsafe {
                k0 += *partial_k.as_ptr().add(from);
                k1 += *partial_k.as_ptr().add(from + 1);
                v0 += *partial_v.as_ptr().add(from);
                v1 += *partial_v.as_ptr().add(from + 1);
            }
            g += 1;
        }
        unsafe {
            *dk.as_mut_ptr().add(i as usize) = if S::BF16 {
                cvt::<true>(k0, k1)
            } else {
                cvt::<false>(k0, k1)
            };
            *dv.as_mut_ptr().add(i as usize) = if S::BF16 {
                cvt::<true>(v0, v1)
            } else {
                cvt::<false>(v0, v1)
            };
        }
        i += step;
    }
}

// Increasing split order, rounded once, so the result is deterministic.
#[kernel(f16, bf16)]
#[launch_bounds(256)]
pub fn backward_dq_merge<S: MergeSpec>(
    partial: &[f32],
    dq: DisjointSlice<u32>,
    pairs: u64,
    splits: u32,
    scale: f32,
) {
    let mut dq = dq;
    let mut i = thread::blockIdx_x() as u64 * 256 + thread::threadIdx_x() as u64;
    let step = thread::gridDim_x() as u64 * 256;
    while i < pairs {
        let mut x = 0.0f32;
        let mut y = 0.0f32;
        let mut split = 0;
        while split < splits {
            let from = ((split as u64 * pairs + i) * 2) as usize;
            unsafe {
                x += *partial.as_ptr().add(from);
                y += *partial.as_ptr().add(from + 1);
            }
            split += 1;
        }
        unsafe {
            *dq.as_mut_ptr().add(i as usize) = if S::BF16 {
                cvt::<true>(x * scale, y * scale)
            } else {
                cvt::<false>(x * scale, y * scale)
            };
        }
        i += step;
    }
}
