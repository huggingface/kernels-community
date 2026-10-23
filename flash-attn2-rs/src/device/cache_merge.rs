use crate::device::forward::LOG2_E;
use crate::device::inlined::{cvt, fast_exp2, fmax, lg2_approx};
use cuda_device::{
    DisjointSlice, kernel, launch_bounds, shared::DynamicSharedArray, thread, warp::shuffle_xor_f32,
};

pub trait Spec {
    const BF16: bool;
    const GROUPED: bool;
    const PACKED: bool;
}
pub struct Merge<const BF16: bool, const GROUPED: bool = false, const PACKED: bool = false>;
impl<const BF16: bool, const GROUPED: bool, const PACKED: bool> Spec
    for Merge<BF16, GROUPED, PACKED>
{
    const BF16: bool = BF16;
    const GROUPED: bool = GROUPED;
    const PACKED: bool = PACKED;
}
#[allow(non_camel_case_types)]
pub type f16 = Merge<false>;
#[allow(non_camel_case_types)]
pub type bf16 = Merge<true>;

#[allow(non_camel_case_types)]
pub type grouped_f16 = Merge<false, true>;
#[allow(non_camel_case_types)]
pub type grouped_bf16 = Merge<true, true>;

#[allow(non_camel_case_types)]
pub type packed_f16 = Merge<false, false, true>;
#[allow(non_camel_case_types)]
pub type packed_bf16 = Merge<true, false, true>;
#[allow(non_camel_case_types)]
pub type grouped_packed_f16 = Merge<false, true, true>;
#[allow(non_camel_case_types)]
pub type grouped_packed_bf16 = Merge<true, true, true>;

#[kernel(
    f16,
    bf16,
    grouped_f16,
    grouped_bf16,
    packed_f16,
    packed_bf16,
    grouped_packed_f16,
    grouped_packed_bf16
)]
#[launch_bounds(128)]
pub fn cache_merge<S: Spec>(
    partial: &[f32],
    stats: &[f32],
    out: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    sq: u32,
    heads: u32,
    dim: u32,
    splits: u32,
) {
    merge_rows::<S, false>(
        partial,
        stats,
        out,
        lse,
        sq,
        heads,
        dim,
        splits,
        core::ptr::null(),
        0,
    );
}

#[kernel(f16, bf16)]
#[launch_bounds(128)]
pub fn varlen_merge<S: Spec>(
    partial: &[f32],
    stats: &[f32],
    out: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    sq: u32,
    heads: u32,
    dim: u32,
    splits: u32,
    cuq: &[i32],
    total_q: u32,
) {
    merge_rows::<S, true>(
        partial,
        stats,
        out,
        lse,
        sq,
        heads,
        dim,
        splits,
        cuq.as_ptr(),
        total_q,
    );
}

#[inline(always)]
fn merge_rows<S: Spec, const VARLEN: bool>(
    partial: &[f32],
    stats: &[f32],
    out: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    sq: u32,
    heads: u32,
    dim: u32,
    splits: u32,
    cuq: *const i32,
    total_q: u32,
) {
    let mut out = out;
    let mut lse = lse;
    let row = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    let batch = row / (heads * sq);
    let head = (row / sq) % heads;
    let query = row % sq;
    let start = if VARLEN {
        let start = unsafe { *cuq.add(batch as usize) };
        let end = unsafe { *cuq.add(batch as usize + 1) };
        // Whole-block exit before reading potentially unwritten partial rows.
        if start < 0
            || end < start
            || end as u32 > total_q
            || (end - start) as u32 > sq
            || query >= (end - start) as u32
        {
            return;
        }
        start as u32
    } else {
        0
    };
    let lane = tid % 32;
    let warp = tid / 32;
    unsafe {
        // 128 weights followed by four warp maxima and four warp sums.
        let shared = DynamicSharedArray::<f32, 16>::get() as usize as *mut f32;
        let base = ((batch as usize * splits as usize) * heads as usize + head as usize)
            * sq as usize
            + query as usize;
        let split_stride = heads as usize * sq as usize;
        let value = if tid < splits {
            *stats.as_ptr().add(base + tid as usize * split_stride)
        } else {
            f32::NEG_INFINITY
        };
        let mut maximum = value;
        maximum = fmax(maximum, shuffle_xor_f32(maximum, 16));
        maximum = fmax(maximum, shuffle_xor_f32(maximum, 8));
        maximum = fmax(maximum, shuffle_xor_f32(maximum, 4));
        maximum = fmax(maximum, shuffle_xor_f32(maximum, 2));
        maximum = fmax(maximum, shuffle_xor_f32(maximum, 1));
        if lane == 0 {
            *shared.add(128 + warp as usize) = maximum;
        }
        thread::sync_threads();
        maximum = fmax(
            fmax(*shared.add(128), *shared.add(129)),
            fmax(*shared.add(130), *shared.add(131)),
        );
        let weight = if maximum == f32::NEG_INFINITY {
            0.0
        } else {
            fast_exp2((value - maximum) * LOG2_E)
        };
        let mut sum = weight;
        sum += shuffle_xor_f32(sum, 16);
        sum += shuffle_xor_f32(sum, 8);
        sum += shuffle_xor_f32(sum, 4);
        sum += shuffle_xor_f32(sum, 2);
        sum += shuffle_xor_f32(sum, 1);
        if lane == 0 {
            *shared.add(132 + warp as usize) = sum;
        }
        thread::sync_threads();
        let denom = (*shared.add(132) + *shared.add(133)) + (*shared.add(134) + *shared.add(135));
        *shared.add(tid as usize) = if denom > 0.0 { weight / denom } else { 0.0 };
        if tid == 0 {
            // Packed decode has one query per sequence and head-major LSE.
            let lse_row = if VARLEN {
                head as usize * total_q as usize + start as usize + query as usize
            } else if S::PACKED {
                if S::GROUPED {
                    (head * sq + query) as usize * (thread::gridDim_x() / (heads * sq)) as usize
                        + batch as usize
                } else {
                    head as usize * (thread::gridDim_x() / heads) as usize + batch as usize
                }
            } else {
                row as usize
            };
            *lse.as_mut_ptr().add(lse_row) = if denom > 0.0 {
                maximum + lg2_approx(denom) / LOG2_E
            } else {
                f32::INFINITY
            };
        }
        thread::sync_threads();
        if tid < dim / 2 {
            let mut lo = 0.0;
            let mut hi = 0.0;
            let mut split = 0;
            while split < splits {
                let w = *shared.add(split as usize);
                // Empty partial storage may be unwritten; never read it.
                if w > 0.0 {
                    let p =
                        (base + split as usize * split_stride) * dim as usize + tid as usize * 2;
                    lo += *partial.as_ptr().add(p) * w;
                    hi += *partial.as_ptr().add(p + 1) * w;
                }
                split += 1;
            }
            let target = if VARLEN {
                ((start as usize + query as usize) * heads as usize + head as usize)
                    * (dim as usize / 2)
                    + tid as usize
            } else if S::GROUPED {
                // Restore original head order while merging GQA groups.
                row as usize * (dim as usize / 2) + tid as usize
            } else {
                ((batch as usize * sq as usize + query as usize) * heads as usize + head as usize)
                    * (dim as usize / 2)
                    + tid as usize
            };
            *out.as_mut_ptr().add(target) = if S::BF16 {
                cvt::<true>(lo, hi)
            } else {
                cvt::<false>(lo, hi)
            };
        }
    }
}
