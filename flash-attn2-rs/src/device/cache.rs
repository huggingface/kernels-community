// Writable caches are never replaced with an aligned copy.
use crate::device::inlined::{cvt, unpack};
use cuda_device::{DisjointSlice, kernel, launch_bounds, thread};

pub trait Spec {
    const BF16: bool;
    const APPEND: bool;
    type Query: Spec;
}
pub struct Copy<const BF16: bool, const APPEND: bool>;
impl<const BF16: bool, const APPEND: bool> Spec for Copy<BF16, APPEND> {
    const BF16: bool = BF16;
    const APPEND: bool = APPEND;
    type Query = Copy<BF16, false>;
}
#[allow(non_camel_case_types)]
pub type q_f16 = Copy<false, false>;
#[allow(non_camel_case_types)]
pub type q_bf16 = Copy<true, false>;
#[allow(non_camel_case_types)]
pub type kv_f16 = Copy<false, true>;
#[allow(non_camel_case_types)]
pub type kv_bf16 = Copy<true, true>;

#[inline(always)]
unsafe fn rotate<const BF16: bool>(
    src: *const u16,
    offset: usize,
    dim: u32,
    rotary_dim: u32,
    interleaved: bool,
    cos: *const u16,
    sin: *const u16,
    position: u32,
) -> u16 {
    unsafe {
        let value = *src.add(offset);
        if dim >= rotary_dim {
            return value;
        }
        let half = rotary_dim / 2;
        let first = if interleaved {
            dim % 2 == 0
        } else {
            dim < half
        };
        let other = if interleaved {
            dim ^ 1
        } else if first {
            dim + half
        } else {
            dim - half
        };
        let frequency = if interleaved { dim / 2 } else { dim % half };
        let angle = position as usize * half as usize + frequency as usize;
        let a = unpack::<BF16>(value as u32);
        let b = unpack::<BF16>(*src.add(offset - dim as usize + other as usize) as u32);
        let c = unpack::<BF16>(*cos.add(angle) as u32);
        let s = unpack::<BF16>(*sin.add(angle) as u32);
        cvt::<BF16>(a * c + b * if first { -s } else { s }, 0.0) as u16
    }
}

#[kernel(q_f16, q_bf16, kv_f16, kv_bf16)]
#[launch_bounds(256)]
pub fn cache_copy<S: Spec>(
    src: &[u16],
    values: &[u16],
    dst: DisjointSlice<u16>,
    dst_values: DisjointSlice<u16>,
    cos: &[u16],
    sin: &[u16],
    lengths: &[i32],
    indices: &[i32],
    table: &[i32],
    count: u64,
    seq: u64,
    heads: u64,
    dim: u64,
    src_bs: u64,
    src_ss: u64,
    src_hs: u64,
    value_bs: u64,
    value_ss: u64,
    value_hs: u64,
    dst_bs: u64,
    dst_ss: u64,
    dst_hs: u64,
    dst_value_bs: u64,
    dst_value_ss: u64,
    dst_value_hs: u64,
    capacity: u32,
    cache_batches: u32,
    page_size: u32,
    table_stride: u32,
    table_columns: u32,
    rotary_dim: u32,
    rotary_rows: u32,
    interleaved: u32,
    advance_q: u32,
) {
    copy_core::<S>(
        src,
        values,
        dst,
        dst_values,
        cos,
        sin,
        lengths,
        indices,
        table,
        count,
        seq,
        heads,
        dim,
        src_bs,
        src_ss,
        src_hs,
        value_bs,
        value_ss,
        value_hs,
        dst_bs,
        dst_ss,
        dst_hs,
        dst_value_bs,
        dst_value_ss,
        dst_value_hs,
        capacity,
        cache_batches,
        page_size,
        table_stride,
        table_columns,
        rotary_dim,
        rotary_rows,
        interleaved,
        advance_q,
        thread::blockIdx_x() as u64 * 256 + thread::threadIdx_x() as u64,
        thread::gridDim_x() as u64 * 256,
    );
}

#[inline(always)]
fn copy_core<S: Spec>(
    src: &[u16],
    values: &[u16],
    dst: DisjointSlice<u16>,
    dst_values: DisjointSlice<u16>,
    cos: &[u16],
    sin: &[u16],
    lengths: &[i32],
    indices: &[i32],
    table: &[i32],
    count: u64,
    seq: u64,
    heads: u64,
    dim: u64,
    src_bs: u64,
    src_ss: u64,
    src_hs: u64,
    value_bs: u64,
    value_ss: u64,
    value_hs: u64,
    dst_bs: u64,
    dst_ss: u64,
    dst_hs: u64,
    dst_value_bs: u64,
    dst_value_ss: u64,
    dst_value_hs: u64,
    capacity: u32,
    cache_batches: u32,
    page_size: u32,
    table_stride: u32,
    table_columns: u32,
    rotary_dim: u32,
    rotary_rows: u32,
    interleaved: u32,
    advance_q: u32,
    index: u64,
    step: u64,
) {
    let mut dst = dst;
    let mut dst_values = dst_values;
    let mut i = index;
    while i < count {
        let d = i % dim;
        let h = (i / dim) % heads;
        let row = (i / (dim * heads)) % seq;
        let b = i / (dim * heads * seq);
        unsafe {
            let start = *lengths.as_ptr().add(b as usize);
            let position = (start as u64) + if S::APPEND || advance_q != 0 { row } else { 0 };
            if start < 0
                || (S::APPEND && position >= capacity as u64)
                || (rotary_dim > 0 && position >= rotary_rows as u64)
            {
                i += step;
                continue;
            }
            let mut physical_batch = b;
            let mut physical_row = row;
            if S::APPEND {
                physical_row = position;
                if page_size > 0 {
                    let column = position / page_size as u64;
                    if column >= table_columns as u64 {
                        i += step;
                        continue;
                    }
                    let page = *table
                        .as_ptr()
                        .add((b * table_stride as u64 + column) as usize);
                    if page < 0 || page as u32 >= cache_batches {
                        i += step;
                        continue;
                    }
                    physical_batch = page as u64;
                    physical_row = position % page_size as u64;
                } else {
                    let index = if indices.is_empty() {
                        b as i32
                    } else {
                        *indices.as_ptr().add(b as usize)
                    };
                    if index < 0 || index as u32 >= cache_batches {
                        i += step;
                        continue;
                    }
                    physical_batch = index as u64;
                }
            }
            let from = b * src_bs + row * src_ss + h * src_hs + d;
            let to = physical_batch * dst_bs + physical_row * dst_ss + h * dst_hs + d;
            let value = if S::BF16 {
                rotate::<true>(
                    src.as_ptr(),
                    from as usize,
                    d as u32,
                    rotary_dim,
                    interleaved != 0,
                    cos.as_ptr(),
                    sin.as_ptr(),
                    position as u32,
                )
            } else {
                rotate::<false>(
                    src.as_ptr(),
                    from as usize,
                    d as u32,
                    rotary_dim,
                    interleaved != 0,
                    cos.as_ptr(),
                    sin.as_ptr(),
                    position as u32,
                )
            };
            *dst.as_mut_ptr().add(to as usize) = value;
            if S::APPEND {
                let from = b * value_bs + row * value_ss + h * value_hs + d;
                let to = physical_batch * dst_value_bs
                    + physical_row * dst_value_ss
                    + h * dst_value_hs
                    + d;
                *dst_values.as_mut_ptr().add(to as usize) = *values.as_ptr().add(from as usize);
            }
        }
        i += step;
    }
}

// Host checks disjointness, so Q rotation and KV writes run concurrently.
#[kernel(kv_f16, kv_bf16)]
#[launch_bounds(256)]
pub fn cache_update<S: Spec>(
    src: &[u16],
    values: &[u16],
    dst: DisjointSlice<u16>,
    dst_values: DisjointSlice<u16>,
    cos: &[u16],
    sin: &[u16],
    lengths: &[i32],
    indices: &[i32],
    table: &[i32],
    count: u64,
    seq: u64,
    heads: u64,
    dim: u64,
    src_bs: u64,
    src_ss: u64,
    src_hs: u64,
    value_bs: u64,
    value_ss: u64,
    value_hs: u64,
    dst_bs: u64,
    dst_ss: u64,
    dst_hs: u64,
    dst_value_bs: u64,
    dst_value_ss: u64,
    dst_value_hs: u64,
    capacity: u32,
    cache_batches: u32,
    page_size: u32,
    table_stride: u32,
    table_columns: u32,
    rotary_dim: u32,
    rotary_rows: u32,
    interleaved: u32,
    advance_q: u32,
    q: &[u16],
    q_out: DisjointSlice<u16>,
    q_count: u64,
    q_seq: u64,
    q_heads: u64,
    q_bs: u64,
    q_ss: u64,
    q_hs: u64,
    q_out_bs: u64,
    q_out_ss: u64,
    q_out_hs: u64,
) {
    let kv_blocks = ((count + 255) / 256).min(65535);
    let block = thread::blockIdx_x() as u64;
    let lane = thread::threadIdx_x() as u64;
    if block < kv_blocks {
        copy_core::<S>(
            src,
            values,
            dst,
            dst_values,
            cos,
            sin,
            lengths,
            indices,
            table,
            count,
            seq,
            heads,
            dim,
            src_bs,
            src_ss,
            src_hs,
            value_bs,
            value_ss,
            value_hs,
            dst_bs,
            dst_ss,
            dst_hs,
            dst_value_bs,
            dst_value_ss,
            dst_value_hs,
            capacity,
            cache_batches,
            page_size,
            table_stride,
            table_columns,
            rotary_dim,
            rotary_rows,
            interleaved,
            advance_q,
            block * 256 + lane,
            kv_blocks * 256,
        );
    } else {
        copy_core::<S::Query>(
            q,
            values,
            q_out,
            dst_values,
            cos,
            sin,
            lengths,
            indices,
            table,
            q_count,
            q_seq,
            q_heads,
            dim,
            q_bs,
            q_ss,
            q_hs,
            value_bs,
            value_ss,
            value_hs,
            q_out_bs,
            q_out_ss,
            q_out_hs,
            dst_value_bs,
            dst_value_ss,
            dst_value_hs,
            capacity,
            cache_batches,
            page_size,
            table_stride,
            table_columns,
            rotary_dim,
            rotary_rows,
            interleaved,
            advance_q,
            (block - kv_blocks) * 256 + lane,
            (thread::gridDim_x() as u64 - kv_blocks) * 256,
        );
    }
}
