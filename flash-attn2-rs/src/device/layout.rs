// Bit-exact 4-D copy, used when strides don't fit the attention tile.
use cuda_device::{DisjointSlice, kernel, launch_bounds, thread};

#[kernel]
#[launch_bounds(256)]
pub fn copy_attention_layout(
    src: &[u16],
    dst: DisjointSlice<u16>,
    count: u64,
    seq: u64,
    heads: u64,
    dim: u64,
    src_batch: u64,
    src_seq: u64,
    src_head: u64,
    dst_batch: u64,
    dst_seq: u64,
    dst_head: u64,
) {
    let mut dst = dst;
    let mut i = thread::blockIdx_x() as u64 * 256 + thread::threadIdx_x() as u64;
    let step = thread::gridDim_x() as u64 * 256;
    while i < count {
        let d = i % dim;
        let h = (i / dim) % heads;
        let s = (i / (dim * heads)) % seq;
        let b = i / (dim * heads * seq);
        let from = b * src_batch + s * src_seq + h * src_head + d;
        let to = b * dst_batch + s * dst_seq + h * dst_head + d;
        unsafe {
            *dst.as_mut_ptr().add(to as usize) = *src.as_ptr().add(from as usize);
        }
        i += step;
    }
}
