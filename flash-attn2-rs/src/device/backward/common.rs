use super::*;

#[inline(always)]
pub fn unpack<const BF16: bool>(bits: u32) -> f32 {
    if BF16 {
        f32::from_bits((bits & 0xffff) << 16)
    } else {
        let value: f32;
        unsafe {
            ptx_asm!("{ .reg .b16 h; cvt.u16.u32 h, %1; cvt.f32.f16 %0, h; }",
                out("=f") value, in("r") bits, options(register_only));
        }
        value
    }
}

#[inline(always)]
pub(crate) unsafe fn copy_rows<const HD: usize, const ROWS: usize>(
    src: *const u32,
    dst: *mut u32,
    base: usize,
    row_start: u32,
    stride: u32,
    seq: u32,
    dim_pairs: u32,
    tid: u32,
) {
    unsafe {
        let mut i = tid as usize * 4;
        while i < ROWS * (HD / 2) {
            let row = i / (HD / 2);
            let col = i % (HD / 2);
            let global_row = row_start as usize + row;
            let valid = global_row < seq as usize && col < dim_pairs as usize;
            let index = if valid {
                base + global_row * stride as usize + col
            } else {
                0
            };
            cp_async_ca_zfill_16(
                dst.add(row * (HD / 2 + 4) + col),
                src.add(index) as *const u8,
                if valid { 16 } else { 0 },
            );
            i += 128 * 4;
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn query_fragments<const NFQ: usize>(
    shared: *const u32,
    tile: &LaunchTile,
    layout: SmemLayout,
) -> [[u32; 4]; NFQ] {
    let mut fragments = [[0; 4]; NFQ];
    let row = tile.warp_row + (tile.lane_id & 7) + ((tile.lane_id >> 3) & 1) * 8;
    let col = (tile.lane_id >> 4) * 4;
    let address = shared_address(shared);
    let mut kc = 0;
    while kc < NFQ {
        thread::__unroll_config::<0>();
        // Caller supplies a synchronized tile matching layout and NFQ.
        fragments[kc] = unsafe {
            ldmatrix_x4_shared(
                address + ((row as usize * layout.stride + kc * 8 + col as usize) * 4) as u32,
            )
        };
        kc += 1;
    }
    fragments
}

// XOR swizzle of 16-byte groups; stores and ldmatrix share it.
#[inline(always)]
pub(super) unsafe fn copy_rows_swizzled<const ROWS: usize, const KSTRIDE: usize>(
    src: *const u32,
    dst: *mut u32,
    base: usize,
    row_start: u32,
    stride: u32,
    seq: u32,
    dim_pairs: u32,
    tid: u32,
) {
    unsafe {
        let mut i = tid as usize * 4;
        while i < ROWS * KSTRIDE {
            let row = i / KSTRIDE;
            let col = i % KSTRIDE;
            let global_row = row_start as usize + row;
            let valid = global_row < seq as usize && col < dim_pairs as usize;
            let index = if valid {
                base + global_row * stride as usize + col
            } else {
                0
            };
            cp_async_ca_zfill_16(
                dst.add(row * KSTRIDE + (col ^ ((row & 7) * 4))),
                src.add(index).cast(),
                if valid { 16 } else { 0 },
            );
            i += 128 * 4;
        }
    }
}

#[inline(always)]
pub(super) unsafe fn query_fragments_swizzled<const NFQ: usize, const KSTRIDE: usize>(
    shared: *const u32,
    tile: &LaunchTile,
) -> [[u32; 4]; NFQ] {
    let mut fragments = [[0; 4]; NFQ];
    let row = tile.warp_row + (tile.lane_id & 7) + ((tile.lane_id >> 3) & 1) * 8;
    let col = (tile.lane_id >> 4) * 4;
    let address = shared_address(shared);
    let mut kc = 0;
    while kc < NFQ {
        thread::__unroll_config::<0>();
        let index = row as usize * KSTRIDE + ((kc * 8 + col as usize) ^ ((row as usize & 7) * 4));
        // Caller provides a synchronized tile written with this XOR mapping.
        fragments[kc] = unsafe { ldmatrix_x4_shared(address + (index * 4) as u32) };
        kc += 1;
    }
    fragments
}

// Combined to avoid a second live score tile during dout*V.
#[inline(always)]
pub(crate) fn probability_gradient(
    raw: f32,
    row: u32,
    col: u32,
    lse: f32,
    cfg: &FwdConfig,
    slope: f32,
    gradient: bool,
) -> f32 {
    let center = row as i64 + cfg.seqlen_k as i64 - cfg.seqlen_q as i64;
    if row >= cfg.seqlen_q
        || col >= cfg.seqlen_k
        || (cfg.window_right >= 0 && col as i64 > center + cfg.window_right as i64)
        || (cfg.window_left >= 0 && (col as i64) < center - cfg.window_left as i64)
    {
        return 0.0;
    }
    let (mut score, derivative) = if cfg.tanh_scale > 0.0 {
        let t = tanh_approx(raw * cfg.tanh_scale);
        (t * cfg.scale, if gradient { 1.0 - t * t } else { 1.0 })
    } else {
        (raw * cfg.scale, 1.0)
    };
    if cfg.has_alibi != 0 {
        score += if cfg.alibi_causal != 0 {
            slope * col as f32
        } else {
            -slope * (center - col as i64).abs() as f32
        };
    }
    fast_exp2((score - lse) * LOG2_E) * derivative
}

// Forward Philox stream read in transposed score coordinates.
#[inline(always)]
pub(super) fn random_word(random: [u32; 4], index: u32) -> u32 {
    if index == 0 {
        random[0]
    } else if index == 1 {
        random[1]
    } else if index == 2 {
        random[2]
    } else {
        random[3]
    }
}

#[inline(always)]
pub(crate) fn transposed_keep(dropout: Dropout, bh: u32, query: u32, key: u32) -> [f32; 4] {
    if dropout.threshold == 255 {
        return [1.0; 4];
    }
    let lane = (query % 8) * 4 + (key % 8) / 2;
    let offset = dropout.offset.wrapping_add(bh as u64 * 32 + lane as u64);
    let a = dropout.random(query / 16, key / 32, offset);
    let b = dropout.random(query / 16, key / 32, offset.wrapping_add(4));
    let word = (key % 32) / 8;
    let shift = ((query % 16) / 8 * 2 + key % 2) * 8;
    [
        (((random_word(a, word) >> shift) & 255) <= dropout.threshold) as u32 as f32,
        (((random_word(b, word) >> shift) & 255) <= dropout.threshold) as u32 as f32,
        (((random_word(a, word + 1) >> shift) & 255) <= dropout.threshold) as u32 as f32,
        (((random_word(b, word + 1) >> shift) & 255) <= dropout.threshold) as u32 as f32,
    ]
}

// All lanes must participate; xor4 lanes share Philox counters.
#[inline(always)]
pub(crate) fn transposed_paired_words(dropout: Dropout, bh: u32, query: u32, key: u32) -> [u32; 4] {
    if dropout.threshold == 255 {
        return [0; 4];
    }
    let parity = key & 1;
    let lane = (query % 8) * 4 + (key % 8) / 2;
    let offset = dropout
        .offset
        .wrapping_add(bh as u64 * 32 + lane as u64 + parity as u64 * 4);
    let random = dropout.random(query / 16, key / 32, offset);
    let word = (key % 32) / 8;
    let own0 = random_word(random, word);
    let own1 = random_word(random, word + 1);
    let other0: u32;
    let other1: u32;
    unsafe {
        ptx_asm!("shfl.sync.bfly.b32 %0, %1, 4, 31, -1;", out("=r") other0, in("r") own0);
        ptx_asm!("shfl.sync.bfly.b32 %0, %1, 4, 31, -1;", out("=r") other1, in("r") own1);
    }
    let a0 = if parity == 0 { own0 } else { other0 };
    let b0 = if parity == 0 { other0 } else { own0 };
    let a1 = if parity == 0 { own1 } else { other1 };
    let b1 = if parity == 0 { other1 } else { own1 };
    [a0, b0, a1, b1]
}

#[inline(always)]
pub(crate) fn transposed_word_mask(
    words: [u32; 4],
    threshold: u32,
    query: u32,
    key: u32,
) -> [f32; 4] {
    let shift = ((query % 16) / 8 * 2 + (key & 1)) * 8;
    [
        ((words[0] >> shift & 255) <= threshold) as u32 as f32,
        ((words[1] >> shift & 255) <= threshold) as u32 as f32,
        ((words[2] >> shift & 255) <= threshold) as u32 as f32,
        ((words[3] >> shift & 255) <= threshold) as u32 as f32,
    ]
}
