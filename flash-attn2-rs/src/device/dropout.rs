// FlashAttention v2's Philox4x32-7 stream, independent of CTA tiling.
use crate::device::config::LaunchTile;

#[derive(Clone, Copy)]
pub struct Dropout {
    pub seed: u64,
    pub offset: u64,
    pub threshold: u32,
    pub inv_keep: f32,
    pub probabilities: *mut u32,
    pub rounded_q: u64,
    pub rounded_k: u64,
    pub seqlen_q: u32,
    pub seqlen_k: u32,
}

impl Dropout {
    pub const NONE: Self = Self {
        seed: 0,
        offset: 0,
        threshold: 255,
        inv_keep: 1.0,
        probabilities: core::ptr::null_mut(),
        rounded_q: 0,
        rounded_k: 0,
        seqlen_q: 0,
        seqlen_k: 0,
    };

    #[inline(always)]
    pub(crate) fn random(self, row: u32, col: u32, lane_offset: u64) -> [u32; 4] {
        let mut c = [lane_offset as u32, (lane_offset >> 32) as u32, row, col];
        let mut key = [self.seed as u32, (self.seed >> 32) as u32];
        let mut round = 0;
        while round < 7 {
            cuda_device::thread::__unroll_config::<0>();
            let a = (c[0] as u64) * 0xD2511F53u64;
            let b = (c[2] as u64) * 0xCD9E8D57u64;
            c = [
                (b >> 32) as u32 ^ c[1] ^ key[0],
                b as u32,
                (a >> 32) as u32 ^ c[3] ^ key[1],
                a as u32,
            ];
            key[0] = key[0].wrapping_add(0x9E3779B9);
            key[1] = key[1].wrapping_add(0xBB67AE85);
            round += 1;
        }
        c
    }

    #[inline(always)]
    pub fn apply<const RFRAGS: usize, const NKT: usize>(
        self,
        p: &mut [[u32; 2]],
        tile: &LaunchTile,
        kv_base: u32,
    ) {
        let offset = self
            .offset
            .wrapping_add(tile.bh as u64 * 32 + tile.lane_id as u64);
        let mut rf = 0;
        while rf < RFRAGS {
            cuda_device::thread::__unroll_config::<0>();
            let row = (tile.q_row_base + tile.warp_row) / 16 + rf as u32;
            let mut nt = 0;
            while nt < NKT {
                cuda_device::thread::__unroll_config::<0>();
                let random = self.random(row, kv_base / 32 + nt as u32 / 4, offset);
                let mut j = 0;
                while j < 4 {
                    cuda_device::thread::__unroll_config::<0>();
                    let r = random[j];
                    let mask0 = (if (r & 255) <= self.threshold {
                        0xffff
                    } else {
                        0
                    }) | (if ((r >> 8) & 255) <= self.threshold {
                        0xffff0000
                    } else {
                        0
                    });
                    let mask1 = (if ((r >> 16) & 255) <= self.threshold {
                        0xffff
                    } else {
                        0
                    }) | (if (r >> 24) <= self.threshold {
                        0xffff0000
                    } else {
                        0
                    });
                    // As in v2, the keep decision is encoded in the sign bit.
                    if !self.probabilities.is_null() {
                        let q = tile.q_row_base as u64
                            + tile.warp_row as u64
                            + tile.group as u64
                            + rf as u64 * 16;
                        let k = kv_base as u64 + (nt + j) as u64 * 8 + tile.pair as u64 * 2;
                        let index =
                            ((tile.bh as u64 * self.rounded_q + q) * self.rounded_k + k) / 2;
                        unsafe {
                            if q < self.seqlen_q as u64 && k < self.seqlen_k as u64 {
                                *self.probabilities.add(index as usize) =
                                    p[rf * NKT + nt + j][0] | ((!mask0) & 0x80008000);
                            }
                            if q + 8 < self.seqlen_q as u64 && k < self.seqlen_k as u64 {
                                *self
                                    .probabilities
                                    .add((index + 4 * self.rounded_k) as usize) =
                                    p[rf * NKT + nt + j][1] | ((!mask1) & 0x80008000);
                            }
                        }
                    }
                    p[rf * NKT + nt + j][0] &= mask0;
                    p[rf * NKT + nt + j][1] &= mask1;
                    j += 1;
                }
                nt += 4;
            }
            rf += 1;
        }
    }
}
