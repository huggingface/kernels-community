// D192 dense MHA; host guards full tiles and plain attention.
use crate::device::forward::{LOG2_E, online_softmax_pack_raw};
use crate::device::inlined::{
    cvt, ldmatrix_x4_shared, ldmatrix_x4_trans_shared, lg2_approx, mma, shared_address,
};
use cuda_device::async_copy::cp_async_ca_zfill_16;
use cuda_device::shared::DynamicSharedArray;
use cuda_device::warp::shuffle_xor_f32;
use cuda_device::{ptx_asm, thread};

#[inline(always)]
fn index(row: u32, col: u32) -> u32 {
    let out: u32;
    unsafe {
        ptx_asm!("{ .reg .u32 mask, col2; and.b32 mask, %1, 7; shl.b32 mask, mask, 2; xor.b32 col2, %2, mask; mad.lo.u32 %0, %1, 96, col2; }", out("=r") out, in("r") row, in("r") col, options(register_only));
    }
    out
}

#[inline(always)]
unsafe fn stage<const ROWS: u32>(dst: *mut u32, src: *const u32, stride: u32, tid: u32) {
    unsafe {
        let mut i = tid * 4;
        while i < ROWS * 96 {
            let row = i / 96;
            let col = i % 96;
            cp_async_ca_zfill_16(
                dst.add(index(row, col) as usize),
                src.add((row * stride + col) as usize) as *const u8,
                16,
            );
            i += 1024;
        }
        ptx_asm!("cp.async.commit_group;", clobber("memory"));
    }
}

#[inline(always)]
pub unsafe fn run<const BF16: bool>(
    q: *const u32,
    k: *const u32,
    v: *const u32,
    o: *mut u32,
    lse: *mut f32,
    scale: f32,
    sq: u32,
    sk: u32,
    h: u32,
) {
    unsafe {
        let tid = thread::threadIdx_x();
        let lane = tid & 31;
        let warp_row = (tid / 32) * 16;
        let batch = thread::blockIdx_z();
        let head = thread::blockIdx_y();
        let qbase = thread::blockIdx_x() * 128;
        let stride = h * 96;
        let qb =
            (batch as usize * sq as usize + qbase as usize) * stride as usize + head as usize * 96;
        let kb = batch as usize * sk as usize * stride as usize + head as usize * 96;
        // Rebuild the generic address; helpers take generic pointers.
        let qs = DynamicSharedArray::<u32, 16>::get() as usize as *mut u32;
        let ks = qs.add(128 * 96);
        let vs = ks.add(64 * 96);
        let qa = shared_address(qs);
        let ka = shared_address(ks);
        let va = shared_address(vs);
        // Q and the first K occupy disjoint buffers; one wait covers both.
        stage::<128>(qs, q.add(qb), stride, tid);
        stage::<64>(ks, k.add(kb), stride, tid);
        ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
        thread::sync_threads();
        stage::<64>(vs, v.add(kb), stride, tid);
        let mut out = [[0.0f32; 4]; 24];
        let mut max = [f32::NEG_INFINITY; 2];
        let mut sum = [0.0f32; 2];
        let scale_log2 = scale * LOG2_E;
        let mut j = 0;
        while j < sk {
            let mut s = [[0.0f32; 4]; 8];
            let mut kc = 0;
            while kc < 12 {
                thread::__unroll_config::<2>();
                let a = ldmatrix_x4_shared(
                    qa + index(
                        warp_row + (lane & 7) + (((lane >> 3) & 1) * 8),
                        kc * 8 + (lane >> 4) * 4,
                    ) * 4,
                );
                let mut nt = 0;
                while nt < 8 {
                    thread::__unroll_config::<0>();
                    let b = ldmatrix_x4_shared(
                        ka + index(
                            nt * 8 + (lane & 7) + (lane >> 4) * 8,
                            kc * 8 + ((lane & 15) >> 3) * 4,
                        ) * 4,
                    );
                    s[nt as usize] = mma::<BF16>(s[nt as usize], a, [b[0], b[1]]);
                    s[nt as usize + 1] = mma::<BF16>(s[nt as usize + 1], a, [b[2], b[3]]);
                    nt += 2;
                }
                kc += 1;
            }
            // V readiness also releases K after all QK reads.
            ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
            thread::sync_threads();
            let next = j + 64 < sk;
            if next {
                stage::<64>(
                    ks,
                    k.add(kb + (j as usize + 64) * stride as usize),
                    stride,
                    tid,
                );
            }
            let mut p = [[0u32; 2]; 8];
            online_softmax_pack_raw::<BF16, 1, 8, 24>(
                &s, &mut p, &mut out, &mut max, &mut sum, scale_log2,
            );
            let mut kc = 0usize;
            while kc < 4 {
                thread::__unroll_config::<0>();
                let a = [
                    p[kc * 2][0],
                    p[kc * 2][1],
                    p[kc * 2 + 1][0],
                    p[kc * 2 + 1][1],
                ];
                let mut d = 0usize;
                while d < 24 {
                    thread::__unroll_config::<0>();
                    let b = ldmatrix_x4_trans_shared(
                        va + index(kc as u32 * 16 + (lane & 15), d as u32 * 4 + (lane >> 4) * 4)
                            * 4,
                    );
                    out[d] = mma::<BF16>(out[d], a, [b[0], b[1]]);
                    out[d + 1] = mma::<BF16>(out[d + 1], a, [b[2], b[3]]);
                    d += 2;
                }
                kc += 1;
            }
            // Next K readiness also releases V after all PV reads.
            ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
            thread::sync_threads();
            if next {
                stage::<64>(
                    vs,
                    v.add(kb + (j as usize + 64) * stride as usize),
                    stride,
                    tid,
                );
            }
            j += 64;
        }
        let mut l0 = sum[0];
        let mut l1 = sum[1];
        l0 += shuffle_xor_f32(l0, 1);
        l0 += shuffle_xor_f32(l0, 2);
        l1 += shuffle_xor_f32(l1, 1);
        l1 += shuffle_xor_f32(l1, 2);
        let inv0 = 1.0f32 / l0;
        let inv1 = 1.0f32 / l1;
        let row = warp_row + (lane >> 2);
        let mut d = 0usize;
        while d < 24 {
            thread::__unroll_config::<0>();
            let col = d * 4 + (lane & 3) as usize;
            *qs.add(row as usize * 96 + col) = cvt::<BF16>(out[d][0] * inv0, out[d][1] * inv0);
            *qs.add((row as usize + 8) * 96 + col) =
                cvt::<BF16>(out[d][2] * inv1, out[d][3] * inv1);
            d += 1;
        }
        if lane & 3 == 0 {
            let base = (batch as usize * h as usize + head as usize) * sq as usize
                + qbase as usize
                + row as usize;
            *lse.add(base) = (max[0] * scale_log2 + lg2_approx(l0)) * core::f32::consts::LN_2;
            *lse.add(base + 8) = (max[1] * scale_log2 + lg2_approx(l1)) * core::f32::consts::LN_2;
        }
        thread::sync_threads();
        let mut i = tid * 4;
        while i < 128 * 96 {
            let row = i / 96;
            let col = i % 96;
            let src = qs.add(i as usize);
            let dst = o.add(qb + row as usize * stride as usize + col as usize);
            let x0 = *src;
            let x1 = *src.add(1);
            let x2 = *src.add(2);
            let x3 = *src.add(3);
            ptx_asm!("st.global.v4.u32 [%0], {%1, %2, %3, %4};", in("l") dst, in("r") x0, in("r") x1, in("r") x2, in("r") x3, clobber("memory"));
            i += 1024;
        }
    }
}
