use cuda_device::convert::cvt_f16x2_f32;
use cuda_device::ptx_asm;
use cuda_device::wmma::{mma_m16n8k16_f32_bf16, mma_m16n8k16_f32_f16};

#[inline(always)]
pub fn fast_exp2(x: f32) -> f32 {
    let y: f32;
    unsafe {
        ptx_asm!("ex2.approx.f32 %0, %1;", out("=f") y, in("f") x, options(register_only));
    }
    y
}

#[inline(always)]
pub fn fmax(a: f32, b: f32) -> f32 {
    let out: f32;
    unsafe {
        ptx_asm!("max.f32 %0, %1, %2;", out("=f") out, in("f") a, in("f") b, options(register_only));
    }
    out
}

#[inline(always)]
pub fn tanh_approx(x: f32) -> f32 {
    let out: f32;
    unsafe {
        ptx_asm!("tanh.approx.f32 %0, %1;", out("=f") out, in("f") x, options(register_only));
    }
    out
}

#[inline(always)]
pub fn lg2_approx(x: f32) -> f32 {
    let l2: f32;
    unsafe {
        ptx_asm!("lg2.approx.f32 %0, %1;", out("=f") l2, in("f") x, options(register_only));
    }
    l2
}

#[inline(always)]
pub fn fabs(x: f32) -> f32 {
    let out: f32;
    unsafe {
        ptx_asm!("abs.f32 %0, %1;", out("=f") out, in("f") x, options(register_only));
    }
    out
}

#[inline(always)]
pub fn cvt_bf16x2_f32(lo: f32, hi: f32) -> u32 {
    let out: u32;
    unsafe {
        ptx_asm!("cvt.rn.bf16x2.f32 %0, %2, %1;", out("=r") out, in("f") lo, in("f") hi, options(register_only));
    }
    out
}

#[inline(always)]
pub fn mma<const BF16: bool>(c: [f32; 4], a: [u32; 4], b: [u32; 2]) -> [f32; 4] {
    unsafe {
        if BF16 {
            mma_m16n8k16_f32_bf16(c, a, b)
        } else {
            mma_m16n8k16_f32_f16(c, a, b)
        }
    }
}

#[inline(always)]
pub fn cvt<const BF16: bool>(lo: f32, hi: f32) -> u32 {
    if BF16 {
        cvt_bf16x2_f32(lo, hi)
    } else {
        cvt_f16x2_f32(lo, hi)
    }
}

// Convert the shared base once so unrolled loads use one register.
#[inline(always)]
pub fn shared_address(ptr: *const u32) -> u32 {
    let addr: u32;
    unsafe {
        ptx_asm!("{ .reg .u64 s; cvta.to.shared.u64 s, %1; cvt.u32.u64 %0, s; }",
            out("=r") addr, in("l") ptr as u64, options(register_only));
    }
    addr
}

#[inline(always)]
fn shared_pointer(addr: u32) -> *const u32 {
    let ptr: u64;
    unsafe {
        ptx_asm!("{ .reg .u64 s; cvt.u64.u32 s, %1; cvta.shared.u64 %0, s; }",
            out("=l") ptr, in("r") addr, options(register_only));
    }
    ptr as *const u32
}

#[inline(always)]
pub unsafe fn ldmatrix_x2_shared(addr: u32) -> [u32; 2] {
    unsafe { cuda_device::wmma::ldmatrix_x2(shared_pointer(addr)) }
}

#[inline(always)]
pub unsafe fn ldmatrix_x2_trans_shared(addr: u32) -> [u32; 2] {
    unsafe { cuda_device::wmma::ldmatrix_x2_trans(shared_pointer(addr)) }
}

#[inline(always)]
pub unsafe fn ldmatrix_x4_shared(addr: u32) -> [u32; 4] {
    unsafe { cuda_device::wmma::ldmatrix_x4(shared_pointer(addr)) }
}

#[inline(always)]
pub unsafe fn ldmatrix_x4_trans_shared(addr: u32) -> [u32; 4] {
    unsafe { cuda_device::wmma::ldmatrix_x4_trans(shared_pointer(addr)) }
}

// Empty slices are non-null; map optional metadata to null pointers.
#[inline(always)]
pub fn optional_ptr<T>(slice: &[T]) -> *const T {
    if slice.is_empty() {
        core::ptr::null()
    } else {
        slice.as_ptr()
    }
}

/// Low half of a packed f16/bf16 pair as f32.
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
