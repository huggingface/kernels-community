//! Tensor layout validation, bounded overlap proofs and layout copies.

use std::ffi::c_void;

use tvm_ffi::Tensor;
use tvm_ffi::error::{RUNTIME_ERROR, Result, TYPE_ERROR, VALUE_ERROR};
use tvm_ffi_sys::TVMFFIEnvGetStream;
use tvm_ffi_sys::dlpack::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType};

use crate::host::cuda::launch_kernel;
use crate::host::runtime::runtime;
use crate::host::{alloc_tensor, err};

#[derive(Clone, Copy)]
pub(super) struct Strides {
    pub(super) batch: u32,
    pub(super) seq: u32,
    pub(super) head: u32,
}

pub(super) fn prepare_qkv(name: &str, t: Tensor) -> Result<(Tensor, bool, Strides)> {
    prepare_attention_tensor(name, t, false)
}

pub(super) fn prepare_packed_qkv(name: &str, t: Tensor) -> Result<(Tensor, bool, Strides)> {
    prepare_attention_tensor(name, t, true)
}

// Normalize metadata only: packed tensors need no singleton-batch view.
pub(super) fn attention_shape(t: &Tensor) -> [i64; 4] {
    match *t.shape() {
        [s, h, d] => [1, s, h, d],
        [b, s, h, d] => [b, s, h, d],
        _ => unreachable!("attention tensor rank was validated"),
    }
}

fn attention_strides(t: &Tensor) -> [i64; 4] {
    match *t.strides() {
        [s, h, d] => [0, s, h, d],
        [b, s, h, d] => [b, s, h, d],
        _ => unreachable!("attention tensor rank was validated"),
    }
}

fn prepare_attention_tensor(
    name: &str,
    mut t: Tensor,
    packed: bool,
) -> Result<(Tensor, bool, Strides)> {
    if t.device().device_type != DLDeviceType::kDLCUDA {
        return Err(err(TYPE_ERROR, &format!("{name} must be a CUDA tensor")));
    }
    let dt = t.dtype();
    let is_f16 = dt.code == DLDataTypeCode::kDLFloat as u8 && dt.bits == 16 && dt.lanes == 1;
    let is_bf16 = dt.code == DLDataTypeCode::kDLBfloat as u8 && dt.bits == 16 && dt.lanes == 1;
    if !is_f16 && !is_bf16 {
        return Err(err(
            TYPE_ERROR,
            &format!("{name} must be float16 or bfloat16"),
        ));
    }
    if t.ndim() != 4 && !(packed && t.ndim() == 3) {
        return Err(err(
            VALUE_ERROR,
            &format!("{name} must be 4-D [B, S, H, D]"),
        ));
    }
    let shape = attention_shape(&t);
    let strides = attention_strides(&t);
    if shape[3] > 1 && strides[3] != 1 {
        return Err(err(
            VALUE_ERROR,
            &format!(
                "{name} head dim must be contiguous (last-dim stride 1, got {})",
                strides[3]
            ),
        ));
    }
    let mut pack = (t.data_ptr() as u64) % 16 != 0;
    let mut s = [0u32; 3];
    for i in 0..3 {
        if shape[i] <= 1 {
            continue;
        }
        let st = strides[i];
        if st < 0 {
            return Err(err(
                VALUE_ERROR,
                &format!("{name} stride for dim {i} must be non-negative, got {st}"),
            ));
        }
        let st_u32 = st / 2;
        pack |= st % 8 != 0 || st_u32 > u32::MAX as i64;
        s[i] = st_u32 as u32;
    }
    if pack {
        let packed = alloc_tensor(t.shape(), t.dtype(), t.device())?;
        copy_layout(&t, &packed)?;
        t = packed;
        let shape = attention_shape(&t);
        let strides = attention_strides(&t);
        for i in 0..3 {
            s[i] = if shape[i] <= 1 {
                0
            } else {
                u32::try_from(strides[i] / 2).map_err(|_| {
                    err(
                        VALUE_ERROR,
                        "packed tensor stride exceeds kernel indexing range",
                    )
                })?
            };
        }
    }
    Ok((
        t,
        is_bf16,
        Strides {
            batch: s[0],
            seq: s[1],
            head: s[2],
        },
    ))
}

pub(super) fn tensor_span(t: &Tensor) -> u64 {
    if t.numel() == 0 {
        return 0;
    }
    1 + t
        .shape()
        .iter()
        .zip(t.strides())
        .map(|(&n, &s)| ((n - 1) * s) as u64)
        .sum::<u64>()
}

pub(super) fn overlaps(a: &Tensor, b: &Tensor) -> bool {
    let x = a.data_ptr() as u64;
    let y = b.data_ptr() as u64;
    x < y + tensor_span(b) * 2 && y < x + tensor_span(a) * 2
}

pub(super) fn nonoverlapping_layout(t: &Tensor) -> bool {
    let mut dims: Vec<_> = t
        .shape()
        .iter()
        .zip(t.strides())
        .filter(|&(&n, _)| n > 1)
        .map(|(&n, &s)| (s, n))
        .collect();
    dims.sort_unstable();
    let mut span = 1i64;
    for &(stride, n) in &dims {
        if stride < span {
            return t.numel() == 0
                || nonoverlapping_interleaved(&dims)
                || nonoverlapping_three_axes(&dims)
                || nonoverlapping_small_layout(&dims)
                || nonoverlapping_bounded_differences(&dims);
        }
        let Some(next) = (n - 1)
            .checked_mul(stride)
            .and_then(|v| v.checked_add(span))
        else {
            return t.numel() == 0;
        };
        span = next;
    }
    true
}

// Exact bounded fallback for views not proved by the stride-lattice checks.
pub(super) fn nonoverlapping_small_layout(dims: &[(i64, i64)]) -> bool {
    let mut span = 1i64;
    let mut i = 0;
    while i < dims.len() && dims[i].0 == span {
        let Some(next) = span.checked_mul(dims[i].1) else {
            return false;
        };
        span = next;
        i += 1;
    }
    let axes = &dims[i..];
    let mut count = 1i64;
    for &(stride, n) in axes {
        let Some(next) = count.checked_mul(n) else {
            return false;
        };
        if stride <= 0 || next > 4096 {
            return false;
        }
        count = next;
    }
    let mut offsets = vec![0i64];
    for &(stride, n) in axes {
        let previous = offsets.len();
        for j in 1..n {
            let Some(delta) = j.checked_mul(stride) else {
                return false;
            };
            for k in 0..previous {
                let Some(offset) = offsets[k].checked_add(delta) else {
                    return false;
                };
                offsets.push(offset);
            }
        }
    }
    offsets.sort_unstable();
    offsets.windows(2).all(|pair| pair[1] - pair[0] >= span)
}

// Enumerate the smaller axes, never the long sequence axis.
pub(super) fn nonoverlapping_bounded_differences(dims: &[(i64, i64)]) -> bool {
    let (mut span, mut i) = (1i64, 0usize);
    while i < dims.len() && dims[i].0 == span {
        let Some(next) = span.checked_mul(dims[i].1) else {
            return false;
        };
        span = next;
        i += 1;
    }
    let mut axes = dims[i..].to_vec();
    if axes.is_empty() {
        return true;
    }
    if axes.iter().any(|&(s, _)| s <= 0) {
        return false;
    }
    axes.sort_unstable_by_key(|&(_, n)| n);
    let (stride, n) = axes.pop().unwrap();
    if stride < span {
        return false;
    }
    let (stride, bound, span) = (stride as i128, n as i128 - 1, span as i128);
    let mut count = 1i128;
    for &(_, n) in &axes {
        let Some(next) = count.checked_mul(2 * n as i128 - 1) else {
            return false;
        };
        if next > 4096 {
            return false;
        }
        count = next;
    }
    let mut offsets = vec![0i128];
    for &(step, n) in &axes {
        let mut next = Vec::with_capacity(offsets.len() * (2 * n - 1) as usize);
        for offset in offsets {
            for j in 1 - n..n {
                next.push(offset + j as i128 * step as i128);
            }
        }
        offsets = next;
    }
    // Remove only the zero vector; a second zero offset proves a collision.
    let mut skipped_zero = false;
    for offset in offsets {
        if offset == 0 && !skipped_zero {
            skipped_zero = true;
            continue;
        }
        let z = (-offset).div_euclid(stride).clamp(-bound, bound);
        if (offset + z * stride)
            .abs()
            .min((offset + (z + 1).min(bound) * stride).abs())
            < span
        {
            return false;
        }
    }
    true
}

// Overlapping interleaved axes can still be injective (GCD lattice).
pub(super) fn layout_floor_sum(mut n: i128, mut m: i128, mut a: i128, mut b: i128) -> i128 {
    let mut total = 0;
    loop {
        total += (n - 1) * n * (a / m) / 2 + n * (b / m);
        a %= m;
        b %= m;
        let y = a * n + b;
        if y < m {
            return total;
        }
        n = y / m;
        b = y % m;
        (m, a) = (a, m);
    }
}

pub(super) fn layout_rectangle_count(a: i128, n: i128, b: i128, m: i128, t: i128) -> i128 {
    if t < 0 {
        return 0;
    }
    let full = ((t - b * (m - 1)).div_euclid(a) + 1).clamp(0, n);
    let end = n.min(t / a + 1);
    let partial = end - full;
    if partial == 0 {
        return full * m;
    }
    full * m + partial + layout_floor_sum(partial, b, a, t - a * (end - 1))
}

pub(super) fn nonoverlapping_two_axes(a: i64, n: i64, b: i64, m: i64, span: i64) -> bool {
    if a.min(b) < span || n.checked_mul(m).is_none() {
        return false;
    }
    let (a, n, b, m, span) = (a as i128, n as i128, b as i128, m as i128, span as i128);
    // Only opposite-sign differences can cancel.
    let shift = b * (m - 1) - a;
    layout_rectangle_count(a, n - 1, b, m - 1, shift + span - 1)
        == layout_rectangle_count(a, n - 1, b, m - 1, shift - span)
}

pub(super) fn nonoverlapping_interleaved(dims: &[(i64, i64)]) -> bool {
    let mut span = 1i64;
    let mut i = 0;
    while i < dims.len() {
        let (stride, n) = dims[i];
        if stride < span {
            return false;
        }
        let Some(mut next) = (n - 1)
            .checked_mul(stride)
            .and_then(|x| x.checked_add(span))
        else {
            return false;
        };
        if let Some(&(other, count)) = dims.get(i + 1) {
            if other < next {
                let (mut gcd, mut remainder) = (stride, other);
                while remainder != 0 {
                    (gcd, remainder) = (remainder, gcd % remainder);
                }
                if (gcd < span && !nonoverlapping_two_axes(stride, n, other, count, span))
                    || (gcd >= span && n > other / gcd && count > stride / gcd)
                {
                    return false;
                }
                let Some(combined) = (count - 1)
                    .checked_mul(other)
                    .and_then(|x| x.checked_add(next))
                else {
                    return false;
                };
                next = combined;
                i += 1;
            }
        }
        span = next;
        i += 1;
    }
    true
}

// Contiguous block plus three interleaved axes: solve a*x + b*y = -c*z.
pub(super) fn nonoverlapping_three_axes(dims: &[(i64, i64)]) -> bool {
    let (span, outer) = match dims {
        [(1, n), rest @ ..] if rest.len() == 3 => (*n, rest),
        rest if rest.len() == 3 => (1, rest),
        _ => return false,
    };
    // Bound rare-path work; larger unproved layouts are rejected.
    if outer
        .iter()
        .any(|&(s, n)| s <= 0 || s > u32::MAX as i64 || n > i32::MAX as i64)
    {
        return false;
    }
    let mut axes = outer.to_vec();
    axes.sort_unstable_by_key(|&(_, n)| n);
    let gcd = |mut a: i64, mut b: i64| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let common = gcd(gcd(axes[0].0, axes[1].0), axes[2].0);
    // A cancellation cannot exceed the other axes' combined reach.
    let feasible = |i: usize| {
        let reach: i128 = (0..3)
            .filter(|&j| j != i)
            .map(|j| axes[j].0 as i128 * (axes[j].1 - 1) as i128)
            .sum();
        ((axes[i].1 - 1) as i128).min((reach + span as i128 - 1) / axes[i].0 as i128)
    };
    let index = (0..3)
        .min_by_key(|&i| {
            let step = if common >= span {
                gcd(axes[(i + 1) % 3].0, axes[(i + 2) % 3].0) / common
            } else {
                1
            };
            feasible(i) / step as i128
        })
        .unwrap();
    axes.swap(0, index);
    let (c, count) = axes[0];
    let (a, nx) = axes[1];
    let (b, ny) = axes[2];
    let reach = a as i128 * (nx - 1) as i128 + b as i128 * (ny - 1) as i128;
    let count = (count as i128).min((reach + span as i128 - 1) / c as i128 + 1) as i64;
    let (mut g, mut r) = (a as i128, b as i128);
    let (mut u, mut ur, mut v, mut vr) = (1i128, 0i128, 0i128, 1i128);
    while r != 0 {
        let q = g / r;
        (g, r) = (r, g - q * r);
        (u, ur) = (ur, u - q * ur);
        (v, vr) = (vr, v - q * vr);
    }
    if common < span {
        if count > 1024 {
            return false;
        }
        if !nonoverlapping_two_axes(a, nx, b, ny, span) {
            return false;
        }
        let (a, b, nx, ny, span) = (a as i128, b as i128, nx as i128, ny as i128, span as i128);
        let shift = a * (nx - 1) + b * (ny - 1);
        for z in 1..count {
            let target = shift - c as i128 * z as i128;
            if layout_rectangle_count(a, 2 * nx - 1, b, 2 * ny - 1, target + span - 1)
                != layout_rectangle_count(a, 2 * nx - 1, b, 2 * ny - 1, target - span)
            {
                return false;
            }
        }
        return true;
    }
    let step = (g / common as i128) as i64;
    if (count - 1) / step > 1023 {
        return false;
    }
    let (sx, sy) = (b as i128 / g, a as i128 / g);
    let (mx, my) = (nx as i128 - 1, ny as i128 - 1);
    // z=0: exclude the all-zero difference, but reject any cancelling pair.
    if sx <= mx && sy <= my {
        return false;
    }
    let ceil = |x: i128, d: i128| x.div_euclid(d) + i128::from(x.rem_euclid(d) != 0);
    // Negative z is symmetric: negate all three coordinate differences.
    for z in (step..count).step_by(step as usize) {
        let target = -(c as i128) * z as i128;
        let (x, y) = (u * (target / g), v * (target / g));
        // All solutions are (x + sx*t, y - sy*t). Intersect the two bounds.
        let lo = ceil(-mx - x, sx).max(ceil(y - my, sy));
        let hi = (mx - x).div_euclid(sx).min((y + my).div_euclid(sy));
        if lo <= hi {
            return false;
        }
    }
    true
}

pub(super) fn copy_layout(src: &Tensor, dst: &Tensor) -> Result<()> {
    let mut count = src.numel() as u64;
    if count == 0 {
        return Ok(());
    }
    let rt = runtime(src.device().device_id)?;
    rt.ctx
        .bind_to_thread()
        .map_err(|e| err(RUNTIME_ERROR, &format!("bind layout context: {e:?}")))?;
    let stream =
        unsafe { TVMFFIEnvGetStream(DLDeviceType::kDLCUDA as i32, src.device().device_id) };
    let mut from = src.data_ptr() as u64;
    let mut to = dst.data_ptr() as u64;
    let mut from_len = tensor_span(src);
    let mut to_len = tensor_span(dst);
    let shape = attention_shape(src);
    let from_strides = attention_strides(src);
    let to_strides = attention_strides(dst);
    let mut seq = shape[1] as u64;
    let mut heads = shape[2] as u64;
    let mut dim = shape[3] as u64;
    let mut strides = [
        from_strides[0] as u64,
        from_strides[1] as u64,
        from_strides[2] as u64,
        to_strides[0] as u64,
        to_strides[1] as u64,
        to_strides[2] as u64,
    ];
    let mut params = vec![
        (&mut from as *mut u64).cast::<c_void>(),
        (&mut from_len as *mut u64).cast(),
        (&mut to as *mut u64).cast(),
        (&mut to_len as *mut u64).cast(),
        (&mut count as *mut u64).cast(),
        (&mut seq as *mut u64).cast(),
        (&mut heads as *mut u64).cast(),
        (&mut dim as *mut u64).cast(),
    ];
    for stride in &mut strides {
        params.push((stride as *mut u64).cast());
    }
    unsafe {
        launch_kernel(
            rt.copy_layout.cu_function(),
            (((count + 255) / 256).min(65535) as u32, 1, 1),
            (256, 1, 1),
            0,
            stream.cast(),
            &mut params,
        )
    }
    .map_err(|e| err(RUNTIME_ERROR, &format!("copy attention layout: {e:?}")))
}

/// Varlen metadata vectors (cu_seqlens, lengths, indices) must be contiguous int32 on q's device.
pub(super) fn metadata(name: &str, tensor: &Tensor, shape: &[i64], device: DLDevice) -> Result<()> {
    if tensor.device() != device
        || tensor.dtype() != DLDataType::new(DLDataTypeCode::kDLInt, 32, 1)
        || tensor.shape() != shape
        || !tensor.is_contiguous()
        || tensor.data_ptr() as u64 % 4 != 0
    {
        return Err(err(
            VALUE_ERROR,
            &format!("{name} must be contiguous CUDA int32 {shape:?} on q's device"),
        ));
    }
    Ok(())
}
