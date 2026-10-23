use std::ffi::c_void;

use tvm_ffi::error::{Error, RUNTIME_ERROR, Result, VALUE_ERROR};
use tvm_ffi::{Any, AnyView, Function, Tensor};
use tvm_ffi_sys::TVMFFIEnvGetStream;
use tvm_ffi_sys::dlpack::{DLDataType, DLDataTypeCode, DLDeviceType};

use crate::cuda::{launch_kernel, sys};
use crate::layout::{prepare_qkv, tensor_span};
use crate::runtime::runtime;
use crate::tiles::HEAD_DIMS;
use crate::{EMPTY_SLICE, MAX_GRID_YZ, alloc_tensor, any_into_raw_owned, arg, err};

pub(super) const fn combined_smem(dropout: bool, masked: bool) -> u32 {
    if dropout || masked { 50176 } else { 17408 }
}

fn backward_component(args: &[AnyView], kv: bool, all: bool) -> Result<Vec<Tensor>> {
    if args.len() != 14 {
        return Err(err(
            VALUE_ERROR,
            "backward component expects q, k, v, dout, lse, delta, alibi, scale, causal, left, right, softcap, dropout, rng_state",
        ));
    }
    let (q, bf16, qs) = prepare_qkv("q", arg(args, 0, "q")?)?;
    let (k, kb, ks) = prepare_qkv("k", arg(args, 1, "k")?)?;
    let (v, vb, vs) = prepare_qkv("v", arg(args, 2, "v")?)?;
    let (dout, db, ds) = prepare_qkv("dout", arg(args, 3, "dout")?)?;
    let lse: Tensor = arg(args, 4, "lse")?;
    let delta: Tensor = arg(args, 5, "delta")?;
    let alibi: Option<Tensor> = arg(args, 6, "alibi")?;
    let scale: Option<f64> = arg(args, 7, "scale")?;
    let causal: bool = arg(args, 8, "causal")?;
    let mut left: i64 = arg(args, 9, "left")?;
    let mut right: i64 = arg(args, 10, "right")?;
    let cap: f64 = arg(args, 11, "softcap")?;
    let p: f64 = arg(args, 12, "dropout")?;
    let rng: Option<Tensor> = arg(args, 13, "rng_state")?;
    let [b, sq, h, d] = *q.shape() else {
        unreachable!()
    };
    let sk = k.shape()[1];
    let hk = k.shape()[2];
    let device = q.device();
    if bf16 != kb
        || bf16 != vb
        || bf16 != db
        || k.device() != device
        || v.device() != device
        || dout.device() != device
    {
        return Err(err(
            VALUE_ERROR,
            "backward inputs must share q's dtype and device",
        ));
    }
    if dout.shape() != q.shape() || k.shape() != v.shape() || k.shape()[0] != b || k.shape()[3] != d
    {
        return Err(err(VALUE_ERROR, "inconsistent backward input shapes"));
    }
    if b <= 0
        || h <= 0
        || b > MAX_GRID_YZ
        || h > MAX_GRID_YZ
        || hk <= 0
        || h % hk != 0
        || d < 8
        || d > 256
        || d % 8 != 0
        || sq < 0
        || sq > i32::MAX as i64 - 127
        || sk < 0
        || sk > i32::MAX as i64
    {
        return Err(err(VALUE_ERROR, "unsupported backward dimensions"));
    }
    if !p.is_finite()
        || !(0.0..1.0).contains(&p)
        || p as f32 >= 1.0
        || (p as f32 > 0.0 && cap > 0.0)
    {
        return Err(err(
            VALUE_ERROR,
            "invalid dropout or dropout combined with softcap",
        ));
    }
    let rounded_q = (sq + 127) / 128 * 128;
    let float32 = DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1);
    for (name, tensor, shape) in [
        ("lse", &lse, [b, h, sq]),
        ("delta", &delta, [b, h, rounded_q]),
    ] {
        if tensor.shape() != shape
            || tensor.dtype() != float32
            || tensor.device() != device
            || (tensor.numel() != 0 && !tensor.is_contiguous())
        {
            return Err(err(
                VALUE_ERROR,
                &format!("{name} must be contiguous float32 with shape {shape:?} on q's device"),
            ));
        }
    }
    if p as f32 > 0.0 {
        let state = rng
            .as_ref()
            .ok_or_else(|| err(VALUE_ERROR, "dropout backward requires saved RNG state"))?;
        if state.shape() != [2]
            || state.dtype() != DLDataType::new(DLDataTypeCode::kDLInt, 64, 1)
            || state.device() != device
            || !state.is_contiguous()
            || state.data_ptr() as u64 % 8 != 0
        {
            return Err(err(
                VALUE_ERROR,
                "RNG state must be contiguous CUDA int64 [2] on q's device",
            ));
        }
    }
    let alibi_bs = if let Some(a) = &alibi {
        if a.dtype() != float32 || a.device() != device || !a.is_contiguous() {
            return Err(err(
                VALUE_ERROR,
                "alibi must be contiguous CUDA float32 on q's device",
            ));
        }
        match *a.shape() {
            [n] if n == h => 0,
            [bb, n] if bb == b && n == h => h,
            _ => return Err(err(VALUE_ERROR, "alibi must have shape [H] or [B,H]")),
        }
    } else {
        0
    };
    if i32::try_from(left).is_err() || i32::try_from(right).is_err() {
        return Err(err(VALUE_ERROR, "window bounds must fit int32"));
    }
    if left >= sk {
        left = -1;
    }
    if right >= sk {
        right = -1;
    }
    if causal {
        right = 0;
    }
    let alibi_causal = left < 0 && right == 0;
    if left < 0 && right >= 0 {
        left = sk;
    }
    if left >= 0 && right < 0 {
        right = sk;
    }
    let original_scale = scale.unwrap_or(1.0 / (d as f64).sqrt());
    let (mut score_scale, mut tanh_scale) = if cap > 0.0 && original_scale != 0.0 {
        (
            cap.copysign(original_scale) as f32,
            (original_scale.abs() / cap) as f32,
        )
    } else {
        (original_scale as f32, 0.0f32)
    };
    let mut keep = 1.0f32 - p as f32;
    let mut gradient_scale = original_scale as f32 / keep;
    let fused = if all {
        let rt = runtime(device.device_id)?;
        let q_bytes = q.numel().checked_mul(4);
        let kv_bytes = b
            .checked_mul(sk)
            .and_then(|n| n.checked_mul(h))
            .and_then(|n| n.checked_mul(d))
            .and_then(|n| n.checked_mul(8));
        let key_blocks = b * h * ((sk + 63) / 64);
        // Tiny query tiles benefit from fewer launches on underfilled grids.
        let short_query = sq <= if d == 64 { 128 } else { 32 };
        let enough_parallelism = (short_query || key_blocks >= rt.sm_count / 2)
            && (sq <= sk || key_blocks >= rt.sm_count);
        // Measured dense range only.
        rt.sm89
            && matches!(d, 64 | 128)
            && enough_parallelism
            && cap <= 0.0
            && (32..=1024).contains(&sq)
            && (128..=1024).contains(&sk)
            && ((left < 0 && right < 0)
                || (alibi.is_none() && left >= sk && right == 0)
                || left + right + 1 >= 256)
            && q_bytes.is_some_and(|n| n <= (rt.l2_bytes / 2).max(0) as usize)
            && (h == hk || kv_bytes.is_some_and(|n| n <= rt.l2_bytes / 2))
    } else {
        false
    };
    if all && !fused {
        let mut gradients = backward_component(args, false, false)?;
        gradients.extend(backward_component(args, true, false)?);
        return Ok(gradients);
    }
    let fused_q = if fused {
        Some(alloc_tensor(q.shape(), q.dtype(), device)?)
    } else {
        None
    };
    let dq = alloc_tensor(if kv { k.shape() } else { q.shape() }, q.dtype(), device)?;
    let dv = if kv {
        Some(alloc_tensor(v.shape(), v.dtype(), device)?)
    } else {
        None
    };
    let finish = || -> Result<Vec<Tensor>> {
        let mut outputs = Vec::new();
        if let Some(output) = &fused_q {
            outputs.push(output.clone());
        }
        outputs.push(dq.clone());
        if let Some(dv) = &dv {
            outputs.push(dv.clone());
        }
        Ok(outputs)
    };
    if (if kv { sk } else { sq }) == 0 {
        return finish();
    }
    let bucket = *HEAD_DIMS.iter().find(|&&bucket| bucket >= d).unwrap();
    let rt = runtime(device.device_id)?;
    rt.ctx
        .bind_to_thread()
        .map_err(|e| err(RUNTIME_ERROR, &format!("bind backward context: {e:?}")))?;
    let stream = unsafe { TVMFFIEnvGetStream(DLDeviceType::kDLCUDA as i32, device.device_id) };
    if (if kv { sq } else { sk }) == 0 {
        for output in std::iter::once(&dq).chain(dv.iter()) {
            unsafe {
                sys::cuMemsetD8Async(
                    output.data_ptr() as u64,
                    0,
                    (output.numel() * 2) as usize,
                    stream.cast(),
                )
            }
            .result()
            .map_err(|e| {
                err(
                    RUNTIME_ERROR,
                    &format!("clear empty attention gradients: {e:?}"),
                )
            })?;
        }
        return finish();
    }
    // Expand underfilled GQA grids; FP32 scratch bounded to half of L2.
    let partial_bytes = b
        .checked_mul(sk)
        .and_then(|n| n.checked_mul(h))
        .and_then(|n| n.checked_mul(d))
        .and_then(|n| n.checked_mul(8));
    let parallel_gqa = (fused && h > hk)
        || (kv
            && rt.sm89
            && bucket == 128
            && h > hk
            && sq >= 64
            && (sq >= 256 || sk <= sq || (left < 0 && right < 0))
            && b * hk * ((sk + 31) / 32) <= 2 * rt.sm_count
            && partial_bytes.is_some_and(|bytes| bytes <= rt.l2_bytes / 2));
    let partial_mode = if p as f32 > 0.0 {
        2
    } else if cap <= 0.0 && alibi.is_none() && left < 0 && right < 0 {
        0
    } else {
        1
    };
    let both_masked = alibi.is_some() || left >= 0 || right >= 0;
    // Normalized causal windows (left=Sk, right=0) stay eligible.
    let wide_window = (left < 0 && right < 0) || left + right + 1 >= 256;
    // Count only key blocks intersecting some row's window.
    let first_key = if left >= 0 {
        (sk - sq - left).max(0)
    } else {
        0
    };
    let active_key_blocks = (sk + 63) / 64 - first_key / 64;
    let both_kv = kv
        && rt.sm89
        && bucket == 128
        && cap <= 0.0
        && wide_window
        && sq >= 256
        && sk >= 64
        && b * (if parallel_gqa { h } else { hk }) * active_key_blocks >= rt.sm_count;
    let partials = if parallel_gqa {
        Some((
            alloc_tensor(&[b, sk, h, d], float32, device)?,
            alloc_tensor(&[b, sk, h, d], float32, device)?,
        ))
    } else {
        None
    };
    // Split keys for underfilled dQ grids, >= 4 tiles per split.
    let query_blocks = b * h * ((sq + 63) / 64);
    let mut q_splits = 1u32;
    if !kv && rt.sm89 && bucket == 128 && sk >= 256 && query_blocks <= rt.sm_count / 2 {
        // Mirrors device KvRange rounding.
        let min_key_tiles = if left < 0 && right < 0 {
            (sk + 31) / 32
        } else {
            [0, ((sq - 1) / 64) * 64]
                .into_iter()
                .map(|first_q| {
                    let last_q = (first_q + 63).min(sq - 1);
                    let first_k = if left >= 0 {
                        (first_q + sk - sq - left).clamp(0, sk) / 32
                    } else {
                        0
                    };
                    let end_k = if right >= 0 {
                        ((last_q + 1 + sk - sq + right).clamp(0, sk) + 31) / 32
                    } else {
                        (sk + 31) / 32
                    };
                    (end_k - first_k).max(0)
                })
                .min()
                .unwrap()
        };
        for splits in [2u32, 4, 8] {
            let bytes = dq.numel().checked_mul(4 * splits as usize);
            if query_blocks * splits as i64 <= rt.sm_count
                && min_key_tiles / splits as i64 >= 4
                && bytes.is_some_and(|n| n <= (rt.l2_bytes / 2).max(0) as usize)
            {
                q_splits = splits;
            }
        }
    }
    let q_partials = if fused || q_splits > 1 {
        Some(alloc_tensor(
            &[q_splits as i64, b, sq, h, d],
            float32,
            device,
        )?)
    } else {
        None
    };
    if fused {
        let partial = q_partials.as_ref().unwrap();
        unsafe {
            sys::cuMemsetD8Async(
                partial.data_ptr() as u64,
                0,
                partial.numel() * 4,
                stream.cast(),
            )
        }
        .result()
        .map_err(|e| err(RUNTIME_ERROR, &format!("clear fused dQ: {e:?}")))?;
    }
    let swizzled_q = !kv && rt.sm89 && bucket == 128 && b * h * ((sq + 63) / 64) > rt.sm_count;
    // Slice ABI: device pointer followed by length.
    let mut buffers = Vec::<u64>::new();
    for t in [&q, &k, &v, &dout] {
        buffers.extend([t.data_ptr() as u64, tensor_span(t) / 2]);
    }
    for t in [&lse, &delta] {
        buffers.extend([t.data_ptr() as u64, t.numel() as u64]);
    }
    if let Some((pk, pv)) = &partials {
        buffers.extend([pk.data_ptr() as u64, pk.numel() as u64]);
        buffers.extend([pv.data_ptr() as u64, pv.numel() as u64]);
    } else if !kv && let Some(partial) = &q_partials {
        buffers.extend([partial.data_ptr() as u64, partial.numel() as u64]);
    } else {
        buffers.extend([dq.data_ptr() as u64, dq.numel() as u64 / 2]);
        if let Some(dv) = &dv {
            buffers.extend([dv.data_ptr() as u64, dv.numel() as u64 / 2]);
        }
    }
    if fused {
        buffers.push(q_partials.as_ref().unwrap().data_ptr() as u64);
    }
    buffers.extend(
        alibi
            .as_ref()
            .map_or(EMPTY_SLICE, |t| [t.data_ptr() as u64, t.numel() as u64]),
    );
    buffers.extend(
        rng.as_ref()
            .map_or(EMPTY_SLICE, |t| [t.data_ptr() as u64, t.numel() as u64]),
    );
    let mut params: Vec<*mut c_void> = buffers.iter_mut().map(|v| (v as *mut u64).cast()).collect();
    params.push((&mut score_scale as *mut f32).cast());
    params.push((&mut tanh_scale as *mut f32).cast());
    let mut config = [
        left as u32,
        right as u32,
        sq as u32,
        sk as u32,
        alibi.is_some() as u32,
        alibi_causal as u32,
        h as u32,
        (h / hk) as u32,
        (d / 2) as u32,
        qs.batch,
        qs.seq,
        qs.head,
        ks.batch,
        ks.seq,
        ks.head,
        vs.batch,
        vs.seq,
        vs.head,
        alibi_bs as u32,
        ds.batch,
        ds.seq,
        ds.head,
    ];
    for value in &mut config {
        params.push((value as *mut u32).cast());
    }
    params.push((&mut gradient_scale as *mut f32).cast());
    params.push((&mut keep as *mut f32).cast());
    unsafe {
        launch_kernel(
            if fused {
                // Normalized causal left=Sk never excludes a valid key.
                let mode = if alibi.is_none() && left < 0 && right < 0 {
                    0
                } else if alibi.is_none() && left >= sk && right == 0 {
                    // Opposite key order helps two batches, one key CTA per SM.
                    if b == 2 && sq == sk && sq >= 256 && h * ((sk + 63) / 64) == rt.sm_count {
                        3
                    } else {
                        1
                    }
                } else {
                    2
                };
                let kernels = if d == 64 {
                    &rt.bwd_fused_d64_fns
                } else {
                    &rt.bwd_fused_fns
                };
                kernels.get(&(bf16, parallel_gqa, p as f32 > 0.0, mode))
            } else if both_kv {
                rt.bwd_dkv_both_fns
                    .get(&(bf16, parallel_gqa, p as f32 > 0.0, both_masked))
            } else if parallel_gqa {
                rt.bwd_dkv_specialized_fns.get(&(bf16, true, partial_mode))
            } else if kv {
                if rt.sm89 && bucket == 128 && partial_mode != 2 {
                    rt.bwd_dkv_specialized_fns.get(&(bf16, false, partial_mode))
                } else {
                    rt.bwd_dkv_fns.get(&(bucket, bf16))
                }
            } else if q_splits > 1 {
                rt.bwd_dq_split_fns.get(&(bf16, q_splits))
            } else if swizzled_q {
                rt.bwd_dq_swizzled_fns.get(&(
                    bf16,
                    if cap <= 0.0 && partial_mode != 0 {
                        if p as f32 > 0.0 { 4 } else { 3 }
                    } else {
                        partial_mode
                    },
                ))
            } else {
                rt.bwd_dq_fns.get(&(bucket, bf16))
            }
            .unwrap()
            .cu_function(),
            if kv {
                (
                    (if fused || both_kv {
                        (sk + 63) / 64
                    } else {
                        (sk + 31) / 32
                    }) as u32,
                    (if parallel_gqa { h } else { hk }) as u32,
                    b as u32,
                )
            } else {
                (((sq + 63) / 64) as u32 * q_splits, h as u32, b as u32)
            },
            (128, 1, 1),
            if fused {
                if d == 64 { 25600 } else { 50176 }
            } else if both_kv {
                combined_smem(p as f32 > 0.0, both_masked)
            } else if swizzled_q {
                50176
            } else {
                ((if kv { 128 } else { 192 }) * (bucket / 2 + 4) * 4) as u32
            },
            stream.cast(),
            &mut params,
        )
    }
    .map_err(|e| err(RUNTIME_ERROR, &format!("launch backward component: {e:?}")))?;
    if let Some((pk, pv)) = &partials {
        let dv = dv.as_ref().unwrap();
        let mut buffers = [
            pk.data_ptr() as u64,
            pk.numel() as u64,
            pv.data_ptr() as u64,
            pv.numel() as u64,
            dq.data_ptr() as u64,
            dq.numel() as u64 / 2,
            dv.data_ptr() as u64,
            dv.numel() as u64 / 2,
            dq.numel() as u64 / 2,
        ];
        let mut shape = [hk as u32, (h / hk) as u32, (d / 2) as u32];
        let mut params: Vec<*mut c_void> =
            buffers.iter_mut().map(|x| (x as *mut u64).cast()).collect();
        params.extend(shape.iter_mut().map(|x| (x as *mut u32).cast()));
        unsafe {
            launch_kernel(
                rt.bwd_dkv_merge_fns.get(&bf16).unwrap().cu_function(),
                (
                    ((dq.numel() as u64 / 2).div_ceil(256).min(65535)) as u32,
                    1,
                    1,
                ),
                (256, 1, 1),
                0,
                stream.cast(),
                &mut params,
            )
        }
        .map_err(|e| err(RUNTIME_ERROR, &format!("merge GQA gradients: {e:?}")))?;
    }
    if let Some(partial) = &q_partials {
        let dq = fused_q.as_ref().unwrap_or(&dq);
        let mut buffers = [
            partial.data_ptr() as u64,
            partial.numel() as u64,
            dq.data_ptr() as u64,
            dq.numel() as u64 / 2,
            dq.numel() as u64 / 2,
        ];
        let mut params: Vec<*mut c_void> =
            buffers.iter_mut().map(|x| (x as *mut u64).cast()).collect();
        params.push((&mut q_splits as *mut u32).cast());
        params.push((&mut gradient_scale as *mut f32).cast());
        unsafe {
            launch_kernel(
                rt.bwd_dq_merge_fns.get(&bf16).unwrap().cu_function(),
                (
                    ((dq.numel() as u64 / 2).div_ceil(256).min(65535)) as u32,
                    1,
                    1,
                ),
                (256, 1, 1),
                0,
                stream.cast(),
                &mut params,
            )
        }
        .map_err(|e| err(RUNTIME_ERROR, &format!("merge dQ key splits: {e:?}")))?;
    }
    finish()
}

fn backward_result(args: &[AnyView], kv: bool, all: bool) -> Result<Any> {
    let outputs = backward_component(args, kv, all)?;
    if kv {
        let views: Vec<AnyView> = outputs.iter().map(AnyView::from).collect();
        Function::get_global("ffi.Array")?.call_packed(&views)
    } else {
        Ok(Any::from(outputs.into_iter().next().unwrap()))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tvm_ffi_bwd_dq(
    _handle: *mut c_void,
    args: *const tvm_ffi_sys::TVMFFIAny,
    num_args: i32,
    result: *mut tvm_ffi_sys::TVMFFIAny,
) -> i32 {
    let packed = unsafe { std::slice::from_raw_parts(args as *const AnyView, num_args as usize) };
    match backward_result(packed, false, false) {
        Ok(value) => {
            unsafe { *result = any_into_raw_owned(value) };
            0
        }
        Err(error) => {
            Error::set_raised(&error);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tvm_ffi_bwd_dkv(
    _handle: *mut c_void,
    args: *const tvm_ffi_sys::TVMFFIAny,
    num_args: i32,
    result: *mut tvm_ffi_sys::TVMFFIAny,
) -> i32 {
    let packed = unsafe { std::slice::from_raw_parts(args as *const AnyView, num_args as usize) };
    match backward_result(packed, true, false) {
        Ok(value) => {
            unsafe { *result = any_into_raw_owned(value) };
            0
        }
        Err(error) => {
            Error::set_raised(&error);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tvm_ffi_bwd_all(
    _handle: *mut c_void,
    args: *const tvm_ffi_sys::TVMFFIAny,
    num_args: i32,
    result: *mut tvm_ffi_sys::TVMFFIAny,
) -> i32 {
    let packed = unsafe { std::slice::from_raw_parts(args as *const AnyView, num_args as usize) };
    match backward_result(packed, true, true) {
        Ok(value) => {
            unsafe { *result = any_into_raw_owned(value) };
            0
        }
        Err(error) => {
            Error::set_raised(&error);
            -1
        }
    }
}

fn bwd_preprocess(args: &[AnyView]) -> Result<Any> {
    if args.len() != 3 {
        return Err(err(
            VALUE_ERROR,
            "bwd_preprocess expects dout, out, p_dropout",
        ));
    }
    let p: f64 = arg(args, 2, "p_dropout")?;
    if !p.is_finite() || !(0.0..1.0).contains(&p) || p as f32 >= 1.0 {
        return Err(err(VALUE_ERROR, "p_dropout must be in [0, 1) as float32"));
    }
    let (dout, bf16, ds) = prepare_qkv("dout", arg(args, 0, "dout")?)?;
    let (out, out_bf16, os) = prepare_qkv("out", arg(args, 1, "out")?)?;
    if dout.shape() != out.shape() || bf16 != out_bf16 || dout.device() != out.device() {
        return Err(err(
            VALUE_ERROR,
            "dout and out must share shape, dtype and device",
        ));
    }
    let [b, seq, h, d] = *out.shape() else {
        unreachable!()
    };
    if b <= 0
        || h <= 0
        || b > MAX_GRID_YZ
        || h > MAX_GRID_YZ
        || seq < 0
        || seq > u32::MAX as i64 - 127
        || d < 8
        || d > 256
        || d % 8 != 0
    {
        return Err(err(VALUE_ERROR, "unsupported backward tensor dimensions"));
    }
    let rounded_seq = (seq + 127) / 128 * 128;
    let delta = alloc_tensor(
        &[b, h, rounded_seq],
        DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1),
        out.device(),
    )?;
    if seq == 0 {
        return Ok(Any::from(delta));
    }
    let lanes = if d <= 64 {
        8
    } else if d <= 128 {
        16
    } else {
        32
    };
    let grid = b * h * rounded_seq / (256 / lanes);
    if grid > i32::MAX as i64 {
        return Err(err(
            VALUE_ERROR,
            "backward grid exceeds CUDA indexing range",
        ));
    }
    let rt = runtime(out.device().device_id)?;
    rt.ctx
        .bind_to_thread()
        .map_err(|e| err(RUNTIME_ERROR, &format!("bind backward context: {e:?}")))?;
    let stream =
        unsafe { TVMFFIEnvGetStream(DLDeviceType::kDLCUDA as i32, out.device().device_id) };
    let mut dp = dout.data_ptr() as u64;
    let mut dl = tensor_span(&dout) / 2;
    let mut op = out.data_ptr() as u64;
    let mut ol = tensor_span(&out) / 2;
    let mut dst = delta.data_ptr() as u64;
    let mut len = delta.numel() as u64;
    let mut dims = [
        seq as u32,
        rounded_seq as u32,
        h as u32,
        (d / 2) as u32,
        ds.batch,
        ds.seq,
        ds.head,
        os.batch,
        os.seq,
        os.head,
    ];
    let mut keep = 1.0f32 - p as f32;
    let mut params = vec![
        (&mut dp as *mut u64).cast::<c_void>(),
        (&mut dl as *mut u64).cast(),
        (&mut op as *mut u64).cast(),
        (&mut ol as *mut u64).cast(),
        (&mut dst as *mut u64).cast(),
        (&mut len as *mut u64).cast(),
    ];
    for d in &mut dims {
        params.push((d as *mut u32).cast());
    }
    params.push((&mut keep as *mut f32).cast());
    unsafe {
        launch_kernel(
            rt.bwd_preprocess_fns
                .get(&(bf16, lanes as u32))
                .unwrap()
                .cu_function(),
            (grid as u32, 1, 1),
            (256, 1, 1),
            0,
            stream.cast(),
            &mut params,
        )
    }
    .map_err(|e| err(RUNTIME_ERROR, &format!("backward preprocess: {e:?}")))?;
    Ok(Any::from(delta))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tvm_ffi_bwd_preprocess(
    _handle: *mut c_void,
    args: *const tvm_ffi_sys::TVMFFIAny,
    num_args: i32,
    result: *mut tvm_ffi_sys::TVMFFIAny,
) -> i32 {
    let packed = unsafe { std::slice::from_raw_parts(args as *const AnyView, num_args as usize) };
    match bwd_preprocess(packed) {
        Ok(value) => {
            unsafe { *result = any_into_raw_owned(value) };
            0
        }
        Err(error) => {
            Error::set_raised(&error);
            -1
        }
    }
}
