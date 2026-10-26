use std::ffi::c_void;

use tvm_ffi::error::{Error, RUNTIME_ERROR, Result, VALUE_ERROR};
use tvm_ffi::{Any, AnyView, Function, Tensor};
use tvm_ffi_sys::TVMFFIEnvGetStream;
use tvm_ffi_sys::dlpack::{DLDataType, DLDataTypeCode, DLDeviceType};

use crate::host::cuda::{launch_kernel, sys};
use crate::host::layout::{attention_shape, prepare_packed_qkv, tensor_span};
use crate::host::runtime::runtime;
use crate::host::tiles::HEAD_DIMS;
use crate::host::layout::metadata;
use crate::host::{EMPTY_SLICE, MAX_GRID_YZ, alloc_tensor, any_into_raw_owned, arg, err};

fn varlen_bwd(args: &[AnyView]) -> Result<Any> {
    if args.len() != 19 {
        return Err(err(
            VALUE_ERROR,
            "varlen_bwd expects q,k,v,dout,out,lse,cuq,cuk,alibi,scale,causal,left,right,cap,dropout,state,max_q,max_k,zero_tensors",
        ));
    }
    let (q, bf16, qs) = prepare_packed_qkv("q", arg(args, 0, "q")?)?;
    let (k, kb, ks) = prepare_packed_qkv("k", arg(args, 1, "k")?)?;
    let (v, vb, vs) = prepare_packed_qkv("v", arg(args, 2, "v")?)?;
    let (dout, db, ds) = prepare_packed_qkv("dout", arg(args, 3, "dout")?)?;
    let (out, ob, os) = prepare_packed_qkv("out", arg(args, 4, "out")?)?;
    let lse: Tensor = arg(args, 5, "lse")?;
    let cuq: Tensor = arg(args, 6, "cu_seqlens_q")?;
    let cuk: Tensor = arg(args, 7, "cu_seqlens_k")?;
    let alibi: Option<Tensor> = arg(args, 8, "alibi")?;
    let scale: Option<f64> = arg(args, 9, "scale")?;
    let causal: bool = arg(args, 10, "causal")?;
    let mut left: i64 = arg(args, 11, "left")?;
    let mut right: i64 = arg(args, 12, "right")?;
    let cap: f64 = arg(args, 13, "cap")?;
    let p: f64 = arg(args, 14, "dropout")?;
    let state: Option<Tensor> = arg(args, 15, "state")?;
    let max_q: i64 = arg(args, 16, "max_q")?;
    let max_k: i64 = arg(args, 17, "max_k")?;
    let zero_tensors: bool = arg(args, 18, "zero_tensors")?;
    let [one, total_q, h, d] = attention_shape(&q);
    let k_shape = attention_shape(&k);
    let total_k = k_shape[1];
    let hk = k_shape[2];
    let device = q.device();
    if cuq.ndim() != 1 || cuq.shape()[0] < 2 {
        return Err(err(VALUE_ERROR, "cu_seqlens_q must have shape [batch + 1]"));
    }
    let batch = cuq.shape()[0] - 1;
    if one != 1
        || k_shape[0] != 1
        || k.shape() != v.shape()
        || k_shape[3] != d
        || bf16 != kb
        || bf16 != vb
        || k.device() != device
        || v.device() != device
    {
        return Err(err(VALUE_ERROR, "inconsistent packed attention inputs"));
    }
    if h <= 0
        || hk <= 0
        || h % hk != 0
        || batch > MAX_GRID_YZ
        || h > MAX_GRID_YZ
        || d < 8
        || d > 256
        || d % 8 != 0
        || total_q > i32::MAX as i64
        || total_k > i32::MAX as i64
        || max_q < 0
        || max_k < 0
        || max_q > i32::MAX as i64 - 127
        || max_k > i32::MAX as i64 - 127
    {
        return Err(err(VALUE_ERROR, "unsupported packed attention dimensions"));
    }
    metadata("cu_seqlens_q", &cuq, &[batch + 1], device)?;
    metadata("cu_seqlens_k", &cuk, &[batch + 1], device)?;
    if !p.is_finite()
        || !(0.0..1.0).contains(&p)
        || p as f32 >= 1.0
        || (p as f32 > 0.0 && cap > 0.0)
    {
        return Err(err(
            VALUE_ERROR,
            "invalid dropout, softcap, or probability request",
        ));
    }
    if p as f32 > 0.0 {
        let rng = state
            .as_ref()
            .ok_or_else(|| err(VALUE_ERROR, "dropout requires RNG state"))?;
        if rng.shape() != [2]
            || rng.device() != device
            || !rng.is_contiguous()
            || rng.dtype() != DLDataType::new(DLDataTypeCode::kDLInt, 64, 1)
            || rng.data_ptr() as u64 % 8 != 0
        {
            return Err(err(
                VALUE_ERROR,
                "RNG state must be contiguous int64 [2] on q's device",
            ));
        }
    }
    if dout.shape() != q.shape()
        || out.shape() != q.shape()
        || db != bf16
        || ob != bf16
        || dout.device() != device
        || out.device() != device
    {
        return Err(err(
            VALUE_ERROR,
            "out and dout must match packed q's shape, dtype and device",
        ));
    }
    if lse.shape() != [h, total_q]
        || lse.device() != device
        || lse.dtype() != DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1)
        || (lse.numel() != 0 && !lse.is_contiguous())
    {
        return Err(err(
            VALUE_ERROR,
            "LSE must be contiguous float32 [heads,total_q] on q's device",
        ));
    }
    let float32 = DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1);
    let alibi_bs = if let Some(a) = &alibi {
        if a.device() != device || a.dtype() != float32 || !a.is_contiguous() {
            return Err(err(
                VALUE_ERROR,
                "alibi must be contiguous float32 on q's device",
            ));
        }
        match *a.shape() {
            [n] if n == h => 0,
            [b, n] if b == batch && n == h => h,
            _ => return Err(err(VALUE_ERROR, "alibi must have shape [H] or [B,H]")),
        }
    } else {
        0
    };
    if i32::try_from(left).is_err() || i32::try_from(right).is_err() {
        return Err(err(VALUE_ERROR, "window bounds must fit int32"));
    }
    if causal {
        right = 0;
    }
    if left >= max_k {
        left = -1;
    }
    if right >= max_k {
        right = -1;
    }
    let alibi_causal = left < 0 && right == 0;
    if left < 0 && right >= 0 {
        left = max_k;
    }
    if left >= 0 && right < 0 {
        right = max_k;
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
    let delta_stride = total_q + 128 * batch;
    let dq = alloc_tensor(q.shape(), q.dtype(), device)?;
    let dk = alloc_tensor(k.shape(), k.dtype(), device)?;
    let dv = alloc_tensor(v.shape(), v.dtype(), device)?;
    let delta = alloc_tensor(&[h, delta_stride], float32, device)?;
    let rt = runtime(device.device_id)?;
    rt.ctx
        .bind_to_thread()
        .map_err(|e| err(RUNTIME_ERROR, &format!("bind packed backward: {e:?}")))?;
    let stream = unsafe { TVMFFIEnvGetStream(DLDeviceType::kDLCUDA as i32, device.device_id) };
    // Expand underfilled GQA grids; FP32 partials bounded to half of L2.
    let partial_bytes = total_k
        .checked_mul(h)
        .and_then(|n| n.checked_mul(d))
        .and_then(|n| n.checked_mul(8));
    let parallel_gqa = rt.sm89
        && h > hk
        && max_q >= 64
        && total_q > 0
        && total_k > 0
        && max_k > 0
        && batch * hk * ((max_k + 31) / 32) <= 2 * rt.sm_count
        && partial_bytes.is_some_and(|bytes| bytes <= rt.l2_bytes / 2);
    let partials = if parallel_gqa {
        Some((
            alloc_tensor(&[total_k, h, d], float32, device)?,
            alloc_tensor(&[total_k, h, d], float32, device)?,
        ))
    } else {
        None
    };
    // Two key partitions for underfilled D64/D128 query grids.
    let q_bytes = total_q
        .checked_mul(h)
        .and_then(|n| n.checked_mul(d))
        .and_then(|n| n.checked_mul(8));
    let mut q_splits = if rt.sm89
        && matches!(d, 64 | 128)
        && max_q >= 64
        && max_k >= 256
        && total_q > 0
        && total_k > 0
        && batch * h * ((max_q + 63) / 64) * 2 <= rt.sm_count
        && q_bytes
            .and_then(|n| {
                n.checked_add(if parallel_gqa {
                    partial_bytes.unwrap()
                } else {
                    0
                })
            })
            .is_some_and(|n| n <= rt.l2_bytes / 2)
    {
        2u32
    } else {
        1u32
    };
    let q_partials = if q_splits > 1 {
        Some(alloc_tensor(
            &[q_splits as i64, total_q, h, d],
            float32,
            device,
        )?)
    } else {
        None
    };
    if zero_tensors && let Some(partial) = &q_partials {
        unsafe {
            sys::cuMemsetD8Async(
                partial.data_ptr() as u64,
                0,
                partial.numel() * 4,
                stream.cast(),
            )
        }
        .result()
        .map_err(|e| err(RUNTIME_ERROR, &format!("clear packed dQ partials: {e:?}")))?;
    }
    if zero_tensors {
        if let Some((pk, pv)) = &partials {
            for t in [pk, pv] {
                unsafe {
                    sys::cuMemsetD8Async(
                        t.data_ptr() as u64,
                        0,
                        t.numel() as usize * 4,
                        stream.cast(),
                    )
                }
                .result()
                .map_err(|e| {
                    err(
                        RUNTIME_ERROR,
                        &format!("clear packed partial gradients: {e:?}"),
                    )
                })?;
            }
        }
    }

    if zero_tensors || max_q == 0 || total_q == 0 {
        // Merges overwrite every element; only partials need zeroing.
        let targets: &[&Tensor] = match (q_partials.is_some(), parallel_gqa) {
            (true, true) => &[&delta],
            (true, false) => &[&dk, &dv, &delta],
            (false, true) => &[&dq, &delta],
            (false, false) => &[&dq, &dk, &dv, &delta],
        };
        for t in targets {
            if t.numel() == 0 {
                continue;
            }
            let bytes = t.numel() as usize * (t.dtype().bits as usize / 8);
            unsafe { sys::cuMemsetD8Async(t.data_ptr() as u64, 0, bytes, stream.cast()) }
                .result()
                .map_err(|e| err(RUNTIME_ERROR, &format!("clear packed gradients: {e:?}")))?;
        }
    }
    if max_q > 0 && total_q > 0 {
        let lanes = if d <= 64 {
            8u32
        } else if d <= 128 {
            16
        } else {
            32
        };
        let mut buffers = Vec::<u64>::new();
        for t in [&dout, &out] {
            buffers.extend([t.data_ptr() as u64, tensor_span(t) / 2]);
        }
        for t in [&delta, &cuq] {
            buffers.extend([t.data_ptr() as u64, t.numel() as u64]);
        }
        let mut params: Vec<*mut c_void> =
            buffers.iter_mut().map(|x| (x as *mut u64).cast()).collect();
        let mut config = [
            total_q as u32,
            delta_stride as u32,
            max_q as u32,
            (d / 2) as u32,
            ds.seq,
            ds.head,
            os.seq,
            os.head,
        ];
        for x in &mut config {
            params.push((x as *mut u32).cast());
        }
        params.push((&mut keep as *mut f32).cast());
        let rounded_q = (max_q + 127) / 128 * 128;
        unsafe {
            launch_kernel(
                rt.varlen_preprocess_fns
                    .get(&(bf16, lanes))
                    .unwrap()
                    .cu_function(),
                (
                    (rounded_q / (256 / lanes) as i64) as u32,
                    h as u32,
                    batch as u32,
                ),
                (256, 1, 1),
                0,
                stream.cast(),
                &mut params,
            )
        }
        .map_err(|e| {
            err(
                RUNTIME_ERROR,
                &format!("launch packed backward preprocessing: {e:?}"),
            )
        })?;
    }
    let bucket = *HEAD_DIMS.iter().find(|&&bucket| bucket >= d).unwrap();
    let mut buffers = Vec::<u64>::new();
    for t in [&q, &k, &v, &dout] {
        buffers.extend([t.data_ptr() as u64, tensor_span(t) / 2]);
    }
    for t in [&lse, &delta] {
        buffers.extend([t.data_ptr() as u64, t.numel() as u64]);
    }
    if let Some(partial) = &q_partials {
        buffers.extend([partial.data_ptr() as u64, partial.numel() as u64]);
    } else {
        buffers.extend([dq.data_ptr() as u64, dq.numel() as u64 / 2]);
    }
    if let Some((pk, pv)) = &partials {
        // Launch ABI uses u32 slices; one word per FP32 partial.
        for t in [pk, pv] {
            buffers.extend([t.data_ptr() as u64, t.numel() as u64]);
        }
    } else {
        for t in [&dk, &dv] {
            buffers.extend([t.data_ptr() as u64, t.numel() as u64 / 2]);
        }
    }
    for t in [&alibi, &state] {
        buffers.extend(
            t.as_ref()
                .map_or(EMPTY_SLICE, |t| [t.data_ptr() as u64, t.numel() as u64]),
        );
    }
    for t in [&cuq, &cuk] {
        buffers.extend([t.data_ptr() as u64, t.numel() as u64]);
    }
    let mut params: Vec<*mut c_void> = buffers.iter_mut().map(|x| (x as *mut u64).cast()).collect();
    params.push((&mut score_scale as *mut f32).cast());
    params.push((&mut tanh_scale as *mut f32).cast());
    let mut config = [
        left as u32,
        right as u32,
        max_q as u32,
        max_k as u32,
        alibi.is_some() as u32,
        alibi_causal as u32,
        h as u32,
        (h / hk) as u32,
        (d / 2) as u32,
        qs.seq,
        qs.head,
        ks.seq,
        ks.head,
        vs.seq,
        vs.head,
        alibi_bs as u32,
        ds.seq,
        ds.head,
        total_q as u32,
        total_k as u32,
        delta_stride as u32,
    ];
    for x in &mut config {
        params.push((x as *mut u32).cast());
    }
    params.push((&mut gradient_scale as *mut f32).cast());
    params.push((&mut keep as *mut f32).cast());
    for kv in [false, true] {
        let seq = if kv { max_k } else { max_q };
        let total = if kv { total_k } else { total_q };
        if seq == 0 || total == 0 || max_q == 0 || total_q == 0 {
            continue;
        }
        let rows = if kv { 32 } else { 64 };
        unsafe {
            launch_kernel(
                if d == 64
                    && cap <= 0.0
                    && alibi.is_none()
                    && alibi_causal
                    && ((!kv && q_splits == 2) || (kv && parallel_gqa))
                {
                    rt.varlen_bwd_causal_fns
                        .get(&(bf16, kv))
                        .unwrap()
                        .cu_function()
                } else if !kv && q_splits > 1 {
                    rt.varlen_dq_split_fns
                        .get(&(bucket, bf16))
                        .unwrap()
                        .cu_function()
                } else if kv && parallel_gqa {
                    rt.varlen_bwd_partial_fns
                        .get(&(bucket, bf16))
                        .unwrap()
                        .cu_function()
                } else {
                    rt.varlen_bwd_fns
                        .get(&(bucket, bf16, kv))
                        .unwrap()
                        .cu_function()
                },
                (
                    ((seq + rows - 1) / rows) as u32 * if kv { 1 } else { q_splits },
                    (if kv && !parallel_gqa { hk } else { h }) as u32,
                    batch as u32,
                ),
                (128, 1, 1),
                ((if kv { 128 } else { 192 }) * (bucket / 2 + 4) * 4) as u32,
                stream.cast(),
                &mut params,
            )
        }
        .map_err(|e| err(RUNTIME_ERROR, &format!("launch packed backward: {e:?}")))?;
    }
    if let Some((pk, pv)) = &partials {
        // Same [row, head, dim] reduction layout as dense GQA.
        let mut buffers = [
            pk.data_ptr() as u64,
            pk.numel() as u64,
            pv.data_ptr() as u64,
            pv.numel() as u64,
            dk.data_ptr() as u64,
            dk.numel() as u64 / 2,
            dv.data_ptr() as u64,
            dv.numel() as u64 / 2,
            dk.numel() as u64 / 2,
        ];
        let mut shape = [hk as u32, (h / hk) as u32, (d / 2) as u32];
        let mut params: Vec<*mut c_void> =
            buffers.iter_mut().map(|x| (x as *mut u64).cast()).collect();
        params.extend(shape.iter_mut().map(|x| (x as *mut u32).cast()));
        unsafe {
            launch_kernel(
                rt.bwd_dkv_merge_fns.get(&bf16).unwrap().cu_function(),
                (
                    (dk.numel() as u64 / 2).div_ceil(256).min(65535) as u32,
                    1,
                    1,
                ),
                (256, 1, 1),
                0,
                stream.cast(),
                &mut params,
            )
        }
        .map_err(|e| err(RUNTIME_ERROR, &format!("merge packed GQA gradients: {e:?}")))?;
    }
    if let Some(partial) = &q_partials {
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
                    (dq.numel() as u64 / 2).div_ceil(256).min(65535) as u32,
                    1,
                    1,
                ),
                (256, 1, 1),
                0,
                stream.cast(),
                &mut params,
            )
        }
        .map_err(|e| err(RUNTIME_ERROR, &format!("merge packed dQ splits: {e:?}")))?;
    }
    Function::get_global("ffi.Array")?.call_packed(&[
        AnyView::from(&dq),
        AnyView::from(&dk),
        AnyView::from(&dv),
        AnyView::from(&delta),
    ])
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tvm_ffi_varlen_bwd_explicit(
    _handle: *mut c_void,
    args: *const tvm_ffi_sys::TVMFFIAny,
    num_args: i32,
    result: *mut tvm_ffi_sys::TVMFFIAny,
) -> i32 {
    let packed = unsafe { std::slice::from_raw_parts(args as *const AnyView, num_args as usize) };
    match varlen_bwd(packed) {
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
