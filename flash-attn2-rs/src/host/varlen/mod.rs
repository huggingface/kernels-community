mod backward;
mod merge;

use std::ffi::c_void;

use tvm_ffi::error::{Error, RUNTIME_ERROR, Result, VALUE_ERROR};
use tvm_ffi::{Any, AnyView, Function, Tensor};
use tvm_ffi_sys::TVMFFIEnvGetStream;
use tvm_ffi_sys::dlpack::{DLDataType, DLDataTypeCode, DLDeviceType};

use crate::host::cuda::{launch_kernel, sys};
use crate::host::layout::{attention_shape, metadata, prepare_packed_qkv, prepare_qkv, tensor_span};
use crate::host::rng::resolve_rng_descriptor;
use crate::host::runtime::runtime;
use crate::host::tiles::{HEAD_DIMS, dropout_tile, flash_smem_bytes};
use crate::host::{EMPTY_SLICE, MAX_GRID_YZ, alloc_tensor, any_into_raw_owned, arg, err};

// Packed rank-three inputs share stride preparation with dense/cache inputs.
fn varlen_fwd(args: &[AnyView], descriptor_mode: bool) -> Result<Any> {
    let cached = !descriptor_mode && matches!(args.len(), 22 | 23 | 24);
    if (descriptor_mode && args.len() != 24)
        || (!descriptor_mode && args.len() != 19 && args.len() != 20 && !cached)
    {
        return Err(err(
            VALUE_ERROR,
            "varlen_fwd expects q, k, v, cuq, cuk, used, leftpad, alibi, scale, causal, left, right, cap, dropout, state, max_q, max_k, return_softmax, zero_tensors",
        ));
    }
    let prepare = if cached {
        prepare_qkv
    } else {
        prepare_packed_qkv
    };
    let (q, bf16, qs) = prepare("q", arg(args, 0, "q")?)?;
    let (k, kb, ks) = prepare("k", arg(args, 1, "k")?)?;
    let (v, vb, vs) = prepare("v", arg(args, 2, "v")?)?;
    let cuq: Option<Tensor> = arg(args, 3, "cu_seqlens_q")?;
    let cuk: Option<Tensor> = arg(args, 4, "cu_seqlens_k")?;
    let used: Option<Tensor> = arg(args, 5, "seqused_k")?;
    let leftpad: Option<Tensor> = arg(args, 6, "leftpad_k")?;
    let alibi: Option<Tensor> = arg(args, 7, "alibi")?;
    let scale: Option<f64> = arg(args, 8, "scale")?;
    let mut causal: bool = arg(args, 9, "causal")?;
    let mut left: i64 = arg(args, 10, "left")?;
    let mut right: i64 = arg(args, 11, "right")?;
    let cap: f64 = arg(args, 12, "cap")?;
    let p: f64 = arg(args, 13, "dropout")?;
    let state: Option<Tensor> = arg(args, 14, "state")?;
    let max_q: i64 = arg(args, 15, "max_q")?;
    let max_k: i64 = arg(args, 16, "max_k")?;
    let return_softmax: bool = arg(args, 17, "return_softmax")?;
    let zero_tensors: bool = arg(args, 18, "zero_tensors")?;
    let [one, sequence_q, h, d] = attention_shape(&q);
    let k_shape = attention_shape(&k);
    let total_q = if cached { one * sequence_q } else { sequence_q };
    let indices: Option<Tensor> = if cached {
        arg(args, 20, "cache_batch_idx")?
    } else {
        None
    };
    let new_keys: i64 = if cached {
        arg(args, 21, "new_keys")?
    } else {
        0
    };
    let requested_splits: i64 = if cached && args.len() >= 23 {
        arg(args, 22, "num_splits")?
    } else {
        1
    };
    if requested_splits < i32::MIN as i64 || requested_splits > 128 {
        return Err(err(
            VALUE_ERROR,
            "num_splits must fit int32 and be at most 128",
        ));
    }
    let grouped: bool = if cached && args.len() == 24 {
        arg(args, 23, "grouped_output")?
    } else {
        false
    };
    let table: Option<Tensor> = if args.len() >= 20 {
        arg(args, 19, "block_table")?
    } else {
        None
    };
    let paged = table.is_some();
    let total_k = if paged || cached {
        k_shape[0] * k_shape[1]
    } else {
        k_shape[1]
    };
    let hk = k_shape[2];
    let device = q.device();
    let batch = if cached {
        one
    } else {
        let t = cuq
            .as_ref()
            .ok_or_else(|| err(VALUE_ERROR, "cu_seqlens_q is required"))?;
        if t.ndim() != 1 || t.shape()[0] < 2 {
            return Err(err(VALUE_ERROR, "cu_seqlens_q must have shape [batch + 1]"));
        }
        t.shape()[0] - 1
    };
    if (!cached && one != 1)
        || (!cached && !paged && k_shape[0] != 1)
        || k.shape() != v.shape()
        || k_shape[3] != d
        || bf16 != kb
        || bf16 != vb
        || k.device() != device
        || v.device() != device
    {
        return Err(err(VALUE_ERROR, "inconsistent packed attention inputs"));
    }
    if batch <= 0
        || h <= 0
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
    if let Some(t) = &table {
        if t.ndim() != 2
            || t.shape()[0] != batch
            || t.device() != device
            || t.dtype() != DLDataType::new(DLDataTypeCode::kDLInt, 32, 1)
            || t.strides()[1] != 1
            || t.strides()[0] < 0
            || t.strides()[0] > u32::MAX as i64
            || t.shape()[1] > u32::MAX as i64
            || t.data_ptr() as u64 % 4 != 0
            || k_shape[1] == 0
            || k_shape[1] % 256 != 0
            || k_shape[1] > u32::MAX as i64
            || leftpad.is_some()
        {
            return Err(err(
                VALUE_ERROR,
                "paged K/V requires int32 [B,blocks] with unit last stride, page size divisible by 256, and no leftpad",
            ));
        }
    }
    if cached {
        let capacity = if paged {
            table.as_ref().unwrap().shape()[1]
                .checked_mul(k_shape[1])
                .ok_or_else(|| err(VALUE_ERROR, "cache capacity overflow"))?
        } else {
            k_shape[1]
        };
        if cuq.is_some()
            || used.is_some()
            || p != 0.0
            || return_softmax
            || max_q != sequence_q
            || max_k != capacity
            || new_keys < 0
            || new_keys > capacity
            || (new_keys > 0 && cuk.is_none())
            || (paged && indices.is_some())
            || (!paged && indices.is_none() && one > k_shape[0])
        {
            return Err(err(VALUE_ERROR, "inconsistent cached attention metadata"));
        }
        if let Some(t) = &cuk {
            metadata("cache lengths", t, &[batch], device)?;
        }
        if let Some(t) = &indices {
            metadata("cache indices", t, &[batch], device)?;
        }
    } else {
        metadata("cu_seqlens_q", cuq.as_ref().unwrap(), &[batch + 1], device)?;
        metadata(
            "cu_seqlens_k",
            cuk.as_ref()
                .ok_or_else(|| err(VALUE_ERROR, "cu_seqlens_k is required"))?,
            &[batch + 1],
            device,
        )?;
    }
    for (name, t) in [("seqused_k", &used), ("leftpad_k", &leftpad)] {
        if let Some(t) = t {
            metadata(name, t, &[batch], device)?;
        }
    }
    if !p.is_finite()
        || !(0.0..1.0).contains(&p)
        || p as f32 >= 1.0
        || (p as f32 > 0.0 && cap > 0.0)
        || (return_softmax && p as f32 == 0.0)
    {
        return Err(err(
            VALUE_ERROR,
            "invalid dropout, softcap, or probability request",
        ));
    }
    if descriptor_mode && p as f32 <= 0.0 {
        return Err(err(VALUE_ERROR, "RNG descriptor requires dropout"));
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
    if max_q == 1 && alibi.is_none() {
        causal = false;
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
    let rt = runtime(device.device_id)?;
    // Resolves state and runs attention in one FFI call.
    let mut descriptor = [0u64; 3];
    let mut captured = 0u32;
    let fused_rng = descriptor_mode
        && !paged
        && (33..=64).contains(&d)
        && total_q > 0
        && max_q > 0
        && max_k > 0;
    if descriptor_mode {
        descriptor = [
            arg::<i64>(args, 20, "seed")? as u64,
            arg::<i64>(args, 21, "offset")? as u64,
            arg::<i64>(args, 22, "intragraph")? as u64,
        ];
        captured = arg::<bool>(args, 23, "captured")? as u32;
        if captured != 0
            && (descriptor[0] == 0
                || descriptor[1] == 0
                || descriptor[0] % 8 != 0
                || descriptor[1] % 8 != 0)
        {
            return Err(err(VALUE_ERROR, "invalid captured RNG pointers"));
        }
        if !fused_rng {
            resolve_rng_descriptor(rt, state.as_ref().unwrap(), descriptor, captured)?;
        }
    }

    let bucket = *HEAD_DIMS.iter().find(|&&bucket| bucket >= d).unwrap();
    // max_q=1, total_q=batch: one query per sequence, reuse cache splits.
    let packed_decode = !cached
        && rt.sm89
        && max_q == 1
        && total_q == batch
        && matches!(bucket, 64 | 128)
        && max_k >= 512
        && batch * h <= rt.sm_count
        && p as f32 == 0.0
        && alibi.is_none()
        && left < 0
        && right < 0;
    let key_tiles =
        (max_k + dropout_tile(bucket).keys as i64 - 1) / dropout_tile(bucket).keys as i64;
    // Reuse K/V across query heads if splits still fill the GPU.
    let packed_grouped = packed_decode
        && bucket == 128
        && h > hk
        && h / hk <= 64
        && batch * hk * key_tiles.min(128) >= rt.sm_count * 2;
    // Multi-query splits: long-key underfilled grids, scratch <= half L2.
    let query_ctas = batch * h * ((max_q + 63) / 64);
    let packed_multi = !cached
        && rt.sm89
        && total_q > 0
        && max_q > 1
        && max_q <= max_k / 2
        && matches!(bucket, 64 | 128)
        && max_k >= 512
        && query_ctas <= rt.sm_count / 2
        && p as f32 == 0.0
        && alibi.is_none()
        && ((left < 0 && right < 0) || (left == max_k && right == 0));
    let multi_splits = if packed_multi {
        let bytes_per_split = batch * h * max_q * (d + 1) * 4;
        cache_splits(query_ctas, rt.sm_count * 2, key_tiles)
            .min(key_tiles / 4)
            .min(16)
            .min((rt.l2_bytes / 2) / bytes_per_split)
            .max(1)
    } else {
        1
    };
    let work_heads = if packed_grouped { hk } else { h };
    let work_queries = if packed_grouped { h / hk } else { max_q };
    let splits = if packed_decode {
        cache_splits(batch * work_heads, rt.sm_count * 2, key_tiles)
    } else if packed_multi {
        multi_splits
    } else if !cached || max_q == 0 || max_k == 0 {
        1
    } else if requested_splits < 1 {
        cache_splits(
            batch * h * ((max_q + 63) / 64),
            rt.sm_count * 2,
            (max_k + dropout_tile(bucket).keys as i64 - 1) / dropout_tile(bucket).keys as i64,
        )
    } else {
        requested_splits
    };
    let split_varlen =
        (packed_decode && (splits > 1 || packed_grouped)) || (packed_multi && splits > 1);
    let partial_output = splits > 1 || grouped || packed_grouped;
    if ((max_q + 63) / 64) * splits > i32::MAX as i64
        || (partial_output && total_q * h > i32::MAX as i64)
    {
        return Err(err(VALUE_ERROR, "cached split grid is too large"));
    }
    let out = alloc_tensor(
        if partial_output {
            vec![batch, splits, work_heads, work_queries, d]
        } else {
            q.shape().to_vec()
        }
        .as_slice(),
        if partial_output { float32 } else { q.dtype() },
        device,
    )?;
    let lse = alloc_tensor(
        if partial_output {
            vec![batch, splits, work_heads, work_queries]
        } else if cached {
            vec![batch, h, max_q]
        } else {
            vec![h, total_q]
        }
        .as_slice(),
        float32,
        device,
    )?;
    let probabilities = alloc_tensor(
        if return_softmax {
            vec![
                batch,
                h,
                (max_q + 127) / 128 * 128,
                (max_k + 127) / 128 * 128,
            ]
        } else {
            vec![0]
        }
        .as_slice(),
        q.dtype(),
        device,
    )?;
    rt.ctx
        .bind_to_thread()
        .map_err(|e| err(RUNTIME_ERROR, &format!("bind varlen context: {e:?}")))?;
    let stream = unsafe { TVMFFIEnvGetStream(DLDeviceType::kDLCUDA as i32, device.device_id) };
    if zero_tensors || max_k == 0 {
        unsafe {
            sys::cuMemsetD8Async(
                out.data_ptr() as u64,
                0,
                (out.numel() * if partial_output { 4 } else { 2 }) as usize,
                stream.cast(),
            )
        }
        .result()
        .map_err(|e| err(RUNTIME_ERROR, &format!("clear varlen output: {e:?}")))?;
        unsafe {
            sys::cuMemsetD32Async(
                lse.data_ptr() as u64,
                if max_k == 0 && !partial_output {
                    f32::INFINITY.to_bits()
                } else {
                    f32::NEG_INFINITY.to_bits()
                },
                lse.numel() as usize,
                stream.cast(),
            )
        }
        .result()
        .map_err(|e| err(RUNTIME_ERROR, &format!("clear varlen LSE: {e:?}")))?;
    }
    if probabilities.numel() > 0 {
        unsafe {
            sys::cuMemsetD8Async(
                probabilities.data_ptr() as u64,
                0,
                (probabilities.numel() * 2) as usize,
                stream.cast(),
            )
        }
        .result()
        .map_err(|e| err(RUNTIME_ERROR, &format!("clear varlen probabilities: {e:?}")))?;
    }
    if total_q > 0 && max_q > 0 && max_k > 0 {
        let mut buffers = Vec::<u64>::new();
        for t in [&q, &k, &v] {
            buffers.extend([t.data_ptr() as u64, tensor_span(t) / 2]);
        }
        buffers.extend([
            out.data_ptr() as u64,
            out.numel() as u64 / if partial_output { 1 } else { 2 },
        ]);
        buffers.extend([lse.data_ptr() as u64, lse.numel() as u64]);
        buffers.extend(
            alibi
                .as_ref()
                .map_or(EMPTY_SLICE, |t| [t.data_ptr() as u64, t.numel() as u64]),
        );
        for t in [&cuq, &cuk] {
            buffers.extend(
                t.as_ref()
                    .map_or(EMPTY_SLICE, |t| [t.data_ptr() as u64, t.numel() as u64]),
            );
        }
        for t in [&used, &leftpad, &state] {
            buffers.extend(
                t.as_ref()
                    .map_or(EMPTY_SLICE, |t| [t.data_ptr() as u64, t.numel() as u64]),
            );
        }
        buffers.extend(if probabilities.numel() > 0 {
            [
                probabilities.data_ptr() as u64,
                probabilities.numel() as u64 / 2,
            ]
        } else {
            EMPTY_SLICE
        });
        let mut params: Vec<*mut c_void> =
            buffers.iter_mut().map(|x| (x as *mut u64).cast()).collect();
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
            total_q as u32,
            total_k as u32,
        ];
        for value in &mut config {
            params.push((value as *mut u32).cast());
        }
        params.push((&mut keep as *mut f32).cast());
        let mut page_buffers = table
            .as_ref()
            .map_or(EMPTY_SLICE, |t| [t.data_ptr() as u64, tensor_span(t)]);
        let mut page_config = table.as_ref().map_or([0u32; 6], |t| {
            [
                t.strides()[0] as u32,
                t.shape()[1] as u32,
                k_shape[1] as u32,
                k_shape[0] as u32,
                ks.batch,
                vs.batch,
            ]
        });
        if cached && !paged {
            page_config[3] = k_shape[0] as u32;
            page_config[4] = ks.batch;
            page_config[5] = vs.batch;
        }
        if paged || cached || split_varlen {
            for value in &mut page_buffers {
                params.push((value as *mut u64).cast());
            }
            for value in &mut page_config {
                params.push((value as *mut u32).cast());
            }
        }
        let mut cache_buffers = indices
            .as_ref()
            .map_or(EMPTY_SLICE, |t| [t.data_ptr() as u64, t.numel() as u64]);
        let mut cache_config = [k_shape[0] as u32, new_keys as u32, qs.batch];
        if cached || split_varlen {
            for value in &mut cache_buffers {
                params.push((value as *mut u64).cast());
            }
            for value in &mut cache_config {
                params.push((value as *mut u32).cast());
            }
        }
        let mut split_count = splits as u32;
        if partial_output {
            params.push((&mut split_count as *mut u32).cast());
        }
        if fused_rng {
            for value in &mut descriptor {
                params.push((value as *mut u64).cast());
            }
            params.push((&mut captured as *mut u32).cast());
        }
        let function = if fused_rng {
            rt.varlen_rng_fns.get(&bf16).unwrap()
        } else if split_varlen {
            rt.varlen_split_fns
                .get(&(bucket, bf16, paged, packed_grouped))
                .unwrap()
        } else if cached {
            (if partial_output {
                &rt.cached_split_fns
            } else {
                &rt.cached_fns
            })
            .get(&(bucket, bf16, paged))
            .unwrap()
        } else {
            (if paged { &rt.paged_fns } else { &rt.varlen_fns })
                .get(&(bucket, bf16, p as f32 > 0.0))
                .unwrap()
        };
        unsafe {
            launch_kernel(
                function.cu_function(),
                (
                    (((work_queries + 63) / 64) * splits) as u32,
                    work_heads as u32,
                    batch as u32,
                ),
                (128, 1, 1),
                flash_smem_bytes(bucket, dropout_tile(bucket)),
                stream.cast(),
                &mut params,
            )
        }
        .map_err(|e| err(RUNTIME_ERROR, &format!("launch varlen forward: {e:?}")))?;
    }
    let (out, lse) = if packed_multi && splits > 1 {
        crate::host::varlen::merge::merge_varlen(
            &out,
            &lse,
            bf16,
            cuq.as_ref().unwrap(),
            total_q,
            q.ndim() == 3,
        )?
    } else if split_varlen {
        crate::host::varlen::merge::merge_packed_decode(&out, &lse, bf16, packed_grouped, q.ndim() == 3)?
    } else if partial_output {
        crate::host::varlen::merge::merge_layout(&out, &lse, bf16, grouped)?
    } else {
        (out, lse)
    };
    Function::get_global("ffi.Array")?.call_packed(&[
        AnyView::from(&out),
        AnyView::from(&lse),
        AnyView::from(&probabilities),
    ])
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tvm_ffi_varlen_fwd_explicit(
    _handle: *mut c_void,
    args: *const tvm_ffi_sys::TVMFFIAny,
    num_args: i32,
    result: *mut tvm_ffi_sys::TVMFFIAny,
) -> i32 {
    let packed = unsafe { std::slice::from_raw_parts(args as *const AnyView, num_args as usize) };
    match varlen_fwd(packed, false) {
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
pub unsafe extern "C" fn __tvm_ffi_varlen_fwd_descriptor(
    _handle: *mut c_void,
    args: *const tvm_ffi_sys::TVMFFIAny,
    num_args: i32,
    result: *mut tvm_ffi_sys::TVMFFIAny,
) -> i32 {
    let packed = unsafe { std::slice::from_raw_parts(args as *const AnyView, num_args as usize) };
    match varlen_fwd(packed, true) {
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

// Smallest count within 85% of the best wave efficiency; capture-safe.
fn cache_splits(ctas: i64, sms: i64, key_tiles: i64) -> i64 {
    if ctas as f64 >= 0.8 * sms as f64 || key_tiles <= 1 {
        return 1;
    }
    let limit = 128.min(sms).min(key_tiles);
    let mut efficiencies = [0.0f64; 128];
    let mut best = 0.0f64;
    for n in 1..=limit {
        if n > 1 && (key_tiles + n - 1) / n == (key_tiles + n - 2) / (n - 1) {
            continue;
        }
        let waves = (ctas * n) as f64 / sms as f64;
        let efficiency = waves / waves.ceil();
        efficiencies[(n - 1) as usize] = efficiency;
        best = best.max(efficiency);
    }
    for n in 1..=limit {
        if efficiencies[(n - 1) as usize] >= 0.85 * best {
            return n;
        }
    }
    1
}
