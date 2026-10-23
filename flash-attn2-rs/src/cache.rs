mod alias;

use std::ffi::c_void;

use tvm_ffi::error::{Error, RUNTIME_ERROR, Result, TYPE_ERROR, VALUE_ERROR};
use tvm_ffi::{Any, AnyView, Tensor};
use tvm_ffi_sys::dlpack::{DLDataType, DLDataTypeCode, DLDeviceType};
use tvm_ffi_sys::{TVMFFIAny, TVMFFIEnvGetStream};

use self::alias::{disjoint_auxiliary, disjoint_views};
use crate::cuda::launch_kernel;
use crate::layout::{copy_layout, nonoverlapping_layout, tensor_span};
use crate::runtime::runtime;
use crate::{EMPTY_SLICE, alloc_tensor, any_into_raw_owned, arg, err};

fn tensor(name: &str, t: &Tensor, q: &Tensor, writable: bool) -> Result<()> {
    if t.ndim() != 4
        || t.device() != q.device()
        || t.dtype() != q.dtype()
        || t.strides()[3] != 1
        || t.strides().iter().any(|&x| x < 0)
        || t.data_ptr() as u64 % 2 != 0
        || (writable && !nonoverlapping_layout(t))
    {
        return Err(err(
            VALUE_ERROR,
            &format!(
                "{name} must be a compatible rank-4 tensor with unit last stride and nonoverlapping writable layout; overlapping or unproven writable layouts are unsupported. Use a separately allocated contiguous cache."
            ),
        ));
    }
    Ok(())
}

// Append into the original cache and return a rotated Q.
fn update(args: &[AnyView]) -> Result<Any> {
    if !matches!(args.len(), 12 | 14) {
        return Err(err(
            VALUE_ERROR,
            "cache_update expects q, kcache, vcache, k, v, lengths, cos, sin, indices, table, interleaved, advance_q, and optional leftpad/alibi",
        ));
    }
    let q: Tensor = arg(args, 0, "q")?;
    let kc: Tensor = arg(args, 1, "kcache")?;
    let vc: Tensor = arg(args, 2, "vcache")?;
    let k: Tensor = arg(args, 3, "k")?;
    let v: Tensor = arg(args, 4, "v")?;
    let lengths: Tensor = arg(args, 5, "lengths")?;
    let cos: Option<Tensor> = arg(args, 6, "cos")?;
    let sin: Option<Tensor> = arg(args, 7, "sin")?;
    let indices: Option<Tensor> = arg(args, 8, "indices")?;
    let table: Option<Tensor> = arg(args, 9, "table")?;
    let interleaved: bool = arg(args, 10, "interleaved")?;
    let advance_q: bool = arg(args, 11, "advance_q")?;
    let leftpad: Option<Tensor> = if args.len() == 14 {
        arg(args, 12, "leftpad")?
    } else {
        None
    };
    let alibi: Option<Tensor> = if args.len() == 14 {
        arg(args, 13, "alibi")?
    } else {
        None
    };
    let dtype = q.dtype();
    let bf16 = dtype.code == DLDataTypeCode::kDLBfloat as u8;
    if q.device().device_type != DLDeviceType::kDLCUDA
        || dtype.bits != 16
        || dtype.lanes != 1
        || (!bf16 && dtype.code != DLDataTypeCode::kDLFloat as u8)
    {
        return Err(err(TYPE_ERROR, "cache update requires CUDA FP16/BF16"));
    }
    // An empty append may still rotate Q, but never writes the caches.
    let writes_cache = k.numel() > 0;
    for (name, t, writable) in [
        ("q", &q, false),
        ("kcache", &kc, writes_cache),
        ("vcache", &vc, writes_cache),
        ("k", &k, false),
        ("v", &v, false),
    ] {
        tensor(name, t, &q, writable)?;
    }
    let [batch, sq, h, d] = *q.shape() else {
        unreachable!()
    };
    if batch <= 0
        || h <= 0
        || d <= 0
        || d > 256
        || kc.shape() != vc.shape()
        || k.shape() != v.shape()
        || k.shape()[0] != batch
        || k.shape()[2] != kc.shape()[2]
        || k.shape()[3] != d
        || kc.shape()[3] != d
        || kc.shape()[2] <= 0
        || h % kc.shape()[2] != 0
    {
        return Err(err(VALUE_ERROR, "inconsistent cache update shapes"));
    }
    if writes_cache {
        if !disjoint_views(&kc, &vc) {
            return Err(err(
                VALUE_ERROR,
                "K and V caches must not overlap; unsupported shared layout: the bounded safety checks cannot prove they are disjoint. Use separately allocated contiguous K and V caches.",
            ));
        }
    }
    super::varlen::metadata("lengths", &lengths, &[batch], q.device())?;
    if let Some(t) = &indices {
        super::varlen::metadata("indices", t, &[batch], q.device())?;
    }
    let mut page_size = 0i64;
    let mut table_stride = 0i64;
    let mut columns = 0i64;
    if let Some(t) = &table {
        if indices.is_some()
            || t.ndim() != 2
            || t.shape()[0] != batch
            || t.device() != q.device()
            || t.dtype() != DLDataType::new(DLDataTypeCode::kDLInt, 32, 1)
            || t.strides()[1] != 1
            || t.strides()[0] < 0
            || t.data_ptr() as u64 % 4 != 0
            || kc.shape()[1] <= 0
            || kc.shape()[1] % 256 != 0
        {
            return Err(err(VALUE_ERROR, "invalid paged cache metadata"));
        }
        page_size = kc.shape()[1];
        table_stride = t.strides()[0];
        columns = t.shape()[1];
    } else if indices.is_none() && kc.shape()[0] < batch {
        return Err(err(VALUE_ERROR, "cache batch must cover Q without indices"));
    }
    let capacity = if table.is_some() {
        columns
            .checked_mul(page_size)
            .ok_or_else(|| err(VALUE_ERROR, "cache capacity overflow"))?
    } else {
        kc.shape()[1]
    };
    if [capacity, kc.shape()[0], page_size, table_stride, columns]
        .iter()
        .any(|&x| x < 0 || x > i32::MAX as i64)
    {
        return Err(err(VALUE_ERROR, "cache dimensions must fit int32"));
    }
    let mut rd = 0i64;
    let mut rr = 0i64;
    match (&cos, &sin) {
        (None, None) => {}
        (Some(c), Some(s)) => {
            if c.ndim() != 2
                || c.shape() != s.shape()
                || c.device() != q.device()
                || s.device() != q.device()
                || c.dtype() != dtype
                || s.dtype() != dtype
                || !c.is_contiguous()
                || !s.is_contiguous()
                || c.data_ptr() as u64 % 2 != 0
                || s.data_ptr() as u64 % 2 != 0
            {
                return Err(err(
                    VALUE_ERROR,
                    "rotary tables must be contiguous [positions,rotary_dim/2] in Q dtype/device",
                ));
            }
            rd = c.shape()[1] * 2;
            rr = c.shape()[0];
            if rd > d || rd % 16 != 0 || rr < capacity || rr > i32::MAX as i64 {
                return Err(err(
                    VALUE_ERROR,
                    "rotary dimension must be divisible by 16 and fit D; positions must cover cache capacity",
                ));
            }
        }
        _ => return Err(err(VALUE_ERROR, "cos and sin must be supplied together")),
    }
    let auxiliaries = [
        cos.as_ref(),
        sin.as_ref(),
        Some(&lengths),
        indices.as_ref(),
        table.as_ref(),
    ];
    if writes_cache
        && auxiliaries
            .into_iter()
            .chain([leftpad.as_ref(), alibi.as_ref()])
            .flatten()
            .any(|t| !disjoint_auxiliary(t, &kc) || !disjoint_auxiliary(t, &vc))
    {
        return Err(err(
            VALUE_ERROR,
            "cache update metadata and rotary tables must not overlap writable caches; unsupported shared layout: the bounded safety checks cannot prove they are disjoint. Allocate metadata and rotary tables separately from the caches.",
        ));
    }
    // Snapshot only aliased read sources: no host sync or full cache copy.
    let snapshot = |source: Tensor| -> Result<Tensor> {
        if writes_cache && (!disjoint_views(&source, &kc) || !disjoint_views(&source, &vc)) {
            let saved = alloc_tensor(source.shape(), source.dtype(), source.device())?;
            copy_layout(&source, &saved)?;
            Ok(saved)
        } else {
            Ok(source)
        }
    };
    let q = snapshot(q)?;
    let k = snapshot(k)?;
    let v = snapshot(v)?;
    let out = if rd > 0 {
        alloc_tensor(q.shape(), dtype, q.device())?
    } else {
        q.clone()
    };
    let rt = runtime(q.device().device_id)?;
    rt.ctx
        .bind_to_thread()
        .map_err(|e| err(RUNTIME_ERROR, &format!("bind cache context: {e:?}")))?;
    let stream = unsafe { TVMFFIEnvGetStream(DLDeviceType::kDLCUDA as i32, q.device().device_id) };
    // Keep separate launch order if auxiliary inputs share cache storage.
    let fused = rd > 0
        && q.numel() > 0
        && k.numel() > 0
        && auxiliaries.into_iter().flatten().all(|t| {
            let start = t.data_ptr() as u64;
            let end = start + tensor_span(t) * (t.dtype().bits as u64 / 8);
            [&kc, &vc].into_iter().all(|cache| {
                let cache_start = cache.data_ptr() as u64;
                t.numel() == 0
                    || cache.numel() == 0
                    || start >= cache_start + tensor_span(cache) * 2
                    || end <= cache_start
            })
        });
    for append in [true, false] {
        let source = if append { &k } else { &q };
        let dest = if append { &kc } else { &out };
        if source.numel() == 0 || (!append && (rd == 0 || fused)) {
            continue;
        }
        let mut buffers = Vec::<u64>::new();
        for t in [
            Some(source),
            if append { Some(&v) } else { None },
            Some(dest),
            if append { Some(&vc) } else { None },
            cos.as_ref(),
            sin.as_ref(),
            Some(&lengths),
            indices.as_ref(),
            table.as_ref(),
        ] {
            buffers.extend(t.map_or(EMPTY_SLICE, |x| [x.data_ptr() as u64, tensor_span(x)]));
        }
        let mut config64 = vec![
            source.numel() as u64,
            source.shape()[1] as u64,
            source.shape()[2] as u64,
            d as u64,
        ];
        for t in [source, &v, dest, &vc] {
            config64.extend(t.strides()[..3].iter().map(|&s| s as u64));
        }
        let mut config32 = [
            capacity as u32,
            kc.shape()[0] as u32,
            page_size as u32,
            table_stride as u32,
            columns as u32,
            rd as u32,
            rr as u32,
            interleaved as u32,
            advance_q as u32,
        ];
        let mut params: Vec<*mut c_void> =
            buffers.iter_mut().map(|x| (x as *mut u64).cast()).collect();
        params.extend(config64.iter_mut().map(|x| (x as *mut u64).cast()));
        params.extend(config32.iter_mut().map(|x| (x as *mut u32).cast()));
        let mut query = vec![];
        if fused {
            for t in [&q, &out] {
                query.extend([t.data_ptr() as u64, tensor_span(t)]);
            }
            query.extend([q.numel() as u64, q.shape()[1] as u64, q.shape()[2] as u64]);
            for t in [&q, &out] {
                query.extend(t.strides()[..3].iter().map(|&s| s as u64));
            }
            params.extend(query.iter_mut().map(|x| (x as *mut u64).cast()));
        }
        let function = if fused {
            rt.cache_update_fns.get(&bf16).unwrap()
        } else {
            rt.cache_copy_fns.get(&(bf16, append)).unwrap()
        };
        let mut blocks = (source.numel() as u64).div_ceil(256).min(65535);
        if fused {
            blocks += (q.numel() as u64).div_ceil(256).min(65535);
        }
        unsafe {
            launch_kernel(
                function.cu_function(),
                (blocks as u32, 1, 1),
                (256, 1, 1),
                0,
                stream.cast(),
                &mut params,
            )
        }
        .map_err(|e| err(RUNTIME_ERROR, &format!("launch cache update: {e:?}")))?;
    }
    let _ = sq;
    Ok(out.into())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tvm_ffi_cache_update_explicit(
    _handle: *mut c_void,
    args: *const TVMFFIAny,
    num_args: i32,
    result: *mut TVMFFIAny,
) -> i32 {
    let packed = unsafe { std::slice::from_raw_parts(args as *const AnyView, num_args as usize) };
    match update(packed) {
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

// Validate original destinations before padding makes new allocations.
fn validate_layouts(args: &[AnyView]) -> Result<Any> {
    if !matches!(args.len(), 2 | 7 | 9) {
        return Err(err(
            VALUE_ERROR,
            "cache_validate_layouts expects K/V caches and optional rotary/length/index/page metadata",
        ));
    }
    let kc: Tensor = arg(args, 0, "kcache")?;
    let vc: Tensor = arg(args, 1, "vcache")?;
    let dtype = kc.dtype();
    if kc.device().device_type != DLDeviceType::kDLCUDA
        || dtype.bits != 16
        || dtype.lanes != 1
        || (dtype.code != DLDataTypeCode::kDLBfloat as u8
            && dtype.code != DLDataTypeCode::kDLFloat as u8)
    {
        return Err(err(TYPE_ERROR, "cache update requires CUDA FP16/BF16"));
    }
    tensor("kcache", &kc, &kc, true)?;
    tensor("vcache", &vc, &kc, true)?;
    if !disjoint_views(&kc, &vc) {
        return Err(err(
            VALUE_ERROR,
            "K and V caches must not overlap; unsupported shared layout: the bounded safety checks cannot prove they are disjoint. Use separately allocated contiguous K and V caches.",
        ));
    }
    for i in 2..args.len() {
        let auxiliary: Option<Tensor> = arg(args, i, "cache auxiliary")?;
        if let Some(t) = auxiliary {
            if !disjoint_auxiliary(&t, &kc) || !disjoint_auxiliary(&t, &vc) {
                return Err(err(
                    VALUE_ERROR,
                    "cache update metadata and rotary tables must not overlap writable caches; unsupported shared layout: the bounded safety checks cannot prove they are disjoint. Allocate metadata and rotary tables separately from the caches.",
                ));
            }
        }
    }
    Ok(Any::default())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tvm_ffi_cache_validate_layouts(
    _handle: *mut c_void,
    args: *const TVMFFIAny,
    num_args: i32,
    result: *mut TVMFFIAny,
) -> i32 {
    let packed = unsafe { std::slice::from_raw_parts(args as *const AnyView, num_args as usize) };
    match validate_layouts(packed) {
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
