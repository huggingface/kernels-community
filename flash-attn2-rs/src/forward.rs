//! Dense forward validation, dispatch and launch.

mod dispatch;

use std::ffi::c_void;

use tvm_ffi::any::TryFromTemp;
use tvm_ffi::error::{Error, RUNTIME_ERROR, Result, TYPE_ERROR, VALUE_ERROR};
use tvm_ffi::{Any, AnyView, Function, Shape, Tensor};
use tvm_ffi_sys::TVMFFIEnvGetStream;
use tvm_ffi_sys::dlpack::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType};

use self::dispatch::{ForwardPlan, plan_forward, select_kernel};
use crate::cuda::{launch_kernel, sys};
use crate::layout::{Strides, copy_layout, nonoverlapping_layout, overlaps, prepare_qkv};
use crate::rng::resolve_rng_descriptor;
use crate::runtime::runtime;
use crate::tiles::{HEAD_DIMS, flash_smem_bytes, split_smem_bytes};
use crate::{EMPTY_SLICE, MAX_GRID_YZ, alloc_tensor, any_into_raw_owned, arg, err};

fn finish_fwd(
    out: Tensor,
    lse: Tensor,
    supplied: Option<Tensor>,
    copy_out: bool,
    probabilities: Option<Tensor>,
    lse_only: bool,
) -> Result<Any> {
    if copy_out {
        copy_layout(&out, supplied.as_ref().unwrap())?;
    }
    if lse_only {
        return Ok(Any::from(lse));
    }
    let out = supplied.as_ref().unwrap_or(&out);
    let mut results = vec![AnyView::from(out), AnyView::from(&lse)];
    if let Some(p) = &probabilities {
        results.push(AnyView::from(p));
    }
    Function::get_global("ffi.Array")?.call_packed(&results)
}

fn fwd(args: &[AnyView]) -> Result<Any> {
    fwd_impl(args, false)
}

/// Validated forward inputs and buffers, kept until all launches finish.
struct PreparedForward {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    _alibi: Option<Tensor>,
    q_str: Strides,
    k_str: Strides,
    v_str: Strides,
    is_bf16: bool,
    b: i64,
    h: i64,
    d: i64,
    seqlen_q: i64,
    seqlen_k: i64,
    hk: i64,
    bh: i64,
    device: DLDevice,
    has_alibi: u32,
    alibi_data: u64,
    alibi_numel: u64,
    alibi_bs: u32,
    scale: f64,
    scale_eff: f64,
    tanh_scale: f64,
    softcap: f64,
    wl: i64,
    wr: i64,
    alibi_causal: u32,
    supplied_out: Option<Tensor>,
    copy_out: bool,
    out: Tensor,
    lse: Tensor,
    probabilities: Option<Tensor>,
    rng_state: Option<Tensor>,
    dropout: bool,
    dropout_seed: u64,
    dropout_offset: u64,
    dropout_keep: f32,
    native_reservation: bool,
    return_softmax: bool,
    descriptor_mode: bool,
    descriptor: [u64; 3],
    descriptor_captured: u32,
    bucket: i64,
    lse_only: bool,
}

fn fwd_impl(args: &[AnyView], lse_only: bool) -> Result<Any> {
    let input = prepare_forward(args, lse_only)?;
    launch_forward(args, input)
}

fn prepare_forward(args: &[AnyView], lse_only: bool) -> Result<PreparedForward> {
    if !matches!(args.len(), 9 | 10 | 12 | 13 | 14 | 16 | 18) {
        return Err(err(
            VALUE_ERROR,
            &format!(
                "fwd expects 9 positional arguments (q, k, v, alibi_slopes, softmax_scale, \
                 is_causal, window_size_left, window_size_right, softcap), optionally followed by p_dropout, seed, offset, and/or an output tensor; got {}",
                args.len()
            ),
        ));
    }
    let q: Tensor = arg(args, 0, "q")?;
    let k: Tensor = arg(args, 1, "k")?;
    let v: Tensor = arg(args, 2, "v")?;
    let alibi: Option<Tensor> = arg(args, 3, "alibi_slopes")?;
    let softmax_scale: Option<f64> = arg(args, 4, "softmax_scale")?;
    let is_causal: bool = arg(args, 5, "is_causal")?;
    let window_size_left: i64 = arg(args, 6, "window_size_left")?;
    let window_size_right: i64 = arg(args, 7, "window_size_right")?;
    let softcap: f64 = arg(args, 8, "softcap")?;

    let p_dropout: f64 = if args.len() >= 12 {
        arg(args, 9, "p_dropout")?
    } else {
        0.0
    };
    if !p_dropout.is_finite() || !(0.0..1.0).contains(&p_dropout) || p_dropout as f32 >= 1.0 {
        return Err(err(
            VALUE_ERROR,
            "p_dropout must be finite and in [0, 1) as float32",
        ));
    }
    let dropout = (p_dropout as f32) > 0.0;
    if dropout && softcap > 0.0 {
        return Err(err(
            VALUE_ERROR,
            "softcap with dropout is not supported by FlashAttention v2",
        ));
    }
    let rng_state = if args.len() >= 12 {
        TryFromTemp::<Tensor>::try_from(args[10])
            .ok()
            .map(TryFromTemp::into_value)
    } else {
        None
    };
    let dropout_seed = if let Some(state) = &rng_state {
        state.data_ptr() as u64
    } else if args.len() >= 12 {
        arg::<i64>(args, 10, "seed")? as u64
    } else {
        0
    };
    let mut dropout_offset = if args.len() >= 12 {
        arg::<i64>(args, 11, "offset")? as u64
    } else {
        0
    };
    if let Some(state) = &rng_state {
        if state.device().device_type != DLDeviceType::kDLCUDA
            || state.device().device_id != q.device().device_id
            || state.dtype() != DLDataType::new(DLDataTypeCode::kDLInt, 64, 1)
            || state.shape() != [2]
            || !state.is_contiguous()
            || (state.data_ptr() as u64) % 8 != 0
        {
            return Err(err(
                VALUE_ERROR,
                "rng_state must be a contiguous CUDA int64 tensor of shape [2] on q's device",
            ));
        }
        if dropout_offset != 0 {
            return Err(err(
                VALUE_ERROR,
                "offset must be zero when seed is an RNG-state tensor",
            ));
        }
        // Device-state export passes pointer/length, not seed/offset.
        dropout_offset = 2;
    }
    let native_reservation = args.len() == 16;
    let return_softmax =
        matches!(args.len(), 14 | 16 | 18) && arg::<bool>(args, 13, "return_softmax")?;
    if native_reservation && (!dropout || rng_state.is_none()) {
        return Err(err(
            VALUE_ERROR,
            "native reservation requires dropout and a state tensor",
        ));
    }
    let descriptor_mode = args.len() == 18
        || (native_reservation
            && !return_softmax
            && q.ndim() == 4
            && k.ndim() == 4
            && q.shape()[3] == 64
            && (1..=128).contains(&q.shape()[1])
            && (1..=128).contains(&k.shape()[1]));
    let mut descriptor = [0u64; 3];
    let mut descriptor_captured = 0u32;
    if descriptor_mode {
        if !native_reservation {
            for (i, value) in descriptor.iter_mut().enumerate() {
                *value = arg::<i64>(args, 14 + i, "RNG descriptor")? as u64;
            }
            descriptor_captured = arg::<bool>(args, 17, "captured")? as u32;
        }
        if !dropout
            || q.ndim() != 4
            || k.ndim() != 4
            || rng_state.is_none()
            || q.shape()[3] != 64
            || q.shape()[1] == 0
            || k.shape()[1] == 0
        {
            return Err(err(
                VALUE_ERROR,
                "RNG descriptor requires nonempty D64 dropout",
            ));
        }
        if descriptor_captured != 0
            && (descriptor[0] == 0
                || descriptor[1] == 0
                || descriptor[0] % 8 != 0
                || descriptor[1] % 8 != 0)
        {
            return Err(err(VALUE_ERROR, "invalid capture descriptor pointers"));
        }
    }

    if lse_only && return_softmax {
        return Err(err(
            VALUE_ERROR,
            "LSE-only forward cannot return diagnostic probabilities",
        ));
    }
    if descriptor_mode && return_softmax {
        return Err(err(
            VALUE_ERROR,
            "RNG descriptor does not support diagnostic probabilities",
        ));
    }
    if return_softmax && (!dropout || rng_state.is_none()) {
        return Err(err(
            VALUE_ERROR,
            "return_softmax requires nonzero dropout and a CUDA RNG-state tensor",
        ));
    }
    let dropout_keep = 1.0f32 - p_dropout as f32;

    let (q, q_bf16, q_str) = prepare_qkv("q", q)?;
    let (k, k_bf16, k_str) = prepare_qkv("k", k)?;
    let (v, v_bf16, v_str) = prepare_qkv("v", v)?;
    if k_bf16 != q_bf16 || v_bf16 != q_bf16 {
        return Err(err(
            TYPE_ERROR,
            "q, k, v must share dtype (all f16 or all bf16)",
        ));
    }
    let is_bf16 = q_bf16;

    let b = q.shape()[0];
    let seqlen_q = q.shape()[1];
    let h = q.shape()[2];
    let d = q.shape()[3];
    let seqlen_k = k.shape()[1];
    let hk = k.shape()[2];
    if k.shape()[0] != b || k.shape()[3] != d {
        return Err(err(
            VALUE_ERROR,
            "k must be [B, Sk, Hk, D] with q's batch and head dim",
        ));
    }
    if v.shape() != k.shape() {
        return Err(err(VALUE_ERROR, "v must have k's shape [B, Sk, Hk, D]"));
    }
    if seqlen_q < 0 || seqlen_k < 0 {
        return Err(err(
            VALUE_ERROR,
            &format!("seqlens must be non-negative, got q={seqlen_q}, k={seqlen_k}"),
        ));
    }
    if d > HEAD_DIMS[HEAD_DIMS.len() - 1] {
        return Err(err(
            VALUE_ERROR,
            &format!(
                "head dim must be <= {}, got {d}",
                HEAD_DIMS[HEAD_DIMS.len() - 1]
            ),
        ));
    }
    if d < 8 || d % 8 != 0 {
        return Err(err(
            VALUE_ERROR,
            &format!("head_size must be a positive multiple of 8 (like the reference), got {d}"),
        ));
    }
    if hk <= 0 || h % hk != 0 {
        return Err(err(
            VALUE_ERROR,
            &format!("num_heads ({h}) must be a multiple of num_heads_k ({hk})"),
        ));
    }
    let bh = b * h;
    if b <= 0 || h <= 0 || b > MAX_GRID_YZ || h > MAX_GRID_YZ {
        return Err(err(
            VALUE_ERROR,
            &format!("batch and heads must each be in 1..={MAX_GRID_YZ}, got {b} and {h}"),
        ));
    }
    let device = q.device();
    for (name, t) in [("k", &k), ("v", &v)] {
        if t.device().device_id != device.device_id {
            return Err(err(
                VALUE_ERROR,
                &format!("{name} must be on the same device as q"),
            ));
        }
    }

    let (has_alibi, alibi_data, alibi_numel, alibi_bs) = match &alibi {
        Some(t) => {
            let dt = t.dtype();
            if t.device().device_type != DLDeviceType::kDLCUDA
                || t.device().device_id != device.device_id
            {
                return Err(err(
                    TYPE_ERROR,
                    "alibi_slopes must be a CUDA tensor on q's device",
                ));
            }
            if dt.code != DLDataTypeCode::kDLFloat as u8 || dt.bits != 32 || dt.lanes != 1 {
                return Err(err(TYPE_ERROR, "alibi_slopes must be float32"));
            }
            if !t.is_contiguous() {
                return Err(err(VALUE_ERROR, "alibi_slopes must be contiguous"));
            }
            let bs = match *t.shape() {
                [n] if n == h => 0i64,
                [bb, n] if bb == b && n == h => h,
                _ => {
                    return Err(err(
                        VALUE_ERROR,
                        &format!("alibi_slopes must be [{h}] or [{b}, {h}]"),
                    ));
                }
            };
            (1u32, t.data_ptr() as u64, t.numel() as u64, bs as u32)
        }
        None => (0u32, q.data_ptr() as u64, 0u64, 0u32),
    };

    let scale = softmax_scale.unwrap_or(1.0 / (d as f64).sqrt());
    // tanh is odd: keep scale positive (0 = no softcap), sign in multiplier.
    let (scale_eff, tanh_scale) = if softcap > 0.0 && scale != 0.0 {
        (softcap.copysign(scale), scale.abs() / softcap)
    } else {
        (scale, 0.0)
    };
    let mut wl = window_size_left;
    let mut wr = window_size_right;
    if wl >= seqlen_k {
        wl = -1;
    }
    if wr >= seqlen_k {
        wr = -1;
    }
    if is_causal {
        wr = 0;
    }
    let alibi_causal = (wl < 0 && wr == 0) as u32;

    if wl < 0 && wr >= 0 {
        wl = seqlen_k;
    }
    if wl >= 0 && wr < 0 {
        wr = seqlen_k;
    }

    let supplied_out: Option<Tensor> = if matches!(args.len(), 10 | 13 | 14 | 16 | 18) {
        arg(
            args,
            if matches!(args.len(), 14 | 16 | 18) {
                12
            } else {
                args.len() - 1
            },
            "out",
        )?
    } else {
        None
    };
    if lse_only && supplied_out.is_none() {
        return Err(err(
            VALUE_ERROR,
            "LSE-only forward requires a supplied output",
        ));
    }
    if let Some(dst) = &supplied_out {
        if dst.shape() != q.shape()
            || dst.dtype() != q.dtype()
            || dst.device().device_type != device.device_type
            || dst.device().device_id != device.device_id
        {
            return Err(err(
                VALUE_ERROR,
                "out must match q's shape, dtype and device",
            ));
        }
        if dst.strides()[3] != 1 || !nonoverlapping_layout(dst) {
            return Err(err(
                VALUE_ERROR,
                "out must have unit last stride and nonoverlapping non-negative strides; unsupported writable layout: elements overlap or the bounded safety checks cannot prove they are disjoint. Use a separately allocated contiguous output.",
            ));
        }
    }
    let copy_out = supplied_out.as_ref().is_some_and(|dst| {
        !dst.is_contiguous()
            || (dst.data_ptr() as u64) % 16 != 0
            || [&q, &k, &v].iter().any(|src| overlaps(dst, src))
    });
    let out = match &supplied_out {
        Some(dst) if !copy_out => dst.clone(),
        _ => alloc_tensor(&[b, seqlen_q, h, d], q.dtype(), device)?,
    };
    let lse = alloc_tensor(
        &[b, h, seqlen_q],
        DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1),
        device,
    )?;

    let probabilities = if !lse_only && matches!(args.len(), 14 | 16 | 18) {
        Some(if return_softmax {
            alloc_tensor(
                &[
                    b,
                    h,
                    (seqlen_q + 127) / 128 * 128,
                    (seqlen_k + 127) / 128 * 128,
                ],
                q.dtype(),
                device,
            )?
        } else {
            alloc_tensor(&[0], q.dtype(), device)?
        })
    } else {
        None
    };
    let bucket = *HEAD_DIMS.iter().find(|&&x| x >= d).unwrap();
    Ok(PreparedForward {
        q,
        k,
        v,
        _alibi: alibi,
        q_str,
        k_str,
        v_str,
        is_bf16,
        b,
        h,
        d,
        seqlen_q,
        seqlen_k,
        hk,
        bh,
        device,
        has_alibi,
        alibi_data,
        alibi_numel,
        alibi_bs,
        scale,
        scale_eff,
        tanh_scale,
        softcap,
        wl,
        wr,
        alibi_causal,
        supplied_out,
        copy_out,
        out,
        lse,
        probabilities,
        rng_state,
        dropout,
        dropout_seed,
        dropout_offset,
        dropout_keep,
        native_reservation,
        return_softmax,
        descriptor_mode,
        descriptor,
        descriptor_captured,
        bucket,
        lse_only,
    })
}

fn launch_forward(args: &[AnyView], input: PreparedForward) -> Result<Any> {
    let PreparedForward {
        ref q,
        ref k,
        ref v,
        q_str,
        k_str,
        v_str,
        is_bf16,
        b,
        h,
        d,
        seqlen_q,
        seqlen_k,
        hk,
        bh,
        device,
        has_alibi,
        alibi_data,
        alibi_numel,
        alibi_bs,
        scale_eff,
        tanh_scale,
        wl,
        wr,
        alibi_causal,
        copy_out,
        ref out,
        ref lse,
        ref probabilities,
        ref rng_state,
        dropout,
        mut dropout_seed,
        mut dropout_offset,
        mut dropout_keep,
        native_reservation,
        return_softmax,
        descriptor_mode,
        mut descriptor,
        mut descriptor_captured,
        bucket,
        lse_only,
        ..
    } = input;
    let rt = runtime(device.device_id)?;
    if native_reservation {
        let reserve: Function = arg(args, 14, "reserve")?;
        let generator: i64 = arg(args, 15, "generator")?;
        let reserved = reserve.call_tuple((generator, device.device_id as i64, b * h * 32))?;
        let values = reserved
            .try_as::<Shape>()
            .ok_or_else(|| err(TYPE_ERROR, "reserve must return a Shape descriptor"))?;
        if values.len() != 4 || !(0..=1).contains(&values[3]) {
            return Err(err(VALUE_ERROR, "invalid reserved RNG descriptor"));
        }
        for (value, raw) in descriptor.iter_mut().zip(values.as_slice()) {
            *value = *raw as u64;
        }
        descriptor_captured = values[3] as u32;
        if descriptor_captured != 0
            && (descriptor[0] == 0
                || descriptor[1] == 0
                || descriptor[0] % 8 != 0
                || descriptor[1] % 8 != 0)
        {
            return Err(err(VALUE_ERROR, "invalid reserved capture pointers"));
        }
        if !descriptor_mode {
            resolve_rng_descriptor(
                rt,
                rng_state.as_ref().unwrap(),
                descriptor,
                descriptor_captured,
            )?;
        }
    }

    if return_softmax && probabilities.as_ref().unwrap().numel() > 0 {
        rt.ctx
            .bind_to_thread()
            .map_err(|e| err(RUNTIME_ERROR, &format!("bind probability context: {e:?}")))?;
        let stream = unsafe { TVMFFIEnvGetStream(DLDeviceType::kDLCUDA as i32, device.device_id) };
        let p = probabilities.as_ref().unwrap();
        unsafe {
            sys::cuMemsetD8Async(
                p.data_ptr() as u64,
                0,
                (p.numel() * 2) as usize,
                stream.cast(),
            )
        }
        .result()
        .map_err(|e| {
            err(
                RUNTIME_ERROR,
                &format!("clear diagnostic probabilities: {e:?}"),
            )
        })?;
    }
    if seqlen_q == 0 || seqlen_k == 0 {
        if seqlen_q > 0 {
            rt.ctx
                .bind_to_thread()
                .map_err(|e| err(RUNTIME_ERROR, &format!("bind context: {e:?}")))?;
            let stream =
                unsafe { TVMFFIEnvGetStream(DLDeviceType::kDLCUDA as i32, device.device_id) };
            unsafe {
                sys::cuMemsetD8Async(
                    out.data_ptr() as u64,
                    0,
                    (out.numel() * 2) as usize,
                    stream.cast(),
                )
                .result()
                .map_err(|e| err(RUNTIME_ERROR, &format!("clear empty-key output: {e:?}")))?;
                sys::cuMemsetD32Async(
                    lse.data_ptr() as u64,
                    f32::INFINITY.to_bits(),
                    lse.numel() as usize,
                    stream.cast(),
                )
                .result()
                .map_err(|e| err(RUNTIME_ERROR, &format!("fill empty-key LSE: {e:?}")))?;
            }
        }
        return finish_fwd(
            input.out,
            input.lse,
            input.supplied_out,
            copy_out,
            input.probabilities,
            lse_only,
        );
    }

    let mut plan = plan_forward(&input, rt);
    let ForwardPlan {
        tile: _,
        short_d32_alt: _,
        cross_split,
        groups,
        wide_queries: _,
        splits,
        wide_keys: _,
    } = plan;
    let partial = if splits > 1 {
        Some(alloc_tensor(
            &[b * splits, seqlen_q, h, d],
            lse.dtype(),
            device,
        )?)
    } else {
        None
    };
    let partial_lse = if splits > 1 {
        Some(alloc_tensor(
            &[b * splits, h, seqlen_q],
            lse.dtype(),
            device,
        )?)
    } else {
        None
    };
    let stream = unsafe { TVMFFIEnvGetStream(DLDeviceType::kDLCUDA as i32, device.device_id) };

    let hs_u32 = (d / 2) as u64;
    let extent = |shape: &[i64], s: &Strides| -> u64 {
        (shape[0] - 1) as u64 * s.batch as u64
            + (shape[1] - 1) as u64 * s.seq as u64
            + (shape[2] - 1) as u64 * s.head as u64
            + hs_u32
    };
    let mut q_ptr = q.data_ptr() as u64;
    let mut q_len = extent(q.shape(), &q_str);
    let mut k_ptr = k.data_ptr() as u64;
    let mut k_len = extent(k.shape(), &k_str);
    let mut v_ptr = v.data_ptr() as u64;
    let mut v_len = extent(v.shape(), &v_str);
    let mut o_ptr = partial.as_ref().unwrap_or(out).data_ptr() as u64;
    let mut o_len = (b * splits * seqlen_q * h) as u64 * hs_u32 * if splits > 1 { 2 } else { 1 };
    let mut lse_ptr = partial_lse.as_ref().unwrap_or(lse).data_ptr() as u64;
    let mut lse_len = (b * splits * h * seqlen_q) as u64;
    let mut alibi_ptr = alibi_data;
    let mut alibi_len = alibi_numel;
    let mut scale_eff = scale_eff as f32;
    let mut tanh_scale = tanh_scale as f32;
    let mut window_left = if splits > 1 { -1 } else { wl as i32 };
    let mut window_right = if splits > 1 { -1 } else { wr as i32 };
    let mut seqlen_q_u = (seqlen_q * groups) as u32;
    let mut seqlen_k_u = seqlen_k as u32;
    let mut has_alibi = has_alibi;
    let mut alibi_causal = alibi_causal;
    let mut h_u = (h / groups) as u32;
    let mut gqa = (h / hk / groups) as u32;
    let mut hs_u32_u = hs_u32 as u32;
    let mut q_bs = q_str.batch;
    let mut q_ss = if groups > 1 { q_str.head } else { q_str.seq };
    let mut q_hs = q_str.head * groups as u32;
    let mut k_bs = k_str.batch;
    let mut k_ss = k_str.seq;
    let mut k_hs = k_str.head;
    let mut v_bs = v_str.batch;
    let mut v_ss = v_str.seq;
    let mut v_hs = v_str.head;
    let mut alibi_bs = alibi_bs;

    let mut split_count = splits as u32;
    let mut kparams = vec![
        (&mut q_ptr as *mut u64).cast::<c_void>(),
        (&mut q_len as *mut u64).cast(),
        (&mut k_ptr as *mut u64).cast(),
        (&mut k_len as *mut u64).cast(),
        (&mut v_ptr as *mut u64).cast(),
        (&mut v_len as *mut u64).cast(),
        (&mut o_ptr as *mut u64).cast(),
        (&mut o_len as *mut u64).cast(),
        (&mut lse_ptr as *mut u64).cast(),
        (&mut lse_len as *mut u64).cast(),
        (&mut alibi_ptr as *mut u64).cast(),
        (&mut alibi_len as *mut u64).cast(),
        (&mut scale_eff as *mut f32).cast(),
        (&mut tanh_scale as *mut f32).cast(),
        (&mut window_left as *mut i32).cast(),
        (&mut window_right as *mut i32).cast(),
        (&mut seqlen_q_u as *mut u32).cast(),
        (&mut seqlen_k_u as *mut u32).cast(),
        (&mut has_alibi as *mut u32).cast(),
        (&mut alibi_causal as *mut u32).cast(),
        (&mut h_u as *mut u32).cast(),
        (&mut gqa as *mut u32).cast(),
        (&mut hs_u32_u as *mut u32).cast(),
        (&mut q_bs as *mut u32).cast(),
        (&mut q_ss as *mut u32).cast(),
        (&mut q_hs as *mut u32).cast(),
        (&mut k_bs as *mut u32).cast(),
        (&mut k_ss as *mut u32).cast(),
        (&mut k_hs as *mut u32).cast(),
        (&mut v_bs as *mut u32).cast(),
        (&mut v_ss as *mut u32).cast(),
        (&mut v_hs as *mut u32).cast(),
        (&mut alibi_bs as *mut u32).cast(),
    ];

    let mut probabilities_ptr = probabilities
        .as_ref()
        .map_or(EMPTY_SLICE[0], |p| p.data_ptr() as u64);
    let mut probabilities_len = probabilities.as_ref().map_or(0, |p| p.numel() as u64 / 2);
    if dropout {
        kparams.push((&mut dropout_seed as *mut u64).cast());
        kparams.push((&mut dropout_offset as *mut u64).cast());
        kparams.push((&mut dropout_keep as *mut f32).cast());
        if descriptor_mode {
            for value in &mut descriptor {
                kparams.push((value as *mut u64).cast());
            }
            kparams.push((&mut descriptor_captured as *mut u32).cast());
        }
        if return_softmax {
            kparams.push((&mut probabilities_ptr as *mut u64).cast());
            kparams.push((&mut probabilities_len as *mut u64).cast());
        }
    }
    if splits > 1 {
        kparams.push((&mut split_count as *mut u32).cast());
    }
    let (flash_fn, dense192_selected) = select_kernel(&input, &mut plan, rt)?;
    let tile = plan.tile;
    rt.ctx
        .bind_to_thread()
        .map_err(|e| err(RUNTIME_ERROR, &format!("bind context: {e:?}")))?;
    let q_blocks = ((seqlen_q * groups + tile.rows - 1) / tile.rows) as u32;
    let head_first = splits == 1 && wr == 0 && wl >= seqlen_k && q_blocks > 1 && q_blocks <= 65535;
    let grid = if head_first {
        (bh as u32, q_blocks, 1)
    } else {
        (q_blocks, (h / groups) as u32, (b * splits) as u32)
    };
    unsafe {
        launch_kernel(
            flash_fn.cu_function(),
            grid,
            (tile.threads, 1, 1),
            if dense192_selected {
                98304
            } else if splits > 1 {
                split_smem_bytes(bucket, tile)
            } else {
                flash_smem_bytes(bucket, tile)
            },
            stream.cast(),
            &mut kparams,
        )
    }
    .map_err(|e| err(RUNTIME_ERROR, &format!("launch flash_attn fwd: {e:?}")))?;

    if splits > 1 {
        let merge4 = cross_split && bucket <= 64;
        let mut final_o = out.data_ptr() as u64;
        let mut final_sq = seqlen_q as u32;
        let mut final_h = h as u32;
        let mut merge_groups = groups as u32;
        let mut final_lse = lse.data_ptr() as u64;
        let mut final_o_len = o_len / (splits as u64 * 2);
        let mut final_lse_len = lse_len / splits as u64;
        let mut merge_params = [
            (&mut o_ptr as *mut u64).cast::<c_void>(),
            (&mut o_len as *mut u64).cast(),
            (&mut lse_ptr as *mut u64).cast(),
            (&mut lse_len as *mut u64).cast(),
            (&mut final_o as *mut u64).cast(),
            (&mut final_o_len as *mut u64).cast(),
            (&mut final_lse as *mut u64).cast(),
            (&mut final_lse_len as *mut u64).cast(),
            (&mut final_sq as *mut u32).cast(),
            (&mut final_h as *mut u32).cast(),
            (&mut hs_u32_u as *mut u32).cast(),
            (&mut split_count as *mut u32).cast(),
            (&mut merge_groups as *mut u32).cast(),
        ];
        unsafe {
            launch_kernel(
                (if merge4 {
                    &rt.merge4_fns
                } else {
                    &rt.merge_fns
                })
                .get(&is_bf16)
                .unwrap()
                .cu_function(),
                (
                    ((b * h * seqlen_q + if merge4 { 3 } else { 0 }) / if merge4 { 4 } else { 1 })
                        as u32,
                    1,
                    1,
                ),
                (
                    if merge4 {
                        128
                    } else if cross_split && bucket <= 64 {
                        32
                    } else if cross_split && bucket <= 128 {
                        64
                    } else {
                        128
                    },
                    1,
                    1,
                ),
                if merge4 { 128 * 4 } else { 32 * 4 },
                stream.cast(),
                &mut merge_params,
            )
        }
        .map_err(|e| err(RUNTIME_ERROR, &format!("launch split merge: {e:?}")))?;
    }

    finish_fwd(
        input.out,
        input.lse,
        input.supplied_out,
        copy_out,
        input.probabilities,
        lse_only,
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tvm_ffi_fwd(
    _handle: *mut c_void,
    args: *const tvm_ffi_sys::TVMFFIAny,
    num_args: i32,
    result: *mut tvm_ffi_sys::TVMFFIAny,
) -> i32 {
    let packed = unsafe { std::slice::from_raw_parts(args as *const AnyView, num_args as usize) };
    match fwd(packed) {
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
pub unsafe extern "C" fn __tvm_ffi_fwd_lse(
    _handle: *mut c_void,
    args: *const tvm_ffi_sys::TVMFFIAny,
    num_args: i32,
    result: *mut tvm_ffi_sys::TVMFFIAny,
) -> i32 {
    let packed = unsafe { std::slice::from_raw_parts(args as *const AnyView, num_args as usize) };
    match fwd_impl(packed, true) {
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
