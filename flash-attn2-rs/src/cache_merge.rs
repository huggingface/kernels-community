use std::ffi::c_void;

use tvm_ffi::error::{Error, RUNTIME_ERROR, Result, VALUE_ERROR};
use tvm_ffi::{Any, AnyView, Function, Tensor};
use tvm_ffi_sys::dlpack::{DLDataType, DLDataTypeCode, DLDeviceType};
use tvm_ffi_sys::{TVMFFIAny, TVMFFIEnvGetStream};

use crate::cuda::launch_kernel;
use crate::runtime::runtime;
use crate::{EMPTY_SLICE, alloc_tensor, any_into_raw_owned, arg, err};

// Partial layout [B,splits,H,Sq,D], with [B,splits,H,Sq] natural-log LSE.
pub(super) fn merge(partial: &Tensor, stats: &Tensor, bf16: bool) -> Result<(Tensor, Tensor)> {
    merge_layout(partial, stats, bf16, false)
}

pub(super) fn merge_layout(
    partial: &Tensor,
    stats: &Tensor,
    bf16: bool,
    grouped: bool,
) -> Result<(Tensor, Tensor)> {
    merge_impl(partial, stats, bf16, grouped, false, None, false)
}

pub(super) fn merge_packed_decode(
    partial: &Tensor,
    stats: &Tensor,
    bf16: bool,
    grouped: bool,
    rank_three: bool,
) -> Result<(Tensor, Tensor)> {
    merge_impl(partial, stats, bf16, grouped, true, None, rank_three)
}

pub(super) fn merge_varlen(
    partial: &Tensor,
    stats: &Tensor,
    bf16: bool,
    cuq: &Tensor,
    total_q: i64,
    rank_three: bool,
) -> Result<(Tensor, Tensor)> {
    merge_impl(
        partial,
        stats,
        bf16,
        false,
        false,
        Some((cuq, total_q)),
        rank_three,
    )
}

fn merge_impl(
    partial: &Tensor,
    stats: &Tensor,
    bf16: bool,
    grouped: bool,
    packed: bool,
    varlen: Option<(&Tensor, i64)>,
    rank_three: bool,
) -> Result<(Tensor, Tensor)> {
    let float32 = DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1);
    if partial.ndim() != 5
        || partial.device().device_type != DLDeviceType::kDLCUDA
        || partial.dtype() != float32
        || stats.dtype() != float32
        || stats.device() != partial.device()
        || stats.shape() != &partial.shape()[..4]
        || (partial.numel() != 0 && !partial.is_contiguous())
        || (stats.numel() != 0 && !stats.is_contiguous())
        || partial.data_ptr() as u64 % 4 != 0
        || stats.data_ptr() as u64 % 4 != 0
    {
        return Err(err(
            VALUE_ERROR,
            "cache merge requires contiguous CUDA float32 [B,splits,H,Sq,D] partials and matching LSE",
        ));
    }
    let [b, splits, h, sq, d] = *partial.shape() else {
        unreachable!()
    };
    if b <= 0
        || h <= 0
        || sq < 0
        || !(1..=128).contains(&splits)
        || !(8..=256).contains(&d)
        || d % 8 != 0
        || (packed && (sq == 0 || (!grouped && sq != 1)))
    {
        return Err(err(VALUE_ERROR, "unsupported cached merge dimensions"));
    }
    if let Some((cuq, total_q)) = varlen {
        crate::varlen::metadata("cu_seqlens_q", cuq, &[b + 1], partial.device())?;
        if sq <= 0 || total_q < 0 || total_q > i32::MAX as i64 {
            return Err(err(VALUE_ERROR, "unsupported packed merge dimensions"));
        }
    }
    let rows = stats.numel() / splits as usize;
    if rows > i32::MAX as usize {
        return Err(err(VALUE_ERROR, "too many cached merge rows"));
    }
    let dtype = DLDataType::new(
        if bf16 {
            DLDataTypeCode::kDLBfloat
        } else {
            DLDataTypeCode::kDLFloat
        },
        16,
        1,
    );
    let shape = if let Some((_, total_q)) = varlen {
        [1, total_q, h, d]
    } else if packed {
        [1, b, h * sq, d]
    } else if grouped {
        [b, h, sq, d]
    } else {
        [b, sq, h, d]
    };
    // Rank-four native callers keep the singleton batch.
    let output_shape = if rank_three { &shape[1..] } else { &shape[..] };
    let out = alloc_tensor(output_shape, dtype, partial.device())?;
    let lse = alloc_tensor(
        if let Some((_, total_q)) = varlen {
            vec![h, total_q]
        } else if packed {
            vec![h * sq, b]
        } else {
            vec![b, h, sq]
        }
        .as_slice(),
        float32,
        partial.device(),
    )?;
    if rows > 0 {
        let rt = runtime(partial.device().device_id)?;
        rt.ctx
            .bind_to_thread()
            .map_err(|e| err(RUNTIME_ERROR, &format!("bind merge context: {e:?}")))?;
        let stream =
            unsafe { TVMFFIEnvGetStream(DLDeviceType::kDLCUDA as i32, partial.device().device_id) };
        let mut buffers = [
            partial.data_ptr() as u64,
            partial.numel() as u64,
            stats.data_ptr() as u64,
            stats.numel() as u64,
            out.data_ptr() as u64,
            out.numel() as u64 / 2,
            lse.data_ptr() as u64,
            lse.numel() as u64,
        ];
        let mut config = [sq as u32, h as u32, d as u32, splits as u32];
        let mut params: Vec<*mut c_void> =
            buffers.iter_mut().map(|x| (x as *mut u64).cast()).collect();
        params.extend(config.iter_mut().map(|x| (x as *mut u32).cast()));
        let mut cuq_buffer = varlen.map_or(EMPTY_SLICE, |(cuq, _)| {
            [cuq.data_ptr() as u64, cuq.numel() as u64]
        });
        let mut total_q = varlen.map_or(0, |(_, total_q)| total_q as u32);
        let function = if varlen.is_some() {
            params.extend(cuq_buffer.iter_mut().map(|x| (x as *mut u64).cast()));
            params.push((&mut total_q as *mut u32).cast());
            rt.varlen_merge_fns.get(&bf16).unwrap()
        } else {
            rt.cache_merge_fns.get(&(bf16, grouped, packed)).unwrap()
        };
        unsafe {
            launch_kernel(
                function.cu_function(),
                (rows as u32, 1, 1),
                (128, 1, 1),
                136 * 4,
                stream.cast(),
                &mut params,
            )
        }
        .map_err(|e| err(RUNTIME_ERROR, &format!("launch cache merge: {e:?}")))?;
    }
    Ok((out, lse))
}

fn op(args: &[AnyView]) -> Result<Any> {
    if args.len() != 3 {
        return Err(err(VALUE_ERROR, "cache_merge expects partials, LSE, bf16"));
    }
    let partial: Tensor = arg(args, 0, "partial")?;
    let stats: Tensor = arg(args, 1, "LSE")?;
    let bf16: bool = arg(args, 2, "bf16")?;
    let (out, lse) = merge(&partial, &stats, bf16)?;
    Function::get_global("ffi.Array")?.call_packed(&[AnyView::from(&out), AnyView::from(&lse)])
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tvm_ffi_cache_merge_explicit(
    _handle: *mut c_void,
    args: *const TVMFFIAny,
    num_args: i32,
    result: *mut TVMFFIAny,
) -> i32 {
    let packed = unsafe { std::slice::from_raw_parts(args as *const AnyView, num_args as usize) };
    match op(packed) {
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
