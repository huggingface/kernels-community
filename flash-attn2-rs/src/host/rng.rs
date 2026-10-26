use std::ffi::c_void;

use tvm_ffi::error::{Error, RUNTIME_ERROR, Result, VALUE_ERROR};
use tvm_ffi::{Any, AnyView, Tensor};
use tvm_ffi_sys::TVMFFIEnvGetStream;
use tvm_ffi_sys::dlpack::{DLDataType, DLDataTypeCode, DLDeviceType};

use crate::host::cuda::launch_kernel;
use crate::host::runtime::{Runtime, runtime};
use crate::host::{alloc_tensor, any_into_raw_owned, arg, err};

// Philox values may be CUDA pointers during graph capture.
pub(super) fn resolve_rng_descriptor(
    rt: &Runtime,
    state: &Tensor,
    descriptor: [u64; 3],
    mut captured: u32,
) -> Result<()> {
    let device = state.device();
    let [mut seed, mut offset, mut intragraph] = descriptor;
    rt.ctx
        .bind_to_thread()
        .map_err(|e| err(RUNTIME_ERROR, &format!("bind RNG context: {e:?}")))?;
    let stream = unsafe { TVMFFIEnvGetStream(DLDeviceType::kDLCUDA as i32, device.device_id) };
    let mut data = state.data_ptr() as u64;
    let mut len = 2u64;
    let mut params = [
        (&mut data as *mut u64).cast::<c_void>(),
        (&mut len as *mut u64).cast(),
        (&mut seed as *mut u64).cast(),
        (&mut offset as *mut u64).cast(),
        (&mut intragraph as *mut u64).cast(),
        (&mut captured as *mut u32).cast(),
    ];
    unsafe {
        launch_kernel(
            rt.resolve_rng.cu_function(),
            (1, 1, 1),
            (1, 1, 1),
            0,
            stream.cast(),
            &mut params,
        )
    }
    .map_err(|e| err(RUNTIME_ERROR, &format!("resolve RNG state: {e:?}")))?;
    Ok(())
}

fn rng_state(args: &[AnyView]) -> Result<Any> {
    if args.len() != 5 {
        return Err(err(
            VALUE_ERROR,
            "rng_state expects reference, seed, offset, intragraph, captured",
        ));
    }
    let reference: Tensor = arg(args, 0, "reference")?;
    let device = reference.device();
    if device.device_type != DLDeviceType::kDLCUDA {
        return Err(err(
            VALUE_ERROR,
            "RNG state requires a CUDA reference tensor",
        ));
    }
    let seed = arg::<i64>(args, 1, "seed")? as u64;
    let offset = arg::<i64>(args, 2, "offset")? as u64;
    let intragraph = arg::<i64>(args, 3, "intragraph")? as u64;
    let captured = arg::<bool>(args, 4, "captured")? as u32;
    if captured != 0 && (seed == 0 || offset == 0 || seed % 8 != 0 || offset % 8 != 0) {
        return Err(err(
            VALUE_ERROR,
            "captured RNG descriptor requires aligned non-null CUDA pointers",
        ));
    }
    let state = alloc_tensor(&[2], DLDataType::new(DLDataTypeCode::kDLInt, 64, 1), device)?;
    let rt = runtime(device.device_id)?;
    resolve_rng_descriptor(rt, &state, [seed, offset, intragraph], captured)?;
    Ok(Any::from(state))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tvm_ffi_rng_state(
    _handle: *mut c_void,
    args: *const tvm_ffi_sys::TVMFFIAny,
    num_args: i32,
    result: *mut tvm_ffi_sys::TVMFFIAny,
) -> i32 {
    let packed = unsafe { std::slice::from_raw_parts(args as *const AnyView, num_args as usize) };
    match rng_state(packed) {
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
