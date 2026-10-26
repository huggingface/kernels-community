//! Host side: argument validation, kernel selection and launches behind the `__tvm_ffi_*` exports.
//! The glue below is shared by every entry-point module.

mod backward;
mod cache;
mod cuda;
mod forward;
mod layout;
mod rng;
mod runtime;
mod tiles;
mod varlen;

use std::ptr;

use tvm_ffi::any::TryFromTemp;
use tvm_ffi::error::{Error, ErrorKind, RUNTIME_ERROR, Result, TYPE_ERROR};
use tvm_ffi::{Any, AnyCompatible, AnyView, Tensor};
use tvm_ffi_sys::dlpack::{DLDataType, DLDevice, DLTensor};
use tvm_ffi_sys::{TVMFFIAny, TVMFFIAnyDataUnion, TVMFFIObjectHandle, TVMFFITypeIndex};

// Non-null aligned sentinel for empty slices; kernels check the length.
const EMPTY_SLICE: [u64; 2] = [16, 0];

// CUDA limits grid.y and grid.z separately, not their product.
const MAX_GRID_YZ: i64 = 65535;

unsafe extern "C" {
    fn TVMFFIEnvTensorAlloc(prototype: *mut DLTensor, out: *mut TVMFFIObjectHandle) -> i32;
}

unsafe fn any_into_raw_owned(mut value: Any) -> TVMFFIAny {
    let raw = unsafe { *Any::as_data_ptr(&mut value) };
    std::mem::forget(value);
    raw
}

fn err(kind: ErrorKind<'_>, msg: &str) -> Error {
    Error::new(kind, msg, "")
}

fn alloc_tensor(shape: &[i64], dtype: DLDataType, device: DLDevice) -> Result<Tensor> {
    let mut shape_vec: Vec<i64> = shape.to_vec();
    let mut proto = DLTensor {
        data: ptr::null_mut(),
        device,
        ndim: shape_vec.len() as i32,
        dtype,
        shape: shape_vec.as_mut_ptr(),
        strides: ptr::null_mut(),
        byte_offset: 0,
    };
    let mut handle: TVMFFIObjectHandle = ptr::null_mut();
    let ret = unsafe { TVMFFIEnvTensorAlloc(&mut proto, &mut handle) };
    if ret != 0 {
        return Err(Error::from_raised());
    }

    let any = unsafe {
        Any::from_raw_ffi_any(TVMFFIAny {
            type_index: TVMFFITypeIndex::kTVMFFITensor as i32,
            small_str_len: 0,
            data_union: TVMFFIAnyDataUnion {
                v_obj: handle.cast(),
            },
        })
    };
    any.try_as::<Tensor>().ok_or_else(|| {
        err(
            RUNTIME_ERROR,
            "environment tensor allocator returned a non-tensor",
        )
    })
}

fn arg<T: AnyCompatible>(args: &[AnyView], i: usize, name: &str) -> Result<T> {
    TryFromTemp::<T>::try_from(args[i])
        .map(TryFromTemp::into_value)
        .map_err(|e| {
            err(
                TYPE_ERROR,
                &format!("fwd argument '{name}' (#{i}): {}", e.message()),
            )
        })
}
