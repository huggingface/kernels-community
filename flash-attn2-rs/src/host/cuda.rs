// CUDA driver API via cudarc, loaded at runtime so builds need no CUDA headers.

use std::ffi::CString;
use std::sync::Arc;

pub use cudarc::driver::result::launch_kernel;
use cudarc::driver::result::module;
pub use cudarc::driver::{CudaContext, DriverError, sys};

// kernel-builder links the SASS fatbin between these symbols.
unsafe extern "C" {
    static KERNEL_BUILDER_CUDA_FATBIN: u8;
    static KERNEL_BUILDER_CUDA_FATBIN_END: u8;
}

/// The kernels' fatbin.
pub fn fatbin() -> &'static [u8] {
    let start = &raw const KERNEL_BUILDER_CUDA_FATBIN;
    let end = &raw const KERNEL_BUILDER_CUDA_FATBIN_END;
    unsafe { std::slice::from_raw_parts(start, end.offset_from(start) as usize) }
}

pub struct CudaModule {
    cu_module: sys::CUmodule,
    ctx: Arc<CudaContext>,
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl CudaModule {
    /// Loads a cubin, PTX, or fatbin image into `ctx`.
    pub fn load(ctx: &Arc<CudaContext>, image: &[u8]) -> Result<Arc<Self>, DriverError> {
        ctx.bind_to_thread()?;
        let cu_module = unsafe { module::load_data(image.as_ptr().cast()) }?;
        Ok(Arc::new(Self {
            cu_module,
            ctx: ctx.clone(),
        }))
    }

    pub fn load_function(self: &Arc<Self>, name: &str) -> Result<CudaFunction, DriverError> {
        let name = CString::new(name).expect("kernel names contain no NUL bytes");
        let cu_function = unsafe { module::get_function(self.cu_module, name) }?;
        Ok(CudaFunction {
            cu_function,
            _module: self.clone(),
        })
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        if self.ctx.bind_to_thread().is_ok() {
            let _ = unsafe { module::unload(self.cu_module) };
        }
    }
}

#[derive(Clone)]
pub struct CudaFunction {
    cu_function: sys::CUfunction,
    _module: Arc<CudaModule>,
}

unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}

impl CudaFunction {
    pub fn cu_function(&self) -> sys::CUfunction {
        self.cu_function
    }
}
