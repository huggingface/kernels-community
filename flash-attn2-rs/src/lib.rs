//! FlashAttention v2 on CUDA, in Rust. `device` holds the kernels; `host` holds the
//! TVM-FFI entry points that validate arguments, pick a specialization and launch them.

mod device;
mod host;
