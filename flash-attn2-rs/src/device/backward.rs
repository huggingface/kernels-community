use crate::device::config::{FwdConfig, GRID_PLAIN, KvRange, LaunchTile, Strides, TensorBases};
use crate::device::dropout::Dropout;
use crate::device::forward::{LOG2_E, SmemLayout, accumulate_values, compute_qk};
use crate::device::inlined::{cvt, fast_exp2, ldmatrix_x4_shared, shared_address, tanh_approx};
use cuda_device::async_copy::cp_async_ca_zfill_16;
use cuda_device::shared::DynamicSharedArray;
use cuda_device::warp::shuffle_xor_f32;
use cuda_device::{DisjointSlice, kernel, launch_bounds, ptx_asm, thread};

mod common;
mod dkv;
mod dq;
mod fused;
mod reduce;

pub(crate) use common::{
    copy_rows, probability_gradient, query_fragments, transposed_keep, unpack,
};
use common::{copy_rows_swizzled, query_fragments_swizzled};
pub(crate) use common::{transposed_paired_words, transposed_word_mask};

use dkv::dkv_both_core;
use fused::{fused_dq_tile, fused_dq_tile_d64, fused_store_ds};
pub(crate) use reduce::{ReduceSpec, bf16_8, bf16_16, bf16_32, f16_8, f16_16, f16_32};
