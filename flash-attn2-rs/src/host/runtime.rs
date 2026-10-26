//! Per-device CUDA modules and specialization lookup tables.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use tvm_ffi::error::{RUNTIME_ERROR, Result};

use crate::host::cuda::{self, CudaContext, CudaFunction, CudaModule, sys};
use crate::host::err;
use crate::host::tiles::{
    ALTERNATE_BUCKETS, HEAD_DIMS, Tile, alternate_tile, combined_smem, default_tile, dropout_tile,
    flash_kernel_name, flash_smem_bytes, split_smem_bytes, split_tile,
};

pub(super) struct Runtime {
    pub(super) ctx: Arc<CudaContext>,
    pub(super) _module: Arc<CudaModule>,

    pub(super) fns: HashMap<(i64, bool, bool), CudaFunction>,
    pub(super) dropout_fns: HashMap<(i64, bool, u8), CudaFunction>,
    pub(super) copy_layout: CudaFunction,
    pub(super) varlen_merge_fns: HashMap<bool, CudaFunction>,
    pub(super) cache_merge_fns: HashMap<(bool, bool, bool), CudaFunction>,
    pub(super) cache_copy_fns: HashMap<(bool, bool), CudaFunction>,
    pub(super) cache_update_fns: HashMap<bool, CudaFunction>,
    pub(super) resolve_rng: CudaFunction,
    pub(super) bwd_preprocess_fns: HashMap<(bool, u32), CudaFunction>,
    pub(super) bwd_dq_fns: HashMap<(i64, bool), CudaFunction>,
    pub(super) bwd_dq_swizzled_fns: HashMap<(bool, u8), CudaFunction>,
    pub(super) bwd_dq_split_fns: HashMap<(bool, u32), CudaFunction>,
    pub(super) bwd_dq_merge_fns: HashMap<bool, CudaFunction>,
    pub(super) bwd_dkv_both_fns: HashMap<(bool, bool, bool, bool), CudaFunction>,
    pub(super) bwd_fused_fns: HashMap<(bool, bool, bool, u8), CudaFunction>,
    pub(super) bwd_fused_d64_fns: HashMap<(bool, bool, bool, u8), CudaFunction>,
    pub(super) bwd_dkv_fns: HashMap<(i64, bool), CudaFunction>,
    pub(super) bwd_dkv_specialized_fns: HashMap<(bool, bool, u8), CudaFunction>,
    pub(super) bwd_dkv_merge_fns: HashMap<bool, CudaFunction>,
    pub(super) cached_fns: HashMap<(i64, bool, bool), CudaFunction>,
    pub(super) cached_split_fns: HashMap<(i64, bool, bool), CudaFunction>,
    pub(super) varlen_split_fns: HashMap<(i64, bool, bool, bool), CudaFunction>,
    pub(super) paged_fns: HashMap<(i64, bool, bool), CudaFunction>,
    pub(super) varlen_rng_fns: HashMap<bool, CudaFunction>,
    pub(super) varlen_fns: HashMap<(i64, bool, bool), CudaFunction>,
    pub(super) varlen_dq_split_fns: HashMap<(i64, bool), CudaFunction>,
    pub(super) varlen_bwd_causal_fns: HashMap<(bool, bool), CudaFunction>,
    pub(super) varlen_bwd_partial_fns: HashMap<(i64, bool), CudaFunction>,
    pub(super) varlen_bwd_fns: HashMap<(i64, bool, bool), CudaFunction>,
    pub(super) varlen_preprocess_fns: HashMap<(bool, u32), CudaFunction>,
    pub(super) scalar_fns: HashMap<(i64, bool, bool), CudaFunction>,
    pub(super) short_d192_fns: HashMap<(bool, bool), CudaFunction>,
    pub(super) dense_d192_fns: HashMap<bool, CudaFunction>,
    pub(super) raw_d128_fns: HashMap<bool, CudaFunction>,
    pub(super) raw_d32_fns: HashMap<bool, CudaFunction>,
    pub(super) wide_d32_fns: HashMap<bool, CudaFunction>,

    pub(super) split_fns: HashMap<(i64, bool, i64), CudaFunction>,
    pub(super) merge_fns: HashMap<bool, CudaFunction>,
    pub(super) merge4_fns: HashMap<bool, CudaFunction>,
    pub(super) causal_split_fns: HashMap<(i64, bool, i64, bool), CudaFunction>,
    pub(super) fast_split_fns: HashMap<(i64, bool, bool), CudaFunction>,
    pub(super) wide_query_split_fns: HashMap<(bool, bool, bool), CudaFunction>,
    pub(super) wide_key_split_fns: HashMap<(i64, bool, bool), CudaFunction>,
    pub(super) sm89: bool,
    pub(super) sm_count: i64,
    pub(super) l2_bytes: i64,
}

pub(super) fn runtime(device: i32) -> Result<&'static Runtime> {
    static RUNTIMES: OnceLock<Mutex<HashMap<i32, &'static Runtime>>> = OnceLock::new();
    let mut map = RUNTIMES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    if let Some(rt) = map.get(&device) {
        return Ok(rt);
    }
    let ctx = CudaContext::new(device as usize).map_err(|e| {
        err(
            RUNTIME_ERROR,
            &format!("cuda context (device {device}): {e:?}"),
        )
    })?;
    let module = CudaModule::load(&ctx, cuda::fatbin())
        .map_err(|e| err(RUNTIME_ERROR, &format!("load kernels image: {e:?}")))?;

    let mut cc = [0i32; 2];
    for (value, attribute) in cc.iter_mut().zip([
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    ]) {
        unsafe { sys::cuDeviceGetAttribute(value, attribute, ctx.cu_device()) }
            .result()
            .map_err(|e| err(RUNTIME_ERROR, &format!("query compute capability: {e:?}")))?;
    }

    let load_flash = |name: &str, smem: u32| -> Result<CudaFunction> {
        let f = module
            .load_function(name)
            .map_err(|e| err(RUNTIME_ERROR, &format!("{name}: {e:?}")))?;
        unsafe {
            sys::cuFuncSetAttribute(
                f.cu_function(),
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem as i32,
            )
            .result()
        }
        .map_err(|e| err(RUNTIME_ERROR, &format!("{name} smem opt-in: {e:?}")))?;
        Ok(f)
    };
    let mut wide_d32_fns = HashMap::new();
    for bf16 in [false, true] {
        let name = format!(
            "flash_attn_fwd_d32_wide_raw{}",
            if bf16 { "_bf16" } else { "" }
        );
        let tile = Tile {
            rows: 64,
            keys: 128,
            ..alternate_tile(32)
        };
        wide_d32_fns.insert(bf16, load_flash(&name, flash_smem_bytes(32, tile))?);
    }
    let mut raw_d32_fns = HashMap::new();
    for bf16 in [false, true] {
        let name = format!("flash_attn_fwd_d32_raw{}", if bf16 { "_bf16" } else { "" });
        raw_d32_fns.insert(
            bf16,
            load_flash(&name, flash_smem_bytes(32, alternate_tile(32)))?,
        );
    }
    let mut dense_d192_fns = HashMap::new();
    for bf16 in [false, true] {
        let name = format!(
            "flash_attn_fwd_d192_dense{}",
            if bf16 { "_bf16" } else { "" }
        );
        dense_d192_fns.insert(bf16, load_flash(&name, 98304)?);
    }
    let mut raw_d128_fns = HashMap::new();
    for bf16 in [false, true] {
        let name = format!("flash_attn_fwd_d128_raw{}", if bf16 { "_bf16" } else { "" });
        raw_d128_fns.insert(
            bf16,
            load_flash(&name, flash_smem_bytes(128, default_tile(128)))?,
        );
    }
    let mut short_d192_fns = HashMap::new();
    for causal in [false, true] {
        for bf16 in [false, true] {
            let name = format!(
                "flash_attn_fwd_d192_short{}{}",
                if causal { "_causal" } else { "" },
                if bf16 { "_bf16" } else { "" },
            );
            short_d192_fns.insert(
                (causal, bf16),
                load_flash(&name, flash_smem_bytes(192, default_tile(192)))?,
            );
        }
    }
    let mut fns = HashMap::new();
    let mut scalar_fns = HashMap::new();
    for &hd in &HEAD_DIMS {
        for &alternate in &[false, true] {
            if alternate && !ALTERNATE_BUCKETS.contains(&hd) {
                continue;
            }

            let tile = if alternate {
                alternate_tile(hd)
            } else {
                default_tile(hd)
            };
            let smem = flash_smem_bytes(hd, tile);
            for &bf16 in &[false, true] {
                let f = load_flash(&flash_kernel_name(hd, alternate, bf16), smem)?;
                fns.insert((hd, alternate, bf16), f);
                if hd > 32 {
                    let name = format!("{}_scalar", flash_kernel_name(hd, alternate, bf16));
                    scalar_fns.insert((hd, alternate, bf16), load_flash(&name, smem)?);
                }
            }
        }
    }
    let mut split_fns = HashMap::new();
    let mut causal_split_fns = HashMap::new();
    for &hd in &HEAD_DIMS {
        for rows in [32, 64] {
            if rows == 64 && hd < 192 {
                continue;
            }
            for bf16 in [false, true] {
                for reverse in [false, true] {
                    if reverse && hd != 256 {
                        continue;
                    }
                    let name = format!(
                        "flash_attn_split_causal_{}split_d{hd}_q{rows}{}",
                        if reverse { "reverse_" } else { "" },
                        if bf16 { "_bf16" } else { "" }
                    );
                    causal_split_fns.insert(
                        (hd, bf16, rows, reverse),
                        load_flash(&name, split_smem_bytes(hd, split_tile(hd, rows)))?,
                    );
                }
            }
        }
    }
    let mut fast_split_fns = HashMap::new();
    for rows in [16, 32] {
        for bf16 in [false, true] {
            for causal in [false, true] {
                if rows == 16 && causal {
                    continue;
                }
                let name = format!(
                    "flash_attn_split{}_split_fast_d128_q{rows}{}",
                    if causal { "_causal" } else { "" },
                    if bf16 { "_bf16" } else { "" }
                );
                fast_split_fns.insert(
                    (rows, bf16, causal),
                    load_flash(&name, split_smem_bytes(128, split_tile(128, rows)))?,
                );
            }
        }
    }
    let mut wide_query_split_fns = HashMap::new();
    for bf16 in [false, true] {
        for causal in [false, true] {
            for unroll in [false, true] {
                let name = format!(
                    "flash_attn_split{}_split_{}_d32_q64{}",
                    if causal { "_causal" } else { "" },
                    if unroll { "fast" } else { "k64" },
                    if bf16 { "_bf16" } else { "" }
                );
                let mut tile = split_tile(32, 64);
                tile.keys = 64;
                wide_query_split_fns.insert(
                    (bf16, causal, unroll),
                    load_flash(&name, split_smem_bytes(32, tile))?,
                );
            }
        }
    }
    let mut wide_key_split_fns = HashMap::new();
    for hd in [32, 64] {
        for bf16 in [false, true] {
            for causal in [false, true] {
                let name = format!(
                    "flash_attn_split{}_split_k64_d{hd}_q32{}",
                    if causal { "_causal" } else { "" },
                    if bf16 { "_bf16" } else { "" }
                );
                let mut tile = split_tile(hd, 32);
                tile.keys = 64;
                wide_key_split_fns.insert(
                    (hd, bf16, causal),
                    load_flash(&name, split_smem_bytes(hd, tile))?,
                );
            }
        }
    }
    let mut merge_fns = HashMap::new();
    let mut merge4_fns = HashMap::new();
    for &bf16 in &[false, true] {
        for &hd in &HEAD_DIMS {
            for rows in [16, 32, 64] {
                if rows == 64 && hd < 192 {
                    continue;
                }
                let suffix = if rows == 16 {
                    String::new()
                } else {
                    format!("_q{rows}")
                };
                let name = format!(
                    "flash_attn_split_split_d{hd}{suffix}{}",
                    if bf16 { "_bf16" } else { "" }
                );
                split_fns.insert(
                    (hd, bf16, rows),
                    load_flash(&name, split_smem_bytes(hd, split_tile(hd, rows)))?,
                );
            }
        }
        let name = if bf16 {
            "flash_attn_merge_merge_bf16"
        } else {
            "flash_attn_merge_merge_f16"
        };
        let name4 = name.replace("flash_attn_merge_", "flash_attn_merge4_");
        merge4_fns.insert(
            bf16,
            module
                .load_function(&name4)
                .map_err(|e| err(RUNTIME_ERROR, &format!("{name4}: {e:?}")))?,
        );
        merge_fns.insert(
            bf16,
            module
                .load_function(name)
                .map_err(|e| err(RUNTIME_ERROR, &format!("{name}: {e:?}")))?,
        );
    }
    let mut dropout_fns = HashMap::new();
    for bucket in HEAD_DIMS {
        for bf16 in [false, true] {
            for rng_kind in 0..if bucket == 64 { 4 } else { 3 } {
                let name = format!(
                    "flash_attn{}_dropout_d{bucket}{}",
                    match rng_kind {
                        3 => "_rng_descriptor",
                        1 => "_rng",
                        2 => "_probs",
                        _ => "",
                    },
                    if bf16 { "_bf16" } else { "" }
                );
                dropout_fns.insert(
                    (bucket, bf16, rng_kind),
                    load_flash(&name, flash_smem_bytes(bucket, dropout_tile(bucket)))?,
                );
            }
        }
    }
    let mut bwd_dq_fns = HashMap::new();
    for bucket in HEAD_DIMS {
        for bf16 in [false, true] {
            let name = format!("backward_dq_d{bucket}{}", if bf16 { "_bf16" } else { "" });
            bwd_dq_fns.insert(
                (bucket, bf16),
                load_flash(&name, (192 * (bucket / 2 + 4) * 4) as u32)?,
            );
        }
    }
    let mut bwd_dq_split_fns = HashMap::new();
    let mut bwd_dq_merge_fns = HashMap::new();
    for bf16 in [false, true] {
        for splits in [2, 4, 8] {
            let name = format!(
                "backward_dq_d128_split{splits}{}",
                if bf16 { "_bf16" } else { "" }
            );
            bwd_dq_split_fns.insert((bf16, splits), load_flash(&name, 52224)?);
        }
        let name = format!("backward_dq_merge_{}", if bf16 { "bf16" } else { "f16" });
        bwd_dq_merge_fns.insert(
            bf16,
            module
                .load_function(&name)
                .map_err(|e| err(RUNTIME_ERROR, &format!("{name}: {e:?}")))?,
        );
    }
    let mut bwd_dq_swizzled_fns = HashMap::new();
    for bf16 in [false, true] {
        for (mode, suffix) in [
            (0, "_plain"),
            (1, "_nodrop"),
            (2, ""),
            (3, "_nodrop_nocap"),
            (4, "_nocap"),
        ] {
            let name = format!(
                "backward_dq_d128_swizzled{}{}",
                suffix,
                if bf16 { "_bf16" } else { "" }
            );
            bwd_dq_swizzled_fns.insert((bf16, mode), load_flash(&name, 50176)?);
        }
    }
    let mut bwd_dkv_fns = HashMap::new();
    for bucket in HEAD_DIMS {
        for bf16 in [false, true] {
            let name = format!(
                "backward_kv_dkv_d{bucket}{}",
                if bf16 { "_bf16" } else { "" }
            );
            bwd_dkv_fns.insert(
                (bucket, bf16),
                load_flash(&name, (128 * (bucket / 2 + 4) * 4) as u32)?,
            );
        }
    }
    let mut bwd_fused_d64_fns = HashMap::new();
    let mut bwd_fused_fns = HashMap::new();
    for bf16 in [false, true] {
        for partial in [false, true] {
            for dropout in [false, true] {
                for (mode, suffix) in [
                    (0u8, "_plain"),
                    (1, "_causal"),
                    (2, ""),
                    (3, "_causal_balanced"),
                ] {
                    let name = format!(
                        "fused_backward_fused_d64_{}{}{}{}",
                        if bf16 { "bf16" } else { "f16" },
                        if dropout { "_drop" } else { "" },
                        if partial { "_partial" } else { "" },
                        suffix,
                    );
                    bwd_fused_d64_fns
                        .insert((bf16, partial, dropout, mode), load_flash(&name, 25600)?);
                    let name = format!(
                        "fused_backward_fused_{}{}{}{}",
                        if bf16 { "bf16" } else { "f16" },
                        if dropout { "_drop" } else { "" },
                        if partial { "_partial" } else { "" },
                        suffix,
                    );
                    bwd_fused_fns.insert((bf16, partial, dropout, mode), load_flash(&name, 50176)?);
                }
            }
        }
    }
    let mut bwd_dkv_both_fns = HashMap::new();
    for bf16 in [false, true] {
        for partial in [false, true] {
            for dropout in [false, true] {
                for masked in [false, true] {
                    let name = format!(
                        "backward_kv_dkv_d128_{}both{}{}{}",
                        if partial { "partial_" } else { "" },
                        if masked { "_masked" } else { "" },
                        if dropout { "_drop" } else { "" },
                        if bf16 { "_bf16" } else { "" }
                    );
                    bwd_dkv_both_fns.insert(
                        (bf16, partial, dropout, masked),
                        load_flash(&name, combined_smem(dropout, masked))?,
                    );
                }
            }
        }
    }
    let mut bwd_dkv_specialized_fns = HashMap::new();
    let mut bwd_dkv_merge_fns = HashMap::new();
    for bf16 in [false, true] {
        for (mode, suffix) in [(0, "_plain"), (1, "_no_dropout"), (2, "")] {
            let name = format!(
                "backward_kv_dkv_d128_partial{suffix}{}",
                if bf16 { "_bf16" } else { "" }
            );
            bwd_dkv_specialized_fns.insert(
                (bf16, true, mode),
                load_flash(&name, (128 * 68 * 4) as u32)?,
            );
            if mode != 2 {
                let name = format!(
                    "backward_kv_dkv_d128{suffix}{}",
                    if bf16 { "_bf16" } else { "" }
                );
                bwd_dkv_specialized_fns.insert(
                    (bf16, false, mode),
                    load_flash(&name, (128 * 68 * 4) as u32)?,
                );
            }
        }
        let name = format!("backward_kv_merge_{}", if bf16 { "bf16" } else { "f16" });
        bwd_dkv_merge_fns.insert(
            bf16,
            module
                .load_function(&name)
                .map_err(|e| err(RUNTIME_ERROR, &format!("{name}: {e:?}")))?,
        );
    }
    let mut varlen_merge_fns = HashMap::new();
    for bf16 in [false, true] {
        let name = format!("varlen_merge_{}", if bf16 { "bf16" } else { "f16" });
        varlen_merge_fns.insert(
            bf16,
            module
                .load_function(&name)
                .map_err(|e| err(RUNTIME_ERROR, &format!("{name}: {e:?}")))?,
        );
    }
    let mut cache_merge_fns = HashMap::new();
    for bf16 in [false, true] {
        for (grouped, packed, prefix) in [
            (false, false, ""),
            (true, false, "grouped_"),
            (false, true, "packed_"),
            (true, true, "grouped_packed_"),
        ] {
            let name = format!(
                "cache_merge_{}{}",
                prefix,
                if bf16 { "bf16" } else { "f16" }
            );
            cache_merge_fns.insert(
                (bf16, grouped, packed),
                module
                    .load_function(&name)
                    .map_err(|e| err(RUNTIME_ERROR, &format!("{name}: {e:?}")))?,
            );
        }
    }
    let mut cache_copy_fns = HashMap::new();
    for bf16 in [false, true] {
        for append in [false, true] {
            let name = format!(
                "cache_copy_{}_{}",
                if append { "kv" } else { "q" },
                if bf16 { "bf16" } else { "f16" }
            );
            cache_copy_fns.insert(
                (bf16, append),
                module
                    .load_function(&name)
                    .map_err(|e| err(RUNTIME_ERROR, &format!("{name}: {e:?}")))?,
            );
        }
    }
    let mut cache_update_fns = HashMap::new();
    for bf16 in [false, true] {
        let name = format!("cache_update_kv_{}", if bf16 { "bf16" } else { "f16" });
        cache_update_fns.insert(
            bf16,
            module
                .load_function(&name)
                .map_err(|e| err(RUNTIME_ERROR, &format!("{name}: {e:?}")))?,
        );
    }
    let mut cached_fns = HashMap::new();
    for bucket in HEAD_DIMS {
        for bf16 in [false, true] {
            for paged in [false, true] {
                let name = format!(
                    "cached_fwd_cache_d{bucket}{}{}",
                    if bf16 { "_bf16" } else { "" },
                    if paged { "_paged" } else { "" }
                );
                cached_fns.insert(
                    (bucket, bf16, paged),
                    load_flash(&name, flash_smem_bytes(bucket, dropout_tile(bucket)))?,
                );
            }
        }
    }
    let mut cached_split_fns = HashMap::new();
    for bucket in HEAD_DIMS {
        for bf16 in [false, true] {
            for paged in [false, true] {
                let name = format!(
                    "cached_split_fwd_split_d{bucket}{}{}",
                    if bf16 { "_bf16" } else { "" },
                    if paged { "_paged" } else { "" }
                );
                cached_split_fns.insert(
                    (bucket, bf16, paged),
                    load_flash(&name, flash_smem_bytes(bucket, dropout_tile(bucket)))?,
                );
            }
        }
    }
    let mut varlen_split_fns = HashMap::new();
    for bucket in [64, 128] {
        for bf16 in [false, true] {
            for paged in [false, true] {
                for grouped in [false, true] {
                    if grouped && bucket != 128 {
                        continue;
                    }
                    let name = format!(
                        "cached_split_fwd_packed_{}split_d{bucket}{}{}",
                        if grouped { "grouped_" } else { "" },
                        if bf16 { "_bf16" } else { "" },
                        if paged { "_paged" } else { "" }
                    );
                    varlen_split_fns.insert(
                        (bucket, bf16, paged, grouped),
                        load_flash(&name, flash_smem_bytes(bucket, dropout_tile(bucket)))?,
                    );
                }
            }
        }
    }
    let mut varlen_rng_fns = HashMap::new();
    for bf16 in [false, true] {
        let name = format!(
            "varlen_rng_descriptor_d64{}_dropout",
            if bf16 { "_bf16" } else { "" }
        );
        varlen_rng_fns.insert(
            bf16,
            load_flash(&name, flash_smem_bytes(64, dropout_tile(64)))?,
        );
    }
    let mut varlen_fns = HashMap::new();
    for bucket in HEAD_DIMS {
        for bf16 in [false, true] {
            for dropout in [false, true] {
                let name = format!(
                    "varlen_fwd_d{bucket}{}{}",
                    if bf16 { "_bf16" } else { "" },
                    if dropout { "_dropout" } else { "" }
                );
                varlen_fns.insert(
                    (bucket, bf16, dropout),
                    load_flash(&name, flash_smem_bytes(bucket, dropout_tile(bucket)))?,
                );
            }
        }
    }
    let mut paged_fns = HashMap::new();
    for bucket in HEAD_DIMS {
        for bf16 in [false, true] {
            for dropout in [false, true] {
                let name = format!(
                    "paged_fwd_paged_d{bucket}{}{}",
                    if bf16 { "_bf16" } else { "" },
                    if dropout { "_dropout" } else { "" }
                );
                paged_fns.insert(
                    (bucket, bf16, dropout),
                    load_flash(&name, flash_smem_bytes(bucket, dropout_tile(bucket)))?,
                );
            }
        }
    }
    let mut varlen_bwd_fns = HashMap::new();
    for bucket in HEAD_DIMS {
        for bf16 in [false, true] {
            for kv in [false, true] {
                let name = format!(
                    "varlen_backward_{}_d{bucket}{}",
                    if kv { "dkv" } else { "dq" },
                    if bf16 { "_bf16" } else { "" }
                );
                varlen_bwd_fns.insert(
                    (bucket, bf16, kv),
                    load_flash(
                        &name,
                        ((if kv { 128 } else { 192 }) * (bucket / 2 + 4) * 4) as u32,
                    )?,
                );
            }
        }
    }
    let mut varlen_bwd_partial_fns = HashMap::new();
    for bucket in HEAD_DIMS {
        for bf16 in [false, true] {
            let name = format!(
                "varlen_backward_dkv_d{bucket}_partial{}",
                if bf16 { "_bf16" } else { "" }
            );
            varlen_bwd_partial_fns.insert(
                (bucket, bf16),
                load_flash(&name, (128 * (bucket / 2 + 4) * 4) as u32)?,
            );
        }
    }
    let mut varlen_bwd_causal_fns = HashMap::new();
    for bf16 in [false, true] {
        for (kv, variant, rows) in [
            (false, "dq_d64_split2_causal", 192),
            (true, "dkv_d64_partial_causal", 128),
        ] {
            let name = format!(
                "varlen_backward_{variant}{}",
                if bf16 { "_bf16" } else { "" }
            );
            varlen_bwd_causal_fns.insert((bf16, kv), load_flash(&name, rows * 36 * 4)?);
        }
    }
    let mut varlen_dq_split_fns = HashMap::new();
    for bucket in [64, 128] {
        for bf16 in [false, true] {
            let name = format!(
                "varlen_backward_dq_d{bucket}_split2{}",
                if bf16 { "_bf16" } else { "" }
            );
            varlen_dq_split_fns.insert(
                (bucket, bf16),
                load_flash(&name, (192 * (bucket / 2 + 4) * 4) as u32)?,
            );
        }
    }
    let mut varlen_preprocess_fns = HashMap::new();
    for bf16 in [false, true] {
        for lanes in [8, 16, 32] {
            let name = format!(
                "varlen_preprocess_{}_{lanes}",
                if bf16 { "bf16" } else { "f16" }
            );
            varlen_preprocess_fns.insert(
                (bf16, lanes),
                module
                    .load_function(&name)
                    .map_err(|e| err(RUNTIME_ERROR, &format!("{name}: {e:?}")))?,
            );
        }
    }
    let mut bwd_preprocess_fns = HashMap::new();
    for bf16 in [false, true] {
        for lanes in [8, 16, 32] {
            let name = format!(
                "bwd_preprocess_{}_{lanes}",
                if bf16 { "bf16" } else { "f16" }
            );
            bwd_preprocess_fns.insert(
                (bf16, lanes),
                module
                    .load_function(&name)
                    .map_err(|e| err(RUNTIME_ERROR, &format!("{name}: {e:?}")))?,
            );
        }
    }
    let sm_count =
        ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|e| err(RUNTIME_ERROR, &format!("query SM count: {e:?}")))? as i64;
    let mut l2_bytes = 0i32;
    unsafe {
        sys::cuDeviceGetAttribute(
            &mut l2_bytes,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
            ctx.cu_device(),
        )
    }
    .result()
    .map_err(|e| err(RUNTIME_ERROR, &format!("query L2 size: {e:?}")))?;
    let rt: &'static Runtime = Box::leak(Box::new(Runtime {
        ctx,
        _module: module.clone(),
        fns,
        dropout_fns,
        bwd_preprocess_fns,
        bwd_dq_fns,
        bwd_dq_swizzled_fns,
        bwd_dq_split_fns,
        bwd_dq_merge_fns,
        bwd_dkv_both_fns,
        bwd_fused_fns,
        bwd_fused_d64_fns,
        bwd_dkv_fns,
        bwd_dkv_specialized_fns,
        bwd_dkv_merge_fns,
        varlen_rng_fns,
        varlen_fns,
        paged_fns,
        cached_fns,
        cached_split_fns,
        varlen_split_fns,
        cache_copy_fns,
        cache_update_fns,
        varlen_merge_fns,
        cache_merge_fns,
        varlen_bwd_fns,
        varlen_bwd_partial_fns,
        varlen_bwd_causal_fns,
        varlen_dq_split_fns,
        varlen_preprocess_fns,
        resolve_rng: module
            .load_function("resolve_rng_state")
            .map_err(|e| err(RUNTIME_ERROR, &format!("resolve RNG: {e:?}")))?,
        copy_layout: module
            .load_function("copy_attention_layout")
            .map_err(|e| err(RUNTIME_ERROR, &format!("copy layout: {e:?}")))?,
        scalar_fns,
        short_d192_fns,
        dense_d192_fns,
        raw_d128_fns,
        raw_d32_fns,
        wide_d32_fns,
        split_fns,
        merge_fns,
        merge4_fns,
        causal_split_fns,
        fast_split_fns,
        wide_key_split_fns,
        wide_query_split_fns,
        sm89: cc == [8, 9],
        sm_count,
        l2_bytes: l2_bytes as i64,
    }));
    map.insert(device, rt);
    Ok(rt)
}
