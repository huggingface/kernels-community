pub(super) const HEAD_DIMS: [i64; 6] = [32, 64, 96, 128, 192, 256];

#[derive(Clone, Copy)]
pub(super) struct Tile {
    pub(super) rows: i64,
    pub(super) threads: u32,
    pub(super) keys: usize,
    pub(super) alternate: bool,
}

pub(super) const ALTERNATE_BUCKETS: [i64; 5] = [32, 96, 128, 192, 256];

pub(super) fn default_tile(bucket: i64) -> Tile {
    let rows = if bucket <= 192 { 64 } else { 128 };
    Tile {
        rows,
        threads: (rows * 2) as u32,
        keys: if bucket == 192 {
            48
        } else if bucket >= 128 {
            32
        } else {
            64
        },
        alternate: false,
    }
}

pub(super) fn dropout_tile(bucket: i64) -> Tile {
    Tile {
        rows: 64,
        threads: 128,
        keys: if bucket <= 64 { 64 } else { 32 },
        alternate: false,
    }
}

pub(super) fn split_tile(bucket: i64, rows: i64) -> Tile {
    Tile {
        rows,
        threads: (rows * 2) as u32,
        keys: if bucket <= 128 { 32 } else { 16 },
        alternate: false,
    }
}

pub(super) fn split_smem_bytes(bucket: i64, tile: Tile) -> u32 {
    (((if bucket == 32 && tile.rows >= 32 {
        3
    } else {
        2
    }) * tile.keys)
        .max(tile.rows as usize)
        * (bucket as usize / 2 + SMEM_PAD_U32)
        * 4) as u32
}

pub(super) fn alternate_tile(bucket: i64) -> Tile {
    let threads = 128;
    Tile {
        rows: if bucket >= 192 { 64 } else { 128 },
        threads,
        keys: if bucket == 32 {
            64
        } else if bucket == 192 {
            48
        } else {
            32
        },
        alternate: true,
    }
}

pub(super) fn flash_tile(bucket: i64, masked: bool, ctas: i64, sm_count: i64) -> Tile {
    match bucket {
        32 if !masked && ctas >= 2 * sm_count => alternate_tile(bucket),
        96 if !masked && ctas >= 2 * sm_count => alternate_tile(bucket),
        // Smaller blocks and single-buffer pipelining help these Ada workloads.
        192 => alternate_tile(bucket),
        256 if masked => alternate_tile(bucket),
        _ => default_tile(bucket),
    }
}

pub(super) const SMEM_BUDGET: usize = 101_376;

pub(super) const SMEM_PAD_U32: usize = 4;

pub(super) fn flash_smem_bytes(bucket: i64, tile: Tile) -> u32 {
    let stride = (bucket / 2) as usize + SMEM_PAD_U32;
    let tile_bytes = tile.keys * stride * 4;
    let nbuf = if (bucket == 192 && tile.rows == 64 && tile.keys == 32)
        || (bucket == 96 && tile.rows == 128 && tile.keys == 32)
    {
        3
    } else if !(bucket == 192 && tile.keys == 48)
        && tile.rows > 32
        && 4 * tile_bytes <= SMEM_BUDGET
        && !((bucket >= 192 && tile.rows == 64 && tile.keys == 32)
            || (bucket == 96 && tile.rows == 64 && tile.keys == 64))
    {
        4
    } else {
        2
    };
    // d96 Q128 triple pipeline needs more scratch for Q/output staging.
    let total = (nbuf * tile_bytes).max(tile.rows as usize * stride * 4);
    debug_assert!(
        total >= tile.rows as usize * stride * 4,
        "Q staging exceeds K/V smem"
    );
    total as u32
}

pub(super) fn flash_kernel_name(bucket: i64, alternate: bool, bf16: bool) -> String {
    format!(
        "flash_attn_fwd_d{bucket}{}{}",
        if alternate && bucket == 32 {
            "_q128"
        } else if alternate {
            "_k32"
        } else {
            ""
        },
        if bf16 { "_bf16" } else { "" }
    )
}

/// Dynamic shared memory of the combined dK/dV backward kernels.
pub(super) const fn combined_smem(dropout: bool, masked: bool) -> u32 {
    if dropout || masked { 50176 } else { 17408 }
}
