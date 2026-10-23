use cuda_device::{thread, warp};

#[derive(Clone, Copy)]
pub struct Strides {
    pub batch: u32,
    pub seq: u32,
    pub head: u32,
}

#[derive(Clone, Copy)]
pub struct FwdConfig {
    pub scale: f32,
    pub tanh_scale: f32,
    pub window_left: i32,
    pub window_right: i32,
    pub seqlen_q: u32,
    pub seqlen_k: u32,
    pub has_alibi: u32,
    pub alibi_causal: u32,
    pub h: u32,
    pub gqa: u32,
    pub hs_u32: u32,
    pub q: Strides,
    pub k: Strides,
    pub v: Strides,
    pub alibi_bs: u32,
}

impl FwdConfig {
    pub fn new(
        scale: f32,
        tanh_scale: f32,
        window_left: i32,
        window_right: i32,
        seqlen_q: u32,
        seqlen_k: u32,
        has_alibi: u32,
        alibi_causal: u32,
        h: u32,
        gqa: u32,
        hs_u32: u32,
        q_bs: u32,
        q_ss: u32,
        q_hs: u32,
        k_bs: u32,
        k_ss: u32,
        k_hs: u32,
        v_bs: u32,
        v_ss: u32,
        v_hs: u32,
        alibi_bs: u32,
    ) -> Self {
        Self {
            scale,
            tanh_scale,
            window_left,
            window_right,
            seqlen_q,
            seqlen_k,
            has_alibi,
            alibi_causal,
            h,
            gqa,
            hs_u32,
            q: Strides {
                batch: q_bs,
                seq: q_ss,
                head: q_hs,
            },
            k: Strides {
                batch: k_bs,
                seq: k_ss,
                head: k_hs,
            },
            v: Strides {
                batch: v_bs,
                seq: v_ss,
                head: v_hs,
            },
            alibi_bs,
        }
    }
}

// Zero means runtime layout; fixed variants save registers.
pub const GRID_PLAIN: u32 = 1;
pub const GRID_CAUSAL: u32 = 2;

#[derive(Clone, Copy)]
pub struct LaunchTile {
    pub tid: u32,
    pub lane_id: u32,
    pub bh: u32,
    pub group: u32,
    pub pair: u32,
    pub q_row_base: u32,
    pub warp_row: u32,
    pub b: u32,
    pub head: u32,
    pub kv_head: u32,
}

impl LaunchTile {
    pub fn new<const BLOCK_ROWS: u32, const RFRAGS: usize, const GRID: u32>(
        h: u32,
        gqa: u32,
        seqlen_q: u32,
        causal: bool,
    ) -> Self {
        let tid = thread::threadIdx_x();
        let warp_id = warp::warp_id();
        let lane_id = tid % 32;
        // Put long causal blocks first, grouping heads at each query tile.
        let last_q_block = (seqlen_q - 1) / BLOCK_ROWS;
        let head_first = if GRID == GRID_PLAIN {
            false
        } else if GRID == GRID_CAUSAL {
            true
        } else {
            causal && last_q_block > 0 && last_q_block < 65535
        };
        let q_block = last_q_block
            - if head_first {
                thread::blockIdx_y()
            } else {
                thread::blockIdx_x()
            };
        let head = if head_first {
            thread::blockIdx_x() % h
        } else {
            thread::blockIdx_y()
        };
        let b = if head_first {
            thread::blockIdx_x() / h
        } else {
            thread::blockIdx_z()
        };
        let bh = b * h + head;
        Self {
            tid,
            lane_id,
            bh,
            group: lane_id >> 2,
            pair: lane_id & 3,
            q_row_base: q_block * BLOCK_ROWS,
            warp_row: warp_id * 16 * RFRAGS as u32,
            b,
            head,
            kv_head: if gqa == 1 { head } else { head / gqa },
        }
    }
}

#[derive(Clone, Copy)]
pub struct TensorBases {
    pub q: usize,
    pub k: usize,
    pub v: usize,
    pub o: usize,
    pub o_ss: usize,
}

impl TensorBases {
    pub fn new(cfg: &FwdConfig, tile: &LaunchTile) -> Self {
        let q = tile.b as usize * cfg.q.batch as usize + tile.head as usize * cfg.q.head as usize;
        let k =
            tile.b as usize * cfg.k.batch as usize + tile.kv_head as usize * cfg.k.head as usize;
        let v =
            tile.b as usize * cfg.v.batch as usize + tile.kv_head as usize * cfg.v.head as usize;
        let o_ss = cfg.h as usize * cfg.hs_u32 as usize;
        let o = tile.b as usize * cfg.seqlen_q as usize * o_ss
            + tile.head as usize * cfg.hs_u32 as usize;
        Self { q, k, v, o, o_ss }
    }
}

#[derive(Clone, Copy)]
pub struct KvRange {
    pub start_tile: u32,
    pub num_tiles: u32,
}

impl KvRange {
    pub fn new<const BLOCK_ROWS: u32, const KEYS: u32>(
        cfg: &FwdConfig,
        q_row_base: u32,
        off: i32,
    ) -> Self {
        let kv_limit = if cfg.window_right >= 0 {
            let last_row = (q_row_base + BLOCK_ROWS - 1).min(cfg.seqlen_q - 1) as i32;
            (last_row + 1 + off + cfg.window_right).clamp(0, cfg.seqlen_k as i32) as u32
        } else {
            cfg.seqlen_k
        };
        let start_tile = if cfg.window_left >= 0 {
            let first =
                (q_row_base as i32 + off - cfg.window_left).clamp(0, cfg.seqlen_k as i32) as u32;
            first / KEYS
        } else {
            0
        };
        Self {
            start_tile,
            num_tiles: kv_limit.div_ceil(KEYS),
        }
    }
}
