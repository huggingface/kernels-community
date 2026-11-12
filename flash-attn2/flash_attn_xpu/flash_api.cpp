#include <torch/all.h>
#include <c10/xpu/XPUStream.h>
#include <cute/util/compat/device.hpp>
#include <ATen/xpu/XPUContext.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <unordered_map>

#include "src/fmha_fwd.hpp"
#include "src/fmha_bwd.hpp"
#include "src/decode/splitkv_decode.hpp"
#include "src/decode/vllm/paged_decode_tuning.h"

namespace FLASH_NAMESPACE {

/// Retrieve Philox RNG seed and offset, advancing the generator state.
inline std::pair<uint64_t, uint64_t> get_philox_state(
    std::optional<at::Generator> gen,
    int64_t counter_offset) {
    auto gen_val = gen.has_value() ? gen.value()
                                  : at::xpu::detail::getDefaultXPUGenerator();
    auto* xpu_gen = at::check_generator<at::XPUGeneratorImpl>(gen_val);
    auto [seed, offset] = xpu_gen->philox_engine_inputs(counter_offset);
    return {seed, offset};
}

/// Round `x` up to the nearest multiple of `m`.
constexpr int round_multiple(int x, int m) {
    const int pad_res = (x + m - 1) / m * m;
    return (pad_res == 224) ? 256 : pad_res;
}

inline at::Tensor ensure_contiguous(const at::Tensor& tensor) {
    return tensor.is_contiguous() ? tensor : tensor.contiguous();
}

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_xpu(), #x " must be on XPU")

// ---------------------------------------------------------------------------
// Split-KV decode (up to kMaxQueries queries per batch). Used when the inputs
// are supported and the plan below expects it to beat the generic forward
// kernels.
// FLASH_ATTN_XPU_DISABLE_SPLITKV_DECODE=1 forces the generic kernels,
// FLASH_ATTN_XPU_DECODE_NUM_SPLITS=N forces split-KV with N splits.
// ---------------------------------------------------------------------------
namespace splitkv {

using flash_xpu_splitkv::decode_args_t;

inline int env_int(const char* name, int dflt) {
    const char* s = std::getenv(name);
    return (s != nullptr && *s != '\0') ? std::atoi(s) : dflt;
}

inline bool disabled() {
    return env_int("FLASH_ATTN_XPU_DISABLE_SPLITKV_DECODE", 0) != 0;
}

inline int head_bucket(int d) {
    for (int b : {64, 96, 128, 192, 256, 512}) {
        if (d <= b) return b;
    }
    return 0;
}

inline int xe_core_count(sycl::queue& queue) {
    static std::mutex mu;
    static std::unordered_map<size_t, int> cache;
    auto dev = queue.get_device();
    const size_t key = std::hash<sycl::device>{}(dev);
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    int cores = 0;
    if (dev.has(sycl::aspect::ext_intel_gpu_slices) &&
        dev.has(sycl::aspect::ext_intel_gpu_subslices_per_slice)) {
        cores = dev.get_info<sycl::ext::intel::info::device::gpu_slices>() *
                dev.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
    }
    if (cores <= 0) cores = 20;
    cache[key] = cores;
    return cores;
}

// Split-count plan for contiguous KV, measured against the generic kernel on
// Arc Pro B60 (20 Xe cores) and an Intel Data Center GPU (32 Xe cores),
// fp16/bf16. Indexed by [head class][w + 3][t], with
//   w = floor(log2(batch * num_heads_kv / xe_cores)) clamped to [-3, 3],
//   t = floor(log2(ceil(seqlen_k / 64))) clamped to [0, 8].
// Each cell holds the split count that was never slower than the generic
// kernel (by more than 1.5% and 2 us) on any measured shape of either
// device: '-' = generic kernel, 'k' = split-KV with 2^k splits. Unmeasured
// cells copy the nearest measured one (the more conservative on ties).
// clang-format off
constexpr const char* kPlan[4][7] = {
    {  // GQA, head_dim <= 128
        "0000000-3",  // w = -3
        "00000--23",  // w = -2
        "000000123",  // w = -1
        "000000121",  // w = 0
        "000000011",  // w = 1
        "000000000",  // w = 2
        "000000001",  // w = 3
    },
    {  // GQA, head_dim > 128
        "------234",  // w = -3
        "000001222",  // w = -2
        "000011111",  // w = -1
        "000000000",  // w = 0
        "000000000",  // w = 1
        "000000000",  // w = 2
        "000000000",  // w = 3
    },
    {  // MHA, head_dim <= 256
        "0000000--",  // w = -3
        "0000000--",  // w = -2
        "0000000--",  // w = -1
        "----00---",  // w = 0
        "00000-234",  // w = 1
        "000000045",  // w = 2
        "000023454",  // w = 3
    },
    {  // MHA, head_dim > 256
        "000001-11",  // w = -3
        "000001-11",  // w = -2
        "000000001",  // w = -1
        "----00000",  // w = 0
        "000000005",  // w = 1
        "000003455",  // w = 2
        "000023444",  // w = 3
    },
};
// clang-format on

inline int floor_log2(int x) {
    int r = -1;
    while (x > 0) { x >>= 1; ++r; }
    return r;
}

// Paged KV: the split count of vllm-xpu-kernels (csrc/flash_attn/flash_api.cpp
// get_num_splits, 64-key tiles with 4 subgroups per work-group), whose
// decode kernel is the same as this one, plus that kernel's single-split
// rule below 32 KV tiles.
inline int vllm_num_splits(int xe_cores, int batch, int num_heads_q,
                           int num_heads_kv, int kv_tiles) {
    if (kv_tiles < 32) return 1;
    const int wgs = std::max(1, batch * num_heads_kv);
    if (wgs >= xe_cores && kv_tiles < 64) return 1;
    int splits = std::max((4 * xe_cores + wgs - 1) / wgs, kv_tiles / 12);
    splits = std::min(
        splits, std::max(2, 128 * xe_cores / std::max(1, batch * num_heads_q)));
    return std::max(1, std::min({splits, kv_tiles / 4, 32}));
}

// Number of KV splits for a decode batch (>= 1), or 0 for the generic kernel.
inline int plan_num_splits(sycl::queue& queue, int batch, int num_heads_q,
                           int num_heads_kv, int head_size, int max_seqlen_k,
                           bool paged) {
    int env = env_int("FLASH_ATTN_XPU_DECODE_NUM_SPLITS", 0);
    if (env > 0) return std::min(env, flash_xpu_splitkv::kMaxSplits);

    const int kv_tiles = (max_seqlen_k + flash_xpu_splitkv::kKVTile - 1) /
                         flash_xpu_splitkv::kKVTile;
    if (paged) {
        return vllm_num_splits(xe_core_count(queue), batch, num_heads_q,
                               num_heads_kv, kv_tiles);
    }
    const int wgs = std::max(1, batch * num_heads_kv);
    const int w = std::clamp(
        int(std::floor(std::log2(double(wgs) / xe_core_count(queue)))), -3, 3);
    const int t = std::clamp(floor_log2(std::max(1, kv_tiles)), 0, 8);
    const bool mha = num_heads_q == num_heads_kv;
    const int cls = mha ? (head_size <= 256 ? 2 : 3) : (head_size <= 128 ? 0 : 1);
    const char c = kPlan[cls][w + 3][t];
    if (c == '-') return 0;
    // At least 4 KV tiles per split.
    const int cap = 1 << floor_log2(std::max(1, kv_tiles / 4));
    return std::min({1 << (c - '0'), cap, flash_xpu_splitkv::kMaxSplits});
}

inline bool stride_ok(int64_t s) {
    // Xe 2D block messages: pitch a multiple of 16 bytes, below 2^24 bytes.
    return s > 0 && s % 8 == 0 && s * 2 < (int64_t(1) << 24);
}

// Longest per-batch query the split-KV path takes (speculative decoding).
constexpr int kMaxQueries = 16;

// The kernel packs the (query, GQA head) rows of a (batch, KV head) into
// kQTile-row work-groups like FlashAttention 3's PackGQA, which it always
// uses when splitting KV (hopper/flash_api.cpp). The split plan is keyed by
// single-query decode, so map `q` queries per batch to the decode batch
// with as many work-groups, and a GQA group as large as the rows a
// work-group fills. Returns the KV split count (0: generic kernel);
// `forced` > 0 overrides the plan.
inline int plan_layout(sycl::queue& queue, int batch, int q, int num_heads_q,
                       int num_heads_kv, int head_size, int max_seqlen_k,
                       bool paged, int64_t q_row_stride, int forced = 0) {
    constexpr int t = flash_xpu_splitkv::kQTile;
    const int g = num_heads_q / num_heads_kv;
    // One head per KV head: the queries of a batch form one Q surface.
    if (g == 1 && q > 1 && !stride_ok(q_row_stride)) return 0;
    const int tiles = (q * g + t - 1) / t;
    const int tiles_1 = (g + t - 1) / t;
    const int eff_batch = (batch * tiles + tiles_1 - 1) / tiles_1;
    int splits = forced > 0
        ? std::min(forced, flash_xpu_splitkv::kMaxSplits)
        : plan_num_splits(queue, eff_batch,
                          num_heads_kv * std::min(q * g, std::max(g, t)),
                          num_heads_kv, head_size, max_seqlen_k, paged);
    // With one head per KV head the fp32 partial rows of a work-group are
    // splits * Hq * D floats apart; keep the pitch inside the 2D block limit.
    while (g == 1 && q > 1 && splits > 1 &&
           int64_t(splits) * num_heads_q * head_size * 4 >= (int64_t(1) << 24))
        splits >>= 1;
    return splits;
}

// Masks the split-KV kernel can apply to a batch with several queries: none,
// or bottom-right causal with an optional left window. window_size_* follow
// the generic kernels (-1 = unbounded; causal folds into window_size_right
// = 0). Sets `causal` on success.
inline bool multi_query_mask(bool is_causal, int window_size_left,
                             int window_size_right, bool& causal) {
    causal = is_causal || window_size_right == 0;
    return causal || (window_size_left == -1 && window_size_right == -1);
}

// Common checks. q: (rows, Hq, D); k/v: (..., seq, Hkv, D) with the seq and
// head dims at -3 and -2.
inline bool supported(const at::Tensor& q, const at::Tensor& k,
                      const at::Tensor& v, bool paged) {
    if (disabled()) return false;
    auto dt = q.scalar_type();
    if (dt != at::kHalf && dt != at::kBFloat16) return false;
    if (k.scalar_type() != dt || v.scalar_type() != dt) return false;
    const int hq = q.size(-2), hkv = k.size(-2), d = q.size(-1);
    if (hkv <= 0 || hq % hkv != 0 || v.size(-2) != hkv) return false;
    if (k.size(-1) != d || v.size(-1) != d) return false;
    if (d < 32 || d % 8 != 0 || head_bucket(d) == 0) return false;
    if (q.stride(-1) != 1 || k.stride(-1) != 1 || v.stride(-1) != 1) return false;
    if (!stride_ok(q.stride(-2))) return false;
    if (!stride_ok(k.stride(-3)) || !stride_ok(v.stride(-3))) return false;
    if (paged) {
        const int64_t page = k.size(1);
        if (page <= 0 || page % flash_xpu_splitkv::kKVTile != 0) return false;
        if (v.size(1) != page) return false;
        if (k.stride(0) % k.stride(1) != 0 || v.stride(0) % v.stride(1) != 0)
            return false;
        const int64_t rows = k.stride(0) / k.stride(1);
        if (rows % flash_xpu_splitkv::kKVTile != 0 ||
            v.stride(0) / v.stride(1) != rows)
            return false;
        if (rows * k.size(0) >= (int64_t(1) << 31)) return false;
    }
    return true;
}

template <int Arch>
void launch(sycl::queue& queue, at::ScalarType dt, int head_size,
            const decode_args_t& args) {
    using namespace flash_xpu_splitkv;
    const bool fp16 = dt == at::kHalf;
#define SPLITKV_CASE(HD)                                                       \
    case HD:                                                                   \
        if (fp16) splitkv_decode_fp16<Arch, HD>(queue, args);                  \
        else      splitkv_decode_bf16<Arch, HD>(queue, args);                  \
        return;
    switch (head_bucket(head_size)) {
        SPLITKV_CASE(64)
        SPLITKV_CASE(96)
        SPLITKV_CASE(128)
        SPLITKV_CASE(192)
        SPLITKV_CASE(256)
        SPLITKV_CASE(512)
        default:
            TORCH_CHECK(false, "split-KV decode: unsupported head size ", head_size);
    }
#undef SPLITKV_CASE
}

// Allocates the split workspace (kept alive by the caller until the kernels
// are enqueued; the caching allocator is stream-ordered) and launches.
inline void run(sycl::queue& queue, const at::Tensor& q, int rows,
                decode_args_t args) {
    at::Tensor oaccum, lse_accum;
    if (args.num_splits > 1) {
        auto fopts = q.options().dtype(at::kFloat);
        oaccum = at::empty({rows, args.num_splits, args.num_heads_q, args.head_size_v}, fopts);
        lse_accum = at::empty({rows, args.num_heads_q, args.num_splits}, fopts);
        args.oaccum = oaccum.data_ptr<float>();
        args.lse_accum = lse_accum.data_ptr<float>();
    }
    args.min_tiles_per_split = 4;
    // Xe35 (CRI) and newer report IP version 35.x and run the CRI image.
    const char* ip = at::xpu::getDeviceProperties(q.device().index())->version.c_str();
    if (std::atoi(ip) >= 35) {
        launch<35>(queue, q.scalar_type(), args.head_size, args);
    } else {
        launch<20>(queue, q.scalar_type(), args.head_size, args);
    }
}

// ---------------------------------------------------------------------------
// Paged single-query decode on Xe35 (CRI) and newer: the vllm-xpu-kernels
// kernel (src/decode/vllm), launched the way vllm's mha_varlen_fwd does.
// FLASH_ATTN_XPU_DISABLE_VLLM_DECODE=1 keeps the kernel above.
// ---------------------------------------------------------------------------

inline bool is_xe35(const at::Tensor& t) {
    return std::atoi(at::xpu::getDeviceProperties(t.device().index())->version.c_str()) >= 35;
}

// vllm-xpu-kernels csrc/flash_attn/flash_api.cpp get_num_splits (16-bit Q/KV).
inline int vllm_get_num_splits(sycl::queue& queue, int batch_size,
                               int num_heads_q, int num_heads_kv,
                               int max_seqlen_k, int block_size,
                               int head_size, int64_t kv_token_stride_bytes) {
    const int num_xe_cores = xe_core_count(queue);
    int kv_tile;
    int sg_per_wg;
    int policy_split_cap;
    if (block_size > 0 && (block_size % 64) == 0) {
        kv_tile = 64;
        sg_per_wg = 4;
        policy_split_cap = 64;
    } else if (block_size == 32) {
        kv_tile = 32;
        sg_per_wg = 2;
        policy_split_cap = 32;
    } else {
        kv_tile = 16;
        sg_per_wg = 1;
        policy_split_cap = 16;
    }
    int kv_tiles = (max_seqlen_k + kv_tile - 1) / kv_tile;

    // Tuned XE3P decode (src/decode/vllm/paged_decode_tuning.h): head-size
    // buckets (96, 128] and (192, 256].
    namespace tuning = flash_xpu_splitkv::paged_decode_tuning;
    const int bucket = (head_size > 96 && head_size <= 128)   ? 128
                       : (head_size > 192 && head_size <= 256) ? 256
                                                               : 0;
    if (tuning::is_tuned(bucket, kv_tile)) {
        return tuning::num_splits(bucket, tuning::Precision::k16Bit, batch_size,
                                  num_heads_q, num_heads_kv, kv_tiles,
                                  num_xe_cores, kv_token_stride_bytes);
    }

    if (kv_tiles < 16) return 1;
    int num_wg_slots = num_xe_cores * 4 / sg_per_wg;
    int wgs_per_split = batch_size * num_heads_kv;
    if (wgs_per_split >= num_wg_slots && kv_tiles < 64) return 1;
    int splits_par =
        std::max(1, (4 * num_wg_slots + wgs_per_split - 1) / wgs_per_split);
    int splits_bw = std::max(1, kv_tiles / 12);
    int splits = std::max(splits_par, splits_bw);
    int red_work = std::max(1, batch_size * num_heads_q);
    int red_cap = std::max(2, 128 * num_xe_cores / red_work);
    splits = std::min(splits, red_cap);
    int max_splits_tiles = std::max(1, kv_tiles / 4);
    return std::max(
        1, std::min({splits, max_splits_tiles, 32, policy_split_cap}));
}

// Inputs as checked by supported(q, k, v, /*paged=*/true), plus the head
// sizes the vllm kernel is built for.
inline bool vllm_supported(const at::Tensor& q) {
    if (env_int("FLASH_ATTN_XPU_DISABLE_VLLM_DECODE", 0) != 0) return false;
    switch (q.size(-1)) {
        case 64: case 96: case 128: case 192: case 256: case 512: break;
        default: return false;
    }
    return is_xe35(q);
}

// One query per batch with a query: q rows (tokens) q_stride_seq apart, out
// contiguous [tokens, Hq, D], softmax_lse[h * lse_stride + token *
// lse_row_stride]. KV lengths: seqused_k[b], or cu_seqlens_k[b + 1] -
// cu_seqlens_k[b] when kv_cumulative. window_size_*: -1 = unbounded.
// `forced_splits` > 0 overrides vllm's split count. Returns false, without
// launching, for sizes the kernel's 32-bit offsets cannot address.
// spec_q_len > 1: q holds that many causal query positions per KV head group
// packed into the head dim, position-major (vllm's packed spec decode).
inline bool run_vllm(sycl::queue& queue, const at::Tensor& q, int tokens,
                     int64_t q_stride_seq, int64_t q_stride_heads,
                     const at::Tensor& k, const at::Tensor& v, at::Tensor& out,
                     const at::Tensor& block_table, const at::Tensor& cu_seqlens_q,
                     const at::Tensor& kv_lens, bool kv_cumulative, int batch,
                     int max_seqlen_k, float softmax_scale, int window_size_left,
                     int window_size_right, int forced_splits,
                     at::Tensor& softmax_lse, int lse_stride, int lse_row_stride,
                     int spec_q_len = 1) {
    const int num_heads_q = q.size(-2);
    const int head_size = q.size(-1);
    const int num_heads_kv = k.size(2);
    const int block_size = k.size(1);

    const bool is_local = (window_size_left != -1) | (window_size_right != -1);
    const int eff_window_left =
        window_size_left == -1 ? max_seqlen_k : window_size_left;
    const int eff_window_right =
        window_size_right == -1 ? max_seqlen_k : window_size_right;
    const int effective_seqlen_k =
        is_local ? std::min(max_seqlen_k, eff_window_left + 1) : max_seqlen_k;

    const int env = env_int("FLASH_ATTN_XPU_DECODE_NUM_SPLITS", 0);
    int num_kv_splits = env > 0 ? env
        : forced_splits > 0 ? forced_splits
        : vllm_get_num_splits(queue, batch, num_heads_q, num_heads_kv,
                              effective_seqlen_k, block_size, head_size,
                              std::max(k.stride(1), v.stride(1)) * k.element_size());
    num_kv_splits = std::min(num_kv_splits, 32);

    constexpr int64_t kIntMax = std::numeric_limits<int>::max();
    if (int64_t(tokens) * q_stride_seq > kIntMax ||
        int64_t(tokens) * num_heads_q * num_kv_splits * head_size > kIntMax)
        return false;

    at::Tensor tmp_out = num_kv_splits == 1
        ? out
        : at::empty({tokens, num_heads_q * num_kv_splits, head_size}, q.options());
    at::Tensor softmax_lse_accum = at::empty(
        {tokens, num_heads_q, num_kv_splits}, q.options().dtype(at::kFloat));

    const int64_t num_blocks = k.size(0);
    const int64_t page_stride_elements = k.stride(0) / k.stride(1);
    const int64_t total_seqlen_k = std::max<int64_t>(
        num_blocks * block_size,
        num_blocks > 0 ? (num_blocks - 1) * page_stride_elements + block_size : 0);

    flash_xpu_splitkv::vllm_decode_args_t a;
    a.query = q.data_ptr();
    a.key = k.data_ptr();
    a.value = v.data_ptr();
    a.out = out.data_ptr();
    a.tem_out = tmp_out.data_ptr();
    a.softmax_lse_accum = softmax_lse_accum.data_ptr();
    a.block_table = block_table.data_ptr();
    a.cu_seqlens_q = cu_seqlens_q.data_ptr();
    a.cu_seqlens_k = kv_lens.data_ptr();
    a.kv_cumulative = kv_cumulative;
    a.max_queries = 1;
    a.max_keys = max_seqlen_k;
    a.total_seqlen_q = tokens;
    a.total_seqlen_k = static_cast<int>(total_seqlen_k);
    a.sm_scale = softmax_scale;
    a.batch_size = batch;
    a.num_heads_q = num_heads_q;
    a.num_heads_k = num_heads_kv;
    a.head_size = head_size;
    a.v_head_size = head_size;
    a.max_blocks_per_seq = block_table.size(1);
    a.block_size = block_size;
    a.window_size_left = eff_window_left;
    a.window_size_right = eff_window_right;
    a.is_varlen = true;
    a.is_paged = true;
    a.is_causal = false;
    a.is_local = is_local;
    a.is_sink = false;
    a.num_kv_splits = num_kv_splits;
    a.k_stride_page = k.stride(0);
    a.k_stride_seq = k.stride(1);
    a.k_stride_heads = k.stride(2);
    a.v_stride_page = v.stride(0);
    a.v_stride_seq = v.stride(1);
    a.v_stride_heads = v.stride(2);
    a.q_stride_seq = q_stride_seq;
    a.q_stride_heads = q_stride_heads;
    a.q_stride_batch = 0;
    a.page_stride_elements = static_cast<int>(page_stride_elements);
    a.softmax_lse = softmax_lse.data_ptr<float>();
    a.lse_stride = lse_stride;
    a.lse_row_stride = lse_row_stride;
    a.spec_q_len = spec_q_len;

    using namespace flash_xpu_splitkv;
    const bool fp16 = q.scalar_type() == at::kHalf;
#define VLLM_CASE(HD)                                                          \
    case HD:                                                                   \
        if (fp16) vllm_paged_decode_fp16<35, HD>(queue, a);                    \
        else      vllm_paged_decode_bf16<35, HD>(queue, a);                    \
        break;
    switch (head_size) {
        VLLM_CASE(64)
        VLLM_CASE(96)
        VLLM_CASE(128)
        VLLM_CASE(192)
        VLLM_CASE(256)
        VLLM_CASE(512)
        default:
            TORCH_CHECK(false, "vllm paged decode: unsupported head size ", head_size);
    }
#undef VLLM_CASE
    return true;
}

// Device int32 tensor [0, 1, ..., n]: cu_seqlens_q of a decode batch.
inline at::Tensor arange_int(const at::Tensor& like, int n) {
    static std::mutex mu;
    static auto* cache = new std::unordered_map<int, at::Tensor>();
    std::lock_guard<std::mutex> lock(mu);
    at::Tensor& t = (*cache)[like.device().index()];
    if (!t.defined() || t.numel() < n + 1) {
        t = at::arange(std::max(n + 1, 1025), like.options().dtype(at::kInt));
        // Later calls may run on other streams.
        c10::xpu::getCurrentXPUStream(like.device().index()).synchronize();
    }
    return t.narrow(0, 0, n + 1);
}

}  // namespace splitkv

std::vector<at::Tensor>
mha_fwd(
        at::Tensor &q,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        std::optional<at::Generator> gen_) {

    auto device_idx = q.device().index();
    compat::select_device(device_idx);

    // check inputs
    TORCH_CHECK(p_dropout < 1.0f, "FlashAttention dropout probability must be less than 1.0");

    q = ensure_contiguous(q);
    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size_og = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(k.size(-1) == head_size_og, "Key head dimension must match Query head dimension");
    TORCH_CHECK(v.size(-1) == head_size_og, "Value head dimension must match Query head dimension");
    TORCH_CHECK(head_size_og <= 256 || head_size_og == 512,
                "FlashAttention XPU only supports head dimension up to 256 or exactly 512. Got: " + std::to_string(head_size_og));

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);

    // Decode and short queries: split-KV kernel.
    // With one query causal masking is a no-op (bottom-right aligned) and
    // window_size_right cannot mask anything.
    bool skv_causal = false;
    if (seqlen_q >= 1 && seqlen_q <= splitkv::kMaxQueries &&
        (seqlen_q == 1 ||
         (splitkv::multi_query_mask(is_causal, window_size_left,
                                    window_size_right, skv_causal) &&
          (!skv_causal || seqlen_k >= seqlen_q))) &&
        p_dropout == 0.0f && !return_softmax &&
        !alibi_slopes_.has_value() && softcap == 0.0f &&
        seqlen_k > 0 && seqlen_k < (1 << 24) &&
        splitkv::supported(q, k, v, /*paged=*/false)) {
        auto queue = c10::xpu::getCurrentXPUStream(device_idx).queue();
        int num_splits = splitkv::plan_layout(
            queue, batch_size, seqlen_q, num_heads, num_heads_k, q.size(-1),
            seqlen_k, /*paged=*/false, q.stride(1));
        if (num_splits > 0) {
            at::Tensor out;
            if (out_.has_value() && out_->is_contiguous() &&
                out_->sizes() == q.sizes() && out_->dtype() == q_dtype) {
                out = out_.value();
            } else {
                out = at::empty_like(q);
            }
            auto opts = q.options().dtype(torch::kFloat32);
            at::Tensor softmax_lse = at::empty({batch_size, num_heads, seqlen_q}, opts);

            splitkv::decode_args_t a;
            a.q = q.data_ptr();
            a.k = k.data_ptr();
            a.v = v.data_ptr();
            a.out = out.data_ptr();
            a.softmax_lse = softmax_lse.data_ptr<float>();
            a.q_per_batch = seqlen_q;
            a.causal = skv_causal;
            a.seqlen_k = seqlen_k;
            a.batch = batch_size;
            a.num_heads_q = num_heads;
            a.num_heads_kv = num_heads_k;
            a.head_size = head_size_og;
            a.head_size_v = head_size_og;
            // stride(1) is arbitrary when seqlen_q == 1.
            a.q_row_stride = seqlen_q == 1 ? q.stride(0) : q.stride(1);
            a.q_head_stride = q.stride(2);
            a.k_batch_stride = k.stride(0);
            a.k_seq_stride = k.stride(1);
            a.k_head_stride = k.stride(2);
            a.v_batch_stride = v.stride(0);
            a.v_seq_stride = v.stride(1);
            a.v_head_stride = v.stride(2);
            // softmax_lse is [batch, Hq, seqlen_q]; row = b * seqlen_q + j.
            a.lse_batch_stride = int64_t(num_heads - 1) * seqlen_q;
            a.lse_head_stride = seqlen_q;
            a.lse_row_stride = 1;
            a.softmax_scale = softmax_scale;
            a.window_size_left =
                (window_size_left >= 0 && window_size_left < seqlen_k) ? window_size_left : -1;
            a.num_splits = num_splits;
            splitkv::run(queue, q, batch_size * seqlen_q, a);

            if (out_.has_value() && !out.is_same(out_.value())) {
                out_.value().copy_(out);
            }
            at::Tensor S_dmask = torch::empty({0}, q.options());
            at::Tensor rng_state = at::empty({2}, q.options().dtype(at::kLong));
            return {out, softmax_lse, S_dmask, rng_state};
        }
    }

    // XPU requires head_size to be a multiple of 32
    const int head_size_padded = round_multiple(head_size_og, 32);
    const bool needs_padding = (head_size_og != head_size_padded);

    at::Tensor q_padded = q;
    at::Tensor k_padded = k;
    at::Tensor v_padded = v;

    // Apply padding if needed
    if (needs_padding) {
        const int pad_size = head_size_padded - head_size_og;
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, pad_size}));
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, pad_size}));
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, pad_size}));
    }

    at::Tensor out_padded;
    if (out_.has_value()) {
        auto out_val = out_.value();
        if (head_size_og != head_size_padded) {
            const int pad_size = head_size_padded - head_size_og;
            out_padded = torch::nn::functional::pad(out_val, torch::nn::functional::PadFuncOptions({0, pad_size}));
        } else {
            out_padded = out_val;
        }
    } else {
        out_padded = torch::zeros_like(q_padded);
    }

    // Allocate softmax_lse output tensor: (batch_size, num_heads, seqlen_q)
    auto opts = q.options().dtype(torch::kFloat32);
    at::Tensor softmax_lse = torch::zeros({batch_size, num_heads, seqlen_q}, opts);

    const bool is_local = (window_size_left != -1) || (window_size_right != -1);

    q_padded = ensure_contiguous(q_padded);
    k_padded = ensure_contiguous(k_padded);
    v_padded = ensure_contiguous(v_padded);

    auto queue = c10::xpu::getCurrentXPUStream(device_idx).queue();

    // Handle dropout RNG state
    uint64_t philox_seed = 0;
    uint64_t philox_offset = 0;

    // Always allocate rng_state to match the meta function and ensure
    // torch.compile compatibility (inductor expects all return values to be
    // valid tensors).
    at::Tensor rng_state = at::empty({2}, q.options().dtype(at::kLong));

    if (p_dropout > 0.0f) {
        // Calculate counter offset for RNG: batch_size * num_heads * 32
        int64_t counter_offset = batch_size * num_heads * 32;
        auto [seed, offset] = get_philox_state(gen_, counter_offset);
        philox_seed = seed;
        philox_offset = offset;
        
        // Store RNG state for backward pass
        rng_state[0] = static_cast<int64_t>(philox_seed);
        rng_state[1] = static_cast<int64_t>(philox_offset);
    }

    // Allocate S_dmask for return_softmax (attention matrix with dropout sign-bit encoding)
    // seqlen_q_rounded must be >= the kernel's Q tile size, otherwise the S_dmask
    // write in the mainloop will go out of bounds for rows in the padding region
    // of the tile.  The Q tile size depends on head_size (see fmha_utils.hpp):
    const int q_tile_size = (head_size_og <= 32) ? 64 : (head_size_og <= 128) ? 128 : 256;
    const int seqlen_q_rounded = round_multiple(seqlen_q, q_tile_size);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);
    at::Tensor S_dmask;
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        S_dmask = torch::empty({batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, q.options());
    } else {
        S_dmask = torch::empty({0}, q.options());
    }

    cutlass_fmha_fwd_fix_impl(
        queue,
        q_padded, k_padded, v_padded, out_padded,
        softmax_lse,
        softmax_scale,
        window_size_left, window_size_right,
        is_causal, is_local,
        p_dropout, philox_seed, philox_offset, nullptr,
        return_softmax ? S_dmask.data_ptr() : nullptr,
        seqlen_q_rounded, seqlen_k_rounded);

    // Strip padding from output back to original head_size
    at::Tensor out = needs_padding
        ? out_padded.index({torch::indexing::Slice(), torch::indexing::Slice(),
                            torch::indexing::Slice(), torch::indexing::Slice(0, head_size_og)})
                    .contiguous()
        : ensure_contiguous(out_padded);

    return {out, softmax_lse, S_dmask, rng_state};
  }


std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &q,     // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,     // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,     // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,  // b x h x seqlen_q
        std::optional<at::Tensor> &dq_,
        std::optional<at::Tensor> &dk_,
        std::optional<at::Tensor> &dv_,
        std::optional<at::Tensor> &alibi_slopes_,
        const float p_dropout,
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic,
        std::optional<at::Generator> gen_,
        std::optional<at::Tensor> &rng_state) {

    auto device_idx = q.device().index();
    compat::select_device(device_idx);

    auto q_dtype = q.dtype();
    TORCH_CHECK(p_dropout < 1.0f, "FlashAttention dropout probability must be less than 1.0");

    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention backward only supports fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);
    CHECK_DEVICE(out);
    CHECK_DEVICE(dout);

    // Ensure inputs are contiguous (k, v may come from kv-packed slices)
    at::Tensor k_contig = ensure_contiguous(k);
    at::Tensor v_contig = ensure_contiguous(v);
    at::Tensor q_contig = ensure_contiguous(q);
    at::Tensor out_contig = ensure_contiguous(out);
    at::Tensor dout_contig = ensure_contiguous(dout);

    TORCH_CHECK(q_contig.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k_contig.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v_contig.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out_contig.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout_contig.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    const auto sizes = q_contig.sizes();
    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size_og = sizes[3];
    const int seqlen_k = k_contig.size(1);
    const int num_heads_k = k_contig.size(2);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size_og <= 256 || head_size_og == 512,
                "FlashAttention XPU backward only supports head dimension up to 256 or exactly 512. Got: " + std::to_string(head_size_og));
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // XPU requires head_size to be a multiple of 32
    const int head_size_padded = round_multiple(head_size_og, 32);
    const bool needs_padding = (head_size_og != head_size_padded);
    const int pad_size = head_size_padded - head_size_og;

    auto maybe_pad = [&](const at::Tensor &t) -> at::Tensor {
        return needs_padding
            ? torch::nn::functional::pad(t, torch::nn::functional::PadFuncOptions({0, pad_size}))
            : t;
    };

    at::Tensor q_padded = maybe_pad(q_contig);
    at::Tensor k_padded = maybe_pad(k_contig);
    at::Tensor v_padded = maybe_pad(v_contig);
    at::Tensor out_padded = maybe_pad(out_contig);
    at::Tensor dout_padded = maybe_pad(dout_contig);

    auto opts = q_contig.options();

    // Allocate output gradients
    // When no padding needed and user provides contiguous tensors, reuse them directly
    at::Tensor dq, dk, dv;
    at::Tensor dq_work, dk_work, dv_work;  // Working tensors for kernel
    bool dq_needs_copy = false, dk_needs_copy = false, dv_needs_copy = false;

    auto get_or_alloc = [&](const c10::optional<at::Tensor> &opt,
                            const std::vector<int64_t> &sizes,
                            bool &needs_copy) -> at::Tensor {
        if (opt.has_value() && opt.value().is_contiguous()) {
            return opt.value();
        }
        needs_copy = opt.has_value();
        return torch::empty(sizes, opts);
    };

    if (!needs_padding && num_heads_k == num_heads) {
        // Optimal path: no padding, no MQA/GQA - can write directly to output
        dq_work = get_or_alloc(dq_, {batch_size, seqlen_q, num_heads, head_size_og}, dq_needs_copy);
        dk_work = get_or_alloc(dk_, {batch_size, seqlen_k, num_heads_k, head_size_og}, dk_needs_copy);
        dv_work = get_or_alloc(dv_, {batch_size, seqlen_k, num_heads_k, head_size_og}, dv_needs_copy);
    } else {
        // Need padding or MQA/GQA - allocate with padded size
        dq_work = torch::empty({batch_size, seqlen_q, num_heads, head_size_padded}, opts);
        dk_work = torch::empty({batch_size, seqlen_k, num_heads_k, head_size_padded}, opts);
        dv_work = torch::empty({batch_size, seqlen_k, num_heads_k, head_size_padded}, opts);
    }

    // Allocate intermediate buffers
    at::Tensor softmax_d = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    // Handle MQA/GQA
    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {
        dk_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size_padded}, opts);
        dv_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size_padded}, opts);
    } else {
        dk_expanded = dk_work;
        dv_expanded = dv_work;
    }

    const bool is_local = (window_size_left != -1) || (window_size_right != -1);

    auto queue = c10::xpu::getCurrentXPUStream(device_idx).queue();

    // Handle dropout - get RNG state from forward pass
    uint64_t philox_seed = 0;
    uint64_t philox_offset = 0;
    if (p_dropout > 0.0f && rng_state.has_value()) {
        auto rng_state_val = rng_state.value();
        philox_seed = static_cast<uint64_t>(rng_state_val[0].item<int64_t>());
        philox_offset = static_cast<uint64_t>(rng_state_val[1].item<int64_t>());
    }

    // Call the cutlass backward implementation
    cutlass_fmha_bwd_fix_impl(
        queue,
        dout_padded, q_padded, k_padded, v_padded, out_padded, softmax_lse,
        dq_work, dk_expanded, dv_expanded, softmax_d,
        softmax_scale, window_size_left, window_size_right, is_causal, is_local,
        p_dropout, philox_seed, philox_offset, deterministic);

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(dk_work, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size_padded}), {3});
        at::sum_out(dv_work, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size_padded}), {3});
    }

    auto slice_head = [&](const at::Tensor &t) -> at::Tensor {
        return t.index({torch::indexing::Slice(), torch::indexing::Slice(),
                        torch::indexing::Slice(), torch::indexing::Slice(0, head_size_og)}).contiguous();
    };
    auto copy_if_provided = [&](const c10::optional<at::Tensor> &opt, const at::Tensor &src) {
        if (opt.has_value()) {
            opt.value().copy_(src);
        }
    };

    // Remove padding from output gradients if needed
    if (needs_padding || num_heads_k != num_heads) {
        dq = slice_head(dq_work);
        dk = slice_head(dk_work);
        dv = slice_head(dv_work);
        copy_if_provided(dq_, dq);
        copy_if_provided(dk_, dk);
        copy_if_provided(dv_, dv);
    } else {
        // Optimal path: no padding, no MQA/GQA - results already in place
        dq = dq_work;
        dk = dk_work;
        dv = dv_work;
        // Copy to non-contiguous user tensors if needed
        if (dq_needs_copy) dq_.value().copy_(dq);
        if (dk_needs_copy) dk_.value().copy_(dk);
        if (dv_needs_copy) dv_.value().copy_(dv);
    }

    return {dq, dk, dv, softmax_d};
}


std::vector<at::Tensor>
mha_varlen_fwd(
              at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
              const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
              const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
              std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
              const at::Tensor &cu_seqlens_q,  // b+1
              const at::Tensor &cu_seqlens_k,  // b+1
              std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
              std::optional<const at::Tensor> &leftpad_k_, // batch_size
              std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
              std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
              int max_seqlen_q,
              const int max_seqlen_k,
              const float p_dropout,
              const float softmax_scale,
              const bool zero_tensors,
              bool is_causal,
              int window_size_left,
              int window_size_right,
              const float softcap,
              const bool return_softmax,
              std::optional<at::Generator> gen_) {

    auto device_idx = q.device().index();
    compat::select_device(device_idx);

    // check inputs
    TORCH_CHECK(p_dropout < 1.0f, "FlashAttention dropout probability must be less than 1.0");

    q = ensure_contiguous(q);
    const auto sizes = q.sizes();
    const int total_q = sizes[0];
    const int num_heads = sizes[1];
    const int head_size_og = sizes[2];
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);
    const int batch_size = cu_seqlens_q.numel() - 1;

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(k.size(-1) == head_size_og, "Key head dimension must match Query head dimension");
    TORCH_CHECK(v.size(-1) == head_size_og, "Value head dimension must match Query head dimension");
    TORCH_CHECK(head_size_og <= 256 || head_size_og == 512,
                "FlashAttention XPU varlen only supports head dimension up to 256 or exactly 512. Got: " + std::to_string(head_size_og));

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);

    // Decode and short queries: split-KV kernel, one decode row per query
    // (see mha_fwd). With longer queries (chunked prefill mixed with decode)
    // the split-KV kernel takes the one-query batches and the generic kernel
    // skips them.
    const bool skv_paged = block_table_.has_value() && block_table_->defined();
    int skv_q_per_batch = 0;  // 0: split-KV kernel not used
    bool skv_mixed = false;
    bool skv_causal = false;
    int skv_splits = 0;
    {
        auto int_dev = [&](const at::Tensor& t) {
            return t.is_xpu() && t.scalar_type() == at::kInt && t.dim() == 1;
        };
        if (max_seqlen_q >= 1 && p_dropout == 0.0f && !return_softmax &&
            !alibi_slopes_.has_value() && softcap == 0.0f &&
            !(seqused_k.has_value() && seqused_k->defined()) &&
            !(leftpad_k_.has_value() && leftpad_k_->defined()) &&
            max_seqlen_k > 0 && (skv_paged || max_seqlen_k < (1 << 24)) &&
            int_dev(cu_seqlens_q) && int_dev(cu_seqlens_k) &&
            cu_seqlens_k.numel() == batch_size + 1 &&
            (!skv_paged || (block_table_->is_xpu() && block_table_->dim() == 2 &&
                            block_table_->scalar_type() == at::kInt &&
                            block_table_->size(0) >= batch_size)) &&
            k.dim() == (skv_paged ? 4 : 3) && v.dim() == k.dim() &&
            splitkv::supported(q, k, v, skv_paged)) {
            if (max_seqlen_q == 1 ||
                (max_seqlen_q <= splitkv::kMaxQueries &&
                 splitkv::multi_query_mask(is_causal, window_size_left,
                                           window_size_right, skv_causal))) {
                skv_q_per_batch = max_seqlen_q;
            } else if (skv_paged || total_q != total_k) {
                // Without paging, total_q == total_k means every batch
                // attends only to its own queries: nothing to decode.
                skv_q_per_batch = 1;
                skv_mixed = true;
                skv_causal = false;
            }
        }
        if (skv_q_per_batch > 0) {
            auto queue = c10::xpu::getCurrentXPUStream(device_idx).queue();
            // k is [total_k, Hkv, D] or, paged, [num_blocks, page, Hkv, D].
            skv_splits = splitkv::plan_layout(
                queue, batch_size, skv_q_per_batch, num_heads, k.size(-2),
                q.size(-1), max_seqlen_k, skv_paged, q.stride(0));
            // The generic kernel runs the one-query batches of a mixed batch
            // with its prefill tiles; the split-KV kernel is always cheaper.
            if (skv_mixed) skv_splits = std::max(skv_splits, 1);
            if (skv_splits == 0) skv_q_per_batch = 0;
        }
    }

    // Runs the split-KV kernel into out ([total_q, Hq, D], contiguous) and
    // softmax_lse ([Hq, total_q]).
    auto run_splitkv = [&](at::Tensor& out, at::Tensor& softmax_lse) {
        auto queue = c10::xpu::getCurrentXPUStream(device_idx).queue();
        at::Tensor cu_q = cu_seqlens_q.contiguous();
        at::Tensor cu_k = cu_seqlens_k.contiguous();
        at::Tensor block_table;

        splitkv::decode_args_t a;
        a.q = q.data_ptr();
        a.k = k.data_ptr();
        a.v = v.data_ptr();
        a.out = out.data_ptr();
        a.softmax_lse = softmax_lse.data_ptr<float>();
        a.cu_seqlens_q = cu_q.data_ptr<int>();
        a.q_per_batch = skv_q_per_batch;
        a.causal = skv_causal;
        a.cu_seqlens_k = cu_k.data_ptr<int>();
        a.seqlen_k = max_seqlen_k;
        a.batch = batch_size;
        a.num_heads_q = num_heads;
        a.num_heads_kv = k.size(-2);
        a.head_size = head_size_og;
        a.head_size_v = head_size_og;
        a.q_row_stride = q.stride(0);
        a.q_head_stride = q.stride(1);
        if (skv_paged) {
            block_table = block_table_->contiguous();
            a.block_table = block_table.data_ptr<int>();
            a.block_table_batch_stride = block_table.stride(0);
            a.page_size = k.size(1);
            a.max_pages_per_seq = block_table.size(1);
            a.page_stride_rows = k.stride(0) / k.stride(1);
            a.total_rows_kv = int64_t(k.size(0)) * a.page_stride_rows;
            a.k_batch_stride = k.stride(0);
            a.k_seq_stride = k.stride(1);
            a.k_head_stride = k.stride(2);
            a.v_batch_stride = v.stride(0);
            a.v_seq_stride = v.stride(1);
            a.v_head_stride = v.stride(2);
        } else {
            a.k_seq_stride = k.stride(0);
            a.k_head_stride = k.stride(1);
            a.v_seq_stride = v.stride(0);
            a.v_head_stride = v.stride(1);
        }
        a.lse_head_stride = total_q;
        a.lse_row_stride = 1;
        a.softmax_scale = softmax_scale;
        a.window_size_left =
            (window_size_left >= 0 && window_size_left < max_seqlen_k) ? window_size_left : -1;
        a.num_splits = skv_splits;
        splitkv::run(queue, q, total_q, a);
    };

    if (skv_q_per_batch > 0 && !skv_mixed) {
        at::Tensor out;
        if (out_.has_value() && out_->is_contiguous() &&
            out_->sizes() == q.sizes() && out_->dtype() == q_dtype) {
            out = out_.value();
        } else {
            out = at::empty_like(q);
        }
        auto opts = q.options().dtype(torch::kFloat32);
        at::Tensor softmax_lse = torch::empty({num_heads, total_q}, opts);
        // Paged decode on Xe35: vllm's kernel (see mha_fwd_kvcache).
        bool done = false;
        if (skv_paged && skv_q_per_batch == 1 && splitkv::vllm_supported(q)) {
            auto queue = c10::xpu::getCurrentXPUStream(device_idx).queue();
            at::Tensor cu_q = cu_seqlens_q.contiguous();
            at::Tensor cu_k = cu_seqlens_k.contiguous();
            at::Tensor block_table = block_table_->contiguous();
            done = splitkv::run_vllm(
                queue, q, total_q, q.stride(0), q.stride(1), k, v, out,
                block_table, cu_q, cu_k, /*kv_cumulative=*/true, batch_size,
                max_seqlen_k, softmax_scale,
                (window_size_left >= 0 && window_size_left < max_seqlen_k)
                    ? window_size_left : -1,
                -1, /*forced_splits=*/0, softmax_lse,
                /*lse_stride=*/total_q, /*lse_row_stride=*/1);
        }
        if (!done) run_splitkv(out, softmax_lse);

        if (out_.has_value() && !out.is_same(out_.value())) {
            out_.value().copy_(out);
        }
        at::Tensor S_dmask = torch::empty({0}, q.options());
        at::Tensor rng_state = at::empty({2}, q.options().dtype(at::kLong));
        return {out, softmax_lse, S_dmask, rng_state};
    }

    // XPU requires head_size to be a multiple of 32
    const int head_size_padded = round_multiple(head_size_og, 32);
    const bool needs_padding = (head_size_og != head_size_padded);
    const int pad_size = head_size_padded - head_size_og;

    auto maybe_pad = [&](const at::Tensor& t) -> at::Tensor {
        return needs_padding
            ? torch::nn::functional::pad(t, torch::nn::functional::PadFuncOptions({0, pad_size}))
            : t;
    };

    at::Tensor q_padded = maybe_pad(q);
    at::Tensor k_padded = maybe_pad(k);
    at::Tensor v_padded = maybe_pad(v);

    at::Tensor out_padded = out_.has_value() ? maybe_pad(out_.value())
                                             : torch::zeros_like(q_padded);

    const bool is_local = (window_size_left != -1) || (window_size_right != -1);
    const bool is_paged = block_table_.has_value() && block_table_->defined();

    q_padded = ensure_contiguous(q_padded);
    k_padded = ensure_contiguous(k_padded);
    v_padded = ensure_contiguous(v_padded);

    // Allocate softmax_lse output tensor
    auto opts = q.options().dtype(torch::kFloat32);
    at::Tensor softmax_lse = torch::zeros({num_heads, total_q}, opts);

    auto queue = c10::xpu::getCurrentXPUStream(device_idx).queue();

    uint64_t philox_seed = 0;
    uint64_t philox_offset = 0;
    at::Tensor rng_state = at::empty({2}, q.options().dtype(at::kLong));

    if (p_dropout > 0.0f) {
        int64_t counter_offset = batch_size * num_heads * 32;
        auto [seed, offset] = get_philox_state(gen_, counter_offset);
        philox_seed = seed;
        philox_offset = offset;
        rng_state[0] = static_cast<int64_t>(philox_seed);
        rng_state[1] = static_cast<int64_t>(philox_offset);
    }

    const int q_tile_size = (head_size_og <= 32) ? 64 : (head_size_og <= 128) ? 128 : 256;
    const int k_tile_size = (max_seqlen_q == 1 && !is_paged) ? 512 : 128;
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, q_tile_size);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, k_tile_size);
    at::Tensor S_dmask;
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        S_dmask = torch::empty({batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, q.options());
    } else {
        S_dmask = torch::empty({0}, q.options());
    }

    cutlass_fmha_fwd_varlen_impl(
        queue,
        q_padded, k_padded, v_padded, out_padded,
        softmax_lse,
        block_table_,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        softmax_scale,
        window_size_left, window_size_right,
        true, is_paged, is_causal, is_local,
        p_dropout, philox_seed, philox_offset, nullptr,
        return_softmax ? S_dmask.data_ptr() : nullptr,
        seqlen_q_rounded, seqlen_k_rounded,
        /*min_seqlen_q=*/skv_mixed ? 2 : 0);

    // Strip padding from output back to original head_size
    at::Tensor out = needs_padding
        ? out_padded.index({torch::indexing::Slice(), torch::indexing::Slice(),
                            torch::indexing::Slice(0, head_size_og)})
                    .contiguous()
        : ensure_contiguous(out_padded);

    if (skv_mixed) {
        run_splitkv(out, softmax_lse);
    }

    return {out, softmax_lse, S_dmask, rng_state};
  }

std::vector<at::Tensor>
mha_varlen_bwd(
        const at::Tensor &dout,
        const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        const at::Tensor &out,
        const at::Tensor &softmax_lse,
        std::optional<at::Tensor> &dq_,
        std::optional<at::Tensor> &dk_,
        std::optional<at::Tensor> &dv_,
        const at::Tensor &cu_seqlens_q,
        const at::Tensor &cu_seqlens_k,
        std::optional<at::Tensor> &alibi_slopes_,
        const int max_seqlen_q,
        const int max_seqlen_k,
        const float p_dropout,
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic,
        std::optional<at::Generator> gen_,
        std::optional<at::Tensor> &rng_state) {

    auto device_idx = q.device().index();
    compat::select_device(device_idx);

    TORCH_CHECK(p_dropout < 1.0f, "FlashAttention dropout probability must be less than 1.0");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention varlen backward only supports fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);
    CHECK_DEVICE(out);
    CHECK_DEVICE(dout);
    CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.dim() == 3, "q must have shape (total_q, num_heads, head_size)");
    TORCH_CHECK(k.dim() == 3, "k must have shape (total_k, num_heads_k, head_size)");
    TORCH_CHECK(v.dim() == 3, "v must have shape (total_k, num_heads_k, head_size)");
    TORCH_CHECK(out.sizes() == q.sizes(), "out must have the same shape as q");
    TORCH_CHECK(dout.sizes() == q.sizes(), "dout must have the same shape as q");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have the same shape");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1,
                "cu_seqlens_q and cu_seqlens_k must be 1D tensors");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == at::ScalarType::Int &&
                cu_seqlens_k.scalar_type() == at::ScalarType::Int,
                "cu_seqlens_q and cu_seqlens_k must be int32 tensors");
    TORCH_CHECK(cu_seqlens_q.numel() == cu_seqlens_k.numel(),
                "cu_seqlens_q and cu_seqlens_k must have the same length");
    TORCH_CHECK(max_seqlen_q > 0 && max_seqlen_k > 0,
                "max_seqlen_q and max_seqlen_k must be positive");

    at::Tensor q_contig = ensure_contiguous(q);
    at::Tensor k_contig = ensure_contiguous(k);
    at::Tensor v_contig = ensure_contiguous(v);
    at::Tensor out_contig = ensure_contiguous(out);
    at::Tensor dout_contig = ensure_contiguous(dout);
    at::Tensor softmax_lse_contig = ensure_contiguous(softmax_lse);
    at::Tensor cu_q_contig = ensure_contiguous(cu_seqlens_q);
    at::Tensor cu_k_contig = ensure_contiguous(cu_seqlens_k);

    TORCH_CHECK(q_contig.stride(-1) == 1, "q must have contiguous last dimension");
    TORCH_CHECK(k_contig.stride(-1) == 1, "k must have contiguous last dimension");
    TORCH_CHECK(v_contig.stride(-1) == 1, "v must have contiguous last dimension");
    TORCH_CHECK(out_contig.stride(-1) == 1, "out must have contiguous last dimension");
    TORCH_CHECK(dout_contig.stride(-1) == 1, "dout must have contiguous last dimension");

    const int total_q = q_contig.size(0);
    const int total_k = k_contig.size(0);
    const int num_heads = q_contig.size(1);
    const int head_size_og = q_contig.size(2);
    const int num_heads_k = k_contig.size(1);
    const int batch_size = cu_q_contig.numel() - 1;

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size_og <= 256 || head_size_og == 512,
                "FlashAttention XPU varlen backward only supports head dimension up to 256 or exactly 512. Got: " + std::to_string(head_size_og));
    TORCH_CHECK(k_contig.size(2) == head_size_og && v_contig.size(2) == head_size_og,
                "k and v head dimensions must match q");
    TORCH_CHECK(num_heads % num_heads_k == 0,
                "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(softmax_lse_contig.dim() == 2 && softmax_lse_contig.size(0) == num_heads &&
                softmax_lse_contig.size(1) == total_q,
                "softmax_lse must have shape (num_heads, total_q)");
    const int head_size_padded = round_multiple(head_size_og, 32);
    const bool needs_padding = (head_size_og != head_size_padded);
    const int pad_size = head_size_padded - head_size_og;

    auto maybe_pad = [&](const at::Tensor &t) -> at::Tensor {
        return needs_padding
            ? torch::nn::functional::pad(t, torch::nn::functional::PadFuncOptions({0, pad_size}))
            : t;
    };

    at::Tensor q_padded = maybe_pad(q_contig);
    at::Tensor k_padded = maybe_pad(k_contig);
    at::Tensor v_padded = maybe_pad(v_contig);
    at::Tensor out_padded = maybe_pad(out_contig);
    at::Tensor dout_padded = maybe_pad(dout_contig);

    auto opts = q_contig.options();
    at::Tensor dq, dk, dv;
    at::Tensor dq_work, dk_work, dv_work;
    bool dq_needs_copy = false, dk_needs_copy = false, dv_needs_copy = false;

    auto get_or_alloc = [&](const c10::optional<at::Tensor> &opt,
                            const std::vector<int64_t> &sizes,
                            bool &needs_copy) -> at::Tensor {
        if (opt.has_value() && opt.value().is_contiguous()) {
            return opt.value();
        }
        needs_copy = opt.has_value();
        return torch::empty(sizes, opts);
    };

    if (!needs_padding && num_heads_k == num_heads) {
        dq_work = get_or_alloc(dq_, {total_q, num_heads, head_size_og}, dq_needs_copy);
        dk_work = get_or_alloc(dk_, {total_k, num_heads_k, head_size_og}, dk_needs_copy);
        dv_work = get_or_alloc(dv_, {total_k, num_heads_k, head_size_og}, dv_needs_copy);
    } else {
        dq_work = torch::empty({total_q, num_heads, head_size_padded}, opts);
        dk_work = torch::empty({total_k, num_heads_k, head_size_padded}, opts);
        dv_work = torch::empty({total_k, num_heads_k, head_size_padded}, opts);
    }

    dq_work.zero_();
    at::Tensor softmax_d = torch::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {
        dk_expanded = torch::empty({total_k, num_heads, head_size_padded}, opts);
        dv_expanded = torch::empty({total_k, num_heads, head_size_padded}, opts);
    } else {
        dk_expanded = dk_work;
        dv_expanded = dv_work;
    }

    const bool is_local = (window_size_left != -1) || (window_size_right != -1);
    auto queue = c10::xpu::getCurrentXPUStream(device_idx).queue();

    uint64_t philox_seed = 0;
    uint64_t philox_offset = 0;
    if (p_dropout > 0.0f && rng_state.has_value()) {
        auto rng_state_val = rng_state.value();
        philox_seed = static_cast<uint64_t>(rng_state_val[0].item<int64_t>());
        philox_offset = static_cast<uint64_t>(rng_state_val[1].item<int64_t>());
    }

    cutlass_fmha_bwd_varlen_impl(
        queue,
        dout_padded, q_padded, k_padded, v_padded, out_padded, softmax_lse_contig,
        cu_q_contig, cu_k_contig,
        dq_work, dk_expanded, dv_expanded, softmax_d,
        softmax_scale,
        max_seqlen_q, max_seqlen_k,
        window_size_left, window_size_right, is_causal, is_local,
        p_dropout, philox_seed, philox_offset, deterministic);

    if (num_heads_k != num_heads) {
        at::sum_out(dk_work, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size_padded}), {2});
        at::sum_out(dv_work, at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size_padded}), {2});
    }

    auto slice_head = [&](const at::Tensor &t) -> at::Tensor {
        return t.index({torch::indexing::Slice(), torch::indexing::Slice(),
                        torch::indexing::Slice(0, head_size_og)}).contiguous();
    };
    auto copy_if_provided = [&](const c10::optional<at::Tensor> &opt, const at::Tensor &src) {
        if (opt.has_value()) {
            opt.value().copy_(src);
        }
    };

    if (needs_padding || num_heads_k != num_heads) {
        dq = slice_head(dq_work);
        dk = slice_head(dk_work);
        dv = slice_head(dv_work);
        copy_if_provided(dq_, dq);
        copy_if_provided(dk_, dk);
        copy_if_provided(dv_, dv);
    } else {
        dq = dq_work;
        dk = dk_work;
        dv = dv_work;
        if (dq_needs_copy) dq_.value().copy_(dq);
        if (dk_needs_copy) dk_.value().copy_(dk);
        if (dv_needs_copy) dv_.value().copy_(dv);
    }

    return {dq, dk, dv, softmax_d};
}

std::vector<at::Tensor>
mha_fwd_kvcache(
        at::Tensor &q,
        const at::Tensor &kcache,
        const at::Tensor &vcache,
        std::optional<const at::Tensor> &k_,
        std::optional<const at::Tensor> &v_,
        std::optional<const at::Tensor> &seqlens_k_,
        std::optional<const at::Tensor> &rotary_cos_,
        std::optional<const at::Tensor> &rotary_sin_,
        std::optional<const at::Tensor> &cache_batch_idx_,
        std::optional<const at::Tensor> &leftpad_k_,
        std::optional<at::Tensor> &block_table_,
        std::optional<at::Tensor> &alibi_slopes_,
        std::optional<at::Tensor> &out_,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        bool is_rotary_interleaved,
        int num_splits) {
    auto device_idx = q.device().index();
    compat::select_device(device_idx);

    TORCH_CHECK(!alibi_slopes_.has_value(),
                "FlashAttention KVCache on XPU does not support alibi_slopes.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention KVCache only supports fp16 and bf16 data type");
    TORCH_CHECK(kcache.dtype() == q_dtype, "query and key cache must have the same dtype");
    TORCH_CHECK(vcache.dtype() == q_dtype, "query and value cache must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);
    TORCH_CHECK(q.stride(-1) == 1, "Query must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Key cache must have contiguous last dimension");
    TORCH_CHECK(vcache.stride(-1) == 1, "Value cache must have contiguous last dimension");

    const bool paged_KV = block_table_.has_value();
    at::Tensor block_table;
    if (paged_KV) {
        TORCH_CHECK(!cache_batch_idx_.has_value(), "Paged KVcache does not support cache_batch_idx");
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
    }

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size_og = sizes[3];

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int page_block_size = !paged_KV ? 1 : kcache.size(1);
    const int seqlen_k = !paged_KV ? kcache.size(1) : max_num_blocks_per_seq * page_block_size;
    const int num_heads_k = kcache.size(2);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256 || head_size_og == 512,
                "FlashAttention KVCache only supports head dimension up to 256 or exactly 512");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }
    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // Decode and short queries: split-KV kernel (see mha_fwd). num_splits:
    // 0 = heuristic, 1 = generic kernel, N > 1 = N splits.
    bool skv_causal = false;
    if (seqlen_q >= 1 && seqlen_q <= splitkv::kMaxQueries &&
        (seqlen_q == 1 ||
         // The generic kernel ignores window_size_right without a left
         // window.
         splitkv::multi_query_mask(
             is_causal, window_size_left,
             window_size_left < 0 && !is_causal ? -1 : window_size_right,
             skv_causal)) &&
        num_splits != 1 && softcap == 0.0f &&
        !rotary_cos_.has_value() && !cache_batch_idx_.has_value() &&
        !leftpad_k_.has_value() && seqlen_k > 0 &&
        (paged_KV || seqlen_k < (1 << 24)) &&
        (!out_.has_value() ||
         (out_->is_contiguous() && out_->sizes() == q.sizes() &&
          out_->dtype() == q_dtype && out_->is_xpu())) &&
        (!seqlens_k_.has_value() ||
         (seqlens_k_->is_xpu() && seqlens_k_->scalar_type() == at::kInt &&
          seqlens_k_->numel() == batch_size)) &&
        (!paged_KV || block_table.size(0) >= batch_size) &&
        (paged_KV || kcache.size(0) >= batch_size) &&
        splitkv::supported(q, kcache, vcache, paged_KV)) {
        bool knew_ok = true;
        at::Tensor knew, vnew;
        if (k_.has_value()) {
            knew = k_.value();
            vnew = v_.has_value() ? v_.value() : at::Tensor();
            knew_ok = vnew.defined() && seqlens_k_.has_value() &&
                knew.is_xpu() && vnew.is_xpu() &&
                knew.dim() == 4 && knew.sizes() == vnew.sizes() &&
                knew.size(0) == batch_size && knew.size(2) == num_heads_k &&
                knew.size(3) == head_size_og &&
                knew.scalar_type() == q_dtype && vnew.scalar_type() == q_dtype &&
                knew.stride(-1) == 1 && vnew.stride(-1) == 1;
        }
        auto queue = c10::xpu::getCurrentXPUStream(device_idx).queue();
        // Row stride of q as passed to the kernel (qc below).
        const int64_t q_row_stride =
            q.stride(0) == seqlen_q * q.stride(1) ? q.stride(1)
                                                  : int64_t(num_heads) * q.size(3);
        int splits = splitkv::plan_layout(
            queue, batch_size, seqlen_q, num_heads, num_heads_k, head_size_og,
            seqlen_k, paged_KV, q_row_stride,
            num_splits > 1 ? num_splits : 0);
        if (knew_ok && splits > 0) {
            // Paged decode without appended KV on Xe35: vllm's kernel. The
            // right window does not mask anything for a single query.
            if (seqlen_q == 1 && paged_KV && seqlens_k_.has_value() &&
                !knew.defined() && splitkv::vllm_supported(q)) {
                at::Tensor out = out_.has_value() ? out_.value()
                                                  : at::empty(q.sizes(), q.options());
                auto softmax_lse = torch::empty({batch_size, num_heads, 1},
                                                q.options().dtype(at::kFloat));
                at::Tensor seqlens_k = seqlens_k_->contiguous();
                at::Tensor block_table_c = block_table.contiguous();
                at::Tensor cu_q = splitkv::arange_int(q, batch_size);
                if (splitkv::run_vllm(
                        queue, q, batch_size, q.stride(0), q.stride(2), kcache,
                        vcache, out, block_table_c, cu_q, seqlens_k,
                        /*kv_cumulative=*/false, batch_size, seqlen_k,
                        softmax_scale, window_size_left, -1,
                        num_splits > 1 ? num_splits : 0, softmax_lse,
                        /*lse_stride=*/1, /*lse_row_stride=*/num_heads))
                    return {out, softmax_lse};
            }
            // Speculative decode (causal, no window): vllm's packed path folds
            // the queries of a batch into the head dim, position-major per KV
            // head, so the cache is read once for all of them.
            if (seqlen_q > 1 && skv_causal && window_size_left < 0 &&
                paged_KV && seqlens_k_.has_value() && !knew.defined() &&
                splitkv::vllm_supported(q)) {
                const int hg = num_heads / num_heads_k;
                const int hp = num_heads * seqlen_q;
                at::Tensor qp = q.unflatten(2, {num_heads_k, hg})
                                    .permute({0, 2, 1, 3, 4})
                                    .reshape({batch_size, hp, head_size_og});
                at::Tensor outp = at::empty({batch_size, hp, head_size_og}, q.options());
                auto lsep = torch::empty({batch_size, hp}, q.options().dtype(at::kFloat));
                at::Tensor seqlens_k = seqlens_k_->contiguous();
                at::Tensor block_table_c = block_table.contiguous();
                at::Tensor cu_q = splitkv::arange_int(q, batch_size);
                if (splitkv::run_vllm(
                        queue, qp, batch_size, qp.stride(0), qp.stride(1), kcache,
                        vcache, outp, block_table_c, cu_q, seqlens_k,
                        /*kv_cumulative=*/false, batch_size, seqlen_k,
                        softmax_scale, -1, -1, num_splits > 1 ? num_splits : 0,
                        lsep, /*lse_stride=*/1, /*lse_row_stride=*/hp, seqlen_q)) {
                    at::Tensor out = out_.has_value() ? out_.value()
                                                      : at::empty(q.sizes(), q.options());
                    out.unflatten(2, {num_heads_k, hg}).copy_(
                        outp.view({batch_size, num_heads_k, seqlen_q, hg, head_size_og})
                            .permute({0, 2, 1, 3, 4}));
                    at::Tensor softmax_lse =
                        lsep.view({batch_size, num_heads_k, seqlen_q, hg})
                            .permute({0, 1, 3, 2})
                            .reshape({batch_size, num_heads, seqlen_q});
                    return {out, softmax_lse};
                }
            }
            // Several queries: rows of q must be evenly spaced and out
            // contiguous; one query: empty_like keeps q's batch stride.
            at::Tensor qc = (seqlen_q == 1 || q.stride(0) == seqlen_q * q.stride(1))
                ? q : q.contiguous();
            at::Tensor out = out_.has_value() ? out_.value()
                : seqlen_q == 1 ? at::empty_like(q) : at::empty(q.sizes(), q.options());
            auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q},
                                            q.options().dtype(at::kFloat));
            at::Tensor seqlens_k = seqlens_k_.has_value()
                ? seqlens_k_->contiguous() : at::Tensor();
            at::Tensor block_table_c = paged_KV ? block_table.contiguous() : at::Tensor();

            const int seqlen_new = knew.defined() ? knew.size(1) : 0;
            if (seqlen_new > 0) {
                flash_xpu_splitkv::append_args_t ap;
                ap.k_new = knew.data_ptr();
                ap.v_new = vnew.data_ptr();
                ap.k_cache = kcache.data_ptr();
                ap.v_cache = vcache.data_ptr();
                ap.cache_seqlens = seqlens_k.data_ptr<int>();
                if (paged_KV) {
                    ap.block_table = block_table_c.data_ptr<int>();
                    ap.block_table_batch_stride = block_table_c.stride(0);
                    ap.page_size = page_block_size;
                    ap.max_pages_per_seq = max_num_blocks_per_seq;
                }
                ap.batch = batch_size;
                ap.seqlen_new = seqlen_new;
                ap.num_heads_kv = num_heads_k;
                ap.head_size = head_size_og;
                ap.seqlen_cache = seqlen_k;
                ap.kn_batch_stride = knew.stride(0);
                ap.kn_seq_stride = knew.stride(1);
                ap.kn_head_stride = knew.stride(2);
                ap.vn_batch_stride = vnew.stride(0);
                ap.vn_seq_stride = vnew.stride(1);
                ap.vn_head_stride = vnew.stride(2);
                ap.kc_batch_stride = kcache.stride(0);
                ap.kc_seq_stride = kcache.stride(1);
                ap.kc_head_stride = kcache.stride(2);
                ap.vc_batch_stride = vcache.stride(0);
                ap.vc_seq_stride = vcache.stride(1);
                ap.vc_head_stride = vcache.stride(2);
                flash_xpu_splitkv::splitkv_append_kv(queue, ap);
            }

            splitkv::decode_args_t a;
            a.q = qc.data_ptr();
            a.k = kcache.data_ptr();
            a.v = vcache.data_ptr();
            a.out = out.data_ptr();
            a.softmax_lse = softmax_lse.data_ptr<float>();
            a.q_per_batch = seqlen_q;
            a.causal = skv_causal;
            if (seqlens_k.defined()) {
                a.seqused_k = seqlens_k.data_ptr<int>();
                a.seqlen_k_add = seqlen_new;
            }
            a.seqlen_k = seqlen_k;
            if (paged_KV) {
                a.block_table = block_table_c.data_ptr<int>();
                a.block_table_batch_stride = block_table_c.stride(0);
                a.page_size = page_block_size;
                a.max_pages_per_seq = max_num_blocks_per_seq;
                a.page_stride_rows = kcache.stride(0) / kcache.stride(1);
                a.total_rows_kv = int64_t(kcache.size(0)) * a.page_stride_rows;
            }
            a.batch = batch_size;
            a.num_heads_q = num_heads;
            a.num_heads_kv = num_heads_k;
            a.head_size = head_size_og;
            a.head_size_v = head_size_og;
            a.q_row_stride = seqlen_q == 1 ? qc.stride(0) : qc.stride(1);
            a.q_head_stride = qc.stride(2);
            a.k_batch_stride = kcache.stride(0);
            a.k_seq_stride = kcache.stride(1);
            a.k_head_stride = kcache.stride(2);
            a.v_batch_stride = vcache.stride(0);
            a.v_seq_stride = vcache.stride(1);
            a.v_head_stride = vcache.stride(2);
            a.lse_batch_stride = int64_t(num_heads - 1) * seqlen_q;
            a.lse_head_stride = seqlen_q;
            a.lse_row_stride = 1;
            a.softmax_scale = softmax_scale;
            a.empty_lse = -std::numeric_limits<float>::infinity();
            a.window_size_left = window_size_left;
            a.num_splits = splits;
            splitkv::run(queue, qc, batch_size * seqlen_q, a);
            return {out, softmax_lse};
        }
    }

    const int head_size_padded = round_multiple(head_size_og, 32);
    const bool needs_padding = (head_size_og != head_size_padded);
    const int pad_size = head_size_padded - head_size_og;

    auto maybe_pad = [&](const at::Tensor& t) -> at::Tensor {
        return needs_padding
            ? torch::nn::functional::pad(t, torch::nn::functional::PadFuncOptions({0, pad_size}))
            : t;
    };

    at::Tensor q_padded = maybe_pad(ensure_contiguous(q));
    at::Tensor kcache_padded = maybe_pad(ensure_contiguous(kcache));
    at::Tensor vcache_padded = maybe_pad(ensure_contiguous(vcache));

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        if (needs_padding) { out = maybe_pad(out); }
    } else {
        out = torch::zeros_like(q_padded);
    }

    auto opts = q.options();
    auto softmax_lse = torch::full({batch_size, num_heads, seqlen_q},
                                    -std::numeric_limits<float>::infinity(),
                                    opts.dtype(at::kFloat));

    // Handle new K/V
    at::Tensor k_padded, v_padded;
    int seqlen_new = 0;
    if (k_.has_value()) {
        TORCH_CHECK(v_.has_value(), "If key is supplied, value must also be passed in");
        TORCH_CHECK(seqlens_k_.has_value(), "If key is supplied, seqlens_k must also be passed in");
        auto k = k_.value();
        auto v = v_.value();
        TORCH_CHECK(k.dtype() == q_dtype && v.dtype() == q_dtype);
        CHECK_DEVICE(k); CHECK_DEVICE(v);
        TORCH_CHECK(k.stride(-1) == 1 && v.stride(-1) == 1);
        seqlen_new = k.size(1);
        k_padded = maybe_pad(ensure_contiguous(k));
        v_padded = maybe_pad(ensure_contiguous(v));
    }

    at::Tensor seqlens_k;
    if (seqlens_k_.has_value()) {
        seqlens_k = seqlens_k_.value();
        TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
        CHECK_DEVICE(seqlens_k);
    }

    at::Tensor rotary_cos, rotary_sin;
    int rotary_dim = 0;
    const bool has_rotary = rotary_cos_.has_value();
    if (has_rotary) {
        TORCH_CHECK(k_.has_value(), "If rotary cos/sin are provided, new key/value must also be provided");
        TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
        rotary_cos = ensure_contiguous(rotary_cos_.value());
        rotary_sin = ensure_contiguous(rotary_sin_.value());
        CHECK_DEVICE(rotary_cos); CHECK_DEVICE(rotary_sin);
        rotary_dim = rotary_cos.size(1) * 2;
        TORCH_CHECK(rotary_dim <= head_size_og, "rotary_dim must be <= headdim");
        TORCH_CHECK(rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
        TORCH_CHECK(rotary_cos.scalar_type() == q_dtype && rotary_sin.scalar_type() == q_dtype);
    }

    at::Tensor cache_batch_idx;
    if (cache_batch_idx_.has_value()) {
        cache_batch_idx = cache_batch_idx_.value();
        CHECK_DEVICE(cache_batch_idx);
        TORCH_CHECK(cache_batch_idx.scalar_type() == torch::kInt32);
    }

    at::Tensor leftpad_k;
    if (leftpad_k_.has_value()) {
        TORCH_CHECK(!paged_KV, "Paged KV and leftpad_k are not supported together");
        leftpad_k = leftpad_k_.value();
        CHECK_DEVICE(leftpad_k);
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
    }

    bool fuse_knew = k_.has_value() && seqlen_new > 0;

    // Dispatch to kernel.  Paged caches are now passed natively (block_table
    // routed straight through to the kernel, no host gather).
    auto queue = c10::xpu::getCurrentXPUStream(device_idx).queue();
    const bool is_local = (window_size_left >= 0);

    std::optional<at::Tensor> cache_batch_idx_opt;
    if (cache_batch_idx_.has_value()) {
        cache_batch_idx_opt = cache_batch_idx;
    }

    std::optional<at::Tensor> leftpad_k_opt;
    if (leftpad_k_.has_value()) {
        leftpad_k_opt = leftpad_k;
    }

    // For paths where new KV is appended in-kernel, pass knew/vnew through.
    std::optional<at::Tensor> knew_opt, vnew_opt;
    if (fuse_knew) {
        knew_opt = k_padded;
        vnew_opt = v_padded;
    }

    std::optional<at::Tensor> block_table_opt;
    if (paged_KV) {
        block_table_opt = block_table;
    }

    std::optional<at::Tensor> rotary_cos_opt, rotary_sin_opt;
    if (fuse_knew && has_rotary) {
        rotary_cos_opt = rotary_cos;
        rotary_sin_opt = rotary_sin;
    }

    cutlass_fmha_fwd_kvcache_impl(
        queue,
        q_padded, kcache_padded, vcache_padded,
        out, softmax_lse,
        seqlens_k, cache_batch_idx_opt, leftpad_k_opt,
        knew_opt, vnew_opt,
        block_table_opt, rotary_cos_opt, rotary_sin_opt,
        fuse_knew ? rotary_dim : 0, is_rotary_interleaved, seqlen_k,
        softmax_scale, window_size_left, window_size_right,
        is_causal, is_local);

    // Update seqlens_k after kernel completes (for fused scatter path)
    if (fuse_knew) {
        seqlens_k = seqlens_k + seqlen_new;
    }

    if (needs_padding) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(),
                         torch::indexing::Slice(), torch::indexing::Slice(0, head_size_og)})
                 .contiguous();
        if (out_.has_value()) { out_.value().copy_(out); }
        if (fuse_knew) {
            // The fused kernel updates the padded cache buffer; publish valid dims back to the user cache.
            kcache.copy_(kcache_padded.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                              torch::indexing::Slice(), torch::indexing::Slice(0, head_size_og)}));
            vcache.copy_(vcache_padded.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                              torch::indexing::Slice(), torch::indexing::Slice(0, head_size_og)}));
        }
    }

    return {out, softmax_lse};
}

}  // namespace FLASH_NAMESPACE

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
std::vector<torch::Tensor>
mha_fwd(
    torch::Tensor &q, 
    const torch::Tensor &k, 
    const torch::Tensor &v,
    c10::optional<torch::Tensor> out_,
    c10::optional<torch::Tensor> alibi_slopes_,
    const double p_dropout, 
    const double softmax_scale, 
    bool is_causal,
    const int64_t window_size_left, 
    const int64_t window_size_right,
    const double softcap, 
    const bool return_softmax,
    c10::optional<at::Generator> gen_) {
    return FLASH_NAMESPACE::mha_fwd(
      q, 
      k, 
      v, 
      out_, 
      alibi_slopes_, 
      static_cast<float>(p_dropout),
      static_cast<float>(softmax_scale), 
      is_causal,
      static_cast<int>(window_size_left), 
      static_cast<int>(window_size_right),
      static_cast<float>(softcap), 
      return_softmax,
      gen_
    );
}

std::vector<torch::Tensor>
mha_varlen_bwd(
        const torch::Tensor &dout,
        const torch::Tensor &q,
        const torch::Tensor &k,
        const torch::Tensor &v,
        const torch::Tensor &out,
        const torch::Tensor &softmax_lse,
        const c10::optional<torch::Tensor> &dq_,
        const c10::optional<torch::Tensor> &dk_,
        const c10::optional<torch::Tensor> &dv_,
        const torch::Tensor &cu_seqlens_q,
        const torch::Tensor &cu_seqlens_k,
        const c10::optional<torch::Tensor> &alibi_slopes_,
        const int64_t max_seqlen_q,
        const int64_t max_seqlen_k,
        const double p_dropout,
        const double softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        const int64_t window_size_left,
        const int64_t window_size_right,
        const double softcap,
        const bool deterministic,
        c10::optional<at::Generator> gen_,
        const c10::optional<torch::Tensor> &rng_state) {
    auto to_std_opt = [](const c10::optional<at::Tensor>& opt) -> std::optional<at::Tensor> {
        return opt.has_value() ? std::optional<at::Tensor>(opt.value()) : std::nullopt;
    };

    auto dq_opt = to_std_opt(dq_);
    auto dk_opt = to_std_opt(dk_);
    auto dv_opt = to_std_opt(dv_);
    auto alibi_opt = to_std_opt(alibi_slopes_);
    auto rng_opt = to_std_opt(rng_state);

    return FLASH_NAMESPACE::mha_varlen_bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq_opt,
        dk_opt,
        dv_opt,
        cu_seqlens_q,
        cu_seqlens_k,
        alibi_opt,
        static_cast<int>(max_seqlen_q),
        static_cast<int>(max_seqlen_k),
        static_cast<float>(p_dropout),
        static_cast<float>(softmax_scale),
        zero_tensors,
        is_causal,
        static_cast<int>(window_size_left),
        static_cast<int>(window_size_right),
        static_cast<float>(softcap),
        deterministic,
        gen_,
        rng_opt);
}

std::vector<torch::Tensor>
mha_varlen_fwd(
    torch::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const torch::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    const torch::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    std::optional<torch::Tensor> out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const torch::Tensor &cu_seqlens_q,  // b+1
    const torch::Tensor &cu_seqlens_k,  // b+1
    std::optional<torch::Tensor> seqused_k, // b. If given, only this many elements of each batch element's keys are used.
    std::optional<torch::Tensor> leftpad_k_, // batch_size
    std::optional<torch::Tensor> block_table_, // batch_size x max_num_blocks_per_seq
    std::optional<torch::Tensor> alibi_slopes_, // num_heads or b x num_heads
    int64_t max_seqlen_q,
    const int64_t max_seqlen_k,
    const double p_dropout,
    const double softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    const double softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen_) {    
    return FLASH_NAMESPACE::mha_varlen_fwd(
        const_cast<at::Tensor &>(q), 
        k, 
        v, 
        out_, 
        cu_seqlens_q, 
        cu_seqlens_k,
        seqused_k,
        // reinterpret_cast: c10::optional<T> and std::optional<const T> share
        // the same ABI layout; this avoids an unnecessary copy.
        reinterpret_cast<std::optional<const at::Tensor>&>(leftpad_k_),
        block_table_,
        alibi_slopes_,
        static_cast<int>(max_seqlen_q),
        static_cast<int>(max_seqlen_k),
        static_cast<float>(p_dropout),
        static_cast<float>(softmax_scale),
        zero_tensors,
        is_causal,
        static_cast<int>(window_size_left), 
        static_cast<int>(window_size_right),
        static_cast<float>(softcap),
        return_softmax,
        gen_
    );
}

std::vector<torch::Tensor>
mha_bwd(const torch::Tensor &dout,
        const torch::Tensor &q,
        const torch::Tensor &k,
        const torch::Tensor &v,
        const torch::Tensor &out,
        const torch::Tensor &softmax_lse,
        const c10::optional<torch::Tensor> &dq_,
        const c10::optional<torch::Tensor> &dk_,
        const c10::optional<torch::Tensor> &dv_,
        const c10::optional<torch::Tensor> &alibi_slopes_,
        const double p_dropout,
        const double softmax_scale,
        const bool is_causal,
        const int64_t window_size_left,
        const int64_t window_size_right,
        const double softcap,
        const bool deterministic,
        c10::optional<at::Generator> gen_,
        const c10::optional<torch::Tensor> &rng_state) {
    // Helper: convert c10::optional<T> -> std::optional<T>
    auto to_std_opt = [](const c10::optional<at::Tensor>& opt) -> std::optional<at::Tensor> {
        return opt.has_value() ? std::optional<at::Tensor>(opt.value()) : std::nullopt;
    };

    auto dq_opt    = to_std_opt(dq_);
    auto dk_opt    = to_std_opt(dk_);
    auto dv_opt    = to_std_opt(dv_);
    auto alibi_opt = to_std_opt(alibi_slopes_);
    auto rng_opt   = to_std_opt(rng_state);
    
    return FLASH_NAMESPACE::mha_bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq_opt,
        dk_opt,
        dv_opt,
        alibi_opt,
        static_cast<float>(p_dropout),
        static_cast<float>(softmax_scale),
        is_causal,
        static_cast<int>(window_size_left),
        static_cast<int>(window_size_right),
        static_cast<float>(softcap),
        deterministic,
        gen_,
        rng_opt
    );
}

std::vector<torch::Tensor>
mha_fwd_kvcache(
        const torch::Tensor &q,
        const torch::Tensor &kcache,
        const torch::Tensor &vcache,
        const c10::optional<torch::Tensor> &k_,
        const c10::optional<torch::Tensor> &v_,
        const c10::optional<torch::Tensor> &seqlens_k_,
        const c10::optional<torch::Tensor> &rotary_cos_,
        const c10::optional<torch::Tensor> &rotary_sin_,
        const c10::optional<torch::Tensor> &cache_batch_idx_,
        const c10::optional<torch::Tensor> &leftpad_k_,
        const c10::optional<torch::Tensor> &block_table_,
        const c10::optional<torch::Tensor> &alibi_slopes_,
        const c10::optional<torch::Tensor> &out_,
        const double softmax_scale,
        bool is_causal,
        const int64_t window_size_left,
        const int64_t window_size_right,
        const double softcap,
        bool is_rotary_interleaved,
        const int64_t num_splits) {
    // Convert c10::optional -> std::optional for the internal API
    auto to_std_opt = [](const c10::optional<torch::Tensor>& opt) -> std::optional<at::Tensor> {
        return opt.has_value() ? std::optional<at::Tensor>(opt.value()) : std::nullopt;
    };
    auto to_std_opt_const = [](const c10::optional<torch::Tensor>& opt) -> std::optional<const at::Tensor> {
        return opt.has_value() ? std::optional<const at::Tensor>(opt.value()) : std::nullopt;
    };

    at::Tensor q_mut = q;
    auto k_opt = to_std_opt_const(k_);
    auto v_opt = to_std_opt_const(v_);
    auto seqlens_opt = to_std_opt_const(seqlens_k_);
    auto rotary_cos_opt = to_std_opt_const(rotary_cos_);
    auto rotary_sin_opt = to_std_opt_const(rotary_sin_);
    auto cache_batch_idx_opt = to_std_opt_const(cache_batch_idx_);
    auto leftpad_k_opt = to_std_opt_const(leftpad_k_);
    auto block_table_opt = to_std_opt(block_table_);
    auto alibi_opt = to_std_opt(alibi_slopes_);
    auto out_opt = to_std_opt(out_);

    return FLASH_NAMESPACE::mha_fwd_kvcache(
        q_mut, kcache, vcache,
        k_opt, v_opt, seqlens_opt,
        rotary_cos_opt, rotary_sin_opt,
        cache_batch_idx_opt, leftpad_k_opt,
        block_table_opt, alibi_opt, out_opt,
        static_cast<float>(softmax_scale),
        is_causal,
        static_cast<int>(window_size_left),
        static_cast<int>(window_size_right),
        static_cast<float>(softcap),
        is_rotary_interleaved,
        static_cast<int>(num_splits)
    );
}