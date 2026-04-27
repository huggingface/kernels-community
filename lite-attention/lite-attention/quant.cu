/******************************************************************************
 * TMA-based Q/K quantization with mean centering for attention (FP16/BF16)
 * 
 * - Q: per-block quantization (kBlockM tokens share a scale per head)
 * - K: smooth by subtracting channel-wise mean, then per-block quantization
 ******************************************************************************/

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/array.h>

#include <cub/cub.cuh>

#include <assert.h>
#include <stdint.h>
#include <float.h>
#include <math.h>

#include "tile_size.h"

using namespace cute;

// -----------------------------------------------------------------------------
// Layouts & Storage
// -----------------------------------------------------------------------------
using Shape4D = cute::Shape<int, int, int, int>;

template<int ROWS, int HEAD_DIM, typename Element>
struct SmemConfig {
    using SmemLayoutIn = Layout<Shape<Int<ROWS>, Int<HEAD_DIM>>,
                                Stride<Int<HEAD_DIM>, _1>>;
    using SmemLayoutOut = Layout<Shape<Int<ROWS>, Int<HEAD_DIM>>,
                                 Stride<Int<HEAD_DIM>, _1>>;
};

template <int HEAD_DIM, int NUM_THREADS, typename Element, typename SmemLayoutIn, typename SmemLayoutOut>
struct __align__(128) TmaSharedStorage {
    using BlockReduce = cub::BlockReduce<float, NUM_THREADS>;

    __align__(128) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutIn>>   smem_in;
    __align__(128) cute::ArrayEngine<int8_t,  cute::cosize_v<SmemLayoutOut>>  smem_out;

    __align__(16) typename BlockReduce::TempStorage cub_storage;

    __align__(16) cute::uint64_t tma_load_barrier;
    __align__(16) float tile_inv_scale;
    __align__(16) float tile_means[HEAD_DIM];
};

// Convert Vector<Element> -> Vector<float>
template <typename Element, size_t N>
__device__ __forceinline__ cute::array<float, N>
vec_to_float(cute::array<Element, N> const& src) {
    cute::array<float, N> dst;
    #pragma unroll
    for (size_t i = 0; i < N; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
    return dst;
}

// Quantize Vector<float> -> Packed uint64_t (8 x int8)
__device__ __forceinline__ uint64_t
float_vec_to_packed_int8(cute::array<float, 8UL> const& src, float scale) {
    union {
        int8_t  i8[8];
        uint64_t u64;
    } packed;

    #pragma unroll
    for (size_t i = 0; i < 8; ++i) {
        int32_t res;
        float x = src[i] * scale;
        // Converts float32 to signed int8 with round-to-nearest-integer, clamps to -128 to 127
        asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(res) : "f"(x));
        packed.i8[i] = static_cast<int8_t>(res);
    }
    return packed.u64;
}

// -----------------------------------------------------------------------------
// Kernel 1: Quantize Q
// -----------------------------------------------------------------------------
template <
    int HEAD_DIM,
    int BLOCK_M,
    int NUM_THREADS,
    typename Element,
    typename TmaLoad,
    typename TmaStore>
__global__ void __launch_bounds__(NUM_THREADS)
quantize_q_kernel(
    CUTE_GRID_CONSTANT TmaLoad const tma_load,
    CUTE_GRID_CONSTANT TmaStore const tma_store,
    int seqlen_q,
    int num_heads,
    int batch,
    float* __restrict__ q_scales,
    int num_seq_tiles_q,
    double q_scale)
{
    using SmemLayoutIn = typename SmemConfig<BLOCK_M, HEAD_DIM, Element>::SmemLayoutIn;
    using SmemLayoutOut = typename SmemConfig<BLOCK_M, HEAD_DIM, Element>::SmemLayoutOut;
    using SharedStorage = TmaSharedStorage<HEAD_DIM, NUM_THREADS, Element, SmemLayoutIn, SmemLayoutOut>;
    using BlockReduce = typename SharedStorage::BlockReduce;

    extern __shared__ char shared_mem[];
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_mem);

    int seq_tile_idx = blockIdx.x;
    int bidh         = blockIdx.y;
    int bidb         = blockIdx.z;

    Shape4D shape_Q = make_shape(seqlen_q, HEAD_DIM, num_heads, batch);
    Tensor mQ       = tma_load.get_tma_tensor(shape_Q)(_, _, bidh, bidb);
    Tensor gQ       = local_tile(mQ, Shape<Int<BLOCK_M>, Int<HEAD_DIM>>{}, make_coord(seq_tile_idx, 0));
    Tensor sQ_in    = make_tensor(make_smem_ptr(storage.smem_in.begin()), SmemLayoutIn{});

    auto tma_load_slice = tma_load.get_slice(_0{});
    if (threadIdx.x == 0) {
        initialize_barrier(storage.tma_load_barrier, 1);
        set_barrier_transaction_bytes(storage.tma_load_barrier, sizeof(Element) * size(sQ_in));
        copy(tma_load.with(storage.tma_load_barrier), tma_load_slice.partition_S(gQ), tma_load_slice.partition_D(sQ_in));
    }
    __syncthreads();
    wait_barrier(storage.tma_load_barrier, 0);
    __syncthreads();

    Tensor sQ_flat = make_tensor(sQ_in.data(), make_layout(size(sQ_in)));
    Tensor sQ_vec  = recast<cute::array<Element, 8>>(sQ_flat);
    uint64_t* smem_out_ptr = reinterpret_cast<uint64_t*>(storage.smem_out.begin());

    float local_max = 0.0f;

    #pragma unroll
    for (int i = threadIdx.x; i < size(sQ_vec); i += NUM_THREADS) {
        cute::array<float, 8> vals = vec_to_float(sQ_vec(i));
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            local_max = fmaxf(local_max, fabsf(vals[k]));
        }
    }

    float tile_max_raw = BlockReduce(storage.cub_storage).Reduce(local_max, cub::Max());

    if (threadIdx.x == 0) {
        double real_scale = (static_cast<double>(tile_max_raw) * q_scale) / 127.0;
        real_scale = fmax(real_scale, 1e-6);

        int idx_scale = bidb * num_heads * num_seq_tiles_q + bidh * num_seq_tiles_q + seq_tile_idx;
        q_scales[idx_scale] = static_cast<float>(real_scale);

        double inv_scale_d = 127.0 / fmax(static_cast<double>(tile_max_raw), 1e-6);
        storage.tile_inv_scale = static_cast<float>(inv_scale_d);
    }
    __syncthreads();

    float inv_scale = storage.tile_inv_scale;

    #pragma unroll
    for (int i = threadIdx.x; i < size(sQ_vec); i += NUM_THREADS) {
        cute::array<float, 8> vals = vec_to_float(sQ_vec(i));
        smem_out_ptr[i] = float_vec_to_packed_int8(vals, inv_scale);
    }
    __syncthreads();

    // TMA Store
    Tensor mQ_out = tma_store.get_tma_tensor(shape_Q)(_, _, bidh, bidb);
    Tensor gQ_out = local_tile(mQ_out, Shape<Int<BLOCK_M>, Int<HEAD_DIM>>{}, make_coord(seq_tile_idx, 0));
    Tensor sQ_out = make_tensor(make_smem_ptr(storage.smem_out.begin()), SmemLayoutOut{});

    auto tma_store_slice = tma_store.get_slice(_0{});
    if (threadIdx.x == 0) {
        tma_store_fence();
        copy(tma_store, tma_store_slice.partition_S(sQ_out), tma_store_slice.partition_D(gQ_out));
        tma_store_wait<0>();
    }
}

// -----------------------------------------------------------------------------
// Kernel 2: Quantize K
// -----------------------------------------------------------------------------
template <
    int HEAD_DIM,
    int BLOCK_N,
    int NUM_THREADS,
    typename Element,
    typename TmaLoad,
    typename TmaStore>
__global__ void __launch_bounds__(NUM_THREADS)
quantize_k_kernel(
    CUTE_GRID_CONSTANT TmaLoad const tma_load,
    CUTE_GRID_CONSTANT TmaStore const tma_store,
    int seqlen_k,
    int num_heads,
    int batch,
    Shape4D shape_K,
    const float* __restrict__ k_mean,
    float* __restrict__ k_scales,
    int num_seq_tiles_k)
{
    static_assert((NUM_THREADS * 8) % HEAD_DIM == 0, "NUM_THREADS * 8 must be divisible by HEAD_DIM to eliminate cycle logic");

    using SmemLayoutIn = typename SmemConfig<BLOCK_N, HEAD_DIM, Element>::SmemLayoutIn;
    using SmemLayoutOut = typename SmemConfig<BLOCK_N, HEAD_DIM, Element>::SmemLayoutOut;
    using SharedStorage = TmaSharedStorage<HEAD_DIM, NUM_THREADS, Element, SmemLayoutIn, SmemLayoutOut>;
    using BlockReduce = typename SharedStorage::BlockReduce;

    extern __shared__ char shared_mem[];
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_mem);

    int seq_tile_idx = blockIdx.x;
    int bidh         = blockIdx.y;
    int bidb         = blockIdx.z;

    int bh_idx = bidb * num_heads + bidh;

    // Load Means (pre-computed externally)
    for (int c = threadIdx.x; c < HEAD_DIM; c += NUM_THREADS) {
        storage.tile_means[c] = k_mean[bh_idx * HEAD_DIM + c];
    }
    __syncthreads();

    // 1. TMA Load
    Tensor mK = tma_load.get_tma_tensor(shape_K)(_, _, bidh, bidb);
    Tensor gK = local_tile(mK, Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{}, make_coord(seq_tile_idx, 0));
    Tensor sK_in = make_tensor(make_smem_ptr(storage.smem_in.begin()), SmemLayoutIn{});

    auto tma_load_slice = tma_load.get_slice(_0{});
    if (threadIdx.x == 0) {
        initialize_barrier(storage.tma_load_barrier, 1);
        set_barrier_transaction_bytes(storage.tma_load_barrier, sizeof(Element) * size(sK_in));
        copy(tma_load.with(storage.tma_load_barrier), tma_load_slice.partition_S(gK), tma_load_slice.partition_D(sK_in));
    }
    __syncthreads();
    wait_barrier(storage.tma_load_barrier, 0);
    __syncthreads();

    float mean_cache[8];
    int start_col_idx = (threadIdx.x * 8) % HEAD_DIM;

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        mean_cache[k] = storage.tile_means[start_col_idx + k];
    }

    Tensor sK_flat = make_tensor(sK_in.data(), make_layout(size(sK_in)));
    Tensor sK_vec  = recast<cute::array<Element, 8>>(sK_flat);
    uint64_t* smem_out_ptr = reinterpret_cast<uint64_t*>(storage.smem_out.begin());

    float local_max = 0.0f;
    int valid_elems = min(BLOCK_N, seqlen_k - seq_tile_idx * BLOCK_N) * HEAD_DIM;

    // Pass 1: Find Max
    #pragma unroll
    for (int i = threadIdx.x; i < size(sK_vec); i += NUM_THREADS) {
        int flat_elem_idx = i * 8;

        if (flat_elem_idx < valid_elems) {
            cute::array<float, 8> vals = vec_to_float(sK_vec(i));

            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float centered = vals[k] - mean_cache[k];
                local_max = fmaxf(local_max, fabsf(centered));
            }
        }
    }

    float tile_max = BlockReduce(storage.cub_storage).Reduce(local_max, cub::Max());

    if (threadIdx.x == 0) {
        double scale = static_cast<double>(tile_max) / 127.0;
        scale = fmax(scale, 1e-6);
        double inv_scale_d = 1.0 / scale;
        storage.tile_inv_scale = static_cast<float>(inv_scale_d);

        int idx_scale = bh_idx * num_seq_tiles_k + seq_tile_idx;
        k_scales[idx_scale] = static_cast<float>(scale);
    }
    __syncthreads();

    // Pass 2: Quantize
    float inv_scale = storage.tile_inv_scale;

    #pragma unroll
    for (int i = threadIdx.x; i < size(sK_vec); i += NUM_THREADS) {
        int flat_elem_idx = i * 8;

        if (flat_elem_idx < valid_elems) {
            cute::array<float, 8> vals = vec_to_float(sK_vec(i));

            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                vals[k] = vals[k] - mean_cache[k];
            }

            smem_out_ptr[i] = float_vec_to_packed_int8(vals, inv_scale);
        } else {
            smem_out_ptr[i] = 0;
        }
    }
    __syncthreads();

    // 3. TMA Store
    Tensor mK_out = tma_store.get_tma_tensor(shape_K)(_, _, bidh, bidb);
    Tensor gK_out = local_tile(mK_out, Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{}, make_coord(seq_tile_idx, 0));
    Tensor sK_out = make_tensor(make_smem_ptr(storage.smem_out.begin()), SmemLayoutOut{});

    auto tma_store_slice = tma_store.get_slice(_0{});
    if (threadIdx.x == 0) {
        tma_store_fence();
        copy(tma_store, tma_store_slice.partition_S(sK_out), tma_store_slice.partition_D(gK_out));
        tma_store_wait<0>();
    }
}

// -----------------------------------------------------------------------------
// Host Dispatchers
// -----------------------------------------------------------------------------

// Thread count: 288 for HEAD_DIM 96/192 (divisibility), 256 otherwise
template<int HEAD_DIM>
constexpr int get_num_threads() { return (HEAD_DIM == 96 || HEAD_DIM == 192) ? 288 : 256; }

// Q-only launcher
template <int HEAD_DIM, int BLOCK_M, typename Element>
void launch_quantize_q_config(
    const Element* Q, int8_t* Q_q, float* q_scales,
    int batch, int seqlen_q, int num_heads,
    double q_scale,
    cudaStream_t stream)
{
    constexpr int kNumThreads = get_num_threads<HEAD_DIM>();
    using SmemConfigQ = SmemConfig<BLOCK_M, HEAD_DIM, Element>;
    int num_seq_tiles_q = (seqlen_q + BLOCK_M - 1) / BLOCK_M;

    Shape4D shape_Q = make_shape(seqlen_q, HEAD_DIM, num_heads, batch);
    int64_t stride_s_q = (int64_t)num_heads * HEAD_DIM;
    int64_t stride_b_q = (int64_t)seqlen_q * stride_s_q;
    auto stride_Q = make_stride(stride_s_q, Int<1>{}, Int<HEAD_DIM>{}, stride_b_q);

    Tensor mQ   = make_tensor(make_gmem_ptr(Q),   make_layout(shape_Q, stride_Q));
    Tensor mQ_q = make_tensor(make_gmem_ptr(Q_q), make_layout(shape_Q, stride_Q));

    using SmemLayoutInQ  = typename SmemConfigQ::SmemLayoutIn;
    using SmemLayoutOutQ = typename SmemConfigQ::SmemLayoutOut;

    auto tma_load_Q = make_tma_copy<Element>(
        SM90_TMA_LOAD{}, mQ, SmemLayoutInQ{}, Shape<Int<BLOCK_M>, Int<HEAD_DIM>>{}, _1{});
    auto tma_store_Q = make_tma_copy<int8_t>(
        SM90_TMA_STORE{}, mQ_q, SmemLayoutOutQ{}, Shape<Int<BLOCK_M>, Int<HEAD_DIM>>{}, _1{});

    using SharedStorageQ = TmaSharedStorage<HEAD_DIM, kNumThreads, Element, SmemLayoutInQ, SmemLayoutOutQ>;
    int smem_size_q = sizeof(SharedStorageQ);

    if (smem_size_q >= 48 * 1024) {
        cudaFuncSetAttribute(quantize_q_kernel<HEAD_DIM, BLOCK_M, kNumThreads, Element, decltype(tma_load_Q), decltype(tma_store_Q)>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_q);
    }

    dim3 grid_q(num_seq_tiles_q, num_heads, batch);
    quantize_q_kernel<HEAD_DIM, BLOCK_M, kNumThreads, Element>
        <<<grid_q, kNumThreads, smem_size_q, stream>>>(
        tma_load_Q, tma_store_Q, seqlen_q, num_heads, batch, q_scales, num_seq_tiles_q, q_scale);
}

// K-only launcher (assumes k_mean passed in from Python)
template <int HEAD_DIM, int BLOCK_N, typename Element>
void launch_quantize_k_config(
    const Element* K, int8_t* K_q, const float* k_mean, float* k_scales,
    int batch, int seqlen_k, int num_heads,
    cudaStream_t stream)
{
    constexpr int kNumThreads = get_num_threads<HEAD_DIM>();
    using SmemConfigK = SmemConfig<BLOCK_N, HEAD_DIM, Element>;
    int num_seq_tiles_k = (seqlen_k + BLOCK_N - 1) / BLOCK_N;

    Shape4D shape_K = make_shape(seqlen_k, HEAD_DIM, num_heads, batch);
    int64_t stride_s_k = (int64_t)num_heads * HEAD_DIM;
    int64_t stride_b_k = (int64_t)seqlen_k * stride_s_k;
    auto stride_K = make_stride(stride_s_k, Int<1>{}, Int<HEAD_DIM>{}, stride_b_k);

    Tensor mK   = make_tensor(make_gmem_ptr(K),   make_layout(shape_K, stride_K));
    Tensor mK_q = make_tensor(make_gmem_ptr(K_q), make_layout(shape_K, stride_K));

    using SmemLayoutInK  = typename SmemConfigK::SmemLayoutIn;
    using SmemLayoutOutK = typename SmemConfigK::SmemLayoutOut;

    auto tma_load_K = make_tma_copy<Element>(
        SM90_TMA_LOAD{}, mK, SmemLayoutInK{}, Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{}, _1{});
    auto tma_store_K = make_tma_copy<int8_t>(
        SM90_TMA_STORE{}, mK_q, SmemLayoutOutK{}, Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{}, _1{});

    using SharedStorageK = TmaSharedStorage<HEAD_DIM, kNumThreads, Element, SmemLayoutInK, SmemLayoutOutK>;
    int smem_size_k = sizeof(SharedStorageK);

    if (smem_size_k >= 48 * 1024) {
        cudaFuncSetAttribute(quantize_k_kernel<HEAD_DIM, BLOCK_N, kNumThreads, Element, decltype(tma_load_K), decltype(tma_store_K)>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_k);
    }

    dim3 grid_k(num_seq_tiles_k, num_heads, batch);
    quantize_k_kernel<HEAD_DIM, BLOCK_N, kNumThreads, Element>
        <<<grid_k, kNumThreads, smem_size_k, stream>>>(
        tma_load_K, tma_store_K, seqlen_k, num_heads, batch, shape_K, k_mean, k_scales, num_seq_tiles_k);
}

// Combined QK launcher (k_mean must be pre-computed externally)
template <int HEAD_DIM, int BLOCK_M, int BLOCK_N, typename Element>
void launch_quantize_qk_config(
    const Element* Q, const Element* K,
    int8_t* Q_q, int8_t* K_q,
    float* q_scales, float* k_scales, const float* k_mean,
    int batch, int seqlen_q, int seqlen_k, int num_heads,
    double q_scale,
    cudaStream_t stream)
{
    launch_quantize_q_config<HEAD_DIM, BLOCK_M, Element>(Q, Q_q, q_scales, batch, seqlen_q, num_heads, q_scale, stream);
    launch_quantize_k_config<HEAD_DIM, BLOCK_N, Element>(K, K_q, k_mean, k_scales, batch, seqlen_k, num_heads, stream);
}

// -----------------------------------------------------------------------------
// Compile-time tile size helpers using tile_size.h
// -----------------------------------------------------------------------------

// Get tile sizes for quantization kernels using tile_size_fwd_sm90
template <int HEAD_DIM, bool V_COLMAJOR = false, bool IS_SKIPABLE = false>
struct QuantTileConfig {
    static constexpr auto tile_info = tile_size_fwd_sm90(
        HEAD_DIM,           // headdim
        HEAD_DIM,           // headdim_v (same as headdim for standard attention)
        false,              // is_causal
        false,              // is_local
        1,                  // element_size (1 for int8)
        V_COLMAJOR,         // v_colmajor
        false,              // paged_kv_non_TMA
        false,              // softcap
        IS_SKIPABLE,        // is_skipable
        true                // is_int8
    );
    static constexpr int kBlockM = std::get<0>(tile_info);
    static constexpr int kBlockN = std::get<1>(tile_info);
};

// Runtime Dispatcher (k_mean must be pre-computed externally)
// Dispatches based on head_dim, v_colmajor, and is_skipable
template <typename Element>
void launch_quantize_qk_runtime(
    const Element* Q, const Element* K,
    int8_t* Q_q, int8_t* K_q,
    float* q_scales, float* k_scales, const float* k_mean,
    int batch, int seqlen_q, int seqlen_k, int num_heads,
    int head_dim, bool v_colmajor, bool is_skipable,
    double q_scale,
    cudaStream_t stream)
{
    #define DISPATCH_QK(HD, VC, SK) \
        launch_quantize_qk_config<HD, QuantTileConfig<HD, VC, SK>::kBlockM, QuantTileConfig<HD, VC, SK>::kBlockN, Element>( \
            Q, K, Q_q, K_q, q_scales, k_scales, k_mean, \
            batch, seqlen_q, seqlen_k, num_heads, q_scale, stream)

    #define DISPATCH_QK_SKIPABLE(HD, VC) \
        if (is_skipable) { DISPATCH_QK(HD, VC, true); } \
        else { DISPATCH_QK(HD, VC, false); }

    #define DISPATCH_QK_COLMAJOR(HD) \
        if (v_colmajor) { DISPATCH_QK_SKIPABLE(HD, true); } \
        else { DISPATCH_QK_SKIPABLE(HD, false); }

    switch (head_dim) {
        case 32:  DISPATCH_QK_COLMAJOR(32);  break;
        case 64:  DISPATCH_QK_COLMAJOR(64);  break;
        case 96:  DISPATCH_QK_COLMAJOR(96);  break;
        case 128: DISPATCH_QK_COLMAJOR(128); break;
        case 192: DISPATCH_QK_COLMAJOR(192); break;
        case 256: DISPATCH_QK_COLMAJOR(256); break;
        default:
            assert(false && "Unsupported head_dim");
    }
    #undef DISPATCH_QK
    #undef DISPATCH_QK_SKIPABLE
    #undef DISPATCH_QK_COLMAJOR
}

// Q-only runtime dispatcher
template <typename Element>
void launch_quantize_q_runtime(
    const Element* Q, int8_t* Q_q, float* q_scales,
    int batch, int seqlen_q, int num_heads, int head_dim,
    bool v_colmajor, bool is_skipable,
    double q_scale,
    cudaStream_t stream)
{
    #define DISPATCH_Q(HD, VC, SK) \
        launch_quantize_q_config<HD, QuantTileConfig<HD, VC, SK>::kBlockM, Element>(Q, Q_q, q_scales, batch, seqlen_q, num_heads, q_scale, stream)

    #define DISPATCH_Q_SKIPABLE(HD, VC) \
        if (is_skipable) { DISPATCH_Q(HD, VC, true); } \
        else { DISPATCH_Q(HD, VC, false); }

    #define DISPATCH_Q_COLMAJOR(HD) \
        if (v_colmajor) { DISPATCH_Q_SKIPABLE(HD, true); } \
        else { DISPATCH_Q_SKIPABLE(HD, false); }

    switch (head_dim) {
        case 32:  DISPATCH_Q_COLMAJOR(32);  break;
        case 64:  DISPATCH_Q_COLMAJOR(64);  break;
        case 96:  DISPATCH_Q_COLMAJOR(96);  break;
        case 128: DISPATCH_Q_COLMAJOR(128); break;
        case 192: DISPATCH_Q_COLMAJOR(192); break;
        case 256: DISPATCH_Q_COLMAJOR(256); break;
        default:
            assert(false && "Unsupported head_dim for Q");
    }
    #undef DISPATCH_Q
    #undef DISPATCH_Q_SKIPABLE
    #undef DISPATCH_Q_COLMAJOR
}

// K-only runtime dispatcher
template <typename Element>
void launch_quantize_k_runtime(
    const Element* K, int8_t* K_q, const float* k_mean, float* k_scales,
    int batch, int seqlen_k, int num_heads, int head_dim,
    bool v_colmajor, bool is_skipable,
    cudaStream_t stream)
{
    #define DISPATCH_K(HD, VC, SK) \
        launch_quantize_k_config<HD, QuantTileConfig<HD, VC, SK>::kBlockN, Element>(K, K_q, k_mean, k_scales, batch, seqlen_k, num_heads, stream)

    #define DISPATCH_K_SKIPABLE(HD, VC) \
        if (is_skipable) { DISPATCH_K(HD, VC, true); } \
        else { DISPATCH_K(HD, VC, false); }

    #define DISPATCH_K_COLMAJOR(HD) \
        if (v_colmajor) { DISPATCH_K_SKIPABLE(HD, true); } \
        else { DISPATCH_K_SKIPABLE(HD, false); }

    switch (head_dim) {
        case 32:  DISPATCH_K_COLMAJOR(32);  break;
        case 64:  DISPATCH_K_COLMAJOR(64);  break;
        case 96:  DISPATCH_K_COLMAJOR(96);  break;
        case 128: DISPATCH_K_COLMAJOR(128); break;
        case 192: DISPATCH_K_COLMAJOR(192); break;
        case 256: DISPATCH_K_COLMAJOR(256); break;
        default:
            assert(false && "Unsupported head_dim for K");
    }
    #undef DISPATCH_K
    #undef DISPATCH_K_SKIPABLE
    #undef DISPATCH_K_COLMAJOR
}

// Instantiations
template void launch_quantize_qk_runtime<cutlass::half_t>(
    const cutlass::half_t*, const cutlass::half_t*,
    int8_t*, int8_t*,
    float*, float*, const float*,
    int, int, int, int,
    int, bool, bool,
    double,
    cudaStream_t);

template void launch_quantize_qk_runtime<cutlass::bfloat16_t>(
    const cutlass::bfloat16_t*, const cutlass::bfloat16_t*,
    int8_t*, int8_t*,
    float*, float*, const float*,
    int, int, int, int,
    int, bool, bool,
    double,
    cudaStream_t);

template void launch_quantize_q_runtime<cutlass::half_t>(
    const cutlass::half_t*, int8_t*, float*,
    int, int, int, int,
    bool, bool,
    double,
    cudaStream_t);

template void launch_quantize_q_runtime<cutlass::bfloat16_t>(
    const cutlass::bfloat16_t*, int8_t*, float*,
    int, int, int, int,
    bool, bool,
    double,
    cudaStream_t);

template void launch_quantize_k_runtime<cutlass::half_t>(
    const cutlass::half_t*, int8_t*, const float*, float*,
    int, int, int, int,
    bool, bool,
    cudaStream_t);

template void launch_quantize_k_runtime<cutlass::bfloat16_t>(
    const cutlass::bfloat16_t*, int8_t*, const float*, float*,
    int, int, int, int,
    bool, bool,
    cudaStream_t);

// -----------------------------------------------------------------------------
// Python Bindings (for standalone quant_tma module)
// -----------------------------------------------------------------------------
#ifdef QUANT_STANDALONE

#include <pybind11/pybind11.h>

namespace py = pybind11;

void quantize_qk_bf16(
    uintptr_t Q_ptr, uintptr_t K_ptr,
    uintptr_t Q_q_ptr, uintptr_t K_q_ptr,
    uintptr_t q_scales_ptr, uintptr_t k_scales_ptr, uintptr_t k_mean_ptr,
    int batch, int seqlen_q, int seqlen_k, int num_heads, int head_dim,
    bool v_colmajor, bool is_skipable, double q_scale)
{
    launch_quantize_qk_runtime<cutlass::bfloat16_t>(
        reinterpret_cast<const cutlass::bfloat16_t*>(Q_ptr),
        reinterpret_cast<const cutlass::bfloat16_t*>(K_ptr),
        reinterpret_cast<int8_t*>(Q_q_ptr),
        reinterpret_cast<int8_t*>(K_q_ptr),
        reinterpret_cast<float*>(q_scales_ptr),
        reinterpret_cast<float*>(k_scales_ptr),
        reinterpret_cast<const float*>(k_mean_ptr),
        batch, seqlen_q, seqlen_k, num_heads,
        head_dim, v_colmajor, is_skipable,
        q_scale,
        0  // default stream
    );
    cudaDeviceSynchronize();
}

void quantize_qk_f16(
    uintptr_t Q_ptr, uintptr_t K_ptr,
    uintptr_t Q_q_ptr, uintptr_t K_q_ptr,
    uintptr_t q_scales_ptr, uintptr_t k_scales_ptr, uintptr_t k_mean_ptr,
    int batch, int seqlen_q, int seqlen_k, int num_heads, int head_dim,
    bool v_colmajor, bool is_skipable, double q_scale)
{
    launch_quantize_qk_runtime<cutlass::half_t>(
        reinterpret_cast<const cutlass::half_t*>(Q_ptr),
        reinterpret_cast<const cutlass::half_t*>(K_ptr),
        reinterpret_cast<int8_t*>(Q_q_ptr),
        reinterpret_cast<int8_t*>(K_q_ptr),
        reinterpret_cast<float*>(q_scales_ptr),
        reinterpret_cast<float*>(k_scales_ptr),
        reinterpret_cast<const float*>(k_mean_ptr),
        batch, seqlen_q, seqlen_k, num_heads,
        head_dim, v_colmajor, is_skipable,
        q_scale,
        0  // default stream
    );
    cudaDeviceSynchronize();
}

// Separate Q-only wrappers
void quantize_q_bf16(
    uintptr_t Q_ptr, uintptr_t Q_q_ptr, uintptr_t q_scales_ptr,
    int batch, int seqlen_q, int num_heads, int head_dim,
    bool v_colmajor, bool is_skipable, double q_scale)
{
    launch_quantize_q_runtime<cutlass::bfloat16_t>(
        reinterpret_cast<const cutlass::bfloat16_t*>(Q_ptr),
        reinterpret_cast<int8_t*>(Q_q_ptr),
        reinterpret_cast<float*>(q_scales_ptr),
        batch, seqlen_q, num_heads, head_dim,
        v_colmajor, is_skipable,
        q_scale,
        0  // default stream
    );
    cudaDeviceSynchronize();
}

void quantize_q_f16(
    uintptr_t Q_ptr, uintptr_t Q_q_ptr, uintptr_t q_scales_ptr,
    int batch, int seqlen_q, int num_heads, int head_dim,
    bool v_colmajor, bool is_skipable, double q_scale)
{
    launch_quantize_q_runtime<cutlass::half_t>(
        reinterpret_cast<const cutlass::half_t*>(Q_ptr),
        reinterpret_cast<int8_t*>(Q_q_ptr),
        reinterpret_cast<float*>(q_scales_ptr),
        batch, seqlen_q, num_heads, head_dim,
        v_colmajor, is_skipable,
        q_scale,
        0  // default stream
    );
    cudaDeviceSynchronize();
}

// Separate K-only wrappers
void quantize_k_bf16(
    uintptr_t K_ptr, uintptr_t K_q_ptr, uintptr_t k_mean_ptr, uintptr_t k_scales_ptr,
    int batch, int seqlen_k, int num_heads, int head_dim,
    bool v_colmajor, bool is_skipable)
{
    launch_quantize_k_runtime<cutlass::bfloat16_t>(
        reinterpret_cast<const cutlass::bfloat16_t*>(K_ptr),
        reinterpret_cast<int8_t*>(K_q_ptr),
        reinterpret_cast<const float*>(k_mean_ptr),
        reinterpret_cast<float*>(k_scales_ptr),
        batch, seqlen_k, num_heads, head_dim,
        v_colmajor, is_skipable,
        0  // default stream
    );
    cudaDeviceSynchronize();
}

void quantize_k_f16(
    uintptr_t K_ptr, uintptr_t K_q_ptr, uintptr_t k_mean_ptr, uintptr_t k_scales_ptr,
    int batch, int seqlen_k, int num_heads, int head_dim,
    bool v_colmajor, bool is_skipable)
{
    launch_quantize_k_runtime<cutlass::half_t>(
        reinterpret_cast<const cutlass::half_t*>(K_ptr),
        reinterpret_cast<int8_t*>(K_q_ptr),
        reinterpret_cast<const float*>(k_mean_ptr),
        reinterpret_cast<float*>(k_scales_ptr),
        batch, seqlen_k, num_heads, head_dim,
        v_colmajor, is_skipable,
        0  // default stream
    );
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(quant_tma, m) {
    m.doc() = "CUDA quantization kernels for Q/K tensors";

    m.def("quantize_qk_bf16", &quantize_qk_bf16,
          "Quantize Q and K tensors (bfloat16)",
          py::arg("Q_ptr"), py::arg("K_ptr"),
          py::arg("Q_q_ptr"), py::arg("K_q_ptr"),
          py::arg("q_scales_ptr"), py::arg("k_scales_ptr"), py::arg("k_mean_ptr"),
          py::arg("batch"), py::arg("seqlen_q"), py::arg("seqlen_k"),
          py::arg("num_heads"), py::arg("head_dim"),
          py::arg("v_colmajor") = false, py::arg("is_skipable") = false,
          py::arg("q_scale") = 1.0);

    m.def("quantize_qk_f16", &quantize_qk_f16,
          "Quantize Q and K tensors (float16)",
          py::arg("Q_ptr"), py::arg("K_ptr"),
          py::arg("Q_q_ptr"), py::arg("K_q_ptr"),
          py::arg("q_scales_ptr"), py::arg("k_scales_ptr"), py::arg("k_mean_ptr"),
          py::arg("batch"), py::arg("seqlen_q"), py::arg("seqlen_k"),
          py::arg("num_heads"), py::arg("head_dim"),
          py::arg("v_colmajor") = false, py::arg("is_skipable") = false,
          py::arg("q_scale") = 1.0);

    m.def("quantize_q_bf16", &quantize_q_bf16,
          "Quantize Q tensor only (bfloat16)",
          py::arg("Q_ptr"), py::arg("Q_q_ptr"), py::arg("q_scales_ptr"),
          py::arg("batch"), py::arg("seqlen_q"), py::arg("num_heads"), py::arg("head_dim"),
          py::arg("v_colmajor") = false, py::arg("is_skipable") = false,
          py::arg("q_scale") = 1.0);

    m.def("quantize_q_f16", &quantize_q_f16,
          "Quantize Q tensor only (float16)",
          py::arg("Q_ptr"), py::arg("Q_q_ptr"), py::arg("q_scales_ptr"),
          py::arg("batch"), py::arg("seqlen_q"), py::arg("num_heads"), py::arg("head_dim"),
          py::arg("v_colmajor") = false, py::arg("is_skipable") = false,
          py::arg("q_scale") = 1.0);

    m.def("quantize_k_bf16", &quantize_k_bf16,
          "Quantize K tensor only (bfloat16)",
          py::arg("K_ptr"), py::arg("K_q_ptr"), py::arg("k_mean_ptr"), py::arg("k_scales_ptr"),
          py::arg("batch"), py::arg("seqlen_k"), py::arg("num_heads"), py::arg("head_dim"),
          py::arg("v_colmajor") = false, py::arg("is_skipable") = false);

    m.def("quantize_k_f16", &quantize_k_f16,
          "Quantize K tensor only (float16)",
          py::arg("K_ptr"), py::arg("K_q_ptr"), py::arg("k_mean_ptr"), py::arg("k_scales_ptr"),
          py::arg("batch"), py::arg("seqlen_k"), py::arg("num_heads"), py::arg("head_dim"),
          py::arg("v_colmajor") = false, py::arg("is_skipable") = false);
}

#endif