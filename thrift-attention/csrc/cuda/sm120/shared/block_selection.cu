#include <cstdint>
#include <cstdio>
#include <float.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "thriftattention/sm120/cuda_common.cuh"

namespace {

constexpr int SQ_TOPK_CHUNK_BLOCKS = 64;
constexpr int SQ_TOPK_LOCAL_WARPS = 8;
constexpr int SQ_TOPK_LOCAL_THREADS = SQ_TOPK_LOCAL_WARPS * TA_WARP_SIZE;

// Per-query-block selection.
//
// q_mean: [flat_heads, num_q_blocks, head_dim]
// k_mean: [flat_heads, num_kv_blocks, head_dim]
// out:    [flat_heads, num_q_blocks, topk_count]
template<typename T, bool CAUSAL, int HEAD_DIM, int MAX_KV_BLOCKS>
__global__ __launch_bounds__(TA_WARP_SIZE)
void block_mean_topk_kernel(
    const T* __restrict__ q_mean,
    const T* __restrict__ k_mean,
    int32_t* __restrict__ topk_out,
    int num_q_blocks,
    int num_kv_blocks,
    int topk_count) {
    using Traits = PrecisionTraits<T>;
    constexpr int ITEMS_PER_LANE = MAX_KV_BLOCKS / TA_WARP_SIZE;
    constexpr int ELEMS_PER_LANE = HEAD_DIM / TA_WARP_SIZE;

    const int bid = blockIdx.x;
    const int flat_head_id = bid / num_q_blocks;
    const int q_block_id = bid % num_q_blocks;
    const int lane_id = threadIdx.x;

    const T* q_row =
        q_mean + (static_cast<int64_t>(flat_head_id) * num_q_blocks + q_block_id) * HEAD_DIM;
    float q_reg[ELEMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; i++) {
        q_reg[i] = Traits::to_float(q_row[lane_id + i * TA_WARP_SIZE]);
    }

    const T* k_head = k_mean + static_cast<int64_t>(flat_head_id) * num_kv_blocks * HEAD_DIM;
    extern __shared__ uint8_t smem[];
    float* block_scores = reinterpret_cast<float*>(smem);

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const T* k_row = k_head + kv_block * HEAD_DIM;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_LANE; i++) {
            dot += q_reg[i] * Traits::to_float(k_row[lane_id + i * TA_WARP_SIZE]);
        }

        #pragma unroll
        for (int delta = 16; delta >= 1; delta >>= 1) {
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, delta);
        }

        if (lane_id == 0) {
            block_scores[kv_block] = dot;
        }
    }
    __syncwarp();

    float values[ITEMS_PER_LANE];
    int indices[ITEMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; i++) {
        const int idx = i * TA_WARP_SIZE + lane_id;
        indices[i] = idx;
        values[i] = (idx < num_kv_blocks && (!CAUSAL || idx <= q_block_id))
            ? block_scores[idx]
            : -FLT_MAX;
    }

    int32_t* out =
        topk_out + (static_cast<int64_t>(flat_head_id) * num_q_blocks + q_block_id) * topk_count;

    for (int k = 0; k < topk_count; k++) {
        float best = values[0];
        int best_idx = indices[0];
        #pragma unroll
        for (int i = 1; i < ITEMS_PER_LANE; i++) {
            if (values[i] > best) {
                best = values[i];
                best_idx = indices[i];
            }
        }

        #pragma unroll
        for (int delta = 16; delta >= 1; delta >>= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFF, best, delta);
            const int other_idx = __shfl_xor_sync(0xFFFFFFFF, best_idx, delta);
            if (other > best) {
                best = other;
                best_idx = other_idx;
            }
        }

        const bool found = best > -FLT_MAX * 0.5f;
        if (lane_id == 0) out[k] = found ? best_idx : -1;
        best_idx = __shfl_sync(0xFFFFFFFF, best_idx, 0);

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_LANE; i++) {
            if (found && indices[i] == best_idx) values[i] = -FLT_MAX;
        }
    }
}

// Single-token decode selection for grouped-query attention.
//
// q_grouped: [flat_kv_heads, groups, head_dim]
// k_mean:    [flat_kv_heads, k_mean_capacity_blocks, head_dim]
// out:       [flat_kv_heads, topk_count]
//
// Scores are max_g dot(q[flat_head, g], k_mean[flat_head, block]), matching
// the Python grouped decode selector.
template<typename T, int HEAD_DIM, int MAX_KV_BLOCKS>
__global__ __launch_bounds__(TA_WARP_SIZE)
void single_query_key_mean_topk_kernel(
    const T* __restrict__ q_grouped,
    const T* __restrict__ k_mean,
    int32_t* __restrict__ topk_out,
    int groups,
    int num_kv_blocks,
    int k_mean_capacity_blocks,
    int topk_count) {
    using Traits = PrecisionTraits<T>;
    constexpr int ITEMS_PER_LANE = MAX_KV_BLOCKS / TA_WARP_SIZE;
    constexpr int ELEMS_PER_LANE = HEAD_DIM / TA_WARP_SIZE;

    const int flat_head_id = blockIdx.x;
    const int lane_id = threadIdx.x;

    const T* q_head =
        q_grouped + static_cast<int64_t>(flat_head_id) * groups * HEAD_DIM;
    const T* k_head =
        k_mean + static_cast<int64_t>(flat_head_id) * k_mean_capacity_blocks * HEAD_DIM;

    extern __shared__ uint8_t smem[];
    float* block_scores = reinterpret_cast<float*>(smem);

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const T* k_row = k_head + kv_block * HEAD_DIM;
        float best = -FLT_MAX;

        for (int group_id = 0; group_id < groups; group_id++) {
            const T* q_row = q_head + group_id * HEAD_DIM;
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_LANE; i++) {
                const int elem = lane_id + i * TA_WARP_SIZE;
                dot += Traits::to_float(q_row[elem]) * Traits::to_float(k_row[elem]);
            }

            #pragma unroll
            for (int delta = 16; delta >= 1; delta >>= 1) {
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, delta);
            }

            if (lane_id == 0 && dot > best) {
                best = dot;
            }
        }

        if (lane_id == 0) {
            block_scores[kv_block] = best;
        }
    }
    __syncwarp();

    float values[ITEMS_PER_LANE];
    int indices[ITEMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; i++) {
        const int idx = i * TA_WARP_SIZE + lane_id;
        indices[i] = idx;
        values[i] = (idx < num_kv_blocks) ? block_scores[idx] : -FLT_MAX;
    }

    int32_t* out = topk_out + static_cast<int64_t>(flat_head_id) * topk_count;

    for (int k = 0; k < topk_count; k++) {
        float best = values[0];
        int best_idx = indices[0];
        #pragma unroll
        for (int i = 1; i < ITEMS_PER_LANE; i++) {
            if (values[i] > best) {
                best = values[i];
                best_idx = indices[i];
            }
        }

        #pragma unroll
        for (int delta = 16; delta >= 1; delta >>= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFF, best, delta);
            const int other_idx = __shfl_xor_sync(0xFFFFFFFF, best_idx, delta);
            if (other > best) {
                best = other;
                best_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            out[k] = best_idx;
        }
        best_idx = __shfl_sync(0xFFFFFFFF, best_idx, 0);

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_LANE; i++) {
            if (indices[i] == best_idx) values[i] = -FLT_MAX;
        }
    }
}

// Per-query-block QUEST min/max selection.
//
// q_mean: [flat_heads, num_q_blocks, head_dim]
// k_min:  [flat_heads, num_kv_blocks, head_dim]
// k_max:  [flat_heads, num_kv_blocks, head_dim]
// out:    [flat_heads, num_q_blocks, topk_count]
template<typename T, bool CAUSAL, int HEAD_DIM, int MAX_KV_BLOCKS>
__global__ __launch_bounds__(TA_WARP_SIZE)
void quest_block_topk_kernel(
    const T* __restrict__ q_mean,
    const T* __restrict__ k_min,
    const T* __restrict__ k_max,
    int32_t* __restrict__ topk_out,
    int num_q_blocks,
    int num_kv_blocks,
    int topk_count) {
    using Traits = PrecisionTraits<T>;
    constexpr int ITEMS_PER_LANE = MAX_KV_BLOCKS / TA_WARP_SIZE;
    constexpr int ELEMS_PER_LANE = HEAD_DIM / TA_WARP_SIZE;

    const int bid = blockIdx.x;
    const int flat_head_id = bid / num_q_blocks;
    const int q_block_id = bid % num_q_blocks;
    const int lane_id = threadIdx.x;

    const T* q_row =
        q_mean + (static_cast<int64_t>(flat_head_id) * num_q_blocks + q_block_id) * HEAD_DIM;
    float q_reg[ELEMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; i++) {
        q_reg[i] = Traits::to_float(q_row[lane_id + i * TA_WARP_SIZE]);
    }

    const T* k_min_head = k_min + static_cast<int64_t>(flat_head_id) * num_kv_blocks * HEAD_DIM;
    const T* k_max_head = k_max + static_cast<int64_t>(flat_head_id) * num_kv_blocks * HEAD_DIM;
    extern __shared__ uint8_t smem[];
    float* block_scores = reinterpret_cast<float*>(smem);

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const T* k_min_row = k_min_head + kv_block * HEAD_DIM;
        const T* k_max_row = k_max_head + kv_block * HEAD_DIM;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_LANE; i++) {
            const int elem = lane_id + i * TA_WARP_SIZE;
            const float q = q_reg[i];
            const float k = q >= 0.0f
                ? Traits::to_float(k_max_row[elem])
                : Traits::to_float(k_min_row[elem]);
            dot += q * k;
        }

        #pragma unroll
        for (int delta = 16; delta >= 1; delta >>= 1) {
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, delta);
        }

        if (lane_id == 0) {
            block_scores[kv_block] = dot;
        }
    }
    __syncwarp();

    float values[ITEMS_PER_LANE];
    int indices[ITEMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; i++) {
        const int idx = i * TA_WARP_SIZE + lane_id;
        indices[i] = idx;
        values[i] = (idx < num_kv_blocks && (!CAUSAL || idx <= q_block_id))
            ? block_scores[idx]
            : -FLT_MAX;
    }

    int32_t* out =
        topk_out + (static_cast<int64_t>(flat_head_id) * num_q_blocks + q_block_id) * topk_count;

    for (int k = 0; k < topk_count; k++) {
        float best = values[0];
        int best_idx = indices[0];
        #pragma unroll
        for (int i = 1; i < ITEMS_PER_LANE; i++) {
            if (values[i] > best) {
                best = values[i];
                best_idx = indices[i];
            }
        }

        #pragma unroll
        for (int delta = 16; delta >= 1; delta >>= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFF, best, delta);
            const int other_idx = __shfl_xor_sync(0xFFFFFFFF, best_idx, delta);
            if (other > best) {
                best = other;
                best_idx = other_idx;
            }
        }

        const bool found = best > -FLT_MAX * 0.5f;
        if (lane_id == 0) out[k] = found ? best_idx : -1;
        best_idx = __shfl_sync(0xFFFFFFFF, best_idx, 0);

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_LANE; i++) {
            if (found && indices[i] == best_idx) values[i] = -FLT_MAX;
        }
    }
}

// Single-token decode QUEST min/max selection for grouped-query attention.
//
// q_grouped: [flat_kv_heads, groups, head_dim]
// k_min/max: [flat_kv_heads, k_stat_capacity_blocks, head_dim]
// out:       [flat_kv_heads, topk_count]
template<typename T, int HEAD_DIM, int MAX_KV_BLOCKS>
__global__ __launch_bounds__(TA_WARP_SIZE)
void single_query_quest_topk_kernel(
    const T* __restrict__ q_grouped,
    const T* __restrict__ k_min,
    const T* __restrict__ k_max,
    int32_t* __restrict__ topk_out,
    int groups,
    int num_kv_blocks,
    int k_stat_capacity_blocks,
    int topk_count) {
    using Traits = PrecisionTraits<T>;
    constexpr int ITEMS_PER_LANE = MAX_KV_BLOCKS / TA_WARP_SIZE;
    constexpr int ELEMS_PER_LANE = HEAD_DIM / TA_WARP_SIZE;

    const int flat_head_id = blockIdx.x;
    const int lane_id = threadIdx.x;

    const T* q_head =
        q_grouped + static_cast<int64_t>(flat_head_id) * groups * HEAD_DIM;
    const T* k_min_head =
        k_min + static_cast<int64_t>(flat_head_id) * k_stat_capacity_blocks * HEAD_DIM;
    const T* k_max_head =
        k_max + static_cast<int64_t>(flat_head_id) * k_stat_capacity_blocks * HEAD_DIM;

    extern __shared__ uint8_t smem[];
    float* block_scores = reinterpret_cast<float*>(smem);

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const T* k_min_row = k_min_head + kv_block * HEAD_DIM;
        const T* k_max_row = k_max_head + kv_block * HEAD_DIM;
        float best = -FLT_MAX;

        for (int group_id = 0; group_id < groups; group_id++) {
            const T* q_row = q_head + group_id * HEAD_DIM;
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_LANE; i++) {
                const int elem = lane_id + i * TA_WARP_SIZE;
                const float q = Traits::to_float(q_row[elem]);
                const float k = q >= 0.0f
                    ? Traits::to_float(k_max_row[elem])
                    : Traits::to_float(k_min_row[elem]);
                dot += q * k;
            }

            #pragma unroll
            for (int delta = 16; delta >= 1; delta >>= 1) {
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, delta);
            }

            if (lane_id == 0 && dot > best) {
                best = dot;
            }
        }

        if (lane_id == 0) {
            block_scores[kv_block] = best;
        }
    }
    __syncwarp();

    float values[ITEMS_PER_LANE];
    int indices[ITEMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; i++) {
        const int idx = i * TA_WARP_SIZE + lane_id;
        indices[i] = idx;
        values[i] = (idx < num_kv_blocks) ? block_scores[idx] : -FLT_MAX;
    }

    int32_t* out = topk_out + static_cast<int64_t>(flat_head_id) * topk_count;

    for (int k = 0; k < topk_count; k++) {
        float best = values[0];
        int best_idx = indices[0];
        #pragma unroll
        for (int i = 1; i < ITEMS_PER_LANE; i++) {
            if (values[i] > best) {
                best = values[i];
                best_idx = indices[i];
            }
        }

        #pragma unroll
        for (int delta = 16; delta >= 1; delta >>= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFF, best, delta);
            const int other_idx = __shfl_xor_sync(0xFFFFFFFF, best_idx, delta);
            if (other > best) {
                best = other;
                best_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            out[k] = best_idx;
        }
        best_idx = __shfl_sync(0xFFFFFFFF, best_idx, 0);

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_LANE; i++) {
            if (indices[i] == best_idx) values[i] = -FLT_MAX;
        }
    }
}

// Chunk-parallel exact single-query selection. Each local CTA scores a
// contiguous 64-block chunk for one KV head, emits that chunk's local top-k,
// then the last completed chunk CTA merges the final top-k from all local
// candidates. Keeping top-k candidates per chunk is exact: any global top-k
// element must be in the top-k of its own chunk.
template<typename T, int HEAD_DIM>
__global__ __launch_bounds__(SQ_TOPK_LOCAL_THREADS)
void single_query_key_mean_topk_local_kernel(
    const T* __restrict__ q_grouped,
    const T* __restrict__ k_mean,
    int32_t* __restrict__ topk_out,
    float* __restrict__ local_scores,
    int32_t* __restrict__ local_indices,
    int32_t* __restrict__ done_counts,
    int groups,
    int num_kv_blocks,
    int k_mean_capacity_blocks,
    int chunk_count,
    int local_count,
    int topk_count) {
    using Traits = PrecisionTraits<T>;
    constexpr int ELEMS_PER_LANE = HEAD_DIM / TA_WARP_SIZE;
    constexpr int ITEMS_PER_LANE = SQ_TOPK_CHUNK_BLOCKS / TA_WARP_SIZE;

    const int chunk_id = blockIdx.x % chunk_count;
    const int flat_head_id = blockIdx.x / chunk_count;
    const int chunk_start = chunk_id * SQ_TOPK_CHUNK_BLOCKS;
    const int tid = threadIdx.x;
    const int lane_id = tid & (TA_WARP_SIZE - 1);
    const int warp_id = tid / TA_WARP_SIZE;

    __shared__ float block_scores[SQ_TOPK_CHUNK_BLOCKS];
    if (tid < SQ_TOPK_CHUNK_BLOCKS) {
        block_scores[tid] = -FLT_MAX;
    }
    __syncthreads();

    const T* q_head =
        q_grouped + static_cast<int64_t>(flat_head_id) * groups * HEAD_DIM;
    const T* k_head =
        k_mean + static_cast<int64_t>(flat_head_id) * k_mean_capacity_blocks * HEAD_DIM;

    for (int local_block = warp_id; local_block < SQ_TOPK_CHUNK_BLOCKS;
         local_block += SQ_TOPK_LOCAL_WARPS) {
        const int kv_block = chunk_start + local_block;
        float best = -FLT_MAX;

        if (kv_block < num_kv_blocks) {
            const T* k_row = k_head + static_cast<int64_t>(kv_block) * HEAD_DIM;
            for (int group_id = 0; group_id < groups; group_id++) {
                const T* q_row = q_head + group_id * HEAD_DIM;
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < ELEMS_PER_LANE; i++) {
                    const int elem = lane_id + i * TA_WARP_SIZE;
                    dot += Traits::to_float(q_row[elem]) * Traits::to_float(k_row[elem]);
                }

                #pragma unroll
                for (int delta = 16; delta >= 1; delta >>= 1) {
                    dot += __shfl_xor_sync(0xFFFFFFFF, dot, delta);
                }

                if (lane_id == 0 && dot > best) {
                    best = dot;
                }
            }
        }

        if (lane_id == 0) {
            block_scores[local_block] = best;
        }
    }
    __syncthreads();

    if (warp_id != 0) {
        return;
    }

    float values[ITEMS_PER_LANE];
    int indices[ITEMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; i++) {
        const int local_idx = i * TA_WARP_SIZE + lane_id;
        const int global_idx = chunk_start + local_idx;
        indices[i] = global_idx;
        values[i] = (global_idx < num_kv_blocks) ? block_scores[local_idx] : -FLT_MAX;
    }

    const int64_t local_base =
        (static_cast<int64_t>(flat_head_id) * chunk_count + chunk_id) * local_count;
    for (int rank = 0; rank < local_count; rank++) {
        float best = values[0];
        int best_idx = indices[0];
        #pragma unroll
        for (int i = 1; i < ITEMS_PER_LANE; i++) {
            if (values[i] > best) {
                best = values[i];
                best_idx = indices[i];
            }
        }

        #pragma unroll
        for (int delta = 16; delta >= 1; delta >>= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFF, best, delta);
            const int other_idx = __shfl_xor_sync(0xFFFFFFFF, best_idx, delta);
            if (other > best) {
                best = other;
                best_idx = other_idx;
            }
        }

        const bool found = best > -FLT_MAX * 0.5f;
        if (lane_id == 0) {
            local_scores[local_base + rank] = found ? best : -FLT_MAX;
            local_indices[local_base + rank] = found ? best_idx : -1;
        }
        best_idx = __shfl_sync(0xFFFFFFFF, best_idx, 0);

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_LANE; i++) {
            if (found && indices[i] == best_idx) {
                values[i] = -FLT_MAX;
            }
        }
    }
    __syncthreads();

    __shared__ int is_last_chunk;
    if (warp_id == 0 && lane_id == 0) {
        __threadfence();
        const int prior = atomicAdd(done_counts + flat_head_id, 1);
        is_last_chunk = (prior == chunk_count - 1);
    }
    __syncthreads();

    if (!is_last_chunk || warp_id != 0) {
        return;
    }

    const int candidate_count = chunk_count * local_count;
    const int64_t candidate_base = static_cast<int64_t>(flat_head_id) * candidate_count;

    for (int rank = 0; rank < topk_count; rank++) {
        float thread_best = -FLT_MAX;
        int thread_best_idx = -1;

        for (int candidate = lane_id; candidate < candidate_count; candidate += TA_WARP_SIZE) {
            const int64_t offset = candidate_base + candidate;
            const int idx = local_indices[offset];
            const float score = local_scores[offset];
            if (idx >= 0 && score > thread_best) {
                thread_best = score;
                thread_best_idx = idx;
            }
        }

        #pragma unroll
        for (int delta = 16; delta >= 1; delta >>= 1) {
            const float other_score = __shfl_down_sync(0xFFFFFFFF, thread_best, delta);
            const int other_idx = __shfl_down_sync(0xFFFFFFFF, thread_best_idx, delta);
            if (lane_id + delta < TA_WARP_SIZE && other_score > thread_best) {
                thread_best = other_score;
                thread_best_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            topk_out[static_cast<int64_t>(flat_head_id) * topk_count + rank] = thread_best_idx;
        }
        const int selected_idx = __shfl_sync(0xFFFFFFFF, thread_best_idx, 0);

        for (int candidate = lane_id; candidate < candidate_count; candidate += TA_WARP_SIZE) {
            const int64_t offset = candidate_base + candidate;
            if (local_indices[offset] == selected_idx) {
                local_scores[offset] = -FLT_MAX;
            }
        }
        __syncwarp();
    }
}

template<typename T, bool CAUSAL, int HEAD_DIM, int MAX_KV_BLOCKS>
void launch_block_mean_topk(
    const T* q_mean,
    const T* k_mean,
    int32_t* topk_out,
    int flat_heads,
    int num_q_blocks,
    int num_kv_blocks,
    int topk_count) {
    const int grid_size = flat_heads * num_q_blocks;
    const int smem_bytes = num_kv_blocks * static_cast<int>(sizeof(float));

    block_mean_topk_kernel<T, CAUSAL, HEAD_DIM, MAX_KV_BLOCKS>
        <<<grid_size, TA_WARP_SIZE, smem_bytes>>>(
            q_mean, k_mean, topk_out, num_q_blocks, num_kv_blocks, topk_count);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "block_mean_topk kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

template<typename T, bool CAUSAL, int HEAD_DIM>
void dispatch_block_mean_topk(
    const T* q_mean,
    const T* k_mean,
    int32_t* topk_out,
    int flat_heads,
    int num_q_blocks,
    int num_kv_blocks,
    int topk_count) {
    if (num_kv_blocks <= 128) {
        launch_block_mean_topk<T, CAUSAL, HEAD_DIM, 128>(
            q_mean, k_mean, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
    } else if (num_kv_blocks <= 512) {
        launch_block_mean_topk<T, CAUSAL, HEAD_DIM, 512>(
            q_mean, k_mean, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
    } else if (num_kv_blocks <= 1024) {
        launch_block_mean_topk<T, CAUSAL, HEAD_DIM, 1024>(
            q_mean, k_mean, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
    } else {
        launch_block_mean_topk<T, CAUSAL, HEAD_DIM, 2048>(
            q_mean, k_mean, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
    }
}

template<typename T, bool CAUSAL, int HEAD_DIM, int MAX_KV_BLOCKS>
void launch_quest_block_topk(
    const T* q_mean,
    const T* k_min,
    const T* k_max,
    int32_t* topk_out,
    int flat_heads,
    int num_q_blocks,
    int num_kv_blocks,
    int topk_count) {
    const int grid_size = flat_heads * num_q_blocks;
    const int smem_bytes = num_kv_blocks * static_cast<int>(sizeof(float));

    quest_block_topk_kernel<T, CAUSAL, HEAD_DIM, MAX_KV_BLOCKS>
        <<<grid_size, TA_WARP_SIZE, smem_bytes>>>(
            q_mean, k_min, k_max, topk_out, num_q_blocks, num_kv_blocks, topk_count);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "quest_block_topk kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

template<typename T, bool CAUSAL, int HEAD_DIM>
void dispatch_quest_block_topk(
    const T* q_mean,
    const T* k_min,
    const T* k_max,
    int32_t* topk_out,
    int flat_heads,
    int num_q_blocks,
    int num_kv_blocks,
    int topk_count) {
    if (num_kv_blocks <= 128) {
        launch_quest_block_topk<T, CAUSAL, HEAD_DIM, 128>(
            q_mean, k_min, k_max, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
    } else if (num_kv_blocks <= 512) {
        launch_quest_block_topk<T, CAUSAL, HEAD_DIM, 512>(
            q_mean, k_min, k_max, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
    } else if (num_kv_blocks <= 1024) {
        launch_quest_block_topk<T, CAUSAL, HEAD_DIM, 1024>(
            q_mean, k_min, k_max, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
    } else {
        launch_quest_block_topk<T, CAUSAL, HEAD_DIM, 2048>(
            q_mean, k_min, k_max, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
    }
}

template<typename T, int HEAD_DIM, int MAX_KV_BLOCKS>
void launch_single_query_key_mean_topk(
    const T* q_grouped,
    const T* k_mean,
    int32_t* topk_out,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_mean_capacity_blocks,
    int topk_count) {
    const int smem_bytes = num_kv_blocks * static_cast<int>(sizeof(float));

    single_query_key_mean_topk_kernel<T, HEAD_DIM, MAX_KV_BLOCKS>
        <<<flat_heads, TA_WARP_SIZE, smem_bytes>>>(
            q_grouped, k_mean, topk_out, groups, num_kv_blocks,
            k_mean_capacity_blocks, topk_count);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "single_query_key_mean_topk kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

template<typename T, int HEAD_DIM>
void dispatch_single_query_key_mean_topk(
    const T* q_grouped,
    const T* k_mean,
    int32_t* topk_out,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_mean_capacity_blocks,
    int topk_count) {
    if (num_kv_blocks <= 128) {
        launch_single_query_key_mean_topk<T, HEAD_DIM, 128>(
            q_grouped, k_mean, topk_out, flat_heads, groups, num_kv_blocks,
            k_mean_capacity_blocks, topk_count);
    } else if (num_kv_blocks <= 512) {
        launch_single_query_key_mean_topk<T, HEAD_DIM, 512>(
            q_grouped, k_mean, topk_out, flat_heads, groups, num_kv_blocks,
            k_mean_capacity_blocks, topk_count);
    } else if (num_kv_blocks <= 1024) {
        launch_single_query_key_mean_topk<T, HEAD_DIM, 1024>(
            q_grouped, k_mean, topk_out, flat_heads, groups, num_kv_blocks,
            k_mean_capacity_blocks, topk_count);
    } else {
        launch_single_query_key_mean_topk<T, HEAD_DIM, 2048>(
            q_grouped, k_mean, topk_out, flat_heads, groups, num_kv_blocks,
            k_mean_capacity_blocks, topk_count);
    }
}

template<typename T, int HEAD_DIM, int MAX_KV_BLOCKS>
void launch_single_query_quest_topk(
    const T* q_grouped,
    const T* k_min,
    const T* k_max,
    int32_t* topk_out,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_stat_capacity_blocks,
    int topk_count) {
    const int smem_bytes = num_kv_blocks * static_cast<int>(sizeof(float));

    single_query_quest_topk_kernel<T, HEAD_DIM, MAX_KV_BLOCKS>
        <<<flat_heads, TA_WARP_SIZE, smem_bytes>>>(
            q_grouped, k_min, k_max, topk_out, groups, num_kv_blocks,
            k_stat_capacity_blocks, topk_count);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "single_query_quest_topk kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

template<typename T, int HEAD_DIM>
void dispatch_single_query_quest_topk(
    const T* q_grouped,
    const T* k_min,
    const T* k_max,
    int32_t* topk_out,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_stat_capacity_blocks,
    int topk_count) {
    if (num_kv_blocks <= 128) {
        launch_single_query_quest_topk<T, HEAD_DIM, 128>(
            q_grouped, k_min, k_max, topk_out, flat_heads, groups, num_kv_blocks,
            k_stat_capacity_blocks, topk_count);
    } else if (num_kv_blocks <= 512) {
        launch_single_query_quest_topk<T, HEAD_DIM, 512>(
            q_grouped, k_min, k_max, topk_out, flat_heads, groups, num_kv_blocks,
            k_stat_capacity_blocks, topk_count);
    } else if (num_kv_blocks <= 1024) {
        launch_single_query_quest_topk<T, HEAD_DIM, 1024>(
            q_grouped, k_min, k_max, topk_out, flat_heads, groups, num_kv_blocks,
            k_stat_capacity_blocks, topk_count);
    } else {
        launch_single_query_quest_topk<T, HEAD_DIM, 2048>(
            q_grouped, k_min, k_max, topk_out, flat_heads, groups, num_kv_blocks,
            k_stat_capacity_blocks, topk_count);
    }
}

template<typename T, int HEAD_DIM>
void launch_single_query_key_mean_topk_chunked(
    const T* q_grouped,
    const T* k_mean,
    int32_t* topk_out,
    float* local_scores,
    int32_t* local_indices,
    int32_t* done_counts,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_mean_capacity_blocks,
    int topk_count,
    int chunk_count,
    int local_count) {
    cudaMemsetAsync(done_counts, 0, static_cast<size_t>(flat_heads) * sizeof(int32_t));

    const int local_grid = flat_heads * chunk_count;
    single_query_key_mean_topk_local_kernel<T, HEAD_DIM>
        <<<local_grid, SQ_TOPK_LOCAL_THREADS>>>(
            q_grouped, k_mean, topk_out, local_scores, local_indices, done_counts,
            groups, num_kv_blocks, k_mean_capacity_blocks, chunk_count, local_count,
            topk_count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "single_query_key_mean_topk_local kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

}  // namespace

template<typename T>
static void dispatch_block_mean_topk_typed(
    const void* q_mean_raw,
    const void* k_mean_raw,
    void* topk_out_raw,
    int flat_heads,
    int num_q_blocks,
    int num_kv_blocks,
    int head_dim,
    int topk_count,
    bool causal) {
    const T* q_mean = reinterpret_cast<const T*>(q_mean_raw);
    const T* k_mean = reinterpret_cast<const T*>(k_mean_raw);
    int32_t* topk_out = reinterpret_cast<int32_t*>(topk_out_raw);

    if (num_kv_blocks > 2048) {
        fprintf(stderr, "block_mean_topk: num_kv_blocks=%d > 2048 not supported\n", num_kv_blocks);
        return;
    }

    if (head_dim == 64) {
        if (causal) {
            dispatch_block_mean_topk<T, true, 64>(
                q_mean, k_mean, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
        } else {
            dispatch_block_mean_topk<T, false, 64>(
                q_mean, k_mean, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
        }
    } else if (head_dim == 128) {
        if (causal) {
            dispatch_block_mean_topk<T, true, 128>(
                q_mean, k_mean, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
        } else {
            dispatch_block_mean_topk<T, false, 128>(
                q_mean, k_mean, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
        }
    } else {
        fprintf(stderr, "block_mean_topk: unsupported head_dim=%d\n", head_dim);
    }
}

void block_mean_topk(
    const void* q_mean_raw,
    const void* k_mean_raw,
    void* topk_out_raw,
    int flat_heads,
    int num_q_blocks,
    int num_kv_blocks,
    int head_dim,
    int topk_count,
    bool causal,
    bool is_bf16) {
    if (is_bf16) {
        dispatch_block_mean_topk_typed<__nv_bfloat16>(
            q_mean_raw, k_mean_raw, topk_out_raw, flat_heads, num_q_blocks,
            num_kv_blocks, head_dim, topk_count, causal);
    } else {
        dispatch_block_mean_topk_typed<half>(
            q_mean_raw, k_mean_raw, topk_out_raw, flat_heads, num_q_blocks,
            num_kv_blocks, head_dim, topk_count, causal);
    }
}

template<typename T>
static void dispatch_quest_block_topk_typed(
    const void* q_mean_raw,
    const void* k_min_raw,
    const void* k_max_raw,
    void* topk_out_raw,
    int flat_heads,
    int num_q_blocks,
    int num_kv_blocks,
    int head_dim,
    int topk_count,
    bool causal) {
    const T* q_mean = reinterpret_cast<const T*>(q_mean_raw);
    const T* k_min = reinterpret_cast<const T*>(k_min_raw);
    const T* k_max = reinterpret_cast<const T*>(k_max_raw);
    int32_t* topk_out = reinterpret_cast<int32_t*>(topk_out_raw);

    if (num_kv_blocks > 2048) {
        fprintf(stderr, "quest_block_topk: num_kv_blocks=%d > 2048 not supported\n", num_kv_blocks);
        return;
    }

    if (head_dim == 64) {
        if (causal) {
            dispatch_quest_block_topk<T, true, 64>(
                q_mean, k_min, k_max, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
        } else {
            dispatch_quest_block_topk<T, false, 64>(
                q_mean, k_min, k_max, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
        }
    } else if (head_dim == 128) {
        if (causal) {
            dispatch_quest_block_topk<T, true, 128>(
                q_mean, k_min, k_max, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
        } else {
            dispatch_quest_block_topk<T, false, 128>(
                q_mean, k_min, k_max, topk_out, flat_heads, num_q_blocks, num_kv_blocks, topk_count);
        }
    } else {
        fprintf(stderr, "quest_block_topk: unsupported head_dim=%d\n", head_dim);
    }
}

void quest_block_topk(
    const void* q_mean_raw,
    const void* k_min_raw,
    const void* k_max_raw,
    void* topk_out_raw,
    int flat_heads,
    int num_q_blocks,
    int num_kv_blocks,
    int head_dim,
    int topk_count,
    bool causal,
    bool is_bf16) {
    if (is_bf16) {
        dispatch_quest_block_topk_typed<__nv_bfloat16>(
            q_mean_raw, k_min_raw, k_max_raw, topk_out_raw, flat_heads, num_q_blocks,
            num_kv_blocks, head_dim, topk_count, causal);
    } else {
        dispatch_quest_block_topk_typed<half>(
            q_mean_raw, k_min_raw, k_max_raw, topk_out_raw, flat_heads, num_q_blocks,
            num_kv_blocks, head_dim, topk_count, causal);
    }
}

template<typename T>
static void dispatch_single_query_key_mean_topk_typed(
    const void* q_grouped_raw,
    const void* k_mean_raw,
    void* topk_out_raw,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_mean_capacity_blocks,
    int head_dim,
    int topk_count) {
    const T* q_grouped = reinterpret_cast<const T*>(q_grouped_raw);
    const T* k_mean = reinterpret_cast<const T*>(k_mean_raw);
    int32_t* topk_out = reinterpret_cast<int32_t*>(topk_out_raw);

    if (num_kv_blocks > 2048) {
        fprintf(stderr, "single_query_key_mean_topk: num_kv_blocks=%d > 2048 not supported\n",
                num_kv_blocks);
        return;
    }

    if (head_dim == 64) {
        dispatch_single_query_key_mean_topk<T, 64>(
            q_grouped, k_mean, topk_out, flat_heads, groups, num_kv_blocks,
            k_mean_capacity_blocks, topk_count);
    } else if (head_dim == 128) {
        dispatch_single_query_key_mean_topk<T, 128>(
            q_grouped, k_mean, topk_out, flat_heads, groups, num_kv_blocks,
            k_mean_capacity_blocks, topk_count);
    } else {
        fprintf(stderr, "single_query_key_mean_topk: unsupported head_dim=%d\n", head_dim);
    }
}

void single_query_key_mean_topk(
    const void* q_grouped_raw,
    const void* k_mean_raw,
    void* topk_out_raw,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_mean_capacity_blocks,
    int head_dim,
    int topk_count,
    bool is_bf16) {
    if (is_bf16) {
        dispatch_single_query_key_mean_topk_typed<__nv_bfloat16>(
            q_grouped_raw, k_mean_raw, topk_out_raw, flat_heads, groups,
            num_kv_blocks, k_mean_capacity_blocks, head_dim, topk_count);
    } else {
        dispatch_single_query_key_mean_topk_typed<half>(
            q_grouped_raw, k_mean_raw, topk_out_raw, flat_heads, groups,
            num_kv_blocks, k_mean_capacity_blocks, head_dim, topk_count);
    }
}

template<typename T>
static void dispatch_single_query_quest_topk_typed(
    const void* q_grouped_raw,
    const void* k_min_raw,
    const void* k_max_raw,
    void* topk_out_raw,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_stat_capacity_blocks,
    int head_dim,
    int topk_count) {
    const T* q_grouped = reinterpret_cast<const T*>(q_grouped_raw);
    const T* k_min = reinterpret_cast<const T*>(k_min_raw);
    const T* k_max = reinterpret_cast<const T*>(k_max_raw);
    int32_t* topk_out = reinterpret_cast<int32_t*>(topk_out_raw);

    if (num_kv_blocks > 2048) {
        fprintf(stderr, "single_query_quest_topk: num_kv_blocks=%d > 2048 not supported\n",
                num_kv_blocks);
        return;
    }

    if (head_dim == 64) {
        dispatch_single_query_quest_topk<T, 64>(
            q_grouped, k_min, k_max, topk_out, flat_heads, groups, num_kv_blocks,
            k_stat_capacity_blocks, topk_count);
    } else if (head_dim == 128) {
        dispatch_single_query_quest_topk<T, 128>(
            q_grouped, k_min, k_max, topk_out, flat_heads, groups, num_kv_blocks,
            k_stat_capacity_blocks, topk_count);
    } else {
        fprintf(stderr, "single_query_quest_topk: unsupported head_dim=%d\n", head_dim);
    }
}

void single_query_quest_topk(
    const void* q_grouped_raw,
    const void* k_min_raw,
    const void* k_max_raw,
    void* topk_out_raw,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_stat_capacity_blocks,
    int head_dim,
    int topk_count,
    bool is_bf16) {
    if (is_bf16) {
        dispatch_single_query_quest_topk_typed<__nv_bfloat16>(
            q_grouped_raw, k_min_raw, k_max_raw, topk_out_raw, flat_heads, groups,
            num_kv_blocks, k_stat_capacity_blocks, head_dim, topk_count);
    } else {
        dispatch_single_query_quest_topk_typed<half>(
            q_grouped_raw, k_min_raw, k_max_raw, topk_out_raw, flat_heads, groups,
            num_kv_blocks, k_stat_capacity_blocks, head_dim, topk_count);
    }
}

template<typename T>
static void dispatch_single_query_key_mean_topk_chunked_typed(
    const void* q_grouped_raw,
    const void* k_mean_raw,
    void* topk_out_raw,
    void* local_scores_raw,
    void* local_indices_raw,
    void* done_counts_raw,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_mean_capacity_blocks,
    int head_dim,
    int topk_count,
    int chunk_count,
    int local_count) {
    const T* q_grouped = reinterpret_cast<const T*>(q_grouped_raw);
    const T* k_mean = reinterpret_cast<const T*>(k_mean_raw);
    int32_t* topk_out = reinterpret_cast<int32_t*>(topk_out_raw);
    float* local_scores = reinterpret_cast<float*>(local_scores_raw);
    int32_t* local_indices = reinterpret_cast<int32_t*>(local_indices_raw);
    int32_t* done_counts = reinterpret_cast<int32_t*>(done_counts_raw);

    if (num_kv_blocks > 2048) {
        fprintf(stderr, "single_query_key_mean_topk_chunked: num_kv_blocks=%d > 2048 not supported\n",
                num_kv_blocks);
        return;
    }
    if (chunk_count <= 0 || local_count <= 0) {
        return;
    }

    if (head_dim == 64) {
        launch_single_query_key_mean_topk_chunked<T, 64>(
            q_grouped, k_mean, topk_out, local_scores, local_indices, done_counts,
            flat_heads, groups, num_kv_blocks, k_mean_capacity_blocks,
            topk_count, chunk_count, local_count);
    } else if (head_dim == 128) {
        launch_single_query_key_mean_topk_chunked<T, 128>(
            q_grouped, k_mean, topk_out, local_scores, local_indices, done_counts,
            flat_heads, groups, num_kv_blocks, k_mean_capacity_blocks,
            topk_count, chunk_count, local_count);
    } else {
        fprintf(stderr, "single_query_key_mean_topk_chunked: unsupported head_dim=%d\n", head_dim);
    }
}

void single_query_key_mean_topk_chunked(
    const void* q_grouped_raw,
    const void* k_mean_raw,
    void* topk_out_raw,
    void* local_scores_raw,
    void* local_indices_raw,
    void* done_counts_raw,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_mean_capacity_blocks,
    int head_dim,
    int topk_count,
    int chunk_count,
    int local_count,
    bool is_bf16) {
    if (is_bf16) {
        dispatch_single_query_key_mean_topk_chunked_typed<__nv_bfloat16>(
            q_grouped_raw, k_mean_raw, topk_out_raw, local_scores_raw,
            local_indices_raw, done_counts_raw, flat_heads, groups, num_kv_blocks,
            k_mean_capacity_blocks, head_dim, topk_count, chunk_count, local_count);
    } else {
        dispatch_single_query_key_mean_topk_chunked_typed<half>(
            q_grouped_raw, k_mean_raw, topk_out_raw, local_scores_raw,
            local_indices_raw, done_counts_raw, flat_heads, groups, num_kv_blocks,
            k_mean_capacity_blocks, head_dim, topk_count, chunk_count, local_count);
    }
}

__global__ __launch_bounds__(256)
void pack_topk_mask_kernel(
    const int32_t* __restrict__ topk,
    unsigned long long* __restrict__ topk_mask,
    int topk_count,
    int total_units,
    int word_count) {
    const int row_id = blockIdx.x;
    const int tid = threadIdx.x;
    unsigned long long* mask_row = topk_mask + static_cast<int64_t>(row_id) * word_count;

    for (int word = tid; word < word_count; word += blockDim.x) {
        mask_row[word] = 0ULL;
    }
    __syncthreads();

    const int32_t* topk_row = topk + static_cast<int64_t>(row_id) * topk_count;
    for (int i = tid; i < topk_count; i += blockDim.x) {
        const int unit_id = topk_row[i];
        if (unit_id >= 0 && unit_id < total_units) {
            const int word = unit_id >> 6;
            if (word < word_count) {
                atomicOr(mask_row + word, 1ULL << (unit_id & 63));
            }
        }
    }
}

void pack_topk_mask(
    const int32_t* topk,
    void* topk_mask_raw,
    int row_count,
    int topk_count,
    int total_units,
    int word_count) {
    auto topk_mask = reinterpret_cast<unsigned long long*>(topk_mask_raw);
    if (row_count <= 0 || word_count <= 0) {
        return;
    }

    pack_topk_mask_kernel<<<row_count, 256>>>(
        topk, topk_mask, topk_count, total_units, word_count);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "pack_topk_mask kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
