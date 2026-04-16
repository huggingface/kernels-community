// Fused int4 SDPA Metal kernel for Apple Silicon
//
// Computes: output = softmax(Q @ dequant(K_int4)^T / sqrt(d)) @ dequant(V_int4)
// in a single kernel dispatch with online softmax — no intermediate score
// matrix is ever materialized.
//
// Architecture (adapted from MLX sdpa_vector):
//   - BN simdgroups per threadgroup, each processes a different key token
//   - BD=32 SIMD lanes per simdgroup, each handles D/32 elements
//   - Online softmax with running max + sum_exp, reduced across simdgroups
//
// int4 format (MLX-compatible):
//   - uint32 packs 8 x 4-bit values
//   - group_size=64: every 64 elements share one (scale, bias) pair
//   - dequantize: value = scale * nibble + bias
//
// Key optimization: "qdot" pattern — pre-scale query elements by
// {1, 1/16, 1/256, 1/4096} so that dot product with raw packed nibbles
// (without shifting) gives the correct result. This eliminates per-nibble
// shift operations in the inner loop.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Fused int4 SDPA — single-token decode (vector query)
// ---------------------------------------------------------------------------
// Template parameters:
//   D:  head dimension (256 or 512)
//   BN: number of simdgroups per threadgroup (32 for D=256, 16 for D=512)
//
// Buffers:
//   queries:     (num_heads, D) float32
//   k_quant:     (num_kv_heads, N, D/8) uint32  — packed int4 keys
//   k_scales:    (num_kv_heads, N, D/64) float32 — per-group scale
//   k_biases:    (num_kv_heads, N, D/64) float32 — per-group bias
//   v_quant:     (num_kv_heads, N, D/8) uint32  — packed int4 values
//   v_scales:    (num_kv_heads, N, D/64) float32
//   v_biases:    (num_kv_heads, N, D/64) float32
//   out:         (num_heads, D) float32
//   gqa_factor:  num_heads / num_kv_heads
//   N:           sequence length (number of KV cache entries)
//   scale:       attention scale (typically 1/sqrt(D))
//   sliding_window: if > 0, only attend to last `sliding_window` tokens

template <int D, int BN = 32>
[[kernel]] void sdpa_int4_vector(
    const device float* queries        [[buffer(0)]],
    const device uint32_t* k_quant     [[buffer(1)]],
    const device float* k_scales       [[buffer(2)]],
    const device float* k_biases       [[buffer(3)]],
    const device uint32_t* v_quant     [[buffer(4)]],
    const device float* v_scales       [[buffer(5)]],
    const device float* v_biases       [[buffer(6)]],
    device float* out                  [[buffer(7)]],
    const constant int& gqa_factor     [[buffer(8)]],
    const constant int& N              [[buffer(9)]],
    const constant float& scale        [[buffer(10)]],
    const constant int& sliding_window [[buffer(11)]],
    const constant int& num_heads      [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr int BD = 32;
    constexpr int elems_per_lane = D / BD;

    const int head_idx = tid.x;
    // For batched decode: head_idx may exceed num_heads, use modulo for KV mapping
    const int kv_head_idx = (head_idx % num_heads) / gqa_factor;

    const int k_packed_dim = D / 8;
    const int k_scale_dim = D / 64;

    // Base pointers for this KV head
    const device uint32_t* k_q_base = k_quant + kv_head_idx * N * k_packed_dim;
    const device float* k_s_base = k_scales + kv_head_idx * N * k_scale_dim;
    const device float* k_b_base = k_biases + kv_head_idx * N * k_scale_dim;
    const device uint32_t* v_q_base = v_quant + kv_head_idx * N * k_packed_dim;
    const device float* v_s_base = v_scales + kv_head_idx * N * k_scale_dim;
    const device float* v_b_base = v_biases + kv_head_idx * N * k_scale_dim;

    int lane_start = simd_lid * elems_per_lane;

    // qdot pattern: pre-scale query so dot with raw nibbles works
    // q_scaled[i+j] = q[i+j] / 16^j, then:
    //   q_scaled[i] * (w & 0xF) + q_scaled[i+1] * (w & 0xF0) + ...
    //   = q[i]*nib0 + q[i+1]*nib1 + q[i+2]*nib2 + q[i+3]*nib3
    thread float q_scaled[D / BD];
    for (int i = 0; i < elems_per_lane; i += 4) {
        q_scaled[i]     = scale * queries[head_idx * D + lane_start + i];
        q_scaled[i + 1] = scale * queries[head_idx * D + lane_start + i + 1] / 16.0f;
        q_scaled[i + 2] = scale * queries[head_idx * D + lane_start + i + 2] / 256.0f;
        q_scaled[i + 3] = scale * queries[head_idx * D + lane_start + i + 3] / 4096.0f;
    }

    // Pre-compute qsum (bias dot-product term) per group — constant across all tokens
    constexpr int groups_per_lane = elems_per_lane / 4;
    thread float qsum_precomp[groups_per_lane];
    thread int group_indices[groups_per_lane];
    for (int g = 0; g < groups_per_lane; g++) {
        int i = g * 4;
        qsum_precomp[g] = q_scaled[i] + q_scaled[i+1]*16.0f
                        + q_scaled[i+2]*256.0f + q_scaled[i+3]*4096.0f;
        group_indices[g] = (lane_start + i) / 64;
    }

    // Accumulator for weighted values
    thread float o[D / BD];
    for (int i = 0; i < elems_per_lane; i++) o[i] = 0.0f;

    float max_score = -1e30f;
    float sum_exp = 0.0f;

    // Fast-skip for sliding window: jump directly to first valid token
    int ki_start = simd_gid;
    if (sliding_window > 0) {
        int first_valid = N - sliding_window;
        if (first_valid > 0) {
            // Align to this simdgroup's stride
            int aligned = first_valid - (first_valid % BN) + int(simd_gid);
            if (aligned < first_valid) aligned += BN;
            ki_start = aligned;
        }
    }

    // Main loop: each simdgroup processes tokens ki, ki+BN, ki+2*BN, ...
    for (int ki = ki_start; ki < N; ki += BN) {

        // --- Score: Q · dequant(K[ki]) via qdot ---
        float score = 0.0f;
        const device uint16_t* kw = (const device uint16_t*)(k_q_base + ki * k_packed_dim)
                                    + lane_start / 4;
        int k_row_offset = ki * k_scale_dim;
        for (int g = 0; g < groups_per_lane; g++) {
            int i = g * 4;
            float s = k_s_base[k_row_offset + group_indices[g]];
            float b = k_b_base[k_row_offset + group_indices[g]];
            uint16_t w = kw[g];

            float accum = q_scaled[i]     * float(w & 0x000Fu)
                        + q_scaled[i + 1] * float(w & 0x00F0u)
                        + q_scaled[i + 2] * float(w & 0x0F00u)
                        + q_scaled[i + 3] * float(w & 0xF000u);

            score += s * accum + b * qsum_precomp[g];
        }
        score = simd_sum(score);

        // --- Online softmax ---
        float new_max = max(max_score, score);
        float factor = metal::fast::exp2(1.4426950408889634f * (max_score - new_max));
        float exp_score = metal::fast::exp2(1.4426950408889634f * (score - new_max));
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;

        // --- Dequantize V[ki] + weighted accumulation ---
        const device uint16_t* vw = (const device uint16_t*)(v_q_base + ki * k_packed_dim)
                                    + lane_start / 4;
        int v_row_offset = ki * k_scale_dim;
        for (int g = 0; g < groups_per_lane; g++) {
            int i = g * 4;
            float evs = exp_score * v_s_base[v_row_offset + group_indices[g]];
            float evb = exp_score * v_b_base[v_row_offset + group_indices[g]];
            uint16_t w = vw[g];

            o[i]     = o[i]     * factor + evs * float(w & 0x000Fu) + evb;
            o[i + 1] = o[i + 1] * factor + evs * float((w >> 4) & 0xFu) + evb;
            o[i + 2] = o[i + 2] * factor + evs * float((w >> 8) & 0xFu) + evb;
            o[i + 3] = o[i + 3] * factor + evs * float((w >> 12) & 0xFu) + evb;
        }
    }

    // --- Cross-simdgroup reduction ---

    // Step 1: find global max and sum across all simdgroups
    threadgroup float tg_max_arr[BN];
    threadgroup float tg_sum_arr[BN];
    if (simd_lid == 0) {
        tg_max_arr[simd_gid] = max_score;
        tg_sum_arr[simd_gid] = sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sg_max = (simd_lid < uint(BN)) ? tg_max_arr[simd_lid] : -1e30f;
    float new_max = simd_max(sg_max);
    float sg_factor = metal::fast::exp2(1.4426950408889634f * (sg_max - new_max));
    float sg_sum_val = (simd_lid < uint(BN)) ? tg_sum_arr[simd_lid] : 0.0f;
    float global_sum = simd_sum(sg_sum_val * sg_factor);

    // Step 2: rescale this simdgroup's partial output
    float my_factor = metal::fast::exp2(1.4426950408889634f * (max_score - new_max));
    for (int i = 0; i < elems_per_lane; i++) {
        o[i] *= my_factor;
    }

    // Step 3: tree reduction of partial outputs — O(log2(BN)) barriers instead of O(BN)
    threadgroup float tg_stage[(BN / 2) * D];

    for (int step = 1; step < BN; step *= 2) {
        if (int(simd_gid % (2 * step)) == step) {
            int slot = simd_gid / (2 * step);
            for (int i = 0; i < elems_per_lane; i++) {
                tg_stage[slot * D + simd_lid * elems_per_lane + i] = o[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if ((simd_gid % (2 * step)) == 0 && (simd_gid + step < BN)) {
            int slot = simd_gid / (2 * step);
            for (int i = 0; i < elems_per_lane; i++) {
                o[i] += tg_stage[slot * D + simd_lid * elems_per_lane + i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 4: simdgroup 0 now holds the fully-reduced output — normalize and write
    if (simd_gid == 0) {
        for (int i = 0; i < elems_per_lane; i++) {
            int elem = simd_lid * elems_per_lane + i;
            out[head_idx * D + elem] = global_sum == 0 ? 0.0f : (o[i] / global_sum);
        }
    }
}

// ---------------------------------------------------------------------------
// Split-K variant: each threadgroup processes a chunk of tokens
// Writes partial (output, max_score, sum_exp) to device memory
// A second kernel reduces across splits
// ---------------------------------------------------------------------------
template <int D, int BN = 16>
[[kernel]] void sdpa_int4_splitk(
    const device float* queries        [[buffer(0)]],
    const device uint32_t* k_quant     [[buffer(1)]],
    const device float* k_scales       [[buffer(2)]],
    const device float* k_biases       [[buffer(3)]],
    const device uint32_t* v_quant     [[buffer(4)]],
    const device float* v_scales       [[buffer(5)]],
    const device float* v_biases       [[buffer(6)]],
    device float* partial_out          [[buffer(7)]],   // (num_heads, num_splits, D)
    device float* partial_max          [[buffer(8)]],   // (num_heads, num_splits)
    device float* partial_sum          [[buffer(9)]],   // (num_heads, num_splits)
    const constant int& gqa_factor     [[buffer(10)]],
    const constant int& N              [[buffer(11)]],
    const constant float& scale        [[buffer(12)]],
    const constant int& sliding_window [[buffer(13)]],
    const constant int& num_splits     [[buffer(14)]],
    const constant int& num_heads      [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr int BD = 32;
    constexpr int elems_per_lane = D / BD;

    const int head_idx = tid.x;
    const int split_idx = tid.y;
    const int kv_head_idx = (head_idx % num_heads) / gqa_factor;

    const int k_packed_dim = D / 8;
    const int k_scale_dim = D / 64;

    // Token range for this split
    int tokens_per_split = (N + num_splits - 1) / num_splits;
    int ki_start = split_idx * tokens_per_split;
    int ki_end = min(ki_start + tokens_per_split, N);

    const device uint32_t* k_q_base = k_quant + kv_head_idx * N * k_packed_dim;
    const device float* k_s_base = k_scales + kv_head_idx * N * k_scale_dim;
    const device float* k_b_base = k_biases + kv_head_idx * N * k_scale_dim;
    const device uint32_t* v_q_base = v_quant + kv_head_idx * N * k_packed_dim;
    const device float* v_s_base = v_scales + kv_head_idx * N * k_scale_dim;
    const device float* v_b_base = v_biases + kv_head_idx * N * k_scale_dim;

    int lane_start = simd_lid * elems_per_lane;

    thread float q_scaled[D / BD];
    for (int i = 0; i < elems_per_lane; i += 4) {
        q_scaled[i]     = scale * queries[head_idx * D + lane_start + i];
        q_scaled[i + 1] = scale * queries[head_idx * D + lane_start + i + 1] / 16.0f;
        q_scaled[i + 2] = scale * queries[head_idx * D + lane_start + i + 2] / 256.0f;
        q_scaled[i + 3] = scale * queries[head_idx * D + lane_start + i + 3] / 4096.0f;
    }

    constexpr int groups_per_lane = elems_per_lane / 4;
    thread float qsum_precomp[groups_per_lane];
    thread int group_indices[groups_per_lane];
    for (int g = 0; g < groups_per_lane; g++) {
        int i = g * 4;
        qsum_precomp[g] = q_scaled[i] + q_scaled[i+1]*16.0f
                        + q_scaled[i+2]*256.0f + q_scaled[i+3]*4096.0f;
        group_indices[g] = (lane_start + i) / 64;
    }

    thread float o[D / BD];
    for (int i = 0; i < elems_per_lane; i++) o[i] = 0.0f;
    float max_score = -1e30f;
    float sum_exp = 0.0f;

    // Fast-skip for sliding window in split-K
    int loop_start = ki_start + int(simd_gid);
    if (sliding_window > 0) {
        int first_valid = N - sliding_window;
        if (first_valid > loop_start) {
            int aligned = first_valid - ((first_valid - loop_start) % BN);
            if (aligned < first_valid) aligned += BN;
            loop_start = max(loop_start, aligned);
        }
    }

    // Process only this split's token range
    for (int ki = loop_start; ki < ki_end; ki += BN) {

        float score = 0.0f;
        const device uint16_t* kw = (const device uint16_t*)(k_q_base + ki * k_packed_dim)
                                    + lane_start / 4;
        int k_row_offset = ki * k_scale_dim;
        for (int g = 0; g < groups_per_lane; g++) {
            int i = g * 4;
            float s = k_s_base[k_row_offset + group_indices[g]];
            float b = k_b_base[k_row_offset + group_indices[g]];
            uint16_t w = kw[g];
            float accum = q_scaled[i]     * float(w & 0x000Fu)
                        + q_scaled[i + 1] * float(w & 0x00F0u)
                        + q_scaled[i + 2] * float(w & 0x0F00u)
                        + q_scaled[i + 3] * float(w & 0xF000u);
            score += s * accum + b * qsum_precomp[g];
        }
        score = simd_sum(score);

        float new_max = max(max_score, score);
        float factor = metal::fast::exp2(1.4426950408889634f * (max_score - new_max));
        float exp_score = metal::fast::exp2(1.4426950408889634f * (score - new_max));
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;

        const device uint16_t* vw = (const device uint16_t*)(v_q_base + ki * k_packed_dim)
                                    + lane_start / 4;
        int v_row_offset = ki * k_scale_dim;
        for (int g = 0; g < groups_per_lane; g++) {
            int i = g * 4;
            float evs = exp_score * v_s_base[v_row_offset + group_indices[g]];
            float evb = exp_score * v_b_base[v_row_offset + group_indices[g]];
            uint16_t w = vw[g];
            o[i]     = o[i]     * factor + evs * float(w & 0x000Fu) + evb;
            o[i + 1] = o[i + 1] * factor + evs * float((w >> 4) & 0xFu) + evb;
            o[i + 2] = o[i + 2] * factor + evs * float((w >> 8) & 0xFu) + evb;
            o[i + 3] = o[i + 3] * factor + evs * float((w >> 12) & 0xFu) + evb;
        }
    }

    // Cross-simdgroup reduction (same as main kernel)
    threadgroup float tg_max_arr[BN];
    threadgroup float tg_sum_arr[BN];
    if (simd_lid == 0) {
        tg_max_arr[simd_gid] = max_score;
        tg_sum_arr[simd_gid] = sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sg_max = (simd_lid < uint(BN)) ? tg_max_arr[simd_lid] : -1e30f;
    float new_max = simd_max(sg_max);
    float sg_factor = metal::fast::exp2(1.4426950408889634f * (sg_max - new_max));
    float sg_sum_val = (simd_lid < uint(BN)) ? tg_sum_arr[simd_lid] : 0.0f;
    float global_sum = simd_sum(sg_sum_val * sg_factor);

    float my_factor = metal::fast::exp2(1.4426950408889634f * (max_score - new_max));
    for (int i = 0; i < elems_per_lane; i++) o[i] *= my_factor;

    threadgroup float tg_stage[(BN / 2) * D];
    for (int step = 1; step < BN; step *= 2) {
        if (int(simd_gid % (2 * step)) == step) {
            int slot = simd_gid / (2 * step);
            for (int i = 0; i < elems_per_lane; i++)
                tg_stage[slot * D + simd_lid * elems_per_lane + i] = o[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((simd_gid % (2 * step)) == 0 && (simd_gid + step < BN)) {
            int slot = simd_gid / (2 * step);
            for (int i = 0; i < elems_per_lane; i++)
                o[i] += tg_stage[slot * D + simd_lid * elems_per_lane + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partial results to device memory (unnormalized)
    if (simd_gid == 0) {
        int out_offset = (head_idx * num_splits + split_idx) * D;
        for (int i = 0; i < elems_per_lane; i++) {
            partial_out[out_offset + simd_lid * elems_per_lane + i] = o[i];
        }
        if (simd_lid == 0) {
            partial_max[head_idx * num_splits + split_idx] = new_max;
            partial_sum[head_idx * num_splits + split_idx] = global_sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Reduction kernel: combine split-K partials
// One threadgroup per head, single simdgroup
// ---------------------------------------------------------------------------
template <int D>
[[kernel]] void sdpa_int4_reduce(
    const device float* partial_out   [[buffer(0)]],   // (num_heads, num_splits, D)
    const device float* partial_max   [[buffer(1)]],   // (num_heads, num_splits)
    const device float* partial_sum   [[buffer(2)]],   // (num_heads, num_splits)
    device float* out                 [[buffer(3)]],   // (num_heads, D)
    const constant int& num_splits    [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr int BD = 32;
    constexpr int elems_per_lane = D / BD;
    const int head_idx = tid.x;

    // Find global max across splits
    float global_max = -1e30f;
    for (int s = 0; s < num_splits; s++) {
        global_max = max(global_max, partial_max[head_idx * num_splits + s]);
    }

    // Accumulate rescaled outputs and sums
    thread float acc[D / BD];
    for (int i = 0; i < elems_per_lane; i++) acc[i] = 0.0f;
    float total_sum = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        float split_max = partial_max[head_idx * num_splits + s];
        float split_sum = partial_sum[head_idx * num_splits + s];
        float factor = metal::fast::exp2(1.4426950408889634f * (split_max - global_max));
        total_sum += split_sum * factor;

        int in_offset = (head_idx * num_splits + s) * D;
        for (int i = 0; i < elems_per_lane; i++) {
            acc[i] += partial_out[in_offset + simd_lid * elems_per_lane + i] * factor;
        }
    }

    // Normalize and write
    for (int i = 0; i < elems_per_lane; i++) {
        int elem = simd_lid * elems_per_lane + i;
        out[head_idx * D + elem] = total_sum == 0 ? 0.0f : (acc[i] / total_sum);
    }
}

// Instantiate for common head dimensions
// Default BN values (production)
template [[host_name("sdpa_int4_128")]] [[kernel]]
void sdpa_int4_vector<128, 16>(
    const device float*, const device uint32_t*, const device float*, const device float*,
    const device uint32_t*, const device float*, const device float*,
    device float*, const constant int&, const constant int&, const constant float&,
    const constant int&, const constant int&, uint3, uint, uint);

template [[host_name("sdpa_int4_256")]] [[kernel]]
void sdpa_int4_vector<256, 16>(
    const device float*, const device uint32_t*, const device float*, const device float*,
    const device uint32_t*, const device float*, const device float*,
    device float*, const constant int&, const constant int&, const constant float&,
    const constant int&, const constant int&, uint3, uint, uint);

template [[host_name("sdpa_int4_512")]] [[kernel]]
void sdpa_int4_vector<512, 16>(
    const device float*, const device uint32_t*, const device float*, const device float*,
    const device uint32_t*, const device float*, const device float*,
    device float*, const constant int&, const constant int&, const constant float&,
    const constant int&, const constant int&, uint3, uint, uint);

// Split-K instantiations
template [[host_name("sdpa_int4_splitk_128")]] [[kernel]]
void sdpa_int4_splitk<128, 16>(
    const device float*, const device uint32_t*, const device float*, const device float*,
    const device uint32_t*, const device float*, const device float*,
    device float*, device float*, device float*,
    const constant int&, const constant int&, const constant float&,
    const constant int&, const constant int&, const constant int&, uint3, uint, uint);

template [[host_name("sdpa_int4_splitk_256")]] [[kernel]]
void sdpa_int4_splitk<256, 16>(
    const device float*, const device uint32_t*, const device float*, const device float*,
    const device uint32_t*, const device float*, const device float*,
    device float*, device float*, device float*,
    const constant int&, const constant int&, const constant float&,
    const constant int&, const constant int&, const constant int&, uint3, uint, uint);

template [[host_name("sdpa_int4_splitk_512")]] [[kernel]]
void sdpa_int4_splitk<512, 16>(
    const device float*, const device uint32_t*, const device float*, const device float*,
    const device uint32_t*, const device float*, const device float*,
    device float*, device float*, device float*,
    const constant int&, const constant int&, const constant float&,
    const constant int&, const constant int&, const constant int&, uint3, uint, uint);

// Reduction instantiations
template [[host_name("sdpa_int4_reduce_128")]] [[kernel]]
void sdpa_int4_reduce<128>(
    const device float*, const device float*, const device float*,
    device float*, const constant int&, uint3, uint);

template [[host_name("sdpa_int4_reduce_256")]] [[kernel]]
void sdpa_int4_reduce<256>(
    const device float*, const device float*, const device float*,
    device float*, const constant int&, uint3, uint);

template [[host_name("sdpa_int4_reduce_512")]] [[kernel]]
void sdpa_int4_reduce<512>(
    const device float*, const device float*, const device float*,
    device float*, const constant int&, uint3, uint);
