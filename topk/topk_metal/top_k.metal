#include <metal_stdlib>

using namespace metal;

/* Top-k over a small row, for a MoE router.
 *
 * Not ggml's: ggml sorts (a bitonic pass plus a merge) because its `argsort` is general. A router
 * wants the largest `k` of a few hundred logits, on one row per token, and torch's MPS `topk` is a
 * full sort too -- 71us for k=8 of 256, against 21us for a plain argmax over the same data. Over 40
 * layers that is ~2.8ms per token spent selecting 8 numbers.
 *
 * One threadgroup per row, one pass per output: each pass reduces the row to its maximum, records
 * it, and masks it out. k*n comparisons, but k and n are small and the launch dominates either way.
 */
kernel void kernel_top_k_f32(
        device const float * logits    [[buffer(0)]],
        device       int   * indices   [[buffer(1)]],
        device       float * values    [[buffer(2)]],
        constant     int   & n         [[buffer(3)]],
        constant     int   & k         [[buffer(4)]],
        constant     int   & softmax   [[buffer(5)]],
        uint  row  [[threadgroup_position_in_grid]],
        uint  tid  [[thread_position_in_threadgroup]],
        uint  nth  [[threads_per_threadgroup]],
        threadgroup float * shared_val [[threadgroup(0)]]) {
    device const float * src = logits + (ulong)row * n;
    threadgroup int * shared_idx = (threadgroup int *)(shared_val + nth);

    // `taken` is tracked by writing -INFINITY into a per-thread running copy of the best candidate,
    // so no scratch buffer is needed for the mask: each pass re-scans and skips already-chosen ones.
    for (int picked = 0; picked < k; ++picked) {
        float best = -INFINITY;
        int   best_i = -1;
        for (int i = tid; i < n; i += nth) {
            float v = src[i];
            bool used = false;
            for (int j = 0; j < picked; ++j) {
                used = used || (indices[(ulong)row * k + j] == i);
            }
            if (!used && v > best) {
                best = v;
                best_i = i;
            }
        }
        shared_val[tid] = best;
        shared_idx[tid] = best_i;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = nth / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                // ties go to the lower index, which is what a stable top-k does
                bool take = shared_val[tid + stride] > shared_val[tid] ||
                            (shared_val[tid + stride] == shared_val[tid] &&
                             shared_idx[tid + stride] < shared_idx[tid]);
                if (take) {
                    shared_val[tid] = shared_val[tid + stride];
                    shared_idx[tid] = shared_idx[tid + stride];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            indices[(ulong)row * k + picked] = shared_idx[0];
            values [(ulong)row * k + picked] = shared_val[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // A router softmaxes exactly the k it selected, and that is another dispatch for k numbers --
    // as expensive as a softmax over the whole row, since it is launch-bound. Done here instead.
    if (softmax != 0 && tid == 0) {
        device float * out = values + (ulong)row * k;
        float top = out[0];  // passes are in descending order, so the first is the maximum
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            out[i] = exp(out[i] - top);
            sum += out[i];
        }
        for (int i = 0; i < k; ++i) {
            out[i] /= sum;
        }
    }
}
