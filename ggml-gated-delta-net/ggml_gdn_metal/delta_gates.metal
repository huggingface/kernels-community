// The two scalar gates a gated-delta-net step needs, in one dispatch.
//
// Per layer the model computes `beta = sigmoid(b)` and `g = -exp(A_log) * softplus(a + dt_bias)` over one
// value per head -- 32 numbers. In torch that is six launches on 32 floats each, and it measures 59 us a
// layer, against 18 us for the recurrence kernel those numbers feed. The arithmetic is free; the launches
// are the cost, so they collapse into one.
//
// `softplus` is written as upstream computes it (log1p(exp(x)) with the large-x branch), so the values
// match `F.softplus` rather than approximating it.

#include <metal_stdlib>
using namespace metal;

// `b` and `a` hold one value per head for every sequence in the batch, `n_values` in all; `a_log` and
// `dt_bias` hold one per head and repeat across the batch.
typedef struct {
    int32_t n_heads;
    int32_t n_values;
} ggml_attn_kargs_delta_gates;

kernel void kernel_delta_gates_f32(
        constant ggml_attn_kargs_delta_gates & args,
        device const float * b,
        device const float * a,
        device const float * a_log,
        device const float * dt_bias,
        device       float * beta,
        device       float * g,
        uint tpig[[thread_position_in_grid]]) {
    if ((int) tpig >= args.n_values) {
        return;
    }
    const int head = (int) tpig % args.n_heads;

    beta[tpig] = 1.0f / (1.0f + exp(-b[tpig]));

    const float x = a[tpig] + dt_bias[head];
    // log1p(exp(x)) saturates to x once exp(x) overflows the mantissa; 20 is where the two agree to f32.
    const float softplus = x > 20.0f ? x : log(1.0f + exp(x));
    g[tpig] = -exp(a_log[head]) * softplus;
}
