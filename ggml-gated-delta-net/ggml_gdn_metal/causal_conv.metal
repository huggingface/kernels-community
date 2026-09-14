#include <metal_stdlib>

using namespace metal;

/* The short causal convolution a linear-attention layer runs before the delta rule, for one token.
 *
 * Eager torch spells this as five dispatches -- concatenate the cached state with the new token,
 * copy the tail back into the cache, a grouped `conv1d`, a slice, and the activation -- and measures
 * 32.5us for 8192 channels of a 4-wide kernel, where the arithmetic is 32k multiply-adds. What costs
 * is the reduction: MPS is slow at those, and the grouped convolution is one group per channel.
 *
 * ggml's own `ssm_conv` is not usable here: it dispatches one thread per threadgroup over as many
 * threadgroups as there are channels, which suits llama.cpp's batched prefill and measured *slower*
 * than torch for a single token.
 *
 * One thread per channel, many channels per threadgroup. Each thread reads its own row, so the
 * shifted state can be written back in place once the window is in registers.
 */
kernel void kernel_causal_conv_update_f32(
        device const float * state    [[buffer(0)]],  // (channels, state_width), updated in place
        device const float * x        [[buffer(1)]],  // (channels,) this token
        device const float * weight   [[buffer(2)]],  // (channels, k)
        device const float * bias     [[buffer(3)]],  // (channels,)
        device       float * out      [[buffer(4)]],  // (channels,)
        device       float * state_out[[buffer(5)]],  // aliases `state`
        constant     int   & channels [[buffer(6)]],
        constant     int   & swidth   [[buffer(7)]],
        constant     int   & k        [[buffer(8)]],
        constant     int   & has_bias [[buffer(9)]],
        constant     int   & silu     [[buffer(10)]],
        uint gid [[thread_position_in_grid]]) {
    if ((int) gid >= channels) {
        return;
    }

    device const float * s = state + (int) gid * swidth;
    const float xv = x[gid];

    // The window is the cached state followed by this token; the convolution reads its last `k`
    // values, which is what `conv1d` then a slice comes to.
    float acc = has_bias ? bias[gid] : 0.0f;
    device const float * w = weight + (int) gid * k;
    const int offset = swidth + 1 - k;
    for (int i = 0; i < k; ++i) {
        const int j = offset + i;
        acc += w[i] * (j < swidth ? s[j] : xv);
    }

    // Roll the cache forward by one: everything the next token still needs, then this token.
    device float * so = state_out + (int) gid * swidth;
    for (int i = 0; i < swidth - 1; ++i) {
        so[i] = s[i + 1];
    }
    so[swidth - 1] = xv;

    out[gid] = silu ? acc / (1.0f + exp(-acc)) : acc;
}
