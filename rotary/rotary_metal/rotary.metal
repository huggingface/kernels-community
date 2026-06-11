#include <metal_stdlib>
using namespace metal;

// Params buffer layout (uint32), see rotary.mm:
// [0] ndim  [1] conj
// [2..9]   sizes (broadcast shape)
// [10..17] strides x1   [18..25] strides x2
// [26..33] strides cos  [34..41] strides sin
// [42..49] strides out1 [50..57] strides out2
constant constexpr uint P_SIZES = 2;
constant constexpr uint P_STRIDES = 10;
constant constexpr uint MAX_DIMS = 8;

// Elementwise rotation with explicit stride arithmetic so that broadcast
// cos/sin and non-contiguous views (e.g. q[..., :rotary_dim]) work like the
// CUDA TensorIterator path. Math is done in float and rounded once, matching
// the CUDA kernel.
template <typename T>
inline void rotary_impl(device const T *x1, device const T *x2,
                        device const T *cos_, device const T *sin_,
                        device T *out1, device T *out2,
                        constant uint *p, uint index) {
    const uint ndim = p[0];
    const bool conj = p[1] != 0;

    uint offs[6] = {0, 0, 0, 0, 0, 0};
    uint rem = index;
    for (uint i = 0; i < ndim; ++i) {
        const uint d = ndim - 1 - i;
        const uint coord = rem % p[P_SIZES + d];
        rem /= p[P_SIZES + d];
        for (uint t = 0; t < 6; ++t) {
            offs[t] += coord * p[P_STRIDES + t * MAX_DIMS + d];
        }
    }

    const float a = float(x1[offs[0]]);
    const float b = float(x2[offs[1]]);
    const float c = float(cos_[offs[2]]);
    const float s = float(sin_[offs[3]]);

    if (!conj) {
        out1[offs[4]] = static_cast<T>(a * c - b * s);
        out2[offs[5]] = static_cast<T>(a * s + b * c);
    } else {
        out1[offs[4]] = static_cast<T>(a * c + b * s);
        out2[offs[5]] = static_cast<T>(-a * s + b * c);
    }
}

#define ROTARY_KERNEL(name, T)                                            \
kernel void name(device const T *x1 [[buffer(0)]],                        \
                 device const T *x2 [[buffer(1)]],                        \
                 device const T *cos_ [[buffer(2)]],                      \
                 device const T *sin_ [[buffer(3)]],                      \
                 device T *out1 [[buffer(4)]],                            \
                 device T *out2 [[buffer(5)]],                            \
                 constant uint *params [[buffer(6)]],                     \
                 uint index [[thread_position_in_grid]]) {                \
    rotary_impl<T>(x1, x2, cos_, sin_, out1, out2, params, index);        \
}

ROTARY_KERNEL(rotary_float, float)
ROTARY_KERNEL(rotary_half, half)
ROTARY_KERNEL(rotary_bfloat, bfloat)
