#include <metal_stdlib>
using namespace metal;

// A&S 7.1.26 polynomial (~1.5e-7 max error), replaces CUDA's built-in erf()
inline float erf_approx(float x) {
    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    x = metal::abs(x);

    constexpr float a1 =  0.254829592f;
    constexpr float a2 = -0.284496736f;
    constexpr float a3 =  1.421413741f;
    constexpr float a4 = -1.453152027f;
    constexpr float a5 =  1.061405429f;
    constexpr float p  =  0.3275911f;

    float t = 1.0f / metal::fma(p, x, 1.0f);

    float poly = metal::fma(a5, t, a4);
    poly = metal::fma(poly, t, a3);
    poly = metal::fma(poly, t, a2);
    poly = metal::fma(poly, t, a1);
    poly = poly * t;

    float y = metal::fma(-poly, metal::exp(-x * x), 1.0f);
    return sign * y;
}

inline float4 erf_approx4(float4 x) {
    float4 sign = select(float4(-1.0f), float4(1.0f), x >= 0.0f);
    x = metal::abs(x);

    constexpr float a1 =  0.254829592f;
    constexpr float a2 = -0.284496736f;
    constexpr float a3 =  1.421413741f;
    constexpr float a4 = -1.453152027f;
    constexpr float a5 =  1.061405429f;
    constexpr float p  =  0.3275911f;

    float4 t = 1.0f / metal::fma(float4(p), x, float4(1.0f));

    float4 poly = metal::fma(float4(a5), t, float4(a4));
    poly = metal::fma(poly, t, float4(a3));
    poly = metal::fma(poly, t, float4(a2));
    poly = metal::fma(poly, t, float4(a1));
    poly = poly * t;

    float4 y = metal::fma(-poly, metal::exp(-x * x), float4(1.0f));
    return sign * y;
}

// Matches CUDA silu_kernel: x * sigmoid(x)
// activation/activation_kernels.cu:35
inline float silu_f(float x) {
    return x / (1.0f + metal::exp(-x));
}

// Matches CUDA gelu_kernel: 0.5x(1 + erf(x/sqrt(2))), uses erf_approx
// activation/activation_kernels.cu:41
inline float gelu_f(float x) {
    constexpr float INV_SQRT2 = 0.7071067811865475f;
    return 0.5f * x * metal::fma(erf_approx(x * INV_SQRT2), 1.0f, 1.0f);
}

// Matches CUDA gelu_tanh_kernel: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
// activation/activation_kernels.cu:51
inline float gelu_tanh_f(float x) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float KAPPA = 0.044715f;
    float x2 = x * x;
    float inner = SQRT_2_OVER_PI * metal::fma(KAPPA * x2, x, x);
    return 0.5f * x * (1.0f + metal::tanh(inner));
}

// Matches CUDA fatrelu_kernel
// activation/activation_kernels.cu:122
inline float fatrelu_f(float x, float threshold) {
    return (x > threshold) ? x : 0.0f;
}

// Matches CUDA gelu_new_kernel, same formula as gelu_tanh with different eval order
// activation/activation_kernels.cu:198
inline float gelu_new_f(float x) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float KAPPA = 0.044715f;
    float x2 = x * x;
    float inner = SQRT_2_OVER_PI * metal::fma(KAPPA * x2, x, x);
    return 0.5f * x * (1.0f + metal::tanh(inner));
}

// Matches CUDA gelu_fast_kernel: 0.5x(1 + tanh(x*sqrt(2/pi)*(1 + 0.044715x^2)))
// activation/activation_kernels.cu:205
inline float gelu_fast_f(float x) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float KAPPA = 0.044715f;
    float x2 = x * x;
    float inner = x * SQRT_2_OVER_PI * metal::fma(KAPPA, x2, 1.0f);
    return 0.5f * x * (1.0f + metal::tanh(inner));
}

// Matches CUDA gelu_quick_kernel: x * sigmoid(1.702x)
// activation/activation_kernels.cu:213
inline float gelu_quick_f(float x) {
    return x / (1.0f + metal::exp(-1.702f * x));
}

inline float4 silu_f4(float4 x) {
    return x / (1.0f + metal::exp(-x));
}

inline float4 gelu_f4(float4 x) {
    constexpr float INV_SQRT2 = 0.7071067811865475f;
    return 0.5f * x * metal::fma(erf_approx4(x * INV_SQRT2), float4(1.0f), float4(1.0f));
}

inline float4 gelu_tanh_f4(float4 x) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float KAPPA = 0.044715f;
    float4 x2 = x * x;
    float4 inner = SQRT_2_OVER_PI * metal::fma(float4(KAPPA) * x2, x, x);
    return 0.5f * x * (1.0f + metal::tanh(inner));
}

inline float4 fatrelu_f4(float4 x, float threshold) {
    return select(float4(0.0f), x, x > threshold);
}

inline float4 gelu_new_f4(float4 x) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float KAPPA = 0.044715f;
    float4 x2 = x * x;
    float4 inner = SQRT_2_OVER_PI * metal::fma(float4(KAPPA) * x2, x, x);
    return 0.5f * x * (1.0f + metal::tanh(inner));
}

inline float4 gelu_fast_f4(float4 x) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float KAPPA = 0.044715f;
    float4 x2 = x * x;
    float4 inner = x * SQRT_2_OVER_PI * metal::fma(float4(KAPPA), x2, float4(1.0f));
    return 0.5f * x * (1.0f + metal::tanh(inner));
}

inline float4 gelu_quick_f4(float4 x) {
    return x / (1.0f + metal::exp(-1.702f * x));
}

kernel void silu_and_mul_f32(
    device float*       out   [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint&      d     [[buffer(2)]],
    uint2 gid                 [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint in_base = tok * 2u * d;
    uint out_base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = *reinterpret_cast<device const float4*>(&input[in_base + base_idx]);
        float4 x1 = *reinterpret_cast<device const float4*>(&input[in_base + base_idx + 4]);
        float4 y0 = *reinterpret_cast<device const float4*>(&input[in_base + d + base_idx]);
        float4 y1 = *reinterpret_cast<device const float4*>(&input[in_base + d + base_idx + 4]);

        *reinterpret_cast<device float4*>(&out[out_base + base_idx]) = silu_f4(x0) * y0;
        *reinterpret_cast<device float4*>(&out[out_base + base_idx + 4]) = silu_f4(x1) * y1;
    }
    else if (base_idx + 4 <= d) {
        float4 x = *reinterpret_cast<device const float4*>(&input[in_base + base_idx]);
        float4 y = *reinterpret_cast<device const float4*>(&input[in_base + d + base_idx]);
        *reinterpret_cast<device float4*>(&out[out_base + base_idx]) = silu_f4(x) * y;

        for (uint i = base_idx + 4; i < d; i++) {
            float xi = input[in_base + i];
            float yi = input[in_base + d + i];
            out[out_base + i] = silu_f(xi) * yi;
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            float xi = input[in_base + i];
            float yi = input[in_base + d + i];
            out[out_base + i] = silu_f(xi) * yi;
        }
    }
}

kernel void silu_and_mul_f16(
    device half*       out   [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint&     d     [[buffer(2)]],
    uint2 gid                [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint in_base = tok * 2u * d;
    uint out_base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = float4(*reinterpret_cast<device const half4*>(&input[in_base + base_idx]));
        float4 x1 = float4(*reinterpret_cast<device const half4*>(&input[in_base + base_idx + 4]));
        float4 y0 = float4(*reinterpret_cast<device const half4*>(&input[in_base + d + base_idx]));
        float4 y1 = float4(*reinterpret_cast<device const half4*>(&input[in_base + d + base_idx + 4]));

        *reinterpret_cast<device half4*>(&out[out_base + base_idx]) = half4(silu_f4(x0) * y0);
        *reinterpret_cast<device half4*>(&out[out_base + base_idx + 4]) = half4(silu_f4(x1) * y1);
    }
    else if (base_idx + 4 <= d) {
        float4 x = float4(*reinterpret_cast<device const half4*>(&input[in_base + base_idx]));
        float4 y = float4(*reinterpret_cast<device const half4*>(&input[in_base + d + base_idx]));
        *reinterpret_cast<device half4*>(&out[out_base + base_idx]) = half4(silu_f4(x) * y);

        for (uint i = base_idx + 4; i < d; i++) {
            float xi = float(input[in_base + i]);
            float yi = float(input[in_base + d + i]);
            out[out_base + i] = half(silu_f(xi) * yi);
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            float xi = float(input[in_base + i]);
            float yi = float(input[in_base + d + i]);
            out[out_base + i] = half(silu_f(xi) * yi);
        }
    }
}

kernel void mul_and_silu_f32(
    device float*       out   [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint&      d     [[buffer(2)]],
    uint2 gid                 [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint in_base = tok * 2u * d;
    uint out_base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = *reinterpret_cast<device const float4*>(&input[in_base + base_idx]);
        float4 x1 = *reinterpret_cast<device const float4*>(&input[in_base + base_idx + 4]);
        float4 y0 = *reinterpret_cast<device const float4*>(&input[in_base + d + base_idx]);
        float4 y1 = *reinterpret_cast<device const float4*>(&input[in_base + d + base_idx + 4]);

        *reinterpret_cast<device float4*>(&out[out_base + base_idx]) = x0 * silu_f4(y0);
        *reinterpret_cast<device float4*>(&out[out_base + base_idx + 4]) = x1 * silu_f4(y1);
    }
    else if (base_idx + 4 <= d) {
        float4 x = *reinterpret_cast<device const float4*>(&input[in_base + base_idx]);
        float4 y = *reinterpret_cast<device const float4*>(&input[in_base + d + base_idx]);
        *reinterpret_cast<device float4*>(&out[out_base + base_idx]) = x * silu_f4(y);

        for (uint i = base_idx + 4; i < d; i++) {
            float xi = input[in_base + i];
            float yi = input[in_base + d + i];
            out[out_base + i] = xi * silu_f(yi);
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            float xi = input[in_base + i];
            float yi = input[in_base + d + i];
            out[out_base + i] = xi * silu_f(yi);
        }
    }
}

kernel void mul_and_silu_f16(
    device half*       out   [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint&     d     [[buffer(2)]],
    uint2 gid                [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint in_base = tok * 2u * d;
    uint out_base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = float4(*reinterpret_cast<device const half4*>(&input[in_base + base_idx]));
        float4 x1 = float4(*reinterpret_cast<device const half4*>(&input[in_base + base_idx + 4]));
        float4 y0 = float4(*reinterpret_cast<device const half4*>(&input[in_base + d + base_idx]));
        float4 y1 = float4(*reinterpret_cast<device const half4*>(&input[in_base + d + base_idx + 4]));

        *reinterpret_cast<device half4*>(&out[out_base + base_idx]) = half4(x0 * silu_f4(y0));
        *reinterpret_cast<device half4*>(&out[out_base + base_idx + 4]) = half4(x1 * silu_f4(y1));
    }
    else if (base_idx + 4 <= d) {
        float4 x = float4(*reinterpret_cast<device const half4*>(&input[in_base + base_idx]));
        float4 y = float4(*reinterpret_cast<device const half4*>(&input[in_base + d + base_idx]));
        *reinterpret_cast<device half4*>(&out[out_base + base_idx]) = half4(x * silu_f4(y));

        for (uint i = base_idx + 4; i < d; i++) {
            float xi = float(input[in_base + i]);
            float yi = float(input[in_base + d + i]);
            out[out_base + i] = half(xi * silu_f(yi));
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            float xi = float(input[in_base + i]);
            float yi = float(input[in_base + d + i]);
            out[out_base + i] = half(xi * silu_f(yi));
        }
    }
}

kernel void gelu_and_mul_f32(
    device float*       out   [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint&      d     [[buffer(2)]],
    uint2 gid                 [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint in_base = tok * 2u * d;
    uint out_base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = *reinterpret_cast<device const float4*>(&input[in_base + base_idx]);
        float4 x1 = *reinterpret_cast<device const float4*>(&input[in_base + base_idx + 4]);
        float4 y0 = *reinterpret_cast<device const float4*>(&input[in_base + d + base_idx]);
        float4 y1 = *reinterpret_cast<device const float4*>(&input[in_base + d + base_idx + 4]);

        *reinterpret_cast<device float4*>(&out[out_base + base_idx]) = gelu_f4(x0) * y0;
        *reinterpret_cast<device float4*>(&out[out_base + base_idx + 4]) = gelu_f4(x1) * y1;
    }
    else if (base_idx + 4 <= d) {
        float4 x = *reinterpret_cast<device const float4*>(&input[in_base + base_idx]);
        float4 y = *reinterpret_cast<device const float4*>(&input[in_base + d + base_idx]);
        *reinterpret_cast<device float4*>(&out[out_base + base_idx]) = gelu_f4(x) * y;

        for (uint i = base_idx + 4; i < d; i++) {
            float xi = input[in_base + i];
            float yi = input[in_base + d + i];
            out[out_base + i] = gelu_f(xi) * yi;
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            float xi = input[in_base + i];
            float yi = input[in_base + d + i];
            out[out_base + i] = gelu_f(xi) * yi;
        }
    }
}

kernel void gelu_and_mul_f16(
    device half*       out   [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint&     d     [[buffer(2)]],
    uint2 gid                [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint in_base = tok * 2u * d;
    uint out_base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = float4(*reinterpret_cast<device const half4*>(&input[in_base + base_idx]));
        float4 x1 = float4(*reinterpret_cast<device const half4*>(&input[in_base + base_idx + 4]));
        float4 y0 = float4(*reinterpret_cast<device const half4*>(&input[in_base + d + base_idx]));
        float4 y1 = float4(*reinterpret_cast<device const half4*>(&input[in_base + d + base_idx + 4]));

        *reinterpret_cast<device half4*>(&out[out_base + base_idx]) = half4(gelu_f4(x0) * y0);
        *reinterpret_cast<device half4*>(&out[out_base + base_idx + 4]) = half4(gelu_f4(x1) * y1);
    }
    else if (base_idx + 4 <= d) {
        float4 x = float4(*reinterpret_cast<device const half4*>(&input[in_base + base_idx]));
        float4 y = float4(*reinterpret_cast<device const half4*>(&input[in_base + d + base_idx]));
        *reinterpret_cast<device half4*>(&out[out_base + base_idx]) = half4(gelu_f4(x) * y);

        for (uint i = base_idx + 4; i < d; i++) {
            float xi = float(input[in_base + i]);
            float yi = float(input[in_base + d + i]);
            out[out_base + i] = half(gelu_f(xi) * yi);
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            float xi = float(input[in_base + i]);
            float yi = float(input[in_base + d + i]);
            out[out_base + i] = half(gelu_f(xi) * yi);
        }
    }
}

kernel void gelu_tanh_and_mul_f32(
    device float*       out   [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint&      d     [[buffer(2)]],
    uint2 gid                 [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint in_base = tok * 2u * d;
    uint out_base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = *reinterpret_cast<device const float4*>(&input[in_base + base_idx]);
        float4 x1 = *reinterpret_cast<device const float4*>(&input[in_base + base_idx + 4]);
        float4 y0 = *reinterpret_cast<device const float4*>(&input[in_base + d + base_idx]);
        float4 y1 = *reinterpret_cast<device const float4*>(&input[in_base + d + base_idx + 4]);

        *reinterpret_cast<device float4*>(&out[out_base + base_idx]) = gelu_tanh_f4(x0) * y0;
        *reinterpret_cast<device float4*>(&out[out_base + base_idx + 4]) = gelu_tanh_f4(x1) * y1;
    }
    else if (base_idx + 4 <= d) {
        float4 x = *reinterpret_cast<device const float4*>(&input[in_base + base_idx]);
        float4 y = *reinterpret_cast<device const float4*>(&input[in_base + d + base_idx]);
        *reinterpret_cast<device float4*>(&out[out_base + base_idx]) = gelu_tanh_f4(x) * y;

        for (uint i = base_idx + 4; i < d; i++) {
            float xi = input[in_base + i];
            float yi = input[in_base + d + i];
            out[out_base + i] = gelu_tanh_f(xi) * yi;
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            float xi = input[in_base + i];
            float yi = input[in_base + d + i];
            out[out_base + i] = gelu_tanh_f(xi) * yi;
        }
    }
}

kernel void gelu_tanh_and_mul_f16(
    device half*       out   [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint&     d     [[buffer(2)]],
    uint2 gid                [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint in_base = tok * 2u * d;
    uint out_base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = float4(*reinterpret_cast<device const half4*>(&input[in_base + base_idx]));
        float4 x1 = float4(*reinterpret_cast<device const half4*>(&input[in_base + base_idx + 4]));
        float4 y0 = float4(*reinterpret_cast<device const half4*>(&input[in_base + d + base_idx]));
        float4 y1 = float4(*reinterpret_cast<device const half4*>(&input[in_base + d + base_idx + 4]));

        *reinterpret_cast<device half4*>(&out[out_base + base_idx]) = half4(gelu_tanh_f4(x0) * y0);
        *reinterpret_cast<device half4*>(&out[out_base + base_idx + 4]) = half4(gelu_tanh_f4(x1) * y1);
    }
    else if (base_idx + 4 <= d) {
        float4 x = float4(*reinterpret_cast<device const half4*>(&input[in_base + base_idx]));
        float4 y = float4(*reinterpret_cast<device const half4*>(&input[in_base + d + base_idx]));
        *reinterpret_cast<device half4*>(&out[out_base + base_idx]) = half4(gelu_tanh_f4(x) * y);

        for (uint i = base_idx + 4; i < d; i++) {
            float xi = float(input[in_base + i]);
            float yi = float(input[in_base + d + i]);
            out[out_base + i] = half(gelu_tanh_f(xi) * yi);
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            float xi = float(input[in_base + i]);
            float yi = float(input[in_base + d + i]);
            out[out_base + i] = half(gelu_tanh_f(xi) * yi);
        }
    }
}

kernel void fatrelu_and_mul_f32(
    device float*       out       [[buffer(0)]],
    device const float* input     [[buffer(1)]],
    constant uint&      d         [[buffer(2)]],
    constant float&     threshold [[buffer(3)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint in_base = tok * 2u * d;
    uint out_base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = *reinterpret_cast<device const float4*>(&input[in_base + base_idx]);
        float4 x1 = *reinterpret_cast<device const float4*>(&input[in_base + base_idx + 4]);
        float4 y0 = *reinterpret_cast<device const float4*>(&input[in_base + d + base_idx]);
        float4 y1 = *reinterpret_cast<device const float4*>(&input[in_base + d + base_idx + 4]);

        *reinterpret_cast<device float4*>(&out[out_base + base_idx]) = fatrelu_f4(x0, threshold) * y0;
        *reinterpret_cast<device float4*>(&out[out_base + base_idx + 4]) = fatrelu_f4(x1, threshold) * y1;
    }
    else if (base_idx + 4 <= d) {
        float4 x = *reinterpret_cast<device const float4*>(&input[in_base + base_idx]);
        float4 y = *reinterpret_cast<device const float4*>(&input[in_base + d + base_idx]);
        *reinterpret_cast<device float4*>(&out[out_base + base_idx]) = fatrelu_f4(x, threshold) * y;

        for (uint i = base_idx + 4; i < d; i++) {
            float xi = input[in_base + i];
            float yi = input[in_base + d + i];
            out[out_base + i] = fatrelu_f(xi, threshold) * yi;
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            float xi = input[in_base + i];
            float yi = input[in_base + d + i];
            out[out_base + i] = fatrelu_f(xi, threshold) * yi;
        }
    }
}

kernel void fatrelu_and_mul_f16(
    device half*        out       [[buffer(0)]],
    device const half*  input     [[buffer(1)]],
    constant uint&      d         [[buffer(2)]],
    constant float&     threshold [[buffer(3)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint in_base = tok * 2u * d;
    uint out_base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = float4(*reinterpret_cast<device const half4*>(&input[in_base + base_idx]));
        float4 x1 = float4(*reinterpret_cast<device const half4*>(&input[in_base + base_idx + 4]));
        float4 y0 = float4(*reinterpret_cast<device const half4*>(&input[in_base + d + base_idx]));
        float4 y1 = float4(*reinterpret_cast<device const half4*>(&input[in_base + d + base_idx + 4]));

        *reinterpret_cast<device half4*>(&out[out_base + base_idx]) = half4(fatrelu_f4(x0, threshold) * y0);
        *reinterpret_cast<device half4*>(&out[out_base + base_idx + 4]) = half4(fatrelu_f4(x1, threshold) * y1);
    }
    else if (base_idx + 4 <= d) {
        float4 x = float4(*reinterpret_cast<device const half4*>(&input[in_base + base_idx]));
        float4 y = float4(*reinterpret_cast<device const half4*>(&input[in_base + d + base_idx]));
        *reinterpret_cast<device half4*>(&out[out_base + base_idx]) = half4(fatrelu_f4(x, threshold) * y);

        for (uint i = base_idx + 4; i < d; i++) {
            float xi = float(input[in_base + i]);
            float yi = float(input[in_base + d + i]);
            out[out_base + i] = half(fatrelu_f(xi, threshold) * yi);
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            float xi = float(input[in_base + i]);
            float yi = float(input[in_base + d + i]);
            out[out_base + i] = half(fatrelu_f(xi, threshold) * yi);
        }
    }
}

kernel void silu_f32(
    device float*       out   [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint&      d     [[buffer(2)]],
    uint2 gid                 [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = *reinterpret_cast<device const float4*>(&input[base + base_idx]);
        float4 x1 = *reinterpret_cast<device const float4*>(&input[base + base_idx + 4]);
        *reinterpret_cast<device float4*>(&out[base + base_idx]) = silu_f4(x0);
        *reinterpret_cast<device float4*>(&out[base + base_idx + 4]) = silu_f4(x1);
    }
    else if (base_idx + 4 <= d) {
        float4 x = *reinterpret_cast<device const float4*>(&input[base + base_idx]);
        *reinterpret_cast<device float4*>(&out[base + base_idx]) = silu_f4(x);
        for (uint i = base_idx + 4; i < d; i++) {
            out[base + i] = silu_f(input[base + i]);
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            out[base + i] = silu_f(input[base + i]);
        }
    }
}

kernel void silu_f16(
    device half*       out   [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint&     d     [[buffer(2)]],
    uint2 gid                [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = float4(*reinterpret_cast<device const half4*>(&input[base + base_idx]));
        float4 x1 = float4(*reinterpret_cast<device const half4*>(&input[base + base_idx + 4]));
        *reinterpret_cast<device half4*>(&out[base + base_idx]) = half4(silu_f4(x0));
        *reinterpret_cast<device half4*>(&out[base + base_idx + 4]) = half4(silu_f4(x1));
    }
    else if (base_idx + 4 <= d) {
        float4 x = float4(*reinterpret_cast<device const half4*>(&input[base + base_idx]));
        *reinterpret_cast<device half4*>(&out[base + base_idx]) = half4(silu_f4(x));
        for (uint i = base_idx + 4; i < d; i++) {
            out[base + i] = half(silu_f(float(input[base + i])));
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            out[base + i] = half(silu_f(float(input[base + i])));
        }
    }
}

kernel void gelu_f32(
    device float*       out   [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint&      d     [[buffer(2)]],
    uint2 gid                 [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = *reinterpret_cast<device const float4*>(&input[base + base_idx]);
        float4 x1 = *reinterpret_cast<device const float4*>(&input[base + base_idx + 4]);
        *reinterpret_cast<device float4*>(&out[base + base_idx]) = gelu_f4(x0);
        *reinterpret_cast<device float4*>(&out[base + base_idx + 4]) = gelu_f4(x1);
    }
    else if (base_idx + 4 <= d) {
        float4 x = *reinterpret_cast<device const float4*>(&input[base + base_idx]);
        *reinterpret_cast<device float4*>(&out[base + base_idx]) = gelu_f4(x);
        for (uint i = base_idx + 4; i < d; i++) {
            out[base + i] = gelu_f(input[base + i]);
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            out[base + i] = gelu_f(input[base + i]);
        }
    }
}

kernel void gelu_f16(
    device half*       out   [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint&     d     [[buffer(2)]],
    uint2 gid                [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = float4(*reinterpret_cast<device const half4*>(&input[base + base_idx]));
        float4 x1 = float4(*reinterpret_cast<device const half4*>(&input[base + base_idx + 4]));
        *reinterpret_cast<device half4*>(&out[base + base_idx]) = half4(gelu_f4(x0));
        *reinterpret_cast<device half4*>(&out[base + base_idx + 4]) = half4(gelu_f4(x1));
    }
    else if (base_idx + 4 <= d) {
        float4 x = float4(*reinterpret_cast<device const half4*>(&input[base + base_idx]));
        *reinterpret_cast<device half4*>(&out[base + base_idx]) = half4(gelu_f4(x));
        for (uint i = base_idx + 4; i < d; i++) {
            out[base + i] = half(gelu_f(float(input[base + i])));
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            out[base + i] = half(gelu_f(float(input[base + i])));
        }
    }
}

kernel void gelu_tanh_f32(
    device float*       out   [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint&      d     [[buffer(2)]],
    uint2 gid                 [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = *reinterpret_cast<device const float4*>(&input[base + base_idx]);
        float4 x1 = *reinterpret_cast<device const float4*>(&input[base + base_idx + 4]);
        *reinterpret_cast<device float4*>(&out[base + base_idx]) = gelu_tanh_f4(x0);
        *reinterpret_cast<device float4*>(&out[base + base_idx + 4]) = gelu_tanh_f4(x1);
    }
    else if (base_idx + 4 <= d) {
        float4 x = *reinterpret_cast<device const float4*>(&input[base + base_idx]);
        *reinterpret_cast<device float4*>(&out[base + base_idx]) = gelu_tanh_f4(x);
        for (uint i = base_idx + 4; i < d; i++) {
            out[base + i] = gelu_tanh_f(input[base + i]);
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            out[base + i] = gelu_tanh_f(input[base + i]);
        }
    }
}

kernel void gelu_tanh_f16(
    device half*       out   [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint&     d     [[buffer(2)]],
    uint2 gid                [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = float4(*reinterpret_cast<device const half4*>(&input[base + base_idx]));
        float4 x1 = float4(*reinterpret_cast<device const half4*>(&input[base + base_idx + 4]));
        *reinterpret_cast<device half4*>(&out[base + base_idx]) = half4(gelu_tanh_f4(x0));
        *reinterpret_cast<device half4*>(&out[base + base_idx + 4]) = half4(gelu_tanh_f4(x1));
    }
    else if (base_idx + 4 <= d) {
        float4 x = float4(*reinterpret_cast<device const half4*>(&input[base + base_idx]));
        *reinterpret_cast<device half4*>(&out[base + base_idx]) = half4(gelu_tanh_f4(x));
        for (uint i = base_idx + 4; i < d; i++) {
            out[base + i] = half(gelu_tanh_f(float(input[base + i])));
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            out[base + i] = half(gelu_tanh_f(float(input[base + i])));
        }
    }
}

kernel void gelu_new_f32(
    device float*       out   [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint&      d     [[buffer(2)]],
    uint2 gid                 [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = *reinterpret_cast<device const float4*>(&input[base + base_idx]);
        float4 x1 = *reinterpret_cast<device const float4*>(&input[base + base_idx + 4]);
        *reinterpret_cast<device float4*>(&out[base + base_idx]) = gelu_new_f4(x0);
        *reinterpret_cast<device float4*>(&out[base + base_idx + 4]) = gelu_new_f4(x1);
    }
    else if (base_idx + 4 <= d) {
        float4 x = *reinterpret_cast<device const float4*>(&input[base + base_idx]);
        *reinterpret_cast<device float4*>(&out[base + base_idx]) = gelu_new_f4(x);
        for (uint i = base_idx + 4; i < d; i++) {
            out[base + i] = gelu_new_f(input[base + i]);
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            out[base + i] = gelu_new_f(input[base + i]);
        }
    }
}

kernel void gelu_new_f16(
    device half*       out   [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint&     d     [[buffer(2)]],
    uint2 gid                [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = float4(*reinterpret_cast<device const half4*>(&input[base + base_idx]));
        float4 x1 = float4(*reinterpret_cast<device const half4*>(&input[base + base_idx + 4]));
        *reinterpret_cast<device half4*>(&out[base + base_idx]) = half4(gelu_new_f4(x0));
        *reinterpret_cast<device half4*>(&out[base + base_idx + 4]) = half4(gelu_new_f4(x1));
    }
    else if (base_idx + 4 <= d) {
        float4 x = float4(*reinterpret_cast<device const half4*>(&input[base + base_idx]));
        *reinterpret_cast<device half4*>(&out[base + base_idx]) = half4(gelu_new_f4(x));
        for (uint i = base_idx + 4; i < d; i++) {
            out[base + i] = half(gelu_new_f(float(input[base + i])));
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            out[base + i] = half(gelu_new_f(float(input[base + i])));
        }
    }
}

kernel void gelu_fast_f32(
    device float*       out   [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint&      d     [[buffer(2)]],
    uint2 gid                 [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = *reinterpret_cast<device const float4*>(&input[base + base_idx]);
        float4 x1 = *reinterpret_cast<device const float4*>(&input[base + base_idx + 4]);
        *reinterpret_cast<device float4*>(&out[base + base_idx]) = gelu_fast_f4(x0);
        *reinterpret_cast<device float4*>(&out[base + base_idx + 4]) = gelu_fast_f4(x1);
    }
    else if (base_idx + 4 <= d) {
        float4 x = *reinterpret_cast<device const float4*>(&input[base + base_idx]);
        *reinterpret_cast<device float4*>(&out[base + base_idx]) = gelu_fast_f4(x);
        for (uint i = base_idx + 4; i < d; i++) {
            out[base + i] = gelu_fast_f(input[base + i]);
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            out[base + i] = gelu_fast_f(input[base + i]);
        }
    }
}

kernel void gelu_fast_f16(
    device half*       out   [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint&     d     [[buffer(2)]],
    uint2 gid                [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = float4(*reinterpret_cast<device const half4*>(&input[base + base_idx]));
        float4 x1 = float4(*reinterpret_cast<device const half4*>(&input[base + base_idx + 4]));
        *reinterpret_cast<device half4*>(&out[base + base_idx]) = half4(gelu_fast_f4(x0));
        *reinterpret_cast<device half4*>(&out[base + base_idx + 4]) = half4(gelu_fast_f4(x1));
    }
    else if (base_idx + 4 <= d) {
        float4 x = float4(*reinterpret_cast<device const half4*>(&input[base + base_idx]));
        *reinterpret_cast<device half4*>(&out[base + base_idx]) = half4(gelu_fast_f4(x));
        for (uint i = base_idx + 4; i < d; i++) {
            out[base + i] = half(gelu_fast_f(float(input[base + i])));
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            out[base + i] = half(gelu_fast_f(float(input[base + i])));
        }
    }
}

kernel void gelu_quick_f32(
    device float*       out   [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint&      d     [[buffer(2)]],
    uint2 gid                 [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = *reinterpret_cast<device const float4*>(&input[base + base_idx]);
        float4 x1 = *reinterpret_cast<device const float4*>(&input[base + base_idx + 4]);
        *reinterpret_cast<device float4*>(&out[base + base_idx]) = gelu_quick_f4(x0);
        *reinterpret_cast<device float4*>(&out[base + base_idx + 4]) = gelu_quick_f4(x1);
    }
    else if (base_idx + 4 <= d) {
        float4 x = *reinterpret_cast<device const float4*>(&input[base + base_idx]);
        *reinterpret_cast<device float4*>(&out[base + base_idx]) = gelu_quick_f4(x);
        for (uint i = base_idx + 4; i < d; i++) {
            out[base + i] = gelu_quick_f(input[base + i]);
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            out[base + i] = gelu_quick_f(input[base + i]);
        }
    }
}

kernel void gelu_quick_f16(
    device half*       out   [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint&     d     [[buffer(2)]],
    uint2 gid                [[thread_position_in_grid]]
) {
    uint idx8 = gid.x;
    uint tok = gid.y;
    uint base_idx = idx8 * 8;

    if (base_idx >= d) return;

    uint base = tok * d;

    if (base_idx + 8 <= d) {
        float4 x0 = float4(*reinterpret_cast<device const half4*>(&input[base + base_idx]));
        float4 x1 = float4(*reinterpret_cast<device const half4*>(&input[base + base_idx + 4]));
        *reinterpret_cast<device half4*>(&out[base + base_idx]) = half4(gelu_quick_f4(x0));
        *reinterpret_cast<device half4*>(&out[base + base_idx + 4]) = half4(gelu_quick_f4(x1));
    }
    else if (base_idx + 4 <= d) {
        float4 x = float4(*reinterpret_cast<device const half4*>(&input[base + base_idx]));
        *reinterpret_cast<device half4*>(&out[base + base_idx]) = half4(gelu_quick_f4(x));
        for (uint i = base_idx + 4; i < d; i++) {
            out[base + i] = half(gelu_quick_f(float(input[base + i])));
        }
    }
    else {
        for (uint i = base_idx; i < d; i++) {
            out[base + i] = half(gelu_quick_f(float(input[base + i])));
        }
    }
}
