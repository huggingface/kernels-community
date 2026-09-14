// RMS norm against a zero-centred weight, in one dispatch.
//
// Upstream's `kernel_rms_norm_mul_f32` multiplies the normalised row by a tensor. Models that store the
// norm weight zero-centred (Gemma, Qwen3-Next) need `1 + weight` instead, and materialising that costs
// either a launch per norm or a cached tensor that goes stale when the weight is written. Folding the
// `1 +` in here removes both: the weight is read as it is stored.
//
// Deliberately a copy of upstream's reduction rather than a call into it: the file is compiled alongside
// ggml-metal.metal, which does not export the pieces, and the arithmetic is four lines.

#include <metal_stdlib>
using namespace metal;

typedef struct {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
} ggml_attn_kargs_norm;

// The gated variant: `rms_norm(x) * weight * silu(gate)`, which is how a gated-delta-net layer finishes.
// Three launches over 32 heads of 128 values in torch; the values are far too small for that to be
// anything but launch cost.
kernel void kernel_rms_norm_gate_f32(
        constant ggml_attn_kargs_norm & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    device const float * x = (device const float *) (src0 + i01*args.nbf1[0]);
    device const float * w = (device const float *) (src1);
    device const float * gate = (device const float *) (src2 + i01*args.nbf1[0]);

    float sumf = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        sumf += x[i00] * x[i00];
    }
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = simd_sum(shmem_f32[tiisg]);

    const float scale = rsqrt(sumf / args.ne00 + args.eps);
    device float * y = (device float *) (dst + i01*args.nb1);
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        const float gv = gate[i00];
        y[i00] = (x[i00] * scale) * w[i00] * (gv / (1.0f + exp(-gv)));
    }
}
