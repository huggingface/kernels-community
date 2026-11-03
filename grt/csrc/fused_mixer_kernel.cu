/*
 * Fused RotationalMixer Kernel - Compilation Unit
 */

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "fused_mixer_kernel.cuh"


extern "C" void fused_mixer_fwd_launcher(
    const float* x_in,
    const float* angles, const float* coupling, const float* raw_scales,
    float* x_out,
    float* save_feat_norm, float* save_inv_norm, float* save_theta,
    int N, int D, int n_groups, int G, float eps
) {
    const int TPB = MIXER_TPB;
    int n_tiles = (N + TPB - 1) / TPB;
    int blocks = n_tiles * n_groups;
    dim3 block(G, TPB);

    // Shared mem:
    //  s_coup [F*G*G] + s_x0[TPB*G] + s_x1[TPB*G] + s_feat[TPB*F*G] + s_scl[2*G]
    // Plus static __shared__ s_inv_norm[TPB] which is declared inside the kernel.
    const int F_FEAT = 4;
    size_t coup_size  = F_FEAT * G * G;
    size_t smem_bytes =
        (coup_size
         + TPB * G
         + TPB * G
         + TPB * F_FEAT * G
         + 2 * G) * sizeof(float);

    fused_mixer_fwd_kernel<<<blocks, block, smem_bytes>>>(
        x_in, angles, coupling, raw_scales, x_out,
        save_feat_norm, save_inv_norm, save_theta,
        N, D, n_groups, G, eps
    );
}


extern "C" void fused_mixer_bwd_launcher(
    const float* grad_out, const float* x_in,
    const float* angles, const float* coupling, const float* raw_scales,
    const float* save_feat_norm, const float* save_inv_norm, const float* save_theta,
    float* grad_x, float* grad_angles, float* grad_coupling, float* grad_raw_scales,
    int N, int D, int n_groups, int G, float eps
) {
    const int TPB = MIXER_TPB;
    int n_tiles = (N + TPB - 1) / TPB;
    int blocks = n_tiles * n_groups;
    dim3 block(G, TPB);

    const int F_FEAT = 4;
    size_t coup_size = F_FEAT * G * G;
    size_t smem_bytes =
        (coup_size                         // s_coup
         + TPB * F_FEAT * G                // s_featn
         + TPB * G                         // s_x0
         + TPB * G                         // s_x1
         + TPB * G                         // s_dmod
         + TPB * F_FEAT * G                // s_dfeat
         + G                               // s_gang
         + 2 * G                           // s_gscl
         + coup_size                       // s_gcoup
        ) * sizeof(float);

    fused_mixer_bwd_kernel<<<blocks, block, smem_bytes>>>(
        grad_out, x_in, angles, coupling, raw_scales,
        save_feat_norm, save_inv_norm, save_theta,
        grad_x, grad_angles, grad_coupling, grad_raw_scales,
        N, D, n_groups, G, eps
    );
}
