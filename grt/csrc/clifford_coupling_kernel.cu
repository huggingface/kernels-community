/*
 * Clifford Coupling Features - CUDA Kernel Compilation Unit
 *
 * PyTorch's cpp_extension handles CUDA→HIP translation on ROCm,
 * so we only include cuda_runtime.h — hipify rewrites it automatically.
 */

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include "clifford_coupling_kernel.cuh"

// ---------------------------------------------------------------------------
// Launch wrappers - extern "C" for cpp binding (default stream)
// ---------------------------------------------------------------------------

extern "C" void clifford_fwd_launcher(
    const float* x_pairs,
    float* features,
    int batch,
    int P
) {
    int threads = (P <= 256) ? P : 256;
    int blocks_x = (P + threads - 1) / threads;

    // Split batch across grid Y and Z to stay within 65535 limit per dim
    int grid_y = (batch <= 65535) ? batch : 65535;
    int grid_z = (batch + grid_y - 1) / grid_y;
    dim3 blocks(blocks_x, grid_y, grid_z);

    int smem_bytes = threads * 2 * sizeof(float);

    clifford_coupling_fwd_kernel<<<blocks, threads, smem_bytes, 0>>>(
        x_pairs, features, batch, P
    );
}

extern "C" void clifford_bwd_launcher(
    const float* x_pairs,
    const float* grad_features,
    float* grad_x_pairs,
    int batch,
    int P
) {
    int threads = (P <= 256) ? P : 256;
    int blocks_x = (P + threads - 1) / threads;

    int grid_y = (batch <= 65535) ? batch : 65535;
    int grid_z = (batch + grid_y - 1) / grid_y;
    dim3 blocks(blocks_x, grid_y, grid_z);

    int smem_bytes = threads * 3 * sizeof(float);

    clifford_coupling_bwd_kernel<<<blocks, threads, smem_bytes, 0>>>(
        x_pairs, grad_features, grad_x_pairs, batch, P
    );
}
