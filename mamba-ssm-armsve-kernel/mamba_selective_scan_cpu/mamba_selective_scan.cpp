#include "mamba_selective_scan.hpp"
#include "mamba_selective_scan_sve.hpp"

#include "cpu_feature.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
namespace mamba {

static inline void mamba_selective_scan_kernel_cpu(
    float* A,
    float* B,
    float* C,
    float* hidden_states,
    float* discrete_time_step,
    float* ssm_state,
    float* scan_output,
    int64_t B_size,
    int64_t D_size,
    int64_t L_size,
    int64_t N_size
) {
    for (int64_t i = 0; i < B_size; i++) {
        for (int64_t j = 0; j < D_size; j++) {
            for (int64_t l = 0; l < L_size; l++) {
                const int64_t dts_offset = i * D_size * L_size + j * L_size + l;

                const float dts = discrete_time_step[dts_offset];
                const float hs  = hidden_states[dts_offset];

                float r1 = 0.0f;

                const int64_t B_base        = i * L_size * N_size + l * N_size;
                const int64_t ssmstate_base = i * D_size * N_size + j * N_size;
                const int64_t A_base        = j * N_size;

                for (int64_t k = 0; k < N_size; k++) {
                    const float a = A[A_base + k];
                    const float b = B[B_base + k];
                    const float c = C[B_base + k];

                    // discrete_A = exp(dts * a)
                    const float discA = std::exp(dts * a);

                    // deltaB_u = hs * (dts * b)
                    const float deltaBu = hs * (dts * b);

                    // ssm = ssm * discA + deltaBu
                    float& s = ssm_state[ssmstate_base + k];
                    s = s * discA + deltaBu;

                    // r1 += s * c
                    r1 += s * c;
                }

                scan_output[dts_offset] += r1;
            }
        }
    }
}

void mamba_selective_scan(
    torch::Tensor& A,
    torch::Tensor& B,
    torch::Tensor& C,
    torch::Tensor& hidden_states,
    torch::Tensor& discrete_time_step,
    torch::Tensor& ssm_state,
    torch::Tensor& scan_output,
    int64_t B_size,
    int64_t D_size,
    int64_t L_size,
    int64_t N_size
) {

#if (defined(__aarch64__) || defined(_M_ARM64))
    if (cpuinfo::CPUFeaturesARM::hasSVE()) {
    
        mamba_selective_scan_sve(
            A, B, C, hidden_states, discrete_time_step, ssm_state, scan_output,
            B_size, D_size, L_size, N_size
        );
        return;
    }
#endif
    mamba_selective_scan_kernel_cpu(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        hidden_states.data_ptr<float>(),
        discrete_time_step.data_ptr<float>(),
        ssm_state.data_ptr<float>(),
        scan_output.data_ptr<float>(),
        B_size, D_size, L_size, N_size
    );
}

} // namespace mamba

void mamba_selective_scan(torch::Tensor &A,
                          torch::Tensor &B,
                          torch::Tensor &C,
                          torch::Tensor &hidden_states,
                          torch::Tensor &discrete_time_step,
                          torch::Tensor &ssm_state,
                          torch::Tensor &scan_output,
                          int64_t B_size,
                          int64_t D_size,
                          int64_t L_size,
                          int64_t N_size) {
    mamba::mamba_selective_scan(
        A, B, C, hidden_states, discrete_time_step, ssm_state, scan_output,
        B_size, D_size, L_size, N_size
    );
}