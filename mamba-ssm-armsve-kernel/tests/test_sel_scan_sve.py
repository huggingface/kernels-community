from typing import Optional
import torch
import mamba_selective_scan
import time

print(dir(mamba_selective_scan))
def mamba_selective_scan_ref(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    hidden_states: torch.Tensor,
    discrete_time_step: torch.Tensor,
    ssm_state: torch.Tensor,
):
    """
    Reference implementation of mamba_selective_scan with broadcasting.
    Shapes:
    A: [D, N]
    B: [B, L, N]
    C: [B, L, N]
    hidden_states: [B, L, D]
    discrete_time_step: [B, D, L]
    ssm_state: [B, D, N]
    """
    dtype = hidden_states.dtype
    B_size, L_size, N_size = B.shape
    D_size = A.shape[0]
    print(dtype, A.dtype, B.dtype, C.dtype, hidden_states.dtype, discrete_time_step.dtype, ssm_state.dtype)
    # Clamp to avoid numerical overflow
    # A = A.clamp(min=-5.0, max=0.0)
    # discrete_time_step = discrete_time_step.clamp(min=-0.1, max=0.1)

    discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])  # [B, D, L, N]
    discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()        # [B, D, L, N]
    deltaB_u = discrete_B * hidden_states[:, :, :, None] 

    scan_output_list = []
    for i in range(L_size):
        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
        out_i = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [B, D, 1]
        scan_output_list.append(out_i[:, :, 0])

    scan_output = torch.stack(scan_output_list, dim=-1)  # [B, D, L]
    return scan_output

def test_kernel():
    # Smaller and safe tensor sizes
    B_size = 1
    D_size = 8
    L_size = 5
    N_size = 4

    A = -torch.linspace(0.1, 1.0, N_size).expand(D_size, N_size)  # [8, 4]
    B = torch.randn(B_size, L_size, N_size) * 0.1
    C = torch.randn(B_size, L_size, N_size) * 0.1
    
    hidden_states = torch.randn(B_size, D_size, L_size, dtype=torch.float32).contiguous()*0.1
    discrete_time_step = torch.randn(B_size, D_size, L_size, dtype=torch.float32) * 0.05
    ssm_state = torch.randn(B_size, D_size, N_size, dtype=torch.float32).contiguous()*0.1
    
    kernel_out = torch.zeros(B_size, D_size, L_size, dtype=torch.float32).contiguous()

    # Reference output
    start_time = time.perf_counter()
    ref_out = mamba_selective_scan_ref(A, B, C, hidden_states, discrete_time_step, ssm_state)
    end_time1 = time.perf_counter()
    # Call your C++ kernel
    mamba_selective_scan.mamba_selective_scan(
        A, B, C, hidden_states, discrete_time_step, ssm_state, kernel_out,
        B_size, D_size, L_size, N_size
    )
    end_time2 = time.perf_counter()
    print(f"Reference implementation time: {end_time1 - start_time:.6f} seconds")
    print(f"C++ kernel time: {end_time2 - end_time1:.6f} seconds")

    # Compare outputs
    err_norm = torch.linalg.norm(ref_out - kernel_out)
    print("Error norm:", err_norm)
    if err_norm >= 1e-6:
        assert err_norm < 1e-6, f"Error too high! norm = {err_norm}"
    else:
        print("Test passed!")
    print ("Speedup over reference: {:.2f}x".format((end_time1 - start_time) / (end_time2 - end_time1)))

if __name__ == "__main__":
    test_kernel()
