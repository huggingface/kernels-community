from ._ops import ops

def mamba_selective_scan(A, B, C, hidden_states, discrete_time_step, ssm_state, scan_output, B_size, D_size, L_size, N_size):
    return ops.mamba_selective_scan(A, B, C, hidden_states, discrete_time_step, ssm_state, scan_output, B_size, D_size, L_size, N_size)

__all__ = ["mamba_selective_scan"]