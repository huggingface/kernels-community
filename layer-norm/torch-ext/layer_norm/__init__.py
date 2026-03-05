import torch
import torch.nn as nn

from ._ops import ops

from . import layers

def dropout_add_ln_fwd(input, gamma, beta, rowscale, colscale, x0_subset, z_subset, dropout_p, epsilon, rowscale_const, z_numrows, gen, residual_in_fp32, is_rms_norm):
    return ops.dropout_add_ln_fwd(input, gamma, beta, rowscale, colscale, x0_subset, z_subset, dropout_p, epsilon, rowscale_const, z_numrows, gen, residual_in_fp32, is_rms_norm)

def dropout_add_ln_bwd(dz, dx, x, mu, rsigma, gamma, rowscale, colscale, x0_subset, z_subset, dropout_p, rowscale_const, x0_numrows, has_residual, is_rms_norm):
    return ops.dropout_add_ln_bwd(dz, dx, x, mu, rsigma, gamma, rowscale, colscale, x0_subset, z_subset, dropout_p, rowscale_const, x0_numrows, has_residual, is_rms_norm)

def dropout_add_ln_parallel_residual_fwd(input, gamma0, beta0, gamma1, beta1, dropout_p, epsilon, gen, residual_in_fp32, is_rms_norm):
    return ops.dropout_add_ln_parallel_residual_fwd(input, gamma0, beta0, gamma1, beta1, dropout_p, epsilon, gen, residual_in_fp32, is_rms_norm)

def dropout_add_ln_parallel_residual_bwd(dz0, dz1, dx, x, mu, rsigma, gamma0, gamma1, dropout_p, has_x1, has_residual, is_rms_norm):
    return ops.dropout_add_ln_parallel_residual_bwd(dz0, dz1, dx, x, mu, rsigma, gamma0, gamma1, dropout_p, has_x1, has_residual, is_rms_norm)

__all__ = [
    "layers",
    "dropout_add_ln_fwd",
    "dropout_add_ln_bwd",
    "dropout_add_ln_parallel_residual_fwd",
    "dropout_add_ln_parallel_residual_bwd",
]