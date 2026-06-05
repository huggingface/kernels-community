# SPDX-License-Identifier: Apache-2.0
"""Adapt AITER's Triton grouped GEMM to MegaBlocks' ``gmm`` calling convention.

MegaBlocks (following tgale96/grouped_gemm) uses a single ``gmm`` entry point
with ``trans_a`` / ``trans_b`` flags:

* ``trans_a=False, trans_b=False``: a(M,K) @ b(G,K,N) -> c(M,N)
* ``trans_a=False, trans_b=True`` : a(M,K) @ b(G,N,K)^T -> c(M,N)   (dgrad)
* ``trans_a=True``                : a(M,K)^T @ b(M,N) per group -> c(G,K,N)  (wgrad)

AITER exposes these as two kernels: ``gmm`` ((M,K)@(G,K,N)->(M,N), transposition
of the 3D operand inferred from strides) and ``ptgmm`` ((K,M)@(M,N)->(G,K,N),
transposition of the 2D operand inferred from strides).
"""

import torch

from .gmm import gmm as _aiter_gmm
from .gmm import ptgmm as _aiter_ptgmm


def gmm(a, b, c, batch_sizes, trans_a=False, trans_b=False):
    # AITER requires group sizes to be int32 and to live on the compute device.
    group_sizes = batch_sizes.to(device=a.device, dtype=torch.int32)

    # AITER asserts exact strides: gmm wants lhs/rhs row-major (a transposed
    # 3D operand must be exactly column-major), tgmm wants rhs row-major and
    # lhs row/column-major. Make operands contiguous first so the transposed
    # views have the precise strides the kernels expect. `.contiguous()` is a
    # no-op when the tensor is already contiguous.
    if trans_a:
        # Weight gradient: a(M,K), b(M,N) -> c(G,K,N).
        # Pass a transposed so AITER sees lhs(K,M) column-major (TRANS_LHS).
        _aiter_ptgmm(
            a.contiguous().transpose(0, 1),
            b.contiguous(),
            group_sizes,
            preferred_element_type=c.dtype,
            existing_out=c,
        )
    else:
        # trans_b contracts b's last dim: pass a column-major (G,K,N) view.
        rhs = b.contiguous()
        if trans_b:
            rhs = rhs.transpose(1, 2)
        _aiter_gmm(
            a.contiguous(),
            rhs,
            group_sizes,
            preferred_element_type=c.dtype,
            existing_out=c,
        )
    return c
