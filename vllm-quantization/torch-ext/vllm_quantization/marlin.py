from typing import TYPE_CHECKING, Optional

import torch

# neuron has torch version that doesn't even have impl_abstract
if TYPE_CHECKING:
    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake


from ._ops import ops, add_op_namespace_prefix
from .scalar_type import ScalarType


# gptq_marlin
def gptq_marlin_gemm(a: torch.Tensor,
                     c: Optional[torch.Tensor],
                     b_q_weight: torch.Tensor,
                     b_scales: torch.Tensor,
                     global_scale: Optional[torch.Tensor],
                     b_zeros: Optional[torch.Tensor],
                     g_idx: Optional[torch.Tensor],
                     perm: Optional[torch.Tensor],
                     workspace: torch.Tensor,
                     b_q_type: ScalarType,
                     size_m: int,
                     size_n: int,
                     size_k: int,
                     is_k_full: bool = True,
                     use_atomic_add: bool = False,
                     use_fp32_reduce: bool = False,
                     is_zp_float: bool = False) -> torch.Tensor:
    return ops.gptq_marlin_gemm(a, c, b_q_weight, b_scales,
                                         global_scale, b_zeros, g_idx, perm,
                                         workspace, b_q_type.id, size_m,
                                         size_n, size_k, is_k_full,
                                         use_atomic_add, use_fp32_reduce,
                                         is_zp_float)

# gptq_marlin
def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    return ops.gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)


# gptq_marlin
def awq_marlin_repack(
    b_q_weight: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    return ops.awq_marlin_repack(b_q_weight, size_k, size_n, num_bits)


# marlin
def marlin_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    size_m: int,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    return ops.marlin_gemm(
        a, b_q_weight, b_scales, workspace, size_m, size_n, size_k
    )


# marlin_24
def gptq_marlin_24_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_meta: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    return ops.gptq_marlin_24_gemm(
        a, b_q_weight, b_meta, b_scales, workspace, b_q_type.id, size_m, size_n, size_k
    )


# qqq ops
def marlin_qqq_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    s_tok: torch.Tensor,
    s_ch: torch.Tensor,
    s_group: torch.Tensor,
    workspace: torch.Tensor,
    size_m: int,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    return ops.marlin_qqq_gemm(
        a, b_q_weight, s_tok, s_ch, s_group, workspace, size_m, size_n, size_k
    )


# Fake ops

if hasattr(ops, "gptq_marlin_24_gemm"):
    @register_fake(add_op_namespace_prefix("gptq_marlin_24_gemm"))
    def _gptq_marlin_24_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                                    b_meta: torch.Tensor, b_scales: torch.Tensor,
                                    workspace: torch.Tensor,
                                    b_q_type: ScalarType, size_m: torch.SymInt,
                                    size_n: torch.SymInt,
                                    size_k: torch.SymInt) -> torch.Tensor:
        return torch.empty((size_m, size_n), device=a.device, dtype=a.dtype)

    @register_fake(add_op_namespace_prefix("gptq_marlin_gemm"))
    def _gptq_marlin_gemm_fake(a: torch.Tensor,
                               c: Optional[torch.Tensor],
                               b_q_weight: torch.Tensor,
                               b_scales: torch.Tensor,
                               global_scale: Optional[torch.Tensor],
                               b_zeros: Optional[torch.Tensor],
                               g_idx: Optional[torch.Tensor],
                               perm: Optional[torch.Tensor],
                               workspace: torch.Tensor,
                               b_q_type_id: int,
                               size_m: torch.SymInt,
                               size_n: torch.SymInt,
                               size_k: torch.SymInt,
                               is_k_full: bool = True,
                               use_atomic_add: bool = False,
                               use_fp32_reduce: bool = False,
                               is_zp_float: bool = False) -> torch.Tensor:
        return torch.empty((size_m, size_n), device=a.device, dtype=a.dtype)

    @register_fake(add_op_namespace_prefix("marlin_qqq_gemm"))
    def _marlin_qqq_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                                s_tok: torch.Tensor, s_ch: torch.Tensor,
                                s_group: torch.Tensor, workspace: torch.Tensor,
                                size_m: torch.SymInt, size_n: torch.SymInt,
                                size_k: torch.SymInt) -> torch.Tensor:
        return torch.empty((size_m, size_n),
                            dtype=torch.float16,
                            device=a.device)

    @register_fake(add_op_namespace_prefix("marlin_gemm"))
    def _marlin_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                            b_scales: torch.Tensor, workspace: torch.Tensor,
                            size_m: torch.SymInt, size_n: torch.SymInt,
                            size_k: torch.SymInt) -> torch.Tensor:
        return torch.empty((size_m, size_n),
                            dtype=torch.float16,
                            device=a.device)
