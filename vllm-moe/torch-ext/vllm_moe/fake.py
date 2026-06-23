from typing import TYPE_CHECKING, Optional

import torch

from ._ops import add_op_namespace_prefix, ops
from .scalar_type import ScalarType

# Register ops
from .utils.custom_ops import direct_register_custom_op
from .fused_moe import inplace_fused_experts, inplace_fused_experts_fake

direct_register_custom_op(
    op_name="inplace_fused_experts",
    op_func=inplace_fused_experts,
    mutates_args=["hidden_states"],
    fake_impl=inplace_fused_experts_fake,
)


from .fused_moe import outplace_fused_experts, outplace_fused_experts_fake

direct_register_custom_op(
    op_name="outplace_fused_experts",
    op_func=outplace_fused_experts,
    mutates_args=[],
    fake_impl=outplace_fused_experts_fake,
)


# neuron has torch version that doesn't even have impl_abstract
if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn

else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake


if hasattr(ops, "single_marlin_gemm_moe"):

    @register_fake(add_op_namespace_prefix("single_marlin_gemm_moe"))
    def single_marlin_moe_fake(
        hidden_states: torch.Tensor,
        w: torch.Tensor,
        scales: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        g_idx: Optional[torch.Tensor] = None,
        sort_indices: Optional[torch.Tensor] = None,
        w_zeros: Optional[torch.Tensor] = None,
        num_bits: int = 8,
        is_k_full: bool = True,
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states)


if hasattr(ops, "fused_marlin_moe"):

    @register_fake(add_op_namespace_prefix("fused_marlin_moe"))
    def fused_marlin_moe_fake(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        gating_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        g_idx1: Optional[torch.Tensor] = None,
        g_idx2: Optional[torch.Tensor] = None,
        sort_indices1: Optional[torch.Tensor] = None,
        sort_indices2: Optional[torch.Tensor] = None,
        w1_zeros: Optional[torch.Tensor] = None,
        w2_zeros: Optional[torch.Tensor] = None,
        num_bits: int = 8,
        is_k_full: bool = True,
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states)


if hasattr(ops, "marlin_gemm_moe"):

    @register_fake(add_op_namespace_prefix("marlin_gemm_moe"))
    def marlin_gemm_moe_fake(
        a: torch.Tensor,
        b_q_weights: torch.Tensor,
        sorted_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        b_scales: torch.Tensor,
        b_zero_points: torch.Tensor,
        g_idx: torch.Tensor,
        perm: torch.Tensor,
        workspace: torch.Tensor,
        b_q_type: ScalarType,
        size_m: torch.SymInt,
        size_n: torch.SymInt,
        size_k: torch.SymInt,
        is_k_full: bool,
        num_experts: int,
        topk: int,
        moe_block_size: int,
        replicate_input: bool,
        apply_weights: bool,
    ) -> torch.Tensor:
        return torch.empty((size_m, topk, size_n), dtype=a.dtype, device=a.device)
