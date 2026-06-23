import torch

from . import layers
from ._ops import ops
from .fp8_utils import per_token_group_quant_fp8, w8a8_block_fp8_matmul
from .fused_marlin_moe import fused_marlin_moe
from .fused_moe import fused_experts, fused_moe, fused_topk, grouped_topk
from .scalar_type import ScalarType, scalar_types

# Do not remove this import, it registers the custom ops.
from . import fake as _


def gptq_marlin_moe_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = torch.empty(
        (num_experts, size_k // 16, size_n * (num_bits // 2)),
        device=b_q_weight.device,
        dtype=b_q_weight.dtype,
    )
    for e in range(num_experts):
        output[e] = ops.gptq_marlin_repack(
            b_q_weight[e], perm[e], size_k, size_n, num_bits
        )
    return output


def awq_marlin_moe_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = torch.empty(
        (num_experts, size_k // 16, size_n * (num_bits // 2)),
        device=b_q_weight.device,
        dtype=b_q_weight.dtype,
    )
    for e in range(num_experts):
        output[e] = ops.awq_marlin_repack(b_q_weight[e], size_k, size_n, num_bits)
    return output


def moe_sum(input: torch.Tensor, output: torch.Tensor):
    ops.moe_sum(input, output)


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indicies: torch.Tensor,
    gating_output: float,
) -> None:
    ops.topk_softmax(topk_weights, topk_ids, token_expert_indicies, gating_output)


__all__ = [
    "ScalarType",
    "awq_marlin_moe_repack",
    "fused_experts",
    "fused_marlin_moe",
    "fused_moe",
    "fused_topk",
    "gptq_marlin_moe_repack",
    "grouped_topk",
    "layers",
    "moe_align_block_size",
    "moe_sum",
    "per_token_group_quant_fp8",
    "scalar_types",
    "topk_softmax",
    "w8a8_block_fp8_matmul",
]
