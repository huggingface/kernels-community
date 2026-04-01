# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end MoE expert layer using unfused grouped GEMM primitives.

Pipeline:
  1. Sort tokens by expert (histc + cumsum + argsort)
  2. Gate+up grouped GEMM → bf16 intermediate
  3. SiLU(gate) * up → bf16 intermediate
  4. Down grouped GEMM → bf16 output
  5. Apply routing weights + unsort + top_k reduce

All operations are deterministic (no atomic_add). This serves as the
baseline implementation; see fused.py for the fused variant that
merges steps 2-3 into a single kernel.
"""

import torch
import torch.nn.functional as F

from .batched import w8a8_block_fp8_matmul_batched
from .grouped import w8a8_block_fp8_matmul_grouped


def moe_grouped(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """End-to-end MoE expert layer using unfused grouped GEMM primitives.

    Input:  unsorted hidden_states + router outputs (top_k_index, top_k_weights)
    Output: (num_tokens, hidden) — accumulated across top_k experts

    Pipeline: sort → gate_up GEMM → SiLU → down GEMM → routing + unsort + reduce
    Deterministic: no atomic_add, all operations have fixed execution order.
    """

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_experts = gate_up_proj.size(0)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    # S is the number of selected token-expert pairs (S = num_tokens * num_top_k)
    token_idx = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1)
        .expand(-1, num_top_k)
        .reshape(-1)
    )  # (S,)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)

    # Sort by expert for grouped processing
    perm = torch.argsort(expert_ids)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)

    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    selected_hidden_states_g = hidden_states[token_idx[perm]]

    # Compute offsets for grouped processing.
    # histc instead of bincount avoids cuda-graph issues;
    # CPU requires float input, CUDA requires int input (deterministic mode).
    histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
    tokens_per_expert = torch.histc(
        histc_input, bins=num_experts, min=0, max=num_experts - 1
    )
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    # --- Gate+up projection per expert (FP8 grouped) ---
    proj_out = w8a8_block_fp8_matmul_grouped(
        selected_hidden_states_g,
        gate_up_proj,
        gate_up_proj_scale_inv,
        offsets,
        tokens_per_expert,
        block_size,
    )  # (S, 2 * intermediate_dim)

    # Apply SiLU gating
    gate, up = proj_out.chunk(2, dim=-1)
    proj_out = F.silu(gate) * up  # (S, intermediate_dim)

    # --- Down projection per expert (FP8 grouped) ---
    proj_out = w8a8_block_fp8_matmul_grouped(
        proj_out, down_proj, down_proj_scale_inv, offsets, tokens_per_expert, block_size
    )  # (S, hidden_dim)

    # Apply routing weights
    weighted_out = proj_out * sample_weights_g.to(proj_out.dtype).unsqueeze(
        -1
    )  # (S, hidden_dim)

    # Restore original order
    weighted_out = weighted_out[inv_perm]

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # (index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd)
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(
        dim=1
    )

    return final_hidden_states.to(hidden_states.dtype)


def moe_batched(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """End-to-end MoE expert layer using unfused batched GEMM primitives.

    Input:  unsorted hidden_states + router outputs (top_k_index, top_k_weights)
    Output: (num_tokens, hidden) — accumulated across top_k experts

    Pipeline: expand → batched_mm(gate_up) → SiLU → batched_mm(down) → routing + reduce
    Deterministic: no sorting, no atomic_add. Each token processed independently.
    """
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    # S is the number of selected token-expert pairs (S = num_tokens * num_top_k)
    token_idx = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1)
        .expand(-1, num_top_k)
        .reshape(-1)
    )  # (S,)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)

    # Get current hidden states for selected samples (no sorting needed)
    selected_hidden_states = hidden_states[token_idx]

    # --- Gate+up projection per expert (FP8 batched) ---
    proj_out = w8a8_block_fp8_matmul_batched(
        selected_hidden_states,
        gate_up_proj,
        gate_up_proj_scale_inv,
        expert_ids,
        block_size,
    )  # (S, 2 * intermediate_dim)

    # Apply SiLU gating
    gate, up = proj_out.chunk(2, dim=-1)
    proj_out = F.silu(gate) * up  # (S, intermediate_dim)

    # --- Down projection per expert (FP8 batched) ---
    proj_out = w8a8_block_fp8_matmul_batched(
        proj_out, down_proj, down_proj_scale_inv, expert_ids, block_size
    )  # (S, hidden_dim)

    # Apply routing weights
    weighted_out = proj_out * sample_weights.to(proj_out.dtype).unsqueeze(
        -1
    )  # (S, hidden_dim)

    # Accumulate results using deterministic reshape+sum instead of index_add_
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(
        dim=1
    )

    return final_hidden_states.to(hidden_states.dtype)
