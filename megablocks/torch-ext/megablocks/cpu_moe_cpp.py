# SPDX-License-Identifier: Apache-2.0
# MegaBlocks C++ Optimized CPU MoE

"""
C++ accelerated MoE with brgemm optimization for Intel AMX.
Direct replacement for cpu_fused_moe.MegaBlocksMoeMLP with better performance.
"""

import torch
from typing import Optional
# Import routing from Python version (lightweight, no performance impact)
from .cpu_fused_moe import route_tokens_cpu
import megablocks_cpu


def fused_moe_cpp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
    alpha: float = 1.702,
    limit: float = 7.0,
    is_interleaved: bool = True,
) -> torch.Tensor:
    """
    C++ Fused MoE with brgemm optimization.
    
    Uses at::native::cpublas::brgemm for efficient batch GEMM on Intel CPUs.
    """
    return megablocks_cpu.fused_moe_cpu(
        hidden_states, w1, w2, topk_weights, topk_ids,
        w1_bias, w2_bias, activation, alpha, limit, is_interleaved
    )


class MegaBlocksMoeMLP(torch.nn.Module):
    """
    C++ optimized MoE MLP using brgemm.
    Drop-in replacement for cpu_fused_moe.MegaBlocksMoeMLP with better performance.
    
    Usage in transformers:
        from megablocks.cpu_moe_cpp import MegaBlocksMoeMLP
        # Will be used via @use_kernel_forward_from_hub decorator
    """
    can_torch_compile: bool = False  # C++ kernels don't support torch.compile
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through the MoE layer using C++ kernel.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            Tuple of (output, expert_weights)
        """
        
        # Get MoE parameters
        moe_top_k = getattr(self.router, "top_k", 4)
        moe_num_experts = getattr(self.experts, "num_experts", 128)
        moe_normalize_expert_weights = getattr(self.experts, "normalize_expert_weights", None)
        
        # Detect activation type
        if hasattr(self.experts, "alpha") and hasattr(self.experts, "limit"):
            activation = "swigluoai"
            alpha = self.experts.alpha
            limit = self.experts.limit
        else:
            activation = getattr(self.experts, "activation", "silu")
            alpha = 1.702
            limit = 7.0
        
        # Get weight tensors
        if hasattr(self.experts, "gate_up_proj"):
            w1 = self.experts.gate_up_proj
            is_interleaved = True  # GptOss uses interleaved layout
        elif hasattr(self.experts, "w1"):
            w1 = self.experts.w1
            w3 = getattr(self.experts, "w3", None)
            if w3 is not None:
                w1 = torch.cat([w1, w3], dim=-1)
            is_interleaved = False
        else:
            raise AttributeError("experts module must have 'gate_up_proj' or 'w1' attribute")
        
        if hasattr(self.experts, "down_proj"):
            w2 = self.experts.down_proj
        elif hasattr(self.experts, "w2"):
            w2 = self.experts.w2
        else:
            raise AttributeError("experts module must have 'down_proj' or 'w2' attribute")
        
        # Get optional bias tensors
        w1_bias = getattr(self.experts, "gate_up_proj_bias", None)
        w2_bias = getattr(self.experts, "down_proj_bias", None)
        
        # Store original shape
        in_shape = x.size()
        
        # Route tokens to experts (Python implementation is fast enough)
        logits, expert_weights, expert_indices = route_tokens_cpu(
            x,
            self.router.weight,
            getattr(self.router, "bias", None),
            moe_top_k,
            moe_num_experts,
            moe_normalize_expert_weights,
        )
        
        # Flatten input
        x_flat = x.view(-1, x.shape[-1])
        
        # Call C++ optimized kernel (main performance bottleneck)
        output = fused_moe_cpp(
            hidden_states=x_flat,
            w1=w1,
            w2=w2,
            topk_weights=expert_weights,
            topk_ids=expert_indices,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            activation=activation,
            alpha=alpha,
            limit=limit,
            is_interleaved=is_interleaved,
        )
        
        # Restore original shape
        output = output.view(in_shape)
        
        return output, expert_weights


__all__ = ["fused_moe_cpp", "MegaBlocksMoeMLP"]
