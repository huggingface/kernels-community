# SPDX-License-Identifier: Apache-2.0
# MegaBlocks CPU Fused MoE Implementation
#
# This is a pure Python/PyTorch implementation for CPU.
# For better performance, consider using the C++ kernel implementation.
#
import torch
import torch.nn.functional as F


def swigluoai_activation(gate: torch.Tensor, up: torch.Tensor, 
                         alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    """
    SwigluOAI activation function used in GptOss models.
    
    Formula:
        gate = clamp(gate, max=limit)
        up = clamp(up, -limit, limit)
        glu = gate * sigmoid(gate * alpha)
        output = (up + 1) * glu
    
    Args:
        gate: Gate tensor from gate projection
        up: Up tensor from up projection  
        alpha: Scaling factor for sigmoid (default: 1.702)
        limit: Clamp limit (default: 7.0)
    
    Returns:
        Activated tensor
    """
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1) * glu


def silu_and_mul_activation(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    SiLU (Swish) activation with element-wise multiplication.
    
    Formula:
        output = silu(gate) * up
    
    Args:
        gate: Gate tensor
        up: Up tensor
    
    Returns:
        Activated tensor
    """
    return F.silu(gate) * up


def route_tokens_cpu(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor | None,
    moe_top_k: int,
    moe_num_experts: int,
    moe_normalize_expert_weights: int | None = None,
) -> tuple:
    """
    Route tokens to experts and compute expert weights and indices (CPU version).
    
    Args:
        x: Input tensor [batch, seq, hidden] or [tokens, hidden]
        router_weight: Router weight [num_experts, hidden]
        router_bias: Router bias [num_experts] or None
        moe_top_k: Number of experts per token
        moe_num_experts: Total number of experts
        moe_normalize_expert_weights: Normalization order or None
    
    Returns:
        Tuple of (logits, expert_weights, expert_indices)
    """
    x_flat = x.view(-1, x.shape[-1])
    logits = F.linear(x_flat, router_weight, router_bias)
    
    if moe_top_k == 1:
        expert_weights, expert_indices = logits.max(dim=-1, keepdim=True)
    else:
        expert_weights, expert_indices = torch.topk(logits, moe_top_k, dim=-1)
    
    expert_weights = expert_weights.softmax(dim=-1)
    
    if moe_normalize_expert_weights is not None:
        expert_weights = expert_weights / torch.norm(
            expert_weights,
            p=moe_normalize_expert_weights,
            dim=-1,
            keepdim=True,
        )
    
    return logits, expert_weights, expert_indices


def cpu_fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    activation: str = "silu",
    alpha: float = 1.702,
    limit: float = 7.0,
    is_interleaved: bool = True,
) -> torch.Tensor:
    """
    CPU Fused MoE using PyTorch operations.
    
    This implementation processes all experts in parallel using batched operations
    instead of sequential for loops, which is more efficient on CPU.
    
    Args:
        hidden_states: [num_tokens, hidden_size]
        w1: [num_experts, hidden_size, 2*inter_size] - gate_up_proj weights
        w2: [num_experts, inter_size, hidden_size] - down_proj weights
        topk_weights: [num_tokens, topk] - routing weights
        topk_ids: [num_tokens, topk] - expert indices
        w1_bias: [num_experts, 2*inter_size] or None
        w2_bias: [num_experts, hidden_size] or None
        activation: "silu" or "swigluoai"
        alpha: swigluoai alpha parameter
        limit: swigluoai limit parameter
        is_interleaved: whether gate_up is interleaved [g0,u0,g1,u1,...] (True for GptOss)
    
    Returns:
        output: [num_tokens, hidden_size]
    """
    num_tokens, hidden_size = hidden_states.shape
    num_experts = w1.shape[0]
    inter_size = w2.shape[1]
    topk = topk_weights.shape[1]
    
    # Initialize output
    output = torch.zeros_like(hidden_states)
    
    # Build expert mask: which tokens go to which expert
    # expert_mask[expert_id] contains indices of (token_idx, topk_pos) pairs
    for expert_idx in range(num_experts):
        # Find tokens assigned to this expert
        # mask shape: [num_tokens, topk], True where topk_ids == expert_idx
        mask = (topk_ids == expert_idx)
        
        if not mask.any():
            continue
        
        # Get token indices and topk positions
        token_indices, topk_positions = torch.where(mask)
        
        if len(token_indices) == 0:
            continue
        
        # Gather input tokens for this expert
        # current_hidden: [num_selected_tokens, hidden_size]
        current_hidden = hidden_states[token_indices]
        
        # Get weights for this expert
        # w1[expert_idx]: [hidden_size, 2*inter_size]
        # w2[expert_idx]: [inter_size, hidden_size]
        expert_w1 = w1[expert_idx]  # [hidden_size, 2*inter_size]
        expert_w2 = w2[expert_idx]  # [inter_size, hidden_size]
        
        # First projection: hidden @ w1 -> [num_selected, 2*inter_size]
        gate_up = current_hidden @ expert_w1
        
        # Add bias if present
        if w1_bias is not None:
            gate_up = gate_up + w1_bias[expert_idx]
        
        # Split gate and up projections
        if is_interleaved:
            # GptOss uses interleaved layout: [g0, u0, g1, u1, ...]
            gate = gate_up[..., ::2]   # [num_selected, inter_size]
            up = gate_up[..., 1::2]    # [num_selected, inter_size]
        else:
            # Standard layout: [gate_all, up_all]
            gate = gate_up[..., :inter_size]
            up = gate_up[..., inter_size:]
        
        # Apply activation
        if activation == "swigluoai":
            activated = swigluoai_activation(gate, up, alpha, limit)
        else:  # silu
            activated = silu_and_mul_activation(gate, up)
        
        # Second projection: activated @ w2 -> [num_selected, hidden_size]
        expert_out = activated @ expert_w2
        
        # Add bias if present
        if w2_bias is not None:
            expert_out = expert_out + w2_bias[expert_idx]
        
        # Apply routing weights and accumulate
        # weights shape: [num_selected]
        weights = topk_weights[token_indices, topk_positions].unsqueeze(-1)
        weighted_out = expert_out * weights
        
        # Accumulate to output
        output.index_add_(0, token_indices, weighted_out.to(output.dtype))
    
    return output


class MegaBlocksMoeMLP(torch.nn.Module):
    """
    CPU MoE MLP module that can be used as a drop-in replacement for
    the transformers GptOssMLP when using @use_kernel_forward_from_hub.
    """
    can_torch_compile: bool = True
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through the MoE layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size] or [tokens, hidden_size]
            
        Returns:
            Tuple of (output, expert_weights) where:
                - output: Tensor of same shape as input
                - expert_weights: Expert weights for each token [tokens, top_k]
        """
        # Get MoE parameters from the wrapped modules
        moe_top_k = getattr(self.router, "top_k", 4)
        moe_num_experts = getattr(self.experts, "num_experts", 128)
        moe_normalize_expert_weights = getattr(
            self.experts, "normalize_expert_weights", None
        )
        
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
        
        # Route tokens to experts
        logits, expert_weights, expert_indices = route_tokens_cpu(
            x,
            self.router.weight,
            getattr(self.router, "bias", None),
            moe_top_k,
            moe_num_experts,
            moe_normalize_expert_weights,
        )
        
        # Reshape input for fused MoE
        x_flat = x.view(-1, x.shape[-1])
        
        # Call CPU fused MoE
        output = cpu_fused_moe(
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


# Export classes and functions
__all__ = [
    "MegaBlocksMoeMLP",
    "cpu_fused_moe",
    "route_tokens_cpu",
    "swigluoai_activation",
    "silu_and_mul_activation",
]
