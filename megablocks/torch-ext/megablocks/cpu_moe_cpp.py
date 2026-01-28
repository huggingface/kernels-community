# SPDX-License-Identifier: Apache-2.0
# MegaBlocks C++ Optimized CPU MoE

"""
C++ accelerated MoE with brgemm optimization for Intel AMX.
Direct replacement for cpu_fused_moe.MegaBlocksMoeMLP with better performance.
"""

import torch
from typing import Optional
from .cpu_fused_moe import route_tokens_cpu
from ._ops import ops


def _to_local_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Convert DTensor to local torch.Tensor if needed for custom ops compatibility."""
    if tensor is None:
        return None
    # Check if it's a DTensor by looking for the to_local() method
    if hasattr(tensor, "to_local"):
        return tensor.to_local()
    return tensor


def fused_moe_cpp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    use_int8_w8a8: bool = False,
    use_fp8_w8a16: bool = False,
    use_mxfp4: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_size: Optional[list] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    alpha: Optional[float] = None,
    limit: Optional[float] = None,
    is_vnni: bool = False,
) -> torch.Tensor:
    """
    C++ Fused MoE with brgemm optimization (sglang compatible interface).
    
    Uses at::native::cpublas::brgemm for efficient batch GEMM on Intel CPUs.
    Supports both silu_and_mul (standard SwiGLU) and swigluoai (GptOss) activations.
    
    Args:
        hidden_states: Input tensor [M, K]
        w1: Gate and up projections [E, 2N, K]
        w2: Down projection [E, K, N]
        topk_weights: Expert weights [M, topk]
        topk_ids: Expert indices [M, topk]
        inplace: Whether to use hidden_states as output
        use_int8_w8a8: Use int8 quantization
        use_fp8_w8a16: Use fp8 quantization
        use_mxfp4: Use mxfp4 quantization
        w1_scale, w2_scale: Quantization scales
        block_size: Block size for fp8
        a1_scale, a2_scale: Activation scales
        w1_bias, w2_bias: Optional biases
        alpha: swigluoai alpha parameter (set to enable swiglu)
        limit: swigluoai limit parameter (set to enable swiglu)
        is_vnni: Whether w1/w2 are already in VNNI packed format
    """
    # MXFP4/FP8 kernels only support bf16, convert if needed
    orig_dtype = hidden_states.dtype
    need_convert = ((use_mxfp4 or use_fp8_w8a16) and orig_dtype != torch.bfloat16) or orig_dtype == torch.float32
    if need_convert:
        hidden_states = hidden_states.to(torch.bfloat16)

    # bias must match hidden_states dtype
    if w1_bias is not None:
        w1_bias = w1_bias.to(hidden_states.dtype)
    if w2_bias is not None:
        w2_bias = w2_bias.to(hidden_states.dtype)

    # Convert DTensor to local tensor for custom ops compatibility (TP mode)
    hidden_states = _to_local_tensor(hidden_states)
    w1 = _to_local_tensor(w1)
    w2 = _to_local_tensor(w2)
    topk_weights = _to_local_tensor(topk_weights)
    topk_ids = _to_local_tensor(topk_ids)
    w1_scale = _to_local_tensor(w1_scale)
    w2_scale = _to_local_tensor(w2_scale)
    a1_scale = _to_local_tensor(a1_scale)
    a2_scale = _to_local_tensor(a2_scale)
    w1_bias = _to_local_tensor(w1_bias)
    w2_bias = _to_local_tensor(w2_bias)
    
    output = ops.fused_experts(
        hidden_states, w1, w2, topk_weights, topk_ids,
        inplace, use_int8_w8a8, use_fp8_w8a16, use_mxfp4,
        w1_scale, w2_scale, block_size, a1_scale, a2_scale,
        w1_bias, w2_bias, alpha, limit, is_vnni
    )
    
    # Convert back to original dtype if needed
    if need_convert:
        output = output.to(orig_dtype)
    return output


class MegaBlocksMoeMLP(torch.nn.Module):
    """
    C++ optimized MoE MLP using brgemm.
    Drop-in replacement for cpu_fused_moe.MegaBlocksMoeMLP with better performance.
    
    Usage in transformers:
        # Will be used via @use_kernel_forward_from_hub decorator
    """
    can_torch_compile: bool = True

    def convert_weight(self, dtype, use_mxfp4: bool = False):
        data_1 = self.experts.gate_up_proj.data.transpose(-1, -2).contiguous()
        data_2 = self.experts.down_proj.data.transpose(-1, -2).contiguous()
        if use_mxfp4:
            self.experts.gate_up_proj.storage.data = ops.convert_weight_packed(data_1)
            self.experts.down_proj.storage.data = ops.convert_weight_packed(data_2)
        else:
            # convert_weight_packed onlu supports bfloat16, float16, int8, fp8_e4m3 or uint8(mxfp4 or int4).
            data_1 = data_1.to(torch.bfloat16) if data_1.dtype == torch.float32 else data_1
            data_2 = data_2.to(torch.bfloat16) if data_2.dtype == torch.float32 else data_2
            self.experts.gate_up_proj.data = ops.convert_weight_packed(data_1)
            self.experts.down_proj.data = ops.convert_weight_packed(data_2)

        # C++ kernel does not support float32.
        dtype = torch.bfloat16 if dtype == torch.float32 else dtype
        if getattr(self.experts, "gate_up_proj_bias", None) is not None:
            self.experts.gate_up_proj_bias.data = self.experts.gate_up_proj_bias.data.to(dtype)
        if getattr(self.experts, "down_proj_bias", None) is not None:
            self.experts.down_proj_bias.data = self.experts.down_proj_bias.data.to(dtype)

    def convert_scales(self):
        data_1 = ops.convert_scale_packed(self.experts.gate_up_proj_precision_config.weight_scale.data.transpose(-1, -2).contiguous())
        data_2 = ops.convert_scale_packed(self.experts.down_proj_precision_config.weight_scale.data.transpose(-1, -2).contiguous())
        self.experts.gate_up_proj_precision_config.weight_scale.storage.data = data_1
        self.experts.down_proj_precision_config.weight_scale.storage.data = data_2
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through the MoE layer using C++ kernel.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            Tuple of (output, expert_weights)
        """
        # Optimization for GPT-OSS model
        if getattr(self, "use_mxfp4", None) is None:
            self.use_mxfp4 = False

        w1_scale = None
        w2_scale = None

        if (
            not getattr(self, "packed_scales", False)
            and hasattr(self.experts, "gate_up_proj")
            and getattr(self.experts, "gate_up_proj_precision_config", None) is not None
        ):
            self.convert_scales()
            self.packed_scales = True
            self.use_mxfp4 = True

        if not getattr(self, "packed_weight", False) and hasattr(
            self.experts, "gate_up_proj"
        ):
            self.convert_weight(x.dtype, self.use_mxfp4)
            self.packed_weight = True

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
        elif hasattr(self.experts, "w1"):
            w1 = self.experts.w1
            w3 = getattr(self.experts, "w3", None)
            if w3 is not None:
                w1 = torch.cat([w1, w3], dim=-1)
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
        w1_bias = w1_bias if w1_bias is None else w1_bias.data
        w2_bias = w2_bias if w2_bias is None else w2_bias.data

        if self.use_mxfp4:
            w1_scale = self.experts.gate_up_proj_precision_config.weight_scale.data
            w2_scale = self.experts.down_proj_precision_config.weight_scale.data
        
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
        
        # Determine alpha/limit for swiglu activation
        use_alpha = alpha if activation == "swigluoai" else None
        use_limit = limit if activation == "swigluoai" else None
        
        # Call C++ optimized kernel
        output = fused_moe_cpp(
            hidden_states=x_flat,
            w1=w1.data,
            w2=w2.data,
            topk_weights=expert_weights,
            topk_ids=expert_indices.to(torch.int32),
            inplace=False,
            use_int8_w8a8=False,
            use_fp8_w8a16=False,
            use_mxfp4=self.use_mxfp4,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            block_size=None,
            a1_scale=None,
            a2_scale=None,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            alpha=use_alpha,
            limit=use_limit,
            is_vnni=getattr(self, "packed_weight", False),
        )
        
        # Restore original shape
        output = output.view(in_shape)
        
        return output, expert_weights


__all__ = ["fused_moe_cpp", "MegaBlocksMoeMLP"]
