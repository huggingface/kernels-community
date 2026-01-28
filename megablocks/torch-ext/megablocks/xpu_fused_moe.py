# SPDX-License-Identifier: Apache-2.0
# MegaBlocks XPU Fused MoE Implementation
import os
import torch

from ._ops import ops


def resolve_dtensor(weight: torch.Tensor):
    """Convert DTensor to local tensor for use with custom ops."""
    from torch.distributed._tensor import DTensor
    if isinstance(weight, DTensor):
        return weight.to_local()
    return weight


# Install meta kernels for torch.compile compatibility
def _install_xpu_meta_kernels():
    """Install meta kernels for XPU MoE operations to support torch.compile"""
    
    # Patch cutlass_grouped_gemm_interface
    if hasattr(ops, "cutlass_grouped_gemm_interface"):
        original_gemm = ops.cutlass_grouped_gemm_interface
        
        def gemm_with_meta(ptr_A, ptr_B, ptr_scales, ptr_bias, ptr_D, 
                          expert_first_token_offset, N, K, num_experts,
                          is_B_int4, is_B_mxfp4):
            if torch.compiler.is_compiling():
                # Meta implementation - ptr_D is the output, return it
                return ptr_D
            return original_gemm(ptr_A, ptr_B, ptr_scales, ptr_bias, ptr_D,
                               expert_first_token_offset, N, K, num_experts,
                               is_B_int4, is_B_mxfp4)
        
        ops.cutlass_grouped_gemm_interface = gemm_with_meta
    
    # Patch fused_moe_prologue
    if hasattr(ops, "fused_moe_prologue"):
        original_prologue = ops.fused_moe_prologue
        
        def prologue_with_meta(input, token_selected_experts, token_final_scales,
                              workspace, hidden_size, inter_size, num_experts_on_rank):
            if torch.compiler.is_compiling():
                # Meta implementation - this op modifies workspace in-place
                return None
            return original_prologue(input, token_selected_experts, token_final_scales,
                                    workspace, hidden_size, inter_size, num_experts_on_rank)
        
        ops.fused_moe_prologue = prologue_with_meta
    
    # Patch moe_gather
    if hasattr(ops, "moe_gather"):
        original_gather = ops.moe_gather
        
        def gather_with_meta(output, moe_output, topk_weights, 
                            unpermuted_row_to_permuted_row, num_experts):
            if torch.compiler.is_compiling():
                # Meta implementation - output is modified in-place
                return None
            return original_gather(output, moe_output, topk_weights,
                                  unpermuted_row_to_permuted_row, num_experts)
        
        ops.moe_gather = gather_with_meta
    
    # Patch activation ops
    for act_name in ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul", 
                     "gelu_fast", "gelu_new", "gelu_quick", "mul_and_silu",
                     "swigluoai_and_mul"]:
        if hasattr(ops, act_name):
            original_act = getattr(ops, act_name)
            
            def make_act_wrapper(orig_fn):
                def act_with_meta(*args, **kwargs):
                    if torch.compiler.is_compiling():
                        # Meta implementation - in-place ops, return None
                        return None
                    return orig_fn(*args, **kwargs)
                return act_with_meta
            
            setattr(ops, act_name, make_act_wrapper(original_act))


# Install meta kernels on module load
_install_xpu_meta_kernels()


# default
def cutlass_grouped_gemm(input_A, input_B, bias, output, expert_token_count, n,
                         k, num_experts):
    # expert_token_count_ = torch.tensor(expert_token_count,
    #                                    dtype=torch.int64,
    #                                    device=input_A.device)
    # if bias is not None:
    #     bias = bias.repeat_interleave(expert_token_count_, dim=0).float()

    def exclusive_prefix_sum(arr):
        prefix = [0]
        for i, x in enumerate(arr):
            prefix.append(prefix[-1] + x)
        return prefix

    expert_offset = torch.tensor(exclusive_prefix_sum(expert_token_count),
                                 dtype=torch.int64,
                                 device="xpu")
    ops.cutlass_grouped_gemm_interface(
        ptr_A=input_A,
        ptr_B=input_B,
        ptr_scales=None,
        ptr_bias=bias,
        ptr_D=output,
        expert_first_token_offset=expert_offset,
        N=n,
        K=k,
        num_experts=num_experts,
        is_B_int4=False,
        is_B_mxfp4=False)


def cutlass_grouped_gemm_xe2(input_A, input_B, scales, bias, output,
                             num_rows_per_expert, n, k, num_experts, is_B_int4,
                             is_B_mxfp4):
    expert_first_token_offset = torch.cat([
        torch.tensor([0],
                     dtype=num_rows_per_expert.dtype,
                     device=num_rows_per_expert.device),
        torch.cumsum(num_rows_per_expert, dim=0)
    ]).to(torch.int64)
    ops.cutlass_grouped_gemm_interface(
        ptr_A=input_A,
        ptr_B=input_B,
        ptr_scales=scales,
        ptr_bias=bias,
        ptr_D=output,
        expert_first_token_offset=expert_first_token_offset,
        N=n,
        K=k,
        num_experts=num_experts,
        is_B_int4=is_B_int4,
        is_B_mxfp4=is_B_mxfp4)


def ceilDiv(a, b):
    return (a + b - 1) // b


def compute_num_tokens_per_block(num_tokens, num_experts_per_node):
    for num_tokens_per_block in [32, 64, 128, 256, 512, 1024]:
        num_blocks_per_seq = ceilDiv(num_tokens, num_tokens_per_block)
        if num_blocks_per_seq * num_experts_per_node <= num_tokens_per_block:
            return num_tokens_per_block
    return 1024


def implement_zp(qweight):
    # change u4 to s4 to avoid zero point in gemm kernel
    # only support default zero point now
    assert qweight.dtype == torch.uint8, "Input tensor must be uint8"

    high_u4 = (qweight >> 4) & 0x0F
    low_u4 = qweight & 0x0F

    high_s8 = high_u4.to(torch.int8)
    low_s8 = low_u4.to(torch.int8)

    high_s8 = high_s8 - 8
    low_s8 = low_s8 - 8

    def pack_compact(a, b):

        def process_number(x):
            sign = (x < 0).to(torch.uint8)
            abs_low3 = (x.view(torch.uint8) & 0x7).to(torch.uint8)
            return (sign << 3) | abs_low3

        packed_a = process_number(a)
        packed_b = process_number(b)

        return (packed_a << 4) | packed_b

    result = pack_compact(high_s8, low_s8)

    return result


def xpu_fused_moe(hidden_states,
                  w13,
                  w13_scales,
                  w13_bias,
                  w2,
                  w2_scales,
                  w2_bias,
                  topk_weights,
                  topk_ids,
                  n_experts_per_token,
                  activation,
                  num_experts,
                  is_fp8=False,
                  is_int4=False,
                  is_mxfp4=False):
    '''
    hidden_states: [num_rows, hidden_size]
    w13: [num_experts, 2*inter_size, hidden_size]
    w13_scales: 
        None for bf16/fp16 
        or [num_experts] for fp8 
        or [num_experts, 2*inter_size, hidden_size // group_size] for 4bits
    w13_bias: [num_experts, 2*inter_size] or None
    w2: [num_experts, hidden_size, inter_size]
    w2_scales:
        None for bf16/fp16 
        or [num_experts] for fp8 
        or [num_experts, hidden_size, inter_size // group_size] for 4bits
    w2_bias: [num_experts, hidden_size] or None
    topk_weights: [num_rows, topk]
    topk_ids: [num_rows, topk]
    n_experts_per_token: int
    activation: str
    num_experts: int
    is_int4: bool
    is_mxfp4: bool
    '''

    # Resolve DTensors to local tensors before passing to custom ops
    hidden_states = resolve_dtensor(hidden_states)
    w13 = resolve_dtensor(w13)
    w2 = resolve_dtensor(w2)
    if w13_scales is not None:
        w13_scales = resolve_dtensor(w13_scales)
    if w13_bias is not None:
        w13_bias = resolve_dtensor(w13_bias)
    if w2_scales is not None:
        w2_scales = resolve_dtensor(w2_scales)
    if w2_bias is not None:
        w2_bias = resolve_dtensor(w2_bias)
    topk_weights = resolve_dtensor(topk_weights)
    topk_ids = resolve_dtensor(topk_ids)

    output = torch.empty_like(hidden_states)
    num_rows, hidden_size = list(hidden_states.shape)

    dim_last = w13.shape[-1]
    dim_second_last = w13.shape[-2]

    # w13 is combined gate+up weights, so one dimension is 2*inter_size
    # Determine which dimension is hidden_size and which is 2*inter_size
    if dim_second_last == hidden_size:
        # w13 is [E, hidden_size, 2*inter_size] - standard layout
        inter_size = dim_last // 2
        needs_transpose = False
    else:
        # w13 is [E, 2*inter_size, hidden_size] - needs transpose
        inter_size = dim_second_last // 2
        needs_transpose = True

    assert w13.is_contiguous() and w2.is_contiguous()

    # 4bits support [E, N, K]
    # other types [E, K, N]
    if not is_int4 and not is_mxfp4:
        if not hasattr(w13, 'xpu_fused_moe'):
            if needs_transpose:
                w13.data = w13.transpose(-1, -2).contiguous()
                w2.data = w2.transpose(-1, -2).contiguous()
            w13.xpu_fused_moe = True
            w13.inter_size = inter_size
        else:
            inter_size = w13.inter_size

    if is_int4 and not hasattr(w13, 'xpu_fused_moe'):
        w13_tmp = torch.empty_like(w13)
        w2_tmp = torch.empty_like(w2)
        for i in range(num_experts):
            w13_tmp[i] = implement_zp(w13[i])
            w2_tmp[i] = implement_zp(w2[i])
        w13_tmp = w13_tmp.contiguous()
        w2_tmp = w2_tmp.contiguous()
        w13.data = w13_tmp
        w2.data = w2_tmp
        w13.xpu_fused_moe = True

    # TODO: will all integrated in Cpp func. Temporary expose before gemm fusion
    num_experts_per_node = num_experts
    experts_per_token = n_experts_per_token
    num_moe_inputs = n_experts_per_token * num_rows
    permuted_elems = num_moe_inputs * hidden_size
    # interbuf_elems = num_moe_inputs * inter_size
    permuted_row_to_unpermuted_row_size = num_moe_inputs * 4
    permuted_token_selected_experts_size = num_moe_inputs * 4
    src_to_dest_map_size = experts_per_token * num_rows * 4
    expert_first_token_offset_size = (num_experts_per_node + 1) * 8
    num_tokens_per_block = compute_num_tokens_per_block(
        num_rows, num_experts_per_node)
    num_blocks_per_seq = ceilDiv(num_rows, num_tokens_per_block)
    blocked_expert_counts_size = num_experts_per_node * num_blocks_per_seq * 4
    blocked_expert_counts_cumsum_size = blocked_expert_counts_size
    blocked_row_to_unpermuted_row_size = num_experts_per_node * num_rows * 4
    permuted_data_size = permuted_elems * hidden_states.element_size()
    permuted_token_final_scales_size = num_moe_inputs * 4

    ws_map = {}
    map_offset = 0

    def config_ws(name, size):
        nonlocal map_offset
        if size % 256 != 0:
            size += 256 - size % 256
        ws_map[name] = (size, map_offset)
        map_offset += size

    config_ws("permuted_row_to_unpermuted_row",
              permuted_row_to_unpermuted_row_size)
    config_ws("permuted_token_selected_experts",
              permuted_token_selected_experts_size)
    config_ws("unpermuted_row_to_permuted_row", src_to_dest_map_size)
    config_ws("blocked_expert_counts", blocked_expert_counts_size)
    config_ws("blocked_expert_counts_cumsum",
              blocked_expert_counts_cumsum_size)
    config_ws("blocked_row_to_unpermuted_row",
              blocked_row_to_unpermuted_row_size)
    config_ws("expert_first_token_offset", expert_first_token_offset_size)
    config_ws("permuted_token_final_scales", permuted_token_final_scales_size)
    config_ws("overlapped_gemm1_gemm2_inputs", permuted_data_size)

    workspace = torch.zeros(map_offset,
                            dtype=torch.uint8,
                            device=hidden_states.device)
    if topk_ids.dtype == torch.int32:
        topk_ids = topk_ids.to(torch.int64)
    ops.fused_moe_prologue(
        input=hidden_states,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        workspace=workspace,
        hidden_size=hidden_size,
        inter_size=inter_size,
        num_experts_on_rank=num_experts_per_node)

    expert_first_token_offset = workspace[
        ws_map["expert_first_token_offset"][1]:
        ws_map["expert_first_token_offset"][1] +
        expert_first_token_offset_size].view(torch.int64)
    unpermuted_row_to_permuted_row = workspace[
        ws_map["unpermuted_row_to_permuted_row"][1]:
        ws_map["unpermuted_row_to_permuted_row"][1] +
        src_to_dest_map_size].view(torch.int32)
    gemm1_input = workspace[ws_map["overlapped_gemm1_gemm2_inputs"][1]:
                            ws_map["overlapped_gemm1_gemm2_inputs"][1] +
                            permuted_data_size].view(hidden_states.dtype).view(
                                num_moe_inputs, hidden_size)
    # permuted_token_final_scales = workspace[
    #     ws_map["permuted_token_final_scales"][1]:
    #     ws_map["permuted_token_final_scales"][1] +
    #     permuted_token_final_scales_size].view(torch.float)
    gemm1_output = torch.empty((num_moe_inputs, 2 * inter_size),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device)

    ########### gemm1 ##################
    input_B = w13

    if not is_fp8 and not is_int4 and not is_mxfp4:
        ops.cutlass_grouped_gemm_interface(
            ptr_A=gemm1_input,
            ptr_B=input_B,
            ptr_scales=None,
            ptr_bias=w13_bias,
            ptr_D=gemm1_output,
            expert_first_token_offset=expert_first_token_offset,
            N=2 * inter_size,
            K=hidden_size,
            num_experts=num_experts_per_node,
            is_B_int4=is_int4,
            is_B_mxfp4=is_mxfp4)
    else:
        ops.cutlass_grouped_gemm_interface(
            ptr_A=gemm1_input,
            ptr_B=input_B,
            ptr_scales=w13_scales,
            ptr_bias=w13_bias,
            ptr_D=gemm1_output,
            expert_first_token_offset=expert_first_token_offset,
            N=2 * inter_size,
            K=hidden_size,
            num_experts=num_experts_per_node,
            is_B_int4=is_int4,
            is_B_mxfp4=is_mxfp4)

    # act
    act_output = torch.empty((num_moe_inputs, inter_size),
                             dtype=gemm1_output.dtype,
                             device=gemm1_output.device)
    if activation == "silu":
        ops.silu_and_mul(act_output, gemm1_output)
    elif activation == "gelu":
        ops.gelu_and_mul(act_output, gemm1_output)
    elif activation == "swigluoai":
        ops.swigluoai_and_mul(act_output, gemm1_output, 1.702, 7.0)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}.")

    ########### gemm2 ##################
    input_A = act_output.contiguous()
    input_B = w2
    gemm2_output = torch.empty((num_moe_inputs, hidden_size),
                               dtype=hidden_states.dtype,
                               device=hidden_states.device)
    if not is_fp8 and not is_int4 and not is_mxfp4:
        ops.cutlass_grouped_gemm_interface(
            ptr_A=input_A,
            ptr_B=input_B,
            ptr_scales=None,
            ptr_bias=w2_bias,
            ptr_D=gemm2_output,
            expert_first_token_offset=expert_first_token_offset,
            N=hidden_size,
            K=inter_size,
            num_experts=num_experts_per_node,
            is_B_int4=is_int4,
            is_B_mxfp4=is_mxfp4)
    else:
        ops.cutlass_grouped_gemm_interface(
            ptr_A=input_A,
            ptr_B=input_B,
            ptr_scales=w2_scales,
            ptr_bias=w2_bias,
            ptr_D=gemm2_output,
            expert_first_token_offset=expert_first_token_offset,
            N=hidden_size,
            K=inter_size,
            num_experts=num_experts_per_node,
            is_B_int4=is_int4,
            is_B_mxfp4=is_mxfp4)

    ops.moe_gather(output, gemm2_output, topk_weights,
                                unpermuted_row_to_permuted_row,
                                num_experts_per_node)
    return output


def apply_jitter(x: torch.Tensor, moe_jitter_eps: float) -> torch.Tensor:
    """Apply jitter to the input tensor for regularization."""
    low = 1.0 - moe_jitter_eps
    high = 1.0 + moe_jitter_eps
    noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
    return x * (low + noise * (high - low))


def compute_top_k(scores: torch.Tensor, moe_top_k: int):
    """Compute the top-k scores from the logits."""
    if moe_top_k == 1:
        return scores.max(dim=-1, keepdim=True)
    return torch.topk(scores, moe_top_k, dim=-1)


def route_tokens_xpu(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    moe_top_k: int,
    moe_num_experts: int,
    moe_jitter_eps: float = None,
    moe_normalize_expert_weights: int = None,
    training: bool = False,
) -> tuple:
    """Route tokens to experts and compute expert weights and indices (XPU version)."""
    if training and moe_jitter_eps is not None:
        x = apply_jitter(x, moe_jitter_eps)

    x_flat = x.view(-1, x.shape[-1])
    logits = torch.nn.functional.linear(x_flat, router_weight, router_bias)
    expert_weights, expert_indices = compute_top_k(logits, moe_top_k)
    expert_weights = expert_weights.softmax(dim=-1)
    if moe_normalize_expert_weights is not None:
        expert_weights = expert_weights / torch.norm(
            expert_weights,
            p=moe_normalize_expert_weights,
            dim=-1,
            keepdim=True,
        )

    return logits, expert_weights, expert_indices


class MegaBlocksMoeMLP(torch.nn.Module):
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
        moe_jitter_eps = getattr(self.experts, "jitter_eps", None)
        moe_normalize_expert_weights = getattr(
            self.experts, "normalize_expert_weights", None
        )
        
        # Detect activation type - check for GptOss-style swigluoai activation
        # GptOssExperts has alpha and limit attributes for swigluoai
        if hasattr(self.experts, "alpha") and hasattr(self.experts, "limit"):
            activation = "swigluoai"
        else:
            activation = getattr(self.experts, "activation", "silu")

        # Get weight tensors - support different naming conventions
        if hasattr(self.experts, "gate_up_proj"):
            w13 = self.experts.gate_up_proj
            # NOTE: swigluoai_and_mul kernel expects interleaved layout [g0,u0,g1,u1,...]
            # which matches GptOss's gate_up_proj layout, so no conversion needed.
                            
        elif hasattr(self.experts, "w1"):
            # Combine w1 and w3 if stored separately
            w1 = self.experts.w1
            w3 = getattr(self.experts, "w3", None)
            if w3 is not None:
                w13 = torch.cat([w1, w3], dim=-2)
            else:
                w13 = w1
        else:
            raise AttributeError("experts module must have 'gate_up_proj' or 'w1' attribute")
        
        if hasattr(self.experts, "down_proj"):
            w2 = self.experts.down_proj
        elif hasattr(self.experts, "w2"):
            w2 = self.experts.w2
        else:
            raise AttributeError("experts module must have 'down_proj' or 'w2' attribute")
        
        # Get optional bias tensors
        w13_bias = getattr(self.experts, "gate_up_proj_bias", None)
        w2_bias = getattr(self.experts, "down_proj_bias", None)
        
        # Get quantization info
        is_fp8 = getattr(self.experts, "is_fp8", False)
        is_int4 = getattr(self.experts, "is_int4", False)
        is_mxfp4 = getattr(self.experts, "is_mxfp4", False)
        
        w13_scales = getattr(self.experts, "gate_up_proj_scales", None)
        w2_scales = getattr(self.experts, "down_proj_scales", None)
        
        # Store original shape
        in_shape = x.size()
        
        # Route tokens to experts
        logits, expert_weights, expert_indices = route_tokens_xpu(
            x,
            self.router.weight,
            getattr(self.router, "bias", None),
            moe_top_k,
            moe_num_experts,
            moe_jitter_eps,
            moe_normalize_expert_weights,
            self.training,
        )
        
        # Reshape input for fused MoE
        x_flat = x.view(-1, x.shape[-1])
        
        # Call XPU fused MoE kernel
        output = xpu_fused_moe(
            hidden_states=x_flat,
            w13=w13,
            w13_scales=w13_scales,
            w13_bias=w13_bias,
            w2=w2,
            w2_scales=w2_scales,
            w2_bias=w2_bias,
            topk_weights=expert_weights.float(),
            topk_ids=expert_indices,
            n_experts_per_token=moe_top_k,
            activation=activation,
            num_experts=moe_num_experts,
            is_fp8=is_fp8,
            is_int4=is_int4,
            is_mxfp4=is_mxfp4,
        )
        
        # Restore original shape
        output = output.view(in_shape)
        
        return output, expert_weights


# Export classes and functions
__all__ = [
    "MegaBlocksMoeMLP",
    "xpu_fused_moe",
    "cutlass_grouped_gemm",
    "cutlass_grouped_gemm_xe2",
]