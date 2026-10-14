from ._ops import ops
import torch
from ._ops import add_op_namespace_prefix


@torch.library.register_fake(add_op_namespace_prefix("fwd"))
def fwd_fake(
    q, k, v, sfq, sfk, sfv, delta_s, unpadded_k, out_,
    softmax_scale, is_causal, per_block_mean, is_bf16,
):
    batch_size, num_heads, seqlen_q, packed_head_size = q.shape
    head_size = packed_head_size * 2
    dtype = torch.bfloat16 if is_bf16 else torch.float16
    out = torch.empty(
        (batch_size, num_heads, seqlen_q, head_size), dtype=dtype, device=q.device
    )
    softmax_lse = torch.empty(
        (batch_size, num_heads, seqlen_q), dtype=torch.float32, device=q.device
    )
    return [out, softmax_lse]


@torch.library.register_fake(add_op_namespace_prefix("scaled_fp4_quant"))
def scaled_fp4_quant_fake(input, output, output_sf, tensor_layout):
    return None


@torch.library.register_fake(add_op_namespace_prefix("scaled_fp4_quant_permute"))
def scaled_fp4_quant_permute_fake(input, output, output_sf, tensor_layout):
    return None


@torch.library.register_fake(add_op_namespace_prefix("scaled_fp4_quant_trans"))
def scaled_fp4_quant_trans_fake(input, output, output_sf, tensor_layout):
    return None


fwd = ops.fwd
scaled_fp4_quant = ops.scaled_fp4_quant
scaled_fp4_quant_permute = ops.scaled_fp4_quant_permute
scaled_fp4_quant_trans = ops.scaled_fp4_quant_trans
