from ._ops import ops
import torch
from ._ops import add_op_namespace_prefix


def _lse_fake_impl(query, tensor_layout, return_lse):
    batch_size = query.size(0)
    if tensor_layout == 0:
        num_qo_heads = query.size(2)
        qo_len = query.size(1)
    else:
        num_qo_heads = query.size(1)
        qo_len = query.size(2)
    if return_lse:
        return torch.empty((batch_size, num_qo_heads, qo_len), dtype=torch.float32, device=query.device)
    return torch.empty((0))


@torch.library.register_fake(add_op_namespace_prefix("qk_int8_sv_f8_accum_f32_attn_inst_buf"))
def qk_int8_sv_f8_accum_f32_attn_inst_buf_fake(
    query, key, value, output, query_scale, key_scale,
    tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse,
):
    return _lse_fake_impl(query, tensor_layout, return_lse)


@torch.library.register_fake(add_op_namespace_prefix("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90"))
def qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90_fake(
    query, key, value, output, query_scale, key_scale, value_scale,
    tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse,
):
    return _lse_fake_impl(query, tensor_layout, return_lse)


qk_int8_sv_f8_accum_f32_attn_inst_buf = ops.qk_int8_sv_f8_accum_f32_attn_inst_buf
qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90 = ops.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90
