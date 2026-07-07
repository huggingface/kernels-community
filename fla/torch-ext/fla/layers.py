import torch.nn as nn

from .modules.fused_norm_gate import rms_norm_gated
from .ops.kda import chunk_kda, fused_recurrent_kda


class FusedRMSNormGated(nn.Module):
    def forward(self, hidden_states, gate=None):
        return rms_norm_gated(
            hidden_states,
            gate,
            self.weight,
            None,  # bias
            self.activation,
            residual=None,
            eps=self.variance_epsilon,
            prenorm=False,
            residual_in_fp32=False,
        )


def chunk_kimi_delta_attention(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    # Keep internal consistency between transformers and fla to allow both
    cu_seqlens = kwargs.pop("cu_seq_lens_q", kwargs.pop("cu_seqlens", None))

    return chunk_kda(
        query,
        key,
        value,
        g=g,
        beta=beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        **kwargs,
    )


def recurrent_kimi_delta_attention(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False, **kwargs,
):
    # Keep internal consistency between transformers and fla to allow both
    cu_seqlens = kwargs.pop("cu_seq_lens_q", kwargs.pop("cu_seqlens", None))

    return fused_recurrent_kda(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        **kwargs,
    )


__all__ = ["FusedRMSNormGated", "chunk_kimi_delta_attention", "recurrent_kimi_delta_attention"]
