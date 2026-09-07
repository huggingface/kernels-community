"""Layer-level entry points, for `kernels`' layer mapping.

`transformers` marks the layers a kernel may replace with `@use_kernel_forward_from_hub("<name>")`, and a
kernel repository supplies a class of that name whose `forward` is grafted onto the real module. The module
keeps its own parameters and constructor -- the class here is stateless, which `kernels` enforces: no
`__init__`, no members besides `forward`, and a signature matching the layer being replaced.

`Qwen3_5GatedDeltaNet` is the one that matters for decode. The layer spends its time in the gated delta
rule, which eager torch has to spell out as a loop of small ops; upstream ggml does it in one kernel, and
the whole point of replacing the forward (rather than just the rule) is that this is the extension point
`transformers` already offers -- `Atlas-Inference/gdn` uses the same one for CUDA.

The body follows `Qwen3_5GatedDeltaNet.forward` and swaps exactly two things: the delta rule becomes one
dispatch, and the L2 normalisation of q/k that the rule folds in becomes another. Everything else -- the
projections, the causal conv, the cache updates, the gated norm -- is the model's own code path, imported
rather than reimplemented, so this stays as close to upstream behaviour as a forward replacement can.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._ops import ops


def _gated_delta_rule(query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False):
    """`fused_recurrent_gated_delta_rule`'s contract, backed by ggml's GGML_OP_GATED_DELTA_NET.

    Upstream's kernel keeps the recurrent state indexed `[value][key]`, which is the layout it also writes,
    so a caller that stores what this returns never transposes. It scales the output by `1/sqrt(head_dim)`
    itself, and it takes `q`/`k` already expanded to one head per value head.
    """
    if use_qk_l2norm_in_kernel:
        query, key = ops.l2_norm(query, 1e-6), ops.l2_norm(key, 1e-6)
    if initial_state is None:
        n_seqs, _, n_heads, head_dim = value.shape
        initial_state = value.new_zeros((n_seqs, n_heads, head_dim, head_dim))
    n_seqs, n_tokens, n_heads, head_dim = value.shape
    # One allocation holding the outputs followed by the final state, which is how the kernel writes them.
    dst = ops.gated_delta_net(query, key, value, g, beta, initial_state)
    n_out = n_seqs * n_tokens * n_heads * head_dim
    out = dst[:n_out].view(n_seqs, n_tokens, n_heads, head_dim)
    state = dst[n_out:].view(n_seqs, n_heads, head_dim, head_dim)
    return out, (state if output_final_state else None)


class Qwen3_5GatedDeltaNet(nn.Module):
    can_torch_compile = False

    # The signature the host declares. `@force_accelerate_hooks` wraps that forward, but it carries
    # `functools.wraps`, so what `kernels` validates against resolves to the real parameters. Replacing
    # the forward drops the accelerate wrapper, which matters only for a model split across devices with
    # offload hooks, not for a single-device run.
    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            apply_mask_to_padding_states,
            causal_conv1d_fn,
            causal_conv1d_update,
        )

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        # The kernels below compute in f32 whatever they are handed, so the layer would otherwise
        # return f32 for a bf16 model and every later layer would run mixed -- which MPS tolerates
        # and CPU does not. A replacement forward has to hand back the dtype it was given.
        dtype = hidden_states.dtype
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            # The whole update in one dispatch rather than five (concatenate, copy the tail back,
            # a grouped conv1d, a slice, the activation): 5.0us against 33.6us for 8192 channels,
            # and a decode step runs it once per linear-attention layer. Batched or non-silu
            # variants keep the model's own path, which is the only thing that has been measured.
            if (
                batch_size == 1
                and self.activation == "silu"
                and conv_state.is_contiguous()
                # The kernel reads and writes the cache in place, so it cannot cast: a model
                # running in bf16 keeps the model's own path.
                and conv_state.dtype == torch.float32
            ):
                mixed_qkv = ops.causal_conv_update(
                    conv_state.view(conv_state.shape[-2], conv_state.shape[-1]),
                    mixed_qkv,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    True,
                ).view(1, -1, 1)
            else:
                mixed_qkv = causal_conv1d_update(
                    mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation
                )
        else:
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
                seq_idx=kwargs.get("seq_idx"),
            )
            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        # `sigmoid(b)` and `-exp(A_log) * softplus(a + dt_bias)` are six launches over one value per head
        # -- more time than the recurrence they feed -- so one kernel produces both. Only for a single
        # token: with more the shapes are per-token and the kernel's one-value-per-head contract does not
        # hold, and prefill is not launch-bound anyway.
        if b.shape[1] == 1:
            beta, g = ops.delta_gates(b.reshape(-1), a.reshape(-1), self.A_log, self.dt_bias)
            beta = beta.view_as(b)
            g = g.view_as(a)
        else:
            beta = b.sigmoid()
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            # The kernel maps a value head to a key head with `i21 % ne01`, so passing them pre-expanded
            # makes that the identity and the caller's own head order is the one that applies.
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None
        if kwargs.get("cu_seq_lens_q") is not None:
            raise NotImplementedError("packed sequences are not supported by this kernel")
        core_attn_out, last_recurrent_state = _gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        # `rms_norm(x) * weight * silu(gate)` as one dispatch rather than three over 32 heads of 128.
        core_attn_out = ops.rms_norm_gate(
            core_attn_out, self.norm.weight, z, self.norm.variance_epsilon
        ).to(core_attn_out.dtype)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1).to(dtype)
        return self.out_proj(core_attn_out)
