"""The layer `kernels` swaps into `transformers`' `WeatherNext2Attention`.

Only `forward` is defined: `kernels` binds it onto the model's own module, so `self.q_proj`,
`self.head_dim` and the rest are the ones `transformers` built.
"""

import torch
from torch import nn

from .banded_attention import banded_attention


def _gather_neighbouring_blocks(states: torch.Tensor) -> torch.Tensor:
    """`[batch, blocks, heads, block, dim]` -> the same with the three neighbours side by side."""
    padding = torch.zeros_like(states[:, :1])
    padded = torch.cat([padding, states, padding], dim=1)
    return torch.cat([padded[:, :-2], padded[:, 1:-1], padded[:, 2:]], dim=3)


def _banded_mask(attention_mask):
    """The geometry's banded mask, as `[blocks, block, 3 * block]`, or None.

    `masking_utils.create_bidirectional_mask` inserts a singleton head axis, so the mask
    reaching the layer is `[blocks, 1, block, 3 * block]` under sdpa and eager. That is the
    same mask, so the axis is dropped rather than treated as a different shape; without this
    the guard rejects every mask the model actually produces and the kernel silently never
    runs. A `BlockMask` from flex attention is not a tensor and is rejected here.
    """
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.dtype != torch.bool:
        return None
    if attention_mask.ndim == 4 and attention_mask.shape[1] == 1:
        attention_mask = attention_mask.squeeze(1)
    return attention_mask if attention_mask.ndim == 3 else None


def _is_banded(attention_mask, hidden_states) -> bool:
    """Is this the geometry's own banded mask, rather than one `masking_utils` expanded?"""
    attention_mask = _banded_mask(attention_mask)
    if attention_mask is None:
        return False
    num_blocks, block_size, key_length = attention_mask.shape
    return (
        key_length == 3 * block_size
        and hidden_states.ndim == 4
        and hidden_states.shape[1] == num_blocks
        and hidden_states.shape[2] == block_size
    )


def _needs_grad(*tensors: torch.Tensor) -> bool:
    """Is autograd going to want a backward through this?

    The kernel has no backward. Its output is written into a fresh tensor, so it carries no
    `grad_fn`: a `loss.backward()` would still succeed, and every parameter upstream of attention
    would silently receive nothing. Falling back is the only safe answer until a backward exists.
    """
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def _reference_attention(query, key, value, attention_mask, scaling):
    """The differentiable path: materialize the three neighbour blocks and use sdpa."""
    batch, blocks, heads, block_size, head_dim = query.shape
    keys = _gather_neighbouring_blocks(key).reshape(batch * blocks, heads, 3 * block_size, head_dim)
    values = _gather_neighbouring_blocks(value).reshape(batch * blocks, heads, 3 * block_size, head_dim)
    queries = query.reshape(batch * blocks, heads, block_size, head_dim)

    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 3:
        # The geometry's banded mask, which sdpa needs broadcast over the folded batch axis.
        attention_mask = attention_mask[None, :, None].expand(batch, blocks, 1, block_size, 3 * block_size)
        attention_mask = attention_mask.reshape(batch * blocks, 1, block_size, 3 * block_size)
    elif not isinstance(attention_mask, torch.Tensor):
        # A `BlockMask` only arrives with `attn_implementation="flex_attention"`, and sdpa cannot
        # consume one. Say so rather than drop the mask, which would attend across the whole band.
        raise ValueError(
            f"WeatherNext2Attention kernel got a {type(attention_mask).__name__} mask, which it "
            'cannot read. Load the model with attn_implementation="sdpa" (the default) when '
            "use_kernels=True."
        )

    out = nn.functional.scaled_dot_product_attention(
        queries, keys, values, attn_mask=attention_mask, scale=scaling
    )
    return out.reshape(batch, blocks, heads, block_size, head_dim)


class WeatherNext2Attention(nn.Module):
    def forward(self, hidden_states: torch.Tensor, attention_mask, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # [batch, blocks, block, hidden] -> [batch, blocks, heads, block, head_dim]
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(2, 3)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(2, 3)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(2, 3)

        banded = _banded_mask(attention_mask)
        if _is_banded(attention_mask, hidden_states) and not _needs_grad(query, key, value):
            # The kernel walks the three neighbouring blocks itself, so the keys and values are
            # never tripled and the mask is never expanded.
            attn_output = banded_attention(query.float(), key.float(), value.float(), banded, self.scaling)
        else:
            attn_output = _reference_attention(query, key, value, attention_mask, self.scaling)

        attn_output = (
            attn_output.to(hidden_states.dtype).transpose(2, 3).reshape(*input_shape, -1).contiguous()
        )
        return self.o_proj(attn_output), None
