"""The layer `kernels` swaps into `transformers`' `WeatherNext2Attention`.

Only `forward` is defined: `kernels` binds it onto the model's own module, so `self.q_proj`,
`self.head_dim` and the rest are the ones `transformers` built.
"""

import os

import torch
from torch import nn

from .banded_attention import banded_attention


# How `tl.dot` should treat the float32 inputs. The kernel's advantage comes from tensor cores,
# which float32 only reaches through tf32. In strict `ieee` Triton has no tensor-core path at all and
# is slower than the fallback it replaces, so that setting is for checking numerics rather than for
# running. `tf32x3` sits in between: tensor cores at close to float32 accuracy, for about a third of
# the tf32 throughput. See the README.
_PRECISION = os.environ.get("WEATHERNEXT2_BANDED_ATTENTION_PRECISION", "tf32").lower()


def _gather_neighbouring_blocks(states: torch.Tensor) -> torch.Tensor:
    """`[batch, blocks, heads, block, dim]` -> the same with the three neighbours side by side."""
    padding = torch.zeros_like(states[:, :1])
    padded = torch.cat([padding, states, padding], dim=1)
    return torch.cat([padded[:, :-2], padded[:, 1:-1], padded[:, 2:]], dim=3)


def _is_banded(attention_mask, hidden_states) -> bool:
    """Is this the geometry's own banded mask, rather than one `masking_utils` expanded?"""
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.dtype != torch.bool:
        return False
    if attention_mask.ndim != 3:
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

        if _is_banded(attention_mask, hidden_states) and not _needs_grad(query, key, value):
            # The kernel walks the three neighbouring blocks itself, so the keys and values are
            # never tripled and the mask is never expanded.
            attn_output = banded_attention(
                query.float(), key.float(), value.float(), attention_mask, self.scaling, precision=_PRECISION
            )
        else:
            attn_output = _reference_attention(query, key, value, attention_mask, self.scaling)

        attn_output = (
            attn_output.to(hidden_states.dtype).transpose(2, 3).reshape(*input_shape, -1).contiguous()
        )
        return self.o_proj(attn_output), None
