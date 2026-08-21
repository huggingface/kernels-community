"""The layer `kernels` swaps into `transformers`' `WeatherNext2Attention`.

Only `forward` is defined: `kernels` binds it onto the model's own module, so `self.q_proj`,
`self.head_dim` and the rest are the ones `transformers` built.
"""

import os

import torch
from torch import nn

from .banded_attention import banded_attention


# The kernel's advantage comes from tensor cores, which float32 inputs only reach through tf32. In
# strict ieee Triton has no tensor-core path at all and is slower than the fallback it replaces, so
# that setting is for checking numerics rather than for running. `tf32x3` sits in between: tensor
# cores at close to float32 accuracy, for about a third of the tf32 throughput. See the README.
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


class WeatherNext2Attention(nn.Module):
    def forward(self, hidden_states: torch.Tensor, attention_mask, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # [batch, blocks, block, hidden] -> [batch, blocks, heads, block, head_dim]
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(2, 3)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(2, 3)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(2, 3)

        if _is_banded(attention_mask, hidden_states):
            # The kernel walks the three neighbouring blocks itself, so the keys and values are
            # never tripled and the mask is never expanded.
            attn_output = banded_attention(
                query.float(),
                key.float(),
                value.float(),
                attention_mask,
                self.scaling,
                precision=_PRECISION,
            )
            attn_weights = None
        else:
            # Anything else is not something the kernel can read, so fall back to what the model does
            # without it. Being slower than the path we replace is recoverable; being wrong is not.
            if not isinstance(attention_mask, torch.Tensor):
                # A `BlockMask` only arrives with `attn_implementation="flex_attention"`, and
                # `scaled_dot_product_attention` cannot consume one. Say so rather than drop the
                # mask, which would silently attend across the whole band.
                raise ValueError(
                    "WeatherNext2Attention kernel got a "
                    f"{type(attention_mask).__name__} mask, which it cannot read. Load the model "
                    'with attn_implementation="sdpa" (the default) when use_kernels=True.'
                )
            batch, blocks, heads, block_size, head_dim = query.shape
            keys = _gather_neighbouring_blocks(key).reshape(batch * blocks, heads, 3 * block_size, head_dim)
            values = _gather_neighbouring_blocks(value).reshape(
                batch * blocks, heads, 3 * block_size, head_dim
            )
            queries = query.reshape(batch * blocks, heads, block_size, head_dim).float()
            out = nn.functional.scaled_dot_product_attention(
                queries, keys.float(), values.float(), attn_mask=attention_mask, scale=self.scaling
            )
            attn_output = out.reshape(batch, blocks, heads, block_size, head_dim)
            attn_weights = None

        attn_output = (
            attn_output.to(hidden_states.dtype).transpose(2, 3).reshape(*input_shape, -1).contiguous()
        )
        return self.o_proj(attn_output), attn_weights
