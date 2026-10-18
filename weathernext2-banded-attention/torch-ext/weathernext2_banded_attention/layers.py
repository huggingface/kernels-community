"""Attention forwards bound to the model's existing modules."""

import torch
from torch import nn

from .banded_attention import PreparedMask, _prepare_mask as _prepare_mask, banded_attention
from .grid import (
    WeatherNext2ConditionedMlp as WeatherNext2ConditionedMlp,
    WeatherNext2ForecastHead as WeatherNext2ForecastHead,
    WeatherNext2GridToMesh as WeatherNext2GridToMesh,
    WeatherNext2MeshToGrid as WeatherNext2MeshToGrid,
)


_prepare_mask = torch.compiler.disable(_prepare_mask)


def _gather_neighbouring_blocks(states: torch.Tensor) -> torch.Tensor:
    """`[batch, blocks, heads, block, dim]` -> the same with the three neighbours side by side."""
    padding = torch.zeros_like(states[:, :1])
    padded = torch.cat([padding, states, padding], dim=1)
    return torch.cat([padded[:, :-2], padded[:, 1:-1], padded[:, 2:]], dim=3)


def _banded_mask(attention_mask):
    """Return the banded mask without its optional singleton head axis, or None."""
    if isinstance(attention_mask, PreparedMask):
        return attention_mask
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
        and (
            hidden_states.shape[1] == num_blocks
            or (
                isinstance(attention_mask, torch.Tensor)
                and hidden_states.shape[0] * hidden_states.shape[1] == num_blocks
            )
        )
        and hidden_states.shape[2] == block_size
    )


def _needs_grad(*tensors: torch.Tensor) -> bool:
    """Whether any input needs the differentiable reference path."""
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def _reference_attention(query, key, value, attention_mask, scaling):
    """The differentiable path: materialize the three neighbour blocks and use sdpa."""
    batch, blocks, heads, block_size, head_dim = query.shape
    keys = _gather_neighbouring_blocks(key).reshape(batch * blocks, heads, 3 * block_size, head_dim)
    values = _gather_neighbouring_blocks(value).reshape(batch * blocks, heads, 3 * block_size, head_dim)
    queries = query.reshape(batch * blocks, heads, block_size, head_dim)

    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 3:
        # The geometry's banded mask, which sdpa needs broadcast over the folded batch axis.
        if attention_mask.shape[0] == blocks:
            attention_mask = attention_mask[None, :, None].expand(batch, blocks, 1, block_size, 3 * block_size)
            attention_mask = attention_mask.reshape(batch * blocks, 1, block_size, 3 * block_size)
        else:
            attention_mask = attention_mask[:, None]

    out = nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=attention_mask, scale=scaling)
    return out.reshape(batch, blocks, heads, block_size, head_dim)


@torch.compiler.assume_constant_result
def _tf32_allowed() -> bool:
    """Read PyTorch's TF32 setting under Dynamo's global-state guard."""
    matmul = torch.backends.cuda.matmul
    # `fp32_precision` reflects both the legacy and the fine-grained precision APIs, whereas
    # `allow_tf32` raises once the two have been mixed.
    if hasattr(matmul, "fp32_precision"):
        return matmul.fp32_precision == "tf32"
    return matmul.allow_tf32


def _float32_precision(device: torch.device) -> str:
    """Follow PyTorch's float32 policy, using TF32x3 for full-precision NVIDIA dots."""
    if device.type != "cuda" or torch.version.hip is not None:
        return "ieee"
    return "default" if _tf32_allowed() else "tf32x3"


def _blockwise_attention(query, key, value, mask, scaling):
    """Run sparse Triton attention without gathering neighbouring key/value blocks."""
    batch, blocks = query.shape[:2]
    precision = _float32_precision(query.device)
    if isinstance(mask, PreparedMask) or mask.shape[0] == blocks:
        return banded_attention(query, key, value, mask, scaling, precision=precision)
    # Expanded masks can differ between ensemble members; never reuse the first member's mask.
    masks = mask.reshape(batch, blocks, *mask.shape[1:])
    return torch.cat(
        [
            banded_attention(
                query[i : i + 1],
                key[i : i + 1],
                value[i : i + 1],
                masks[i],
                scaling,
                precision=precision,
            )
            for i in range(batch)
        ]
    )


class WeatherNext2MeshEncoder(nn.Module):
    """Prepare one packed geometry mask per forward, shared by all transformer layers."""

    can_torch_compile = True

    def forward(self, hidden_states, attention_mask, conditioning, **kwargs):
        if (
            not isinstance(attention_mask, dict)
            and not self.training
            and not torch.is_grad_enabled()
            and not hidden_states.is_cpu
            and self.config._attn_implementation == "sdpa"
        ):
            num_nodes = hidden_states.shape[1]
            block_size = min(self.config.attention_bandwidth, num_nodes)
            num_blocks = (num_nodes + block_size - 1) // block_size
            if attention_mask.shape == (num_blocks, 1, block_size, 3 * block_size):
                attention_mask = {"full_attention": _prepare_mask(attention_mask[:, 0])}
        return type(self).forward(self, hidden_states, attention_mask, conditioning, **kwargs)


class WeatherNext2Attention(nn.Module):
    can_torch_compile = True

    def forward(self, hidden_states: torch.Tensor, attention_mask, **kwargs):
        if self.config._attn_implementation != "sdpa" or not isinstance(attention_mask, (torch.Tensor, PreparedMask)):
            return type(self).forward(self, hidden_states, attention_mask, **kwargs)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # [batch, blocks, block, hidden] -> [batch, blocks, heads, block, head_dim]
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(2, 3)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(2, 3)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(2, 3)
        query, key, value = (states.float() for states in (query, key, value))

        banded = _banded_mask(attention_mask)
        if _is_banded(attention_mask, hidden_states) and not query.is_cpu and not _needs_grad(query, key, value):
            attn_output = _blockwise_attention(query, key, value, banded, self.scaling)
        else:
            mask = banded if banded is not None else attention_mask
            attn_output = _reference_attention(query, key, value, mask, self.scaling)

        attn_output = attn_output.to(hidden_states.dtype).transpose(2, 3).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), None
