"""Shared test helpers: the loaded kernel, device markers, and PyTorch references."""

import kernels
import pytest
import torch


kernel = kernels.get_kernel("kernels-community/weathernext2-banded-attention", version=4)
banded_attention = kernel.banded_attention
prepare_mask = kernel.layers._prepare_mask

DEVICE = kernel.infer_device()
requires_accelerator = pytest.mark.skipif(DEVICE == "cpu", reason="the kernel needs an accelerator")
requires_nvidia = pytest.mark.skipif(
    DEVICE != "cuda" or torch.version.hip is not None,
    reason="TF32x3 is an NVIDIA tensor-core mode",
)


def gather_neighbouring_blocks(states):
    """Concatenates each block with its two neighbours, zero-padded at the ends."""
    padding = torch.zeros_like(states[:, :1])
    padded = torch.cat([padding, states, padding], dim=1)
    return torch.cat([padded[:, :-2], padded[:, 1:-1], padded[:, 2:]], dim=3)


def reference(query, key, value, mask, scaling):
    """Dense sdpa over the three neighbouring blocks, with the geometry mask shared by the batch."""
    batch, blocks, heads, block_size, head_dim = query.shape
    keys = gather_neighbouring_blocks(key).reshape(batch * blocks, heads, 3 * block_size, head_dim)
    values = gather_neighbouring_blocks(value).reshape(batch * blocks, heads, 3 * block_size, head_dim)
    queries = query.reshape(batch * blocks, heads, block_size, head_dim)
    dense = mask[None, :, None].expand(batch, blocks, 1, block_size, 3 * block_size)
    dense = dense.reshape(batch * blocks, 1, block_size, 3 * block_size)
    out = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=dense, scale=scaling)
    return out.reshape(batch, blocks, heads, block_size, head_dim)


def assert_matches_reference(actual, query, key, value, mask, scaling):
    torch.testing.assert_close(actual, reference(query, key, value, mask, scaling), atol=2e-5, rtol=2e-5)


def banded_mask(blocks, block_size, density, device=DEVICE, generator=None):
    """A random geometry mask where every query row reaches at least itself."""
    mask = torch.rand(blocks, block_size, 3 * block_size, device=device, generator=generator) < density
    mask[0, :, :block_size] = False  # the first block has no predecessor
    mask[-1, :, 2 * block_size :] = False  # nor the last a successor
    mask[:, :, block_size : 2 * block_size] |= torch.eye(block_size, dtype=torch.bool, device=device)
    return mask


def random_qkv(shape, generator, device=DEVICE):
    """Query, key and value shaped `[batch, blocks, heads, block_size, head_dim]`.

    They are transposed views of `[batch, blocks, block_size, heads, head_dim]`, matching the model's
    non-contiguous head layout.
    """
    batch, blocks, heads, block_size, head_dim = shape
    source = (batch, blocks, block_size, heads, head_dim)
    return tuple(torch.randn(source, device=device, generator=generator).transpose(2, 3) for _ in range(3))


def attention_case(shape, density=0.05, device=DEVICE, seed=7):
    """Random query/key/value with a matching geometry mask and the default softmax scaling."""
    generator = torch.Generator(device=device).manual_seed(seed)
    query, key, value = random_qkv(shape, generator, device)
    mask = banded_mask(shape[1], shape[3], density, device, generator)
    return query, key, value, mask, shape[-1] ** -0.5
