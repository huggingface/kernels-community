"""The kernel has to agree with the PyTorch path it replaces, on the mask shape the model uses."""

import pytest
import torch

from weathernext2_banded_attention import banded_attention


def gather_neighbouring_blocks(states):
    padding = torch.zeros_like(states[:, :1])
    padded = torch.cat([padding, states, padding], dim=1)
    return torch.cat([padded[:, :-2], padded[:, 1:-1], padded[:, 2:]], dim=3)


def reference(query, key, value, mask, scaling):
    batch, blocks, heads, block_size, head_dim = query.shape
    keys = gather_neighbouring_blocks(key).reshape(batch * blocks, heads, 3 * block_size, head_dim)
    values = gather_neighbouring_blocks(value).reshape(batch * blocks, heads, 3 * block_size, head_dim)
    queries = query.reshape(batch * blocks, heads, block_size, head_dim)
    dense = mask[None, :, None].expand(batch, blocks, 1, block_size, 3 * block_size)
    dense = dense.reshape(batch * blocks, 1, block_size, 3 * block_size)
    out = torch.nn.functional.scaled_dot_product_attention(
        queries, keys, values, attn_mask=dense, scale=scaling
    )
    return out.reshape(batch, blocks, heads, block_size, head_dim)


def banded_mask(blocks, block_size, density, device, generator):
    mask = torch.rand(blocks, block_size, 3 * block_size, device=device, generator=generator) < density
    mask[0, :, :block_size] = False  # the first block has no predecessor
    mask[-1, :, 2 * block_size :] = False  # nor the last a successor
    # Every mesh node reaches itself, so no row is empty and no row softmaxes to NaN.
    mask[:, :, block_size : 2 * block_size] |= torch.eye(block_size, dtype=torch.bool, device=device)
    return mask


@pytest.mark.parametrize("blocks", [2, 4])
@pytest.mark.parametrize("density", [0.05, 0.4])
def test_matches_scaled_dot_product_attention(blocks, density):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    batch, heads, block_size, head_dim = 1, 4, 128, 64
    shape = (batch, blocks, heads, block_size, head_dim)
    query, key, value = (
        torch.randn(shape, device=device, dtype=torch.float32, generator=generator) for _ in range(3)
    )
    mask = banded_mask(blocks, block_size, density, device, generator)
    scaling = head_dim**-0.5

    ours = banded_attention(query, key, value, mask, scaling, precision="ieee")
    torch.testing.assert_close(ours, reference(query, key, value, mask, scaling), atol=2e-5, rtol=2e-5)


def test_rows_that_reach_only_themselves_are_finite():
    """The sparsest legal mask: a node that sees nothing but itself must not produce NaN."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    batch, blocks, heads, block_size, head_dim = 1, 3, 2, 64, 32
    shape = (batch, blocks, heads, block_size, head_dim)
    query, key, value = (
        torch.randn(shape, device=device, dtype=torch.float32, generator=generator) for _ in range(3)
    )
    mask = torch.zeros(blocks, block_size, 3 * block_size, dtype=torch.bool, device=device)
    mask[:, :, block_size : 2 * block_size] = torch.eye(block_size, dtype=torch.bool, device=device)

    out = banded_attention(query, key, value, mask, head_dim**-0.5, precision="ieee")
    assert torch.isfinite(out).all()
    # Attending to yourself alone is just your own value vector.
    torch.testing.assert_close(out, value, atol=2e-5, rtol=2e-5)
