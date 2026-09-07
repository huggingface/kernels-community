"""The kernel has to agree with the PyTorch path it replaces, on the mask shape the model uses."""

import pytest
import torch

import kernels


weathernext2_banded_attention = kernels.get_kernel(
    "kernels-community/weathernext2-banded-attention", version=1
)
WeatherNext2Attention = weathernext2_banded_attention.WeatherNext2Attention
banded_attention = weathernext2_banded_attention.banded_attention


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


# The kernel is pure Triton and the package declares cuda, rocm and xpu, so the tests have to run
# on whichever accelerator is present rather than assuming CUDA. Otherwise XPU or NPU CI would skip
# every kernel launch and the declared backend would go untested.
DEVICE = weathernext2_banded_attention.infer_device()
requires_accelerator = pytest.mark.skipif(DEVICE == "cpu", reason="the kernel needs an accelerator")


@requires_accelerator
@pytest.mark.parametrize("blocks", [2, 4])
@pytest.mark.parametrize("density", [0.05, 0.4])
def test_matches_scaled_dot_product_attention(blocks, density):
    device = torch.device(DEVICE)
    generator = torch.Generator(device=device).manual_seed(0)
    batch, heads, block_size, head_dim = 1, 4, 128, 64
    # The model produces this layout by splitting the hidden dimension into heads and transposing
    # the block and head axes. Keep it non-contiguous so the kernel's stride handling is exercised.
    source_shape = (batch, blocks, block_size, heads, head_dim)
    query, key, value = (
        torch.randn(source_shape, device=device, dtype=torch.float32, generator=generator).transpose(2, 3)
        for _ in range(3)
    )
    assert not query.is_contiguous()
    mask = banded_mask(blocks, block_size, density, device, generator)
    scaling = head_dim**-0.5

    # On NVIDIA, force IEEE so this stays a strict algorithmic comparison rather than a tensor-core
    # precision one. Everywhere else take the backend default: AMD has no TF32 so it is already
    # IEEE, and `input_precision="ieee"` is not guaranteed to be implemented on other backends.
    on_nvidia = torch.version.cuda is not None and torch.version.hip is None
    precision = "ieee" if on_nvidia else "default"
    ours = banded_attention(query, key, value, mask, scaling, precision=precision)
    torch.testing.assert_close(ours, reference(query, key, value, mask, scaling), atol=2e-5, rtol=2e-5)


@requires_accelerator
@pytest.mark.kernels_ci
def test_rows_that_reach_only_themselves_are_finite():
    """The sparsest legal mask: a node that sees nothing but itself must not produce NaN."""
    device = torch.device(DEVICE)
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


class _StubAttention(WeatherNext2Attention):
    """The attributes the layer reads off the `transformers` module it is bound onto."""

    def __init__(self, hidden_size, heads):
        super().__init__()
        self.head_dim = hidden_size // heads
        self.scaling = self.head_dim**-0.5
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, torch.nn.Linear(hidden_size, hidden_size, bias=name == "o_proj"))


@pytest.mark.kernels_ci
def test_backward_reaches_every_projection():
    """The kernel has no backward, so anything needing one must take the differentiable path.

    Without the guard this still runs: the kernel's output carries no `grad_fn`, so `backward()`
    succeeds while `q_proj`, `k_proj` and `v_proj` silently receive nothing.
    """
    torch.manual_seed(0)
    batch, blocks, heads, block_size, hidden = 1, 3, 2, 16, 32
    attention = _StubAttention(hidden, heads)
    hidden_states = torch.randn(batch, blocks, block_size, hidden, requires_grad=True)
    mask = torch.zeros(blocks, block_size, 3 * block_size, dtype=torch.bool)
    mask[:, :, block_size : 2 * block_size] = torch.eye(block_size, dtype=torch.bool)

    attention(hidden_states, mask)[0].square().mean().backward()

    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert getattr(attention, name).weight.grad is not None, f"{name} received no gradient"
    assert hidden_states.grad is not None


@pytest.mark.kernels_ci
def test_inference_takes_the_kernel_and_training_does_not():
    """Under `no_grad` the fast path is available; with grad it must not be."""
    layers = weathernext2_banded_attention.layers

    hidden_states = torch.randn(1, 3, 16, 32)
    mask = torch.zeros(3, 16, 48, dtype=torch.bool)
    assert layers._is_banded(mask, hidden_states)

    leaf = hidden_states.clone().requires_grad_(True)
    assert layers._needs_grad(leaf)
    with torch.no_grad():
        assert not layers._needs_grad(leaf)


@pytest.mark.kernels_ci
def test_rejects_head_dimensions_it_cannot_tile():
    """`HEAD_DIM` is an unmasked constexpr tile width, so a bad value must raise, not read past."""
    mask = torch.zeros(2, 32, 96, dtype=torch.bool)
    for head_dim in (8, 24, 48):
        query = torch.zeros(1, 2, 2, 32, head_dim)
        with pytest.raises(ValueError, match="head_dim must be a power of two"):
            banded_attention(query, query, query, mask, head_dim**-0.5)
