"""The kernel has to agree with the PyTorch path it replaces, on the mask shape the model uses."""

import kernels
import pytest
import torch


weathernext2_banded_attention = kernels.get_kernel("kernels-community/weathernext2-banded-attention", version=2)
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
    out = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=dense, scale=scaling)
    return out.reshape(batch, blocks, heads, block_size, head_dim)


def banded_mask(blocks, block_size, density, device, generator):
    mask = torch.rand(blocks, block_size, 3 * block_size, device=device, generator=generator) < density
    mask[0, :, :block_size] = False  # the first block has no predecessor
    mask[-1, :, 2 * block_size :] = False  # nor the last a successor
    # Every mesh node reaches itself, so no row is empty and no row softmaxes to NaN.
    mask[:, :, block_size : 2 * block_size] |= torch.eye(block_size, dtype=torch.bool, device=device)
    return mask


DEVICE = weathernext2_banded_attention.infer_device()
requires_accelerator = pytest.mark.skipif(DEVICE == "cpu", reason="the kernel needs an accelerator")


@requires_accelerator
@pytest.mark.parametrize("blocks", [2, 4])
@pytest.mark.parametrize("density", [0.05, 0.4])
@pytest.mark.parametrize("prepared", [False, True])
def test_matches_scaled_dot_product_attention(blocks, density, prepared):
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

    banded = weathernext2_banded_attention.layers._prepare_mask(mask) if prepared else mask
    ours = banded_attention(
        query,
        key,
        value,
        banded,
        scaling,
        precision="ieee",
    )
    torch.testing.assert_close(ours, reference(query, key, value, mask, scaling), atol=2e-5, rtol=2e-5)


@requires_accelerator
@pytest.mark.kernels_ci
@pytest.mark.parametrize("prepared", [False, True])
def test_matches_float64_reference(prepared):
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    shape = (2, 3, 2, 129, 128)
    query, key, value = (torch.randn(shape, device=DEVICE, generator=generator) for _ in range(3))
    mask = banded_mask(3, 129, 0.05, DEVICE, generator)
    scaling = 128**-0.5

    # Faster-WeatherNext validates against a float64 QK -> scale -> softmax -> PV reference.
    queries, keys, values = (tensor.cpu().double() for tensor in (query, key, value))
    keys, values = (gather_neighbouring_blocks(tensor) for tensor in (keys, values))
    scores = (queries @ keys.transpose(-1, -2)) * scaling
    scores = scores.masked_fill(~mask.cpu()[None, :, None], float("-inf"))
    expected = scores.softmax(dim=-1) @ values

    banded = weathernext2_banded_attention.layers._prepare_mask(mask) if prepared else mask
    actual = banded_attention(query, key, value, banded, scaling, precision="ieee")
    torch.testing.assert_close(actual.cpu().double(), expected, atol=2e-5, rtol=2e-5)
    assert (actual.cpu().double() - expected).abs().max() / expected.abs().max() < 1e-5


@requires_accelerator
@pytest.mark.kernels_ci
def test_rows_that_reach_only_themselves_are_finite():
    """The sparsest legal mask: a node that sees nothing but itself must not produce NaN."""
    device = torch.device(DEVICE)
    generator = torch.Generator(device=device).manual_seed(0)
    batch, blocks, heads, block_size, head_dim = 1, 3, 2, 64, 32
    shape = (batch, blocks, heads, block_size, head_dim)
    query, key, value = (torch.randn(shape, device=device, dtype=torch.float32, generator=generator) for _ in range(3))
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
            setattr(
                self,
                name,
                torch.nn.Linear(hidden_size, hidden_size, bias=name == "o_proj"),
            )


@requires_accelerator
@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_layer_keeps_attention_in_full_float32(dtype):
    torch.manual_seed(7)
    attention = _StubAttention(64, 2).to(device=DEVICE, dtype=dtype).eval()
    hidden_states = torch.randn(2, 3, 16, 64, device=DEVICE, dtype=dtype)
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    mask = banded_mask(3, 16, 0.1, DEVICE, generator)
    with torch.inference_mode():
        query, key, value = (
            getattr(attention, name)(hidden_states).view(2, 3, 16, 2, 32).transpose(2, 3).float()
            for name in ("q_proj", "k_proj", "v_proj")
        )
        expected = reference(query, key, value, mask, attention.scaling)
        expected = attention.o_proj(expected.to(dtype).transpose(2, 3).reshape_as(hidden_states))
        output, _ = attention(hidden_states, mask)
    torch.testing.assert_close(output, expected)
    assert output.dtype == dtype


@pytest.mark.kernels_ci
@pytest.mark.parametrize("batched_mask", [False, True])
def test_backward_reaches_every_projection(batched_mask):
    """The kernel has no backward, so anything needing one must take the differentiable path.

    Without the guard this still runs: the kernel's output carries no `grad_fn`, so `backward()`
    succeeds while `q_proj`, `k_proj` and `v_proj` silently receive nothing.
    """
    torch.manual_seed(0)
    batch, blocks, heads, block_size, hidden = 2, 3, 2, 16, 32
    attention = _StubAttention(hidden, heads)
    hidden_states = torch.randn(batch, blocks, block_size, hidden, requires_grad=True)
    mask = torch.zeros(blocks, block_size, 3 * block_size, dtype=torch.bool)
    mask[:, :, block_size : 2 * block_size] = torch.eye(block_size, dtype=torch.bool)
    if batched_mask:
        mask = mask.repeat(batch, 1, 1)

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


@requires_accelerator
@pytest.mark.kernels_ci
@pytest.mark.parametrize("block_size", [65, 129])
@pytest.mark.parametrize("pattern", ["self", "random", "empty"])
def test_sparse_tiles_preserve_outputs(block_size, pattern):
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    shape = (2, 3, block_size, 2, 32)
    query, key, value = (torch.randn(shape, device=DEVICE, generator=generator).transpose(2, 3) for _ in range(3))
    mask = banded_mask(3, block_size, 0.01 if pattern == "random" else 0, DEVICE, generator)
    if pattern == "empty":
        mask.zero_()
    # Preserve a noncontiguous mask view, including the partial final query/key tiles.
    storage = torch.zeros(3, block_size, 6 * block_size, dtype=torch.bool, device=DEVICE)
    storage[..., ::2] = mask
    mask = storage[..., ::2]
    assert not mask.is_contiguous()
    expected = banded_attention(query, key, value, mask, 32**-0.5, precision="ieee")
    prepared = weathernext2_banded_attention.layers._prepare_mask(mask)
    actual = banded_attention(
        query,
        key,
        value,
        prepared,
        32**-0.5,
        precision="ieee",
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual, reference(query, key, value, mask, 32**-0.5), atol=2e-5, rtol=2e-5)


@requires_accelerator
@pytest.mark.kernels_ci
@pytest.mark.parametrize("block_size", [65, 129])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("prepared", [False, True])
def test_layer_preserves_sdpa(block_size, batch, prepared):
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    shape = (batch, 3, block_size, 2, 32)
    query, key, value = (torch.randn(shape, device=DEVICE, generator=generator).transpose(2, 3) for _ in range(3))
    mask = banded_mask(3, block_size, 0.1, DEVICE, generator)
    banded = weathernext2_banded_attention.layers._prepare_mask(mask) if prepared else mask
    actual = weathernext2_banded_attention.layers._blockwise_attention(query, key, value, banded, 32**-0.5)
    torch.testing.assert_close(actual, reference(query, key, value, mask, 32**-0.5), atol=2e-5, rtol=2e-5)


@requires_accelerator
@pytest.mark.kernels_ci
@pytest.mark.parametrize("mask_layout", ["geometry", "batched", "packed"])
def test_inference_layer_preserves_reference(mask_layout):
    torch.manual_seed(7)
    attention = _StubAttention(256, 2).to(DEVICE).eval()
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    hidden_states = torch.randn(2, 3, 129, 256, device=DEVICE, generator=generator)
    mask = banded_mask(3, 129, 0.05, DEVICE, generator)
    if mask_layout == "packed":
        banded = weathernext2_banded_attention.layers._prepare_mask(mask)
    elif mask_layout == "batched":
        second_mask = banded_mask(3, 129, 0.1, DEVICE, generator)
        banded = torch.stack((mask, second_mask)).view(6, 1, 129, 387)
    else:
        banded = mask[:, None]
    assert weathernext2_banded_attention.layers._is_banded(banded, hidden_states)
    with torch.inference_mode():
        query, key, value = (
            getattr(attention, name)(hidden_states).view(2, 3, 129, 2, 128).transpose(2, 3)
            for name in ("q_proj", "k_proj", "v_proj")
        )
        if mask_layout == "batched":
            expected = torch.cat(
                [
                    reference(query[:1], key[:1], value[:1], mask, attention.scaling),
                    reference(query[1:], key[1:], value[1:], second_mask, attention.scaling),
                ]
            )
        else:
            expected = reference(query, key, value, mask, attention.scaling)
        expected = attention.o_proj(expected.transpose(2, 3).reshape_as(hidden_states))
        actual, _ = attention(hidden_states, banded)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("head_axis", [False, True])
def test_cpu_inference_preserves_reference(dtype, head_axis):
    torch.manual_seed(7)
    attention = _StubAttention(64, 2).to(dtype=dtype).eval()
    hidden_states = torch.randn(2, 3, 16, 64, dtype=dtype)
    mask = banded_mask(3, 16, 0.1, "cpu", torch.Generator().manual_seed(7))
    with torch.inference_mode():
        query, key, value = (
            getattr(attention, name)(hidden_states).view(2, 3, 16, 2, 32).transpose(2, 3).float()
            for name in ("q_proj", "k_proj", "v_proj")
        )
        expected = reference(query, key, value, mask, attention.scaling)
        expected = attention.o_proj(expected.to(dtype).transpose(2, 3).reshape_as(hidden_states))
        actual, _ = attention(hidden_states, mask[:, None] if head_axis else mask)
    torch.testing.assert_close(actual, expected)


@requires_accelerator
@pytest.mark.kernels_ci
def test_attention_on_noncurrent_device():
    backend = torch.get_device_module(DEVICE)
    if backend.device_count() < 2:
        pytest.skip("requires two accelerators")
    device = torch.device(DEVICE, 1)
    generator = torch.Generator(device=device).manual_seed(7)
    query, key, value = (torch.randn(1, 3, 2, 65, 32, device=device, generator=generator) for _ in range(3))
    mask = banded_mask(3, 65, 0.05, device, generator)
    with backend.device(0):
        prepared = weathernext2_banded_attention.layers._prepare_mask(mask)
        actual = banded_attention(query, key, value, prepared, 32**-0.5, precision="ieee")
        torch.testing.assert_close(actual, reference(query, key, value, mask, 32**-0.5), atol=2e-5, rtol=2e-5)
        assert actual.device == device
        assert backend.current_device() == 0


@requires_accelerator
@pytest.mark.parametrize("prepared", [False, True])
def test_attention_compile(prepared):
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    shape = (2, 3, 2, 65, 32)
    query, key, value = (torch.randn(shape, device=DEVICE, generator=generator) for _ in range(3))
    mask = banded_mask(3, 65, 0.05, DEVICE, generator)
    banded = weathernext2_banded_attention.layers._prepare_mask(mask) if prepared else mask

    def attention(query, key, value, mask):
        return banded_attention(query, key, value, mask, 32**-0.5, precision="ieee")

    compiled = torch.compile(attention, fullgraph=True)
    try:
        with torch.inference_mode():
            for _ in range(2):
                query = torch.randn(shape, device=DEVICE, generator=generator)
                actual = compiled(query, key, value, banded)
                expected = reference(query, key, value, mask, 32**-0.5)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    finally:
        torch._dynamo.reset()
