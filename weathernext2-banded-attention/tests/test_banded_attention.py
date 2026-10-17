"""The kernel has to agree with the PyTorch path it replaces, on the mask shape the model uses."""

from types import SimpleNamespace

import kernels
import pytest
import torch


weathernext2_banded_attention = kernels.get_kernel("kernels-community/weathernext2-banded-attention", version=3)
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
    # Keep every query row nonempty.
    mask[:, :, block_size : 2 * block_size] |= torch.eye(block_size, dtype=torch.bool, device=device)
    return mask


DEVICE = weathernext2_banded_attention.infer_device()
requires_accelerator = pytest.mark.skipif(DEVICE == "cpu", reason="the kernel needs an accelerator")
requires_nvidia = pytest.mark.skipif(
    DEVICE != "cuda" or torch.version.hip is not None, reason="TF32x3 is an NVIDIA tensor-core mode"
)


@requires_accelerator
@pytest.mark.parametrize("blocks", [2, 4])
@pytest.mark.parametrize("density", [0.05, 0.4])
@pytest.mark.parametrize("prepared", [False, True])
def test_matches_scaled_dot_product_attention(blocks, density, prepared):
    device = torch.device(DEVICE)
    generator = torch.Generator(device=device).manual_seed(0)
    batch, heads, block_size, head_dim = 1, 4, 128, 64
    # Match the model's non-contiguous head layout.
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
    torch.testing.assert_close(out, value, atol=2e-5, rtol=2e-5)


class _StubAttention(WeatherNext2Attention):
    def __init__(self, hidden_size, heads):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="sdpa")
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
    """Gradient-enabled calls must use the differentiable reference."""
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
@pytest.mark.parametrize(
    ("batch", "block_size", "mask_layout"),
    [(1, 65, "geometry"), (1, 65, "packed"), (2, 129, "geometry"), (2, 129, "batched"), (2, 129, "packed")],
)
def test_inference_layer_preserves_reference(batch, block_size, mask_layout):
    torch.manual_seed(7)
    attention = _StubAttention(256, 2).to(DEVICE).eval()
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    hidden_states = torch.randn(batch, 3, block_size, 256, device=DEVICE, generator=generator)
    mask = banded_mask(3, block_size, 0.05, DEVICE, generator)
    if mask_layout == "packed":
        banded = weathernext2_banded_attention.layers._prepare_mask(mask)
    elif mask_layout == "batched":
        second_mask = banded_mask(3, block_size, 0.1, DEVICE, generator)
        banded = torch.stack((mask, second_mask)).view(6, 1, block_size, 3 * block_size)
    else:
        banded = mask[:, None]
    assert weathernext2_banded_attention.layers._is_banded(banded, hidden_states)
    with torch.inference_mode():
        query, key, value = (
            getattr(attention, name)(hidden_states).view(batch, 3, block_size, 2, 128).transpose(2, 3)
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


@requires_nvidia
@pytest.mark.kernels_ci
@pytest.mark.parametrize("prepared", [False, True])
def test_tf32x3_matches_ieee(prepared):
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    shape = (2, 3, 2, 129, 128)
    query, key, value = (torch.randn(shape, device=DEVICE, generator=generator) for _ in range(3))
    mask = banded_mask(3, 129, 0.05, DEVICE, generator)
    banded = weathernext2_banded_attention.layers._prepare_mask(mask) if prepared else mask
    expected = banded_attention(query, key, value, banded, 128**-0.5, precision="ieee")
    actual = banded_attention(query, key, value, banded, 128**-0.5, precision="tf32x3")
    torch.testing.assert_close(actual, expected)


@pytest.mark.kernels_ci
def test_layer_precision_follows_set_float32_matmul_precision():
    precision = weathernext2_banded_attention.layers._float32_precision
    on_rocm = torch.version.hip is not None
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        assert precision(torch.device("cuda")) == ("ieee" if on_rocm else "default")
        torch.set_float32_matmul_precision("highest")
        assert precision(torch.device("cuda")) == ("ieee" if on_rocm else "tf32x3")
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.kernels_ci
def test_tf32x3_needs_nvidia():
    query = key = value = torch.zeros(1, 3, 1, 16, 16)
    mask = torch.ones(3, 16, 48, dtype=torch.bool)
    with pytest.raises(ValueError, match="tf32x3"):
        banded_attention(query, key, value, mask, 0.25, precision="tf32x3")


@requires_accelerator
@pytest.mark.kernels_ci
def test_layer_attention_compile():
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    shape = (2, 3, 2, 65, 32)
    query, key, value = (torch.randn(shape, device=DEVICE, generator=generator) for _ in range(3))
    mask = banded_mask(3, 65, 0.05, DEVICE, generator)
    prepared = weathernext2_banded_attention.layers._prepare_mask(mask)
    compiled = torch.compile(weathernext2_banded_attention.layers._blockwise_attention, fullgraph=True)
    try:
        with torch.inference_mode():
            actual = compiled(query, key, value, prepared, 32**-0.5)
        torch.testing.assert_close(actual, reference(query, key, value, mask, 32**-0.5), atol=2e-5, rtol=2e-5)
    finally:
        torch._dynamo.reset()


@requires_nvidia
@pytest.mark.kernels_ci
def test_compiled_layer_attention_follows_precision_changes():
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    shape = (2, 3, 2, 65, 32)
    query, key, value = (torch.randn(shape, device=DEVICE, generator=generator) for _ in range(3))
    mask = banded_mask(3, 65, 0.05, DEVICE, generator)
    prepared = weathernext2_banded_attention.layers._prepare_mask(mask)
    compiled = torch.compile(weathernext2_banded_attention.layers._blockwise_attention, fullgraph=True)

    def eager(precision):
        return banded_attention(query, key, value, prepared, 32**-0.5, precision=precision)

    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with torch.inference_mode():
            assert torch.equal(compiled(query, key, value, prepared, 32**-0.5), eager("tf32x3"))
            torch.set_float32_matmul_precision("high")
            assert torch.equal(compiled(query, key, value, prepared, 32**-0.5), eager("default"))
            torch.set_float32_matmul_precision("highest")
            assert torch.equal(compiled(query, key, value, prepared, 32**-0.5), eager("tf32x3"))
    finally:
        torch.set_float32_matmul_precision(previous)
        torch._dynamo.reset()
