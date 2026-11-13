"""The kernel has to agree with the PyTorch path it replaces, on the mask shape the model uses."""

from types import SimpleNamespace

import pytest
import torch

from .utils import (
    DEVICE,
    assert_matches_reference,
    attention_case,
    banded_attention,
    banded_mask,
    gather_neighbouring_blocks,
    kernel,
    prepare_mask,
    reference,
    requires_accelerator,
    requires_nvidia,
)


class _StubAttention(kernel.WeatherNext2Attention):
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

    def projected(self, hidden_states):
        """Query, key and value as the layer computes them, upcast to float32."""
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        return tuple(
            getattr(self, name)(hidden_states).view(shape).transpose(2, 3).float()
            for name in ("q_proj", "k_proj", "v_proj")
        )


@requires_accelerator
@pytest.mark.kernels_ci
@pytest.mark.parametrize("prepared", [False, True])
def test_matches_float64_reference(prepared):
    query, key, value, mask, scaling = attention_case((2, 3, 2, 129, 128))

    # Faster-WeatherNext validates against a float64 QK -> scale -> softmax -> PV reference.
    queries, keys, values = (tensor.cpu().double() for tensor in (query, key, value))
    keys, values = (gather_neighbouring_blocks(tensor) for tensor in (keys, values))
    scores = (queries @ keys.transpose(-1, -2)) * scaling
    scores = scores.masked_fill(~mask.cpu()[None, :, None], float("-inf"))
    expected = scores.softmax(dim=-1) @ values

    banded = prepare_mask(mask) if prepared else mask
    actual = banded_attention(query, key, value, banded, scaling, precision="ieee").cpu().double()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert (actual - expected).abs().max() / expected.abs().max() < 1e-5


@requires_accelerator
@pytest.mark.kernels_ci
def test_rows_that_reach_only_themselves_are_finite():
    """The sparsest legal mask: a node that sees nothing but itself must not produce NaN."""
    query, key, value, mask, scaling = attention_case((1, 3, 2, 64, 32), density=0)

    out = banded_attention(query, key, value, mask, scaling, precision="ieee")
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, value, atol=2e-5, rtol=2e-5)


@requires_accelerator
@pytest.mark.kernels_ci
@pytest.mark.parametrize("block_size", [65, 129])
@pytest.mark.parametrize("pattern", ["self", "random", "empty"])
def test_sparse_tiles_preserve_outputs(block_size, pattern):
    """Skipped tiles, partial final tiles and a non-contiguous mask view must not change the output."""
    query, key, value, mask, scaling = attention_case((2, 3, 2, block_size, 32), density=0.01 * (pattern == "random"))
    if pattern == "empty":
        mask.zero_()
    storage = torch.zeros(3, block_size, 6 * block_size, dtype=torch.bool, device=DEVICE)
    storage[..., ::2] = mask
    mask = storage[..., ::2]

    expected = banded_attention(query, key, value, mask, scaling, precision="ieee")
    actual = banded_attention(query, key, value, prepare_mask(mask), scaling, precision="ieee")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert_matches_reference(actual, query, key, value, mask, scaling)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("device", list(dict.fromkeys(["cpu", DEVICE])))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mask_layout", ["geometry", "head_axis", "batched", "packed"])
def test_layer_matches_reference(device, dtype, mask_layout):
    """Every mask layout the model can pass, on the kernel and on the CPU fallback, with fp32 attention."""
    if device == "cpu" and mask_layout == "packed":
        pytest.skip("packing needs an accelerator")
    torch.manual_seed(7)
    attention = _StubAttention(64, 2).to(device=device, dtype=dtype).eval()
    hidden_states = torch.randn(2, 3, 65, 64, device=device, dtype=dtype)
    # One geometry mask per ensemble member; only the batched layout lets them differ.
    masks = [banded_mask(3, 65, 0.05, device), banded_mask(3, 65, 0.1, device)]
    if mask_layout == "batched":
        banded = torch.cat(masks)[:, None]
    else:
        masks = masks[:1] * 2
        banded = masks[0]
    if mask_layout == "head_axis":
        banded = banded[:, None]
    elif mask_layout == "packed":
        banded = prepare_mask(banded)

    with torch.inference_mode():
        query, key, value = attention.projected(hidden_states)
        expected = torch.cat(
            [
                reference(
                    query[i : i + 1],
                    key[i : i + 1],
                    value[i : i + 1],
                    masks[i],
                    attention.scaling,
                )
                for i in range(2)
            ]
        )
        expected = attention.o_proj(expected.to(dtype).transpose(2, 3).reshape_as(hidden_states))
        actual, _ = attention(hidden_states, banded)

    assert actual.dtype == dtype
    # Attention runs in fp32 either way; bf16 only rounds the projections around it.
    tolerance = {"atol": 2e-5, "rtol": 2e-5} if dtype == torch.float32 else {"atol": 1e-3, "rtol": 1.6e-2}
    torch.testing.assert_close(actual, expected, **tolerance)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("batched_mask", [False, True])
def test_backward_reaches_every_projection(batched_mask):
    """Gradient-enabled calls must use the differentiable reference."""
    torch.manual_seed(0)
    attention = _StubAttention(32, 2)
    hidden_states = torch.randn(2, 3, 16, 32, requires_grad=True)
    mask = banded_mask(3, 16, 0, device="cpu")
    if batched_mask:
        mask = mask.repeat(2, 1, 1)

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
def test_attention_on_noncurrent_device():
    backend = torch.get_device_module(DEVICE)
    if backend.device_count() < 2:
        pytest.skip("requires two accelerators")
    query, key, value, mask, scaling = attention_case((1, 3, 2, 65, 32), device=torch.device(DEVICE, 1))
    with backend.device(0):
        actual = banded_attention(query, key, value, prepare_mask(mask), scaling, precision="ieee")
        assert_matches_reference(actual, query, key, value, mask, scaling)
        assert actual.device == query.device
        assert backend.current_device() == 0


@requires_accelerator
@pytest.mark.kernels_ci
@pytest.mark.parametrize("prepared", [False, True])
def test_blockwise_attention_compile(prepared, reset_dynamo):
    query, key, value, mask, scaling = attention_case((2, 3, 2, 65, 32))
    banded = prepare_mask(mask) if prepared else mask
    compiled = torch.compile(kernel.layers._blockwise_attention, fullgraph=True)
    with torch.inference_mode():
        for _ in range(2):
            query = torch.randn_like(query)
            assert_matches_reference(
                compiled(query, key, value, banded, scaling),
                query,
                key,
                value,
                mask,
                scaling,
            )


@requires_nvidia
@pytest.mark.kernels_ci
def test_tf32x3_matches_ieee():
    query, key, value, mask, scaling = attention_case((2, 3, 2, 129, 128))
    expected = banded_attention(query, key, value, mask, scaling, precision="ieee")
    actual = banded_attention(query, key, value, mask, scaling, precision="tf32x3")
    torch.testing.assert_close(actual, expected)


@pytest.mark.kernels_ci
def test_layer_precision_follows_set_float32_matmul_precision(float32_matmul_precision):
    precision = kernel.layers._float32_precision
    on_rocm = torch.version.hip is not None
    float32_matmul_precision("high")
    assert precision(torch.device("cuda")) == ("ieee" if on_rocm else "default")
    float32_matmul_precision("highest")
    assert precision(torch.device("cuda")) == ("ieee" if on_rocm else "tf32x3")


@requires_nvidia
@pytest.mark.kernels_ci
def test_compiled_attention_follows_precision_changes(float32_matmul_precision, reset_dynamo):
    """The precision is a Dynamo-guarded constant, so changing it must recompile rather than reuse the graph."""
    query, key, value, mask, scaling = attention_case((2, 3, 2, 65, 32))
    prepared = prepare_mask(mask)
    compiled = torch.compile(kernel.layers._blockwise_attention, fullgraph=True)

    with torch.inference_mode():
        for setting, mode in (
            ("highest", "tf32x3"),
            ("high", "default"),
            ("highest", "tf32x3"),
        ):
            float32_matmul_precision(setting)
            expected = banded_attention(query, key, value, prepared, scaling, precision=mode)
            assert torch.equal(compiled(query, key, value, prepared, scaling), expected)
