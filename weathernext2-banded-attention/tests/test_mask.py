"""The model owns one prepared geometry mask, shared by every attention layer."""

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import kernels
import pytest
import torch
from torch import nn


kernel = kernels.get_kernel("kernels-community/weathernext2-banded-attention", version=2)
DEVICE = kernel.infer_device()
requires_accelerator = pytest.mark.skipif(DEVICE == "cpu", reason="the kernel needs an accelerator")


class MaskPreparer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="sdpa")

    def forward(self, mask, batch_size, dtype):
        return mask


@requires_accelerator
@pytest.mark.kernels_ci
@pytest.mark.parametrize("inference_mode", [False, True])
def test_model_owned_cache(inference_mode):
    layer = MaskPreparer().eval()
    layer.forward = MethodType(kernel.WeatherNext2AttentionMask.forward, layer)
    mask = torch.rand(3, 1, 65, 195, device=DEVICE) > 0.5
    with patch.object(kernel.layers, "_prepare_mask", wraps=kernel.layers._prepare_mask) as prepare:
        with torch.inference_mode(inference_mode), torch.no_grad():
            first = layer(mask, 1, torch.float32)
            second = layer(mask, 2, torch.float32)
            assert first.packed is second.packed
            assert prepare.call_count == 1
        mask.zero_()
        with torch.inference_mode(inference_mode), torch.no_grad():
            third = layer(mask, 1, torch.float32)
            assert third.packed is not first.packed
            assert prepare.call_count == 2
            assert third.packed.numel() == 0
            assert layer._prepared_packed is third.packed


@requires_accelerator
@pytest.mark.kernels_ci
def test_prepared_mask_follows_the_module():
    layer = MaskPreparer().eval()
    layer.forward = MethodType(kernel.WeatherNext2AttentionMask.forward, layer)
    mask = torch.rand(3, 1, 65, 195, device=DEVICE) > 0.5
    with torch.no_grad():
        prepared = layer(mask, 1, torch.float32)
    assert prepared.packed.numel() > 0
    # Non-persistent: never saved, but moved with the module, so `.cpu()` releases the device copy.
    assert not any(name.startswith("_prepared_") for name in layer.state_dict())
    layer.cpu()
    assert all(getattr(layer, f"_prepared_{name}").device.type == "cpu" for name in ("packed", "tiles", "counts"))


@requires_accelerator
@pytest.mark.kernels_ci
def test_inference_tensor_geometry_is_not_cached():
    layer = MaskPreparer().eval()
    layer.forward = MethodType(kernel.WeatherNext2AttentionMask.forward, layer)
    with patch.object(kernel.layers, "_prepare_mask", wraps=kernel.layers._prepare_mask) as prepare:
        with torch.inference_mode():
            # Created under inference mode, so it has no version counter to say it changed.
            mask = torch.rand(3, 1, 65, 195, device=DEVICE) > 0.5
            layer(mask, 1, torch.float32)
            layer(mask, 1, torch.float32)
    assert prepare.call_count == 2


@requires_accelerator
@pytest.mark.kernels_ci
def test_prepared_mask_needs_packed_words():
    mask = torch.ones(2, 32, 96, dtype=torch.bool, device=DEVICE)
    prepared = kernel.layers._prepare_mask(mask, sparse_tiles=True, packed_mask=False)
    query = torch.zeros(1, 2, 2, 32, 32, device=DEVICE)
    with pytest.raises(ValueError, match="needs its packed mask"):
        kernel.banded_attention(query, query, query, prepared, 32**-0.5)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("reason", ["training", "gradients", "flex"])
def test_standard_mask_fallback(reason):
    layer = MaskPreparer().eval()
    layer.forward = MethodType(kernel.WeatherNext2AttentionMask.forward, layer)
    if reason == "training":
        layer.train()
    if reason == "flex":
        layer.config._attn_implementation = "flex_attention"
    mask = torch.ones(3, 1, 65, 195, dtype=torch.bool)
    with torch.set_grad_enabled(reason == "gradients"):
        assert layer(mask, 1, torch.float32) is mask
    assert not hasattr(layer, "_prepared_key")
