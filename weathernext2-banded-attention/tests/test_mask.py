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
            assert first is second
            assert prepare.call_count == 1
        mask.zero_()
        with torch.inference_mode(inference_mode), torch.no_grad():
            third = layer(mask, 1, torch.float32)
            assert third is not first
            assert prepare.call_count == 2
            assert third.packed.numel() == 0
            assert layer._prepared_attention_mask[1] is third


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
    assert not hasattr(layer, "_prepared_attention_mask")
