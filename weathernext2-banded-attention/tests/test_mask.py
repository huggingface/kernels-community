"""The mask layer prepares one compact geometry mask shared by all attention layers."""

from types import MethodType, SimpleNamespace

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


def decode_mask(mask):
    """Independent CPU decoder of the packed representation; never used by inference."""
    packed = mask.packed.cpu().to(torch.int64)
    tiles, offsets = (tensor.cpu() for tensor in (mask.tiles, mask.offsets))
    decoded = torch.zeros(mask.shape, dtype=torch.bool)
    block_size = mask.shape[1]
    query_tiles = (block_size + 63) // 64
    for block in range(mask.shape[0]):
        for query_tile in range(query_tiles):
            start = query_tile * 64
            rows = min(64, block_size - start)
            for neighbour in range(3):
                slot = (block * query_tiles + query_tile) * 3 + neighbour
                for index in range(int(offsets[slot]), int(offsets[slot + 1])):
                    column = int(tiles[index]) * 32
                    columns = min(32, block_size - column)
                    words = packed[index, :rows]
                    bits = ((words[:, None] >> torch.arange(columns)) & 1).bool()
                    column += neighbour * block_size
                    decoded[block, start : start + rows, column : column + columns] = bits
    return decoded


@pytest.mark.kernels_ci
@pytest.mark.parametrize("device", list(dict.fromkeys(["cpu", DEVICE])))
@pytest.mark.parametrize("inference_mode", [False, True])
def test_inference_prepares_one_shared_mask(device, inference_mode):
    layer = MaskPreparer().eval()
    layer.forward = MethodType(kernel.WeatherNext2AttentionMask.forward, layer)
    mask = torch.rand(3, 1, 65, 195, device=device) > 0.5
    mask[0, :, :, :65] = False
    mask[-1, :, :, 130:] = False
    with torch.inference_mode(inference_mode), torch.no_grad():
        banded = layer(mask, 2, torch.float32)
    assert banded.shape == (3, 65, 195)
    if device == "cpu":
        assert banded.data_ptr() == mask.data_ptr()
    else:
        assert banded.packed.dtype == torch.uint32
        torch.testing.assert_close(decode_mask(banded), mask[:, 0].cpu())
    assert list(layer.buffers()) == []


@requires_accelerator
@pytest.mark.kernels_ci
@pytest.mark.parametrize("empty", [False, True])
def test_packed_mask_roundtrip(empty):
    storage = torch.rand(3, 65, 390, device=DEVICE) > 0.5
    mask = storage[..., ::2]
    mask[0, :, :65] = False
    mask[-1, :, 130:] = False
    if empty:
        mask.zero_()
    prepared = kernel.layers._prepare_mask(mask)
    assert prepared.tiles.ndim == 1
    assert prepared.tiles.numel() == prepared.packed.shape[0] == int(prepared.offsets[-1])
    assert prepared.offsets.numel() == 3 * 2 * 3 + 1
    if empty:
        assert prepared.tiles.numel() == prepared.packed.numel() == 0
    torch.testing.assert_close(decode_mask(prepared), mask.cpu(), atol=0, rtol=0)


@requires_accelerator
@pytest.mark.kernels_ci
def test_sparse_mask_stores_only_active_tiles():
    blocks, block_size = 3, 129
    mask = torch.zeros(blocks, block_size, 3 * block_size, dtype=torch.bool, device=DEVICE)
    mask[:, :, block_size : 2 * block_size] = torch.eye(block_size, dtype=torch.bool, device=DEVICE)
    prepared = kernel.layers._prepare_mask(mask)
    assert prepared.tiles.numel() == blocks * ((block_size + 31) // 32)
    assert prepared.packed.shape == (prepared.tiles.numel(), 64)
    assert prepared.offsets[0] == 0
    assert torch.all(prepared.offsets[1:] >= prepared.offsets[:-1])
    storage_bytes = sum(tensor.numel() * tensor.element_size() for tensor in prepared[:3])
    assert storage_bytes < mask.numel() * mask.element_size() // 8
    torch.testing.assert_close(decode_mask(prepared), mask.cpu(), atol=0, rtol=0)


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
