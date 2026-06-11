# Self-contained test for the low-level Triton `apply_rotary` op.
#
# The upstream `tests/test_rotary.py` exercises the high-level
# `flash_attn.layers.rotary` module, which we do not vendor. Here we test the
# vendored Triton kernel directly against a pure-PyTorch reference.

import pytest
import torch

from flash_attn_ops import apply_rotary


def rotary_ref(x, cos, sin, interleaved=False):
    """Pure-PyTorch reference for `apply_rotary` (non-varlen, seqlen_offsets=0)."""
    # x: (batch, seqlen, nheads, headdim); cos/sin: (seqlen, rotary_dim // 2)
    seqlen = x.shape[1]
    rotary_dim = cos.shape[-1] * 2
    cos = cos[:seqlen]
    sin = sin[:seqlen]
    # broadcast to (1, seqlen, 1, rotary_dim // 2)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    xr = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    if interleaved:
        x1 = xr[..., 0::2]
        x2 = xr[..., 1::2]
    else:
        x1, x2 = xr.chunk(2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    if interleaved:
        out_r = torch.stack([o1, o2], dim=-1).flatten(-2)
    else:
        out_r = torch.cat([o1, o2], dim=-1)
    return torch.cat([out_r, x_pass], dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("headdim,rotary_dim", [(64, 64), (128, 64), (256, 192)])
def test_apply_rotary(headdim, rotary_dim, interleaved, dtype):
    torch.manual_seed(0)
    batch, seqlen, nheads = 2, 37, 4
    device = "cuda"
    x = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
    angle = torch.randn(seqlen, rotary_dim // 2, device=device, dtype=torch.float32)
    cos, sin = torch.cos(angle), torch.sin(angle)

    out = apply_rotary(x, cos.to(dtype), sin.to(dtype), interleaved=interleaved)
    ref = rotary_ref(x.float(), cos, sin, interleaved=interleaved).to(dtype)

    atol = 1e-3 if dtype == torch.float32 else 1e-2
    assert torch.allclose(out, ref, atol=atol, rtol=1e-2)
