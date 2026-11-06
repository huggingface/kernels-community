import pytest
import torch

import kernels

triton_kernels = kernels.get_kernel("kernels-community/gpt-oss-triton-kernels", version=1)

BlackwellMXScaleLayout = triton_kernels._private_for_testing.layout.BlackwellMXScaleLayout

# ------------------------------------------------------------
# Torch tests
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (3, 4096, 1024),
        (10, 254, 60),
        (1, 320, 160),
        (2, 16, 512),
        (3, 2, 36),
    ],
)
def test_mxfp4_scale_roundtrip(shape):
    x = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    layout = BlackwellMXScaleLayout(x.shape)
    res = layout.unswizzle_data(layout.swizzle_data(x))
    assert (res == x).all()
