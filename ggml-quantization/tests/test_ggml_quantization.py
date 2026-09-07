"""Both ops against `gguf.quants.dequantize`, the reference implementation of the block layouts.

The blocks are random bytes rather than a quantized tensor: `gguf` can unpack every type but only
pack a couple of them, and unpacking is defined for any byte pattern, so this needs no quantizer and
no checkpoint. The one constraint is that a block's scales are fp16 fields — masked below so a random
pattern cannot land on an exponent of all ones and make the whole block inf/nan.
"""

import numpy as np
import pytest
import torch

from ggml_quantization import GEMV_TYPES, MAX_GEMV_ROWS, dequantize, mul_mat_vec


gguf = pytest.importorskip("gguf", reason="the reference unpacker comes from the `gguf` package")

DEVICE = "cuda" if torch.cuda.is_available() else "mps"

# name -> (ggml type id, values per block, bytes per block), taken from gguf's own quant sizes so a
# block layout is never restated here. Q1_0/Q2_0/NVFP4 are missing: the `gguf` release this tests
# against cannot unpack them, so there is no reference to compare a kernel to.
QUANT_TYPES = {
    name: (int(t), *gguf.GGML_QUANT_SIZES[t])
    for name, t in (
        (n, getattr(gguf.GGMLQuantizationType, n, None))
        for n in (
            "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
            "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
            "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL",
            "IQ3_S", "IQ2_S", "IQ4_XS", "IQ1_M", "MXFP4",
        )
    )
    if t is not None
}

# Which of those this build actually implements a gemv for. Asked rather than assumed: the two
# backends do not cover the same set, and a type routed into a gemv it has no kernel for faults.
SUPPORTED = {name: v for name, v in QUANT_TYPES.items() if v[0] in GEMV_TYPES}


def random_blocks(rows: int, cols: int, ggml_name: str, device=DEVICE):
    """Random blocks and the values `gguf` reads out of them."""
    _, block_values, block_bytes = QUANT_TYPES[ggml_name]
    generator = np.random.default_rng(0)
    packed = generator.integers(0, 256, (rows, cols // block_values * block_bytes), dtype=np.uint8)
    if ggml_name == "MXFP4":
        # The only type whose scale is not an fp16 field: one E8M0 byte per block, worth
        # 2^(byte-127), so a random byte is worth up to 2^128 and the block overflows to inf in the
        # reference before a kernel is even involved. Held near 2^0 instead.
        packed[:, ::block_bytes] = generator.integers(120, 135, packed[:, ::block_bytes].shape)
    else:
        # every fp16 scale sits at an even offset in its block, so clearing bit 6 of each odd byte
        # keeps every possible fp16 field finite whatever the rest of the pattern is
        packed[:, 1::2] &= 0xBF

    quant_type = getattr(gguf.GGMLQuantizationType, ggml_name)
    reference = gguf.quants.dequantize(packed.reshape(-1), quant_type).reshape(rows, cols)
    assert np.isfinite(reference).all(), f"the {ggml_name} reference is not finite"
    return torch.from_numpy(packed).to(device), torch.from_numpy(reference).to(device)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("ggml_name", SUPPORTED)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dequantize_matches_reference(ggml_name, dtype):
    ggml_type = SUPPORTED[ggml_name][0]
    rows, cols = 64, 512
    blocks, reference = random_blocks(rows, cols, ggml_name)

    out = dequantize(blocks, ggml_type, rows, cols, dtype)

    assert out.shape == (rows, cols) and out.dtype == dtype
    # The kernel writes `dtype` directly, so the tolerance is that dtype's own resolution -- but
    # measured against the scale of the values, not each element. Every value in a block shares one
    # scale, so a dequantizer's error is a fraction of that scale; asking a near-zero element to
    # match to its own magnitude asks for an exactness even a different summation order breaks.
    # Most types do come back bit-exact; Q4_0/Q4_1/Q4_K land within an ulp or so of f32.
    scale = reference.abs().max()
    torch.testing.assert_close(
        out.float(), reference, rtol=0, atol=torch.finfo(dtype).eps * 4 * scale
    )


@pytest.mark.kernels_ci
@pytest.mark.parametrize("ggml_name", SUPPORTED)
@pytest.mark.parametrize("n_rows", [1, MAX_GEMV_ROWS])
def test_mul_mat_vec_matches_matmul(ggml_name, n_rows):
    ggml_type = SUPPORTED[ggml_name][0]
    out_features, in_features = 128, 512
    blocks, reference = random_blocks(out_features, in_features, ggml_name)
    x = torch.randn(n_rows, in_features, dtype=torch.bfloat16, device=DEVICE)

    out = mul_mat_vec(blocks, x, ggml_type, out_features)

    assert out.shape == (n_rows, out_features) and out.dtype == torch.float32
    # the kernel quantizes the activations to q8_1, so this is close to a matmul, not equal to one
    expected = x.float() @ reference.T
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2 * expected.abs().max())


@pytest.mark.kernels_ci
def test_gemv_is_compileable():
    """A graph break here would cost more than the kernel saves, so the fake has to be right."""
    ggml_type = SUPPORTED["Q4_K"][0]
    out_features, in_features = 128, 512
    blocks, reference = random_blocks(out_features, in_features, "Q4_K")
    x = torch.randn(1, in_features, dtype=torch.bfloat16, device=DEVICE)

    compiled = torch.compile(
        lambda t: mul_mat_vec(blocks, t, ggml_type, out_features), fullgraph=True
    )

    expected = x.float() @ reference.T
    torch.testing.assert_close(compiled(x), expected, rtol=2e-2, atol=2e-2 * expected.abs().max())
