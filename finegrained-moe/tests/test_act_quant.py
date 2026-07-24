"""Tests for the standalone ``fp8_act_quant_tensor_wide`` op."""

import pytest
import torch

from utils import TEST_DEVICE  # type: ignore

import triton
import triton.language as tl

from finegrained_moe.quant import fp8_act_quant_block_dynamic, fp8_act_quant_tensor_wide, mx_act_quant_inline, mxfp4_act_quant, mxfp8_act_quant, nvfp4_act_quant  # type: ignore


_FP8_DTYPE = torch.float8_e4m3fn


def _ref_fp8_act_quant_tensor_wide(x: torch.Tensor, block_size: int):
    """Pure-PyTorch reference: per-block dynamic FP8 quant.

    ``s = amax / 448`` (returned verbatim, can be 0 for all-zero blocks);
    the divider is floored at 1e-12 so all-zero blocks emit zeros, not NaN.
    """
    *prefix, K = x.shape
    n_blocks = K // block_size
    groups = x.reshape(*prefix, n_blocks, block_size).float()
    s = groups.abs().amax(dim=-1) / 448.0
    y = (groups / s.unsqueeze(-1).clamp(min=1e-12)).to(_FP8_DTYPE)
    return y.reshape(*prefix, K), s


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="Accelerator not available")
@pytest.mark.parametrize(
    "shape",
    [(16, 128), (1, 256), (4, 8, 128)],
    ids=lambda s: "x".join(map(str, s)),
)
@pytest.mark.parametrize("block_size", [32, 64, 128], ids=lambda b: f"bs{b}")
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float16, torch.float32],
    ids=["bf16", "fp16", "fp32"],
)
def test_fp8_act_quant_tensor_wide(shape, block_size, dtype):
    """Kernel matches the pure-PyTorch reference at FP8 granularity."""
    if shape[-1] % block_size != 0:
        pytest.skip(f"shape last dim {shape[-1]} not divisible by {block_size}")
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype, device=TEST_DEVICE)

    y, s = fp8_act_quant_tensor_wide(x, block_size)
    y_ref, s_ref = _ref_fp8_act_quant_tensor_wide(x, block_size)

    assert y.dtype == _FP8_DTYPE
    assert y.shape == x.shape
    assert s.dtype == torch.float32
    assert s.shape == x.shape[:-1] + (x.shape[-1] // block_size,)
    torch.testing.assert_close(s, s_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(y.float(), y_ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="Accelerator not available")
def test_fp8_act_quant_zero_block():
    """All-zero input emits zero quantized output (not NaN) and a zero scale.

    Regression test for the ``tl.maximum(s, 1e-12)`` divider floor: without it,
    a zero block divides by zero and produces NaN.
    """
    x = torch.zeros(2, 128, dtype=torch.bfloat16, device=TEST_DEVICE)
    y, s = fp8_act_quant_tensor_wide(x, block_size=128)
    assert not torch.isnan(y.float()).any()
    assert (y.float() == 0).all()
    assert (s == 0).all()


# ── inline-vs-offline arm parity (every recipe's quant has two implementations) ──────

E2M1_LUT = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _dequant_e2m1(packed, scales, K):
    p = packed.view(torch.uint8)
    lut = torch.tensor(E2M1_LUT, device=packed.device)
    vals = torch.empty(p.shape[0], K, device=p.device)
    vals[:, 0::2] = lut[(p & 0xF).long()]
    vals[:, 1::2] = lut[(p >> 4).long()]
    s = torch.pow(2.0, scales.view(torch.uint8).float() - 127.0)
    return (vals.reshape(-1, K // 32, 32) * s[..., None]).reshape(-1, K)


def _ref_mxfp4_act_quant(x: torch.Tensor):
    """Pure-PyTorch MXFP4 reference: UE8M0 group-32 scale (amax/6 ceil'd to a power of
    two via the exponent-bump bit trick), E2M1 rounding via bucketize, nibble pack."""
    T, K = x.shape
    groups = x.float().reshape(T, K // 32, 32)
    amax = groups.abs().amax(dim=-1)
    bits = (amax / 6.0).view(torch.int32)
    exp_ceil = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    exp_ceil = exp_ceil.clamp(1, 254)
    exp_ceil = torch.where(amax == 0, torch.full_like(exp_ceil, 127), exp_ceil)
    scaled = groups / torch.pow(2.0, exp_ceil.float() - 127.0)[..., None]
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    codes = torch.bucketize(scaled.abs().reshape(T, K), boundaries).to(torch.uint8)
    codes |= (scaled.reshape(T, K) < 0).to(torch.uint8) << 3
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, exp_ceil.to(torch.uint8)


def _ref_nvfp4_act_quant(x: torch.Tensor):
    """Pure-PyTorch NVFP4 reference: E4M3 group-16 scale (amax/6, no power-of-two ceil),
    values divided by the DECODED scale, E2M1 rounding, nibble pack."""
    T, K = x.shape
    groups = x.float().reshape(T, K // 16, 16)
    amax = groups.abs().amax(dim=-1)
    scales = (amax / 6.0).to(torch.float8_e4m3fn)
    scaled = groups / scales.float().clamp(min=2.0**-127)[..., None]
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    codes = torch.bucketize(scaled.abs().reshape(T, K), boundaries).to(torch.uint8)
    codes |= (scaled.reshape(T, K) < 0).to(torch.uint8) << 3
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, scales


# M=100 does not divide the row-tile BLOCK_T {8,16,32,64} — exercises the tail mask.
@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
@pytest.mark.parametrize("M", [64, 100], ids=["Tdiv", "Ttail"])
def test_mxfp4_act_quant_matches_torch_reference(M):
    """The Triton one-pass quant must be bit-identical to an INDEPENDENT pure-PyTorch
    implementation (the inline/offline parity tests share kernel code, so only this
    catches a shared bug)."""
    x = _act_inputs(M=M)
    q, s = mxfp4_act_quant(x)
    q_ref, s_ref = _ref_mxfp4_act_quant(x)
    assert torch.equal(s, s_ref)
    assert torch.equal(q.view(torch.uint8), q_ref)


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
@pytest.mark.parametrize("M", [64, 100], ids=["Tdiv", "Ttail"])
def test_nvfp4_act_quant_matches_torch_reference(M):
    """Same independent-reference check for the NVFP4 quant (E4M3 group-16 scales)."""
    x = _act_inputs(M=M)
    q, s = nvfp4_act_quant(x)
    q_ref, s_ref = _ref_nvfp4_act_quant(x)
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8))
    assert torch.equal(q.view(torch.uint8), q_ref)


@triton.jit
def _mx_inline_harness(X, Q, S, M: tl.constexpr, N: tl.constexpr, RECIPE: tl.constexpr):
    x = tl.load(X + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :])
    q, s = mx_act_quant_inline(x.to(tl.float32), M, N, 32, RECIPE)
    width: tl.constexpr = N // 2 if RECIPE == "mxfp4" else N
    tl.store(Q + tl.arange(0, M)[:, None] * width + tl.arange(0, width)[None, :], q)
    tl.store(
        S + tl.arange(0, M)[:, None] * (N // 32) + tl.arange(0, N // 32)[None, :], s
    )


def _run_mx_inline(x, recipe):
    M, N = x.shape
    width = N // 2 if recipe == "mxfp4" else N
    q = torch.empty(
        M,
        width,
        device=x.device,
        dtype=torch.uint8 if recipe == "mxfp4" else torch.float8_e4m3fn,
    )
    s = torch.empty(M, N // 32, device=x.device, dtype=torch.uint8)
    _mx_inline_harness[(1,)](x, q, s, M=M, N=N, RECIPE=recipe, num_warps=4)
    torch.cuda.synchronize()
    return q, s


def _act_inputs(M=64, N=256, zero_rows=True):
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32) * 8
    if zero_rows:
        x[0] = 0  # whole-row zero: scales must stay neutral, values must dequant to 0
        x[1, :32] = 0  # one zero group inside a live row
    return x


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
def test_mxfp8_inline_matches_offline():
    """The in-kernel E4M3 quant (fused epilogues, inline decode arm) and the offline
    ``mxfp8_act_quant`` pass must be bit-identical — the dtype-branched kernels and the
    fused/unfused parity both rely on it."""
    x = _act_inputs()
    q, s = _run_mx_inline(x, "mxfp8")
    q_ref, s_ref = mxfp8_act_quant(x)
    assert torch.equal(s, s_ref)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
def test_mxfp4_inline_matches_host():
    """The in-kernel packed-E2M1 quant (the "mxfp4" output recipe) and the host
    ``mxfp4_act_quant`` must be bit-identical — the fused intermediate must equal
    quantizing the same values offline."""
    x = _act_inputs()
    q, s = _run_mx_inline(x, "mxfp4")
    q_ref, s_ref = mxfp4_act_quant(x)
    assert torch.equal(s, s_ref)
    assert torch.equal(q, q_ref.view(torch.uint8))


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
def test_nvfp4_inline_matches_offline():
    """The in-kernel NVFP4 quant and the one-pass offline kernel share
    ``mx_act_quant_inline``'s arm, but exercise different tile widths — the packed
    bytes and E4M3 scales must be bit-identical."""
    x = _act_inputs()
    M, N = x.shape
    q = torch.empty(M, N // 2, device="cuda", dtype=torch.uint8)
    s = torch.empty(M, N // 16, device="cuda", dtype=torch.float8_e4m3fn)
    _nv_inline_harness[(1,)](x, q, s, M=M, N=N, num_warps=4)
    torch.cuda.synchronize()
    q_ref, s_ref = nvfp4_act_quant(x)
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8))
    assert torch.equal(q, q_ref.view(torch.uint8))


@triton.jit
def _nv_inline_harness(X, Q, S, M: tl.constexpr, N: tl.constexpr):
    x = tl.load(X + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :])
    q, s = mx_act_quant_inline(x.to(tl.float32), M, N, 16, "nvfp4")
    tl.store(Q + tl.arange(0, M)[:, None] * (N // 2) + tl.arange(0, N // 2)[None, :], q)
    tl.store(
        S + tl.arange(0, M)[:, None] * (N // 16) + tl.arange(0, N // 16)[None, :], s
    )


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
def test_mxfp4_round_trip_bound():
    """Dequantizing a host-quantized tensor stays within one E2M1 grid step of the
    input (half the largest gap, 1.0, times each group's scale) — and zeros round-trip
    to exact zeros."""
    x = _act_inputs()
    q, s = mxfp4_act_quant(x)
    xdq = _dequant_e2m1(q, s, x.shape[1])
    bound = torch.pow(2.0, s.float().amax() - 127.0).item() * 1.0
    assert (xdq - x).abs().max().item() <= bound
    assert (xdq[0] == 0).all()


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
def test_block_dynamic_offline_zero_block():
    """A zero block must produce a finite scale and exact-zero values (the 1e-12 floor),
    with live blocks in the same row unaffected."""
    x = _act_inputs()
    q, s = fp8_act_quant_block_dynamic(x.to(torch.bfloat16), 128)
    assert torch.isfinite(s).all()
    assert (q[0].float() == 0).all()
    assert (q[2:].float() != 0).any()


# T values that don't divide the row-tile BLOCK_T {8,16,32,64} cover the tail mask; the
# one-row-per-program predecessor never had one, so this guards the row-tile rewrite.
@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
@pytest.mark.parametrize(
    "shape",
    [(64, 256), (100, 256), (3, 128), (129, 512), (1, 384)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_fp8_act_quant_block_dynamic_matches_reference(shape):
    """``fp8_act_quant_block_dynamic`` matches the pure-PyTorch per-row, per-block_k
    reference across token counts — including ones that leave a partial row tile."""
    T, K = shape
    block_k = 128
    torch.manual_seed(0)
    x = torch.randn(T, K, device=TEST_DEVICE, dtype=torch.bfloat16)
    y, s = fp8_act_quant_block_dynamic(x, block_k)
    y_ref, s_ref = _ref_fp8_act_quant_tensor_wide(x, block_k)
    assert y.shape == (T, K)
    assert s.shape == (T, K // block_k)
    torch.testing.assert_close(s, s_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(y.float(), y_ref.float(), atol=1e-2, rtol=1e-2)
